#!/usr/bin/env python3
"""
Retrain Trajectron++ on our ETH/UCY leave-one-out splits.

Wrapper around the vendored ``trajectron/train.py``. Everything it needs that
CANVAS did not ship is supplied here, so no vendored file is edited:

  * ``sys.path`` -- upstream ``train.py`` uses root-level imports
    (``model``, ``environment``, ``utils``, ``visualization``), so the
    trajectron package dir goes on the path.
  * a stub ``cp.adaptive_cp`` -- ``trajectron/visualization`` imports it, but
    that module exists nowhere in CANVAS (so upstream ``train.py`` cannot run
    as shipped). It is only used for optional prediction plots.
  * Environment pickles -- produced by ``kpp.baselines.trajectron_data``;
    this script generates them on demand if missing.

Usage:
    python scripts/train_trajectron.py zara1 --epochs 1        # smoke test
    python scripts/train_trajectron.py zara1 --epochs 100      # full run
    python scripts/train_trajectron.py --all --epochs 100

Outputs land in ``runs/trajectron/<scene>/`` as ``model_registrar-<epoch>.pt``
plus ``config.json`` -- the layout ``trajectron_eval.load_trajectron`` reads.
Evaluate a retrained model with:

    from kpp.baselines.trajectron_eval import evaluate_trajectron
    evaluate_trajectron("zara1", model_dir="runs/trajectron/zara1", ts=<epoch>)
"""
import argparse
import pathlib
import runpy
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "kpp" / "baselines" / "vendor"
TRAJ_DIR = VENDOR / "canvas" / "predictors" / "trajectron"
DEFAULT_CONF = VENDOR / "assets" / "models" / "trajectron" / "zara01_vel_ar3" / "config.json"
PKL_ROOT = ROOT / "data" / "trajectron"
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def _install_stubs() -> None:
    """Reconcile upstream's two incompatible import styles, plus the missing ``cp``.

    ``train.py`` imports root-level (``from model.trajectron import ...``) but the
    model files themselves use subpackage-relative imports (``from ...utils import
    block_diag``). Putting the trajectron dir on ``sys.path`` satisfies the first
    and breaks the second ("relative import beyond top-level package") -- which is
    why upstream ``train.py`` cannot run as shipped. So we import the real
    subpackages and alias them under the root-level names instead.
    """
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))

    # stub the nonexistent `cp` first: `visualization` imports it at module load
    if "cp" not in sys.modules:
        cp = types.ModuleType("cp")
        acp = types.ModuleType("cp.adaptive_cp")

        class AdaptiveConformalPredictionModule:  # pragma: no cover - stub
            def __init__(self, *a, **kw):
                pass

            def __getattr__(self, _):
                return lambda *a, **kw: None

        acp.AdaptiveConformalPredictionModule = AdaptiveConformalPredictionModule
        cp.adaptive_cp = acp
        sys.modules["cp"] = cp
        sys.modules["cp.adaptive_cp"] = acp

    import importlib
    base = "canvas.predictors.trajectron"
    for alias in ("model", "environment", "utils", "visualization",
                  "evaluation", "argument_parser"):
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(f"{base}.{alias}")
        except ImportError as e:
            print(f"  [warn] could not alias '{alias}': {e}")
    # submodules referenced as `model.x` must resolve through the alias too
    for sub in ("trajectron", "model_registrar", "model_utils", "dataset"):
        name = f"model.{sub}"
        if name not in sys.modules:
            try:
                sys.modules[name] = importlib.import_module(f"{base}.model.{sub}")
            except ImportError as e:
                print(f"  [warn] could not alias '{name}': {e}")


def _ensure_pickles(scene: str) -> pathlib.Path:
    """Make sure the dill Environments for this scene exist; build if not."""
    sys.path.insert(0, str(ROOT))
    from kpp.baselines.trajectron_data import dump_split

    d = PKL_ROOT / scene
    for phase in ("train", "val"):
        p = d / f"{scene}_{phase}.pkl"
        if not p.exists():
            print(f"  building {p} ...")
            dump_split(scene, phase, p)
    return d


def train(scene: str, epochs: int, device: str, conf: str, workers: int) -> None:
    data_dir = _ensure_pickles(scene)
    log_dir = ROOT / "runs" / "trajectron" / scene
    log_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        str(TRAJ_DIR / "train.py"),
        "--conf", conf,
        "--data_dir", str(data_dir),
        "--train_data_dict", f"{scene}_train.pkl",
        "--eval_data_dict", f"{scene}_val.pkl",
        "--log_dir", str(log_dir),
        "--log_tag", scene,
        "--train_epochs", str(epochs),
        "--device", device,
        "--eval_device", device,
        "--preprocess_workers", str(workers),
    ]
    print(f"\n=== training Trajectron++ on '{scene}' "
          f"({epochs} epochs, {device}) -> {log_dir} ===")
    _install_stubs()
    runpy.run_path(str(TRAJ_DIR / "train.py"), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=None,
                    help="scenes to train (default: zara1)")
    ap.add_argument("--all", action="store_true", help="train every ETH/UCY scene")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", default=str(DEFAULT_CONF))
    ap.add_argument("--workers", type=int, default=4)
    # internal: train exactly one scene in this process (used by the dispatcher)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    scenes = SCENES if args.all else (args.scenes or ["zara1"])

    if args._worker:
        # one scene, this process only -- no cross-scene module state to leak
        assert len(scenes) == 1, "--_worker takes exactly one scene"
        train(scenes[0].strip().lower(), args.epochs, args.device, args.conf, args.workers)
        return

    # dispatcher: one FRESH subprocess per scene. The vendored trajectron
    # `argument_parser` parses sys.argv at import time and is then cached in
    # sys.modules, so training a second scene in-process silently reuses the
    # first scene's args. A separate interpreter per scene avoids that entirely.
    for sc in scenes:
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               sc.strip().lower(), "--_worker",
               "--epochs", str(args.epochs), "--device", args.device,
               "--conf", args.conf, "--workers", str(args.workers)]
        print(f"\n>>> dispatching '{sc}' -> {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! scene '{sc}' exited with code {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
