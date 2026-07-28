import os
import json
import torch
import sys

# The vendored Trajectron code occasionally uses root-level imports (e.g.
# ``from model.trajectron import ...``). Put the vendored ``trajectron`` package
# dir on sys.path so those resolve. Previously this was a hard-coded absolute
# path to a specific machine; use a path relative to this file instead so the
# repo is portable after cloning.
_TRAJECTRON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectron")
if _TRAJECTRON_DIR not in sys.path:
    sys.path.append(_TRAJECTRON_DIR)

import numpy as np
import pandas as pd
from .trajectron.model.model_registrar import ModelRegistrar
from .trajectron.model.trajectron import Trajectron
from .trajectron.environment import Environment, Scene, Node, derivative_of

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

# Define the multi-index columns for the data
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


def process_tracking_results(tracking_results, dt=0.4):
    """
    Processes a dictionary of tracking results into a Scene.

    Parameters:
      tracking_results : dict
         Keys are object ids (or Node objects) and values are histories (array-like with shape (T, 2)
         where the two columns represent x and y positions).
      dt : float
         Time difference between timesteps.

    Returns:
      scene : Scene
         A scene with one Node per object id, where each node's trajectory is padded to the same length.
    """
    # If tracking_results is empty, return an empty Scene
    if not tracking_results:
        return Scene(timesteps=0, dt=dt, name="tracking_results")

    # Convert each object's trajectory to a numpy array and reshape it (if 1-dimensional)
    valid_histories = {}
    for key, history in tracking_results.items():
        arr = np.array(history)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            else:
                continue  # Skip if the format is invalid.
        if arr.shape[0] < 2:
            continue
        valid_histories[key] = arr

    if not valid_histories:
        return Scene(timesteps=0, dt=dt, name="tracking_results")

    # Find the maximum trajectory length among all objects
    max_timesteps = max(arr.shape[0] for arr in valid_histories.values())

    # Pad every object to max_timesteps (repeating the last value)
    padded_histories = {}
    all_x, all_y = [], []
    for key, arr in valid_histories.items():
        if arr.shape[0] < max_timesteps:
            pad_length = max_timesteps - arr.shape[0]
            arr = np.concatenate([arr, np.repeat(arr[-1:, :], pad_length, axis=0)], axis=0)
        padded_histories[key] = arr
        all_x.extend(arr[:, 0])
        all_y.extend(arr[:, 1])

    global_mean_x = np.mean(all_x) if all_x else 0
    global_mean_y = np.mean(all_y) if all_y else 0

    scene = Scene(timesteps=max_timesteps, dt=dt, name="tracking_results")
    scene.pos_x_mean = global_mean_x
    scene.pos_y_mean = global_mean_y

    for key, arr in padded_histories.items():
        # If key is a Node object, extract its id; otherwise normalize to a string
        if hasattr(key, 'id'):
            obj_id = str(key.id)
        else:
            obj_id = str(key)
        # Global mean correction
        arr_centered = arr - np.array([global_mean_x, global_mean_y])
        x = arr_centered[:, 0]
        y = arr_centered[:, 1]
        vx = derivative_of(x, dt)
        vy = derivative_of(y, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        data_dict = {
            ('position', 'x'): x,
            ('position', 'y'): y,
            ('velocity', 'x'): vx,
            ('velocity', 'y'): vy,
            ('acceleration', 'x'): ax,
            ('acceleration', 'y'): ay
        }
        node_data = pd.DataFrame(data_dict, columns=data_columns)
        node = Node(node_type='PEDESTRIAN', node_id=obj_id, data=node_data, first_timestep=0)
        scene.nodes.append(node)

    return scene


#python evaluate_trajectron_lobby_data.py --data ../processed/lobby_data_test.pkl  --model ./models/models_17_Mar_2025_22_52_52lobby_data_ar3

def predict(tracking_result, prediction_len=None,model_dir="./", env=None, eval_stg=None, hyperparams=None):
    """
    Predict trajectories using the Trajectron model.
    :param tracking_result: dict, keys are object ids and values are histories.
    :param prediction_len: The desired prediction length (number of timesteps). If provided, it overrides the hyperparameter.
    :return: predictions dictionary.
    """

    scene = process_tracking_results(tracking_result, dt=0.1)
    env.scenes = [scene]

    eval_stg.set_environment(env)
    eval_stg.set_annealing_params()
    # If a desired prediction length is given here, override the hyperparameter.
    if prediction_len is not None:
        hyperparams['prediction_horizon'] = prediction_len

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius_val = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius_val)

    scenes = env.scenes

    # Also set max_ft passed to get_timesteps_data to prediction_len.
    ph = hyperparams['prediction_horizon']
    with torch.no_grad():
        for scene in scenes:
            t = scene.timesteps - 1  # Perform prediction at the most recent timestep
            timesteps = np.array([t])
            # Keep min_future_timesteps at 0 and pass max_ft as ph.
            predictions = eval_stg.predict(
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_history_timesteps=7,
                min_future_timesteps=0,
                z_mode=True,
                gmm_mode=True,
                full_dist=False
            )
            if not predictions:
                continue
            pos_x_mean = scene.pos_x_mean
            pos_y_mean = scene.pos_y_mean
            for primary_idx, pedestrians in predictions.items():
                for ped_key, ped_data in pedestrians.items():
                    ped_data[..., 0] = ped_data[..., 0] + pos_x_mean
                    ped_data[..., 1] = ped_data[..., 1] + pos_y_mean
    return predictions
