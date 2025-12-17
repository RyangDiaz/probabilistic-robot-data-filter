# Given an existing set of demos and a set of new demos,
# create a new dataset that is the concatenation of the existing set of demos
# and the subset of the new set of demos where each demo in the subset has been
# determined to not be in the dataset by the probabilistic filter

import numpy as np
import argparse
import h5py
import tqdm
import os
import shutil

from filters.local_bloom_filter import LocallySensitiveBloomFilter
from filters.race_sketch import RACESketch
from filters.random_filter import RandomFilter

OBS_KEYS = ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]

def get_dataset_dim(dataset_path):
    f = h5py.File(dataset_path, "r")
    actions = f[f'data/demo_1/actions'][:]
    obs = [f[f'data/demo_1/obs/{modality}'][:] for modality in OBS_KEYS]
    state_actions = np.concatenate(obs + [actions], axis=1)

    return state_actions.shape[1]

def create_filter(dataset_path, filter_type, **filter_kwargs):
    if filter_type == "BloomFilter":
        filter = LocallySensitiveBloomFilter(**filter_kwargs)
    elif filter_type == "RaceSketch":
        filter = RACESketch(**filter_kwargs)
    else:
        filter = RandomFilter(**filter_kwargs)

    f = h5py.File(dataset_path, "r")
    print(list(f['mask'].keys()))
    demo_idx = [a.decode('utf-8') for a in f['mask/20_percent'][:]]

    total_samples = 0

    # Adding samples
    print(f"Constructing {filter_type}...")
    for idx in tqdm.tqdm(demo_idx):
        actions = f[f'data/{idx}/actions'][:]
        obs = [f[f'data/{idx}/obs/{modality}'][:] for modality in OBS_KEYS]
        state_actions = np.concatenate(obs + [actions], axis=1)

        for v in state_actions:
            filter.insert(v)
            total_samples += 1
    
    print(total_samples, "total samples added")

    f.close()

    return filter

def create_filtered_dataset(dataset_path, filter):
    dataset_path_root = '/'.join(dataset_path.split('/')[:-1])
    dataset_name = args.dataset_path.split("/")[-1].split(".")[0]

    filter_type = None
    if isinstance(filter, LocallySensitiveBloomFilter):
        filter_type = "bloom_filter"
    elif isinstance(filter, RACESketch):
        filter_type = "race_sketch"
    else:
        filter_type = "random"

    new_dataset_path = os.path.join(dataset_path_root, dataset_name + f"_{filter_type}.hdf5")
    shutil.copy2(dataset_path, new_dataset_path)
    f_new = h5py.File(new_dataset_path, "r+")

    total_new_samples = 0
    total_kept_new_samples = 0

    demo_idx = [a.decode('utf-8') for a in f_new['mask/20_percent'][:]]
    num_demos = len(list(f_new['data'].keys()))

    # Aggregate all filtered state-actions pairs into one demo (to play nice with SequenceDataset)
    filtered_actions_all = None
    filtered_obs_all = {}

    for idx in tqdm.tqdm(range(num_demos)):
        demo_name = f'demo_{idx}'
        if demo_name not in demo_idx:
            actions = f_new[f'data/{demo_name}/actions'][:]
            obs = [f_new[f'data/{demo_name}/obs/{modality}'][:] for modality in OBS_KEYS]
            state_actions = np.concatenate(obs + [actions], axis=1)
            valid_sample_idx = []

            for i, v in enumerate(state_actions):
                if not filter.query(v):
                    valid_sample_idx.append(i)
                    filter.insert(v)
                    total_kept_new_samples += 1
                total_new_samples += 1
            
            if len(valid_sample_idx) > 0:
                # Filter and update actions and obs
                filtered_actions = actions[valid_sample_idx, :]
                if filtered_actions_all is None:
                    filtered_actions_all = filtered_actions
                else:
                    filtered_actions_all = np.concatenate((filtered_actions_all, filtered_actions), axis=0)

                # f_new.create_dataset(f'data/{demo_name}/actions', data=filtered_actions)
                for modality in f_new[f'data/{demo_name}/obs'].keys():
                    filtered_obs = f_new[f'data/{demo_name}/obs/{modality}'][:][valid_sample_idx, :]
                    if modality not in filtered_obs_all:
                        filtered_obs_all[modality] = filtered_obs
                    else:
                        filtered_obs_all[modality] = np.concatenate((filtered_obs_all[modality], filtered_obs), axis=0)

            del f_new[f'data/{demo_name}']
    
    if total_kept_new_samples > 0:
        new_demo = f_new.create_group(f'data/demo_1000')
        new_demo.create_dataset('actions', data=filtered_actions_all)
        new_demo.attrs['num_samples'] = total_kept_new_samples
        new_obs = new_demo.create_group('obs')
        for o in filtered_obs_all:
            new_obs.create_dataset(o, data=filtered_obs_all[o])

    print(f"Added {total_kept_new_samples} samples to dataset out of {total_new_samples} incoming samples.")
    # print(f"File now has {len(list(f_new['data'].keys()))} demos.")
    f_new.close()

def main(args):
    dim = get_dataset_dim(args.dataset_path)

    rate = None
    if args.filter_type == "Random":
        if "lift" in args.dataset_path:
            rate = 0.62
        elif "can" in args.dataset_path:
            rate = 0.48
        elif "square" in args.dataset_path:
            rate = 0.36

    filter_args = {
        "Random": {
            "rate": rate,
            "seed": 2025
        },
        "BloomFilter": {
            "dim": dim,
            "m": 2**15,
            "k": 8,
            "bits_per_lsh": 64,
            "seed": 2025,
        },
        "RaceSketch": {
            "dim": dim,
            "num_hash": 8,
            "table_length": 2**10,
            "bits_per_lsh": 64,
            "threshold": 0.001,
            "seed": 2025,
        }
    }
    filter_kwargs = filter_args[args.filter_type]
    filter = create_filter(args.dataset_path, args.filter_type, **filter_kwargs)
    create_filtered_dataset(args.dataset_path, filter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset to test membership against
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str
    )

    # Data structure of demo filter
    parser.add_argument(
        "--filter_type",
        required=True,
        choices=["BloomFilter", "RaceSketch", "Random"]
    )

    args = parser.parse_args()
    main(args)
