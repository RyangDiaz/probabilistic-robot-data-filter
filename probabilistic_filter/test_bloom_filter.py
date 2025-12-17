import numpy as np
import argparse
import os
import h5py
import tqdm
import matplotlib.pyplot as plt

from filters.local_bloom_filter import LocallySensitiveBloomFilter

# Perform false postiive/false negative analyses on LSBF

# Add all real demos to dataset
# For each state-action pair, generate N slightly perturbed versions of the pair and test membership
# Generate M completely random state-action pairs and test membership
# Do this for different Bloom filter parameters

def get_dataset_dim(dataset_path):
    f = h5py.File(dataset_path, "r")
    actions = f[f'data/demo_1/actions'][:]
    obs = [f[f'data/demo_1/obs/{modality}'][:] for modality in ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]]
    state_actions = np.concatenate(obs + [actions], axis=1)

    return state_actions.shape[1]

def create_bloom_filter(dataset_path, dim=30, m=2**10, k=8, bits_per_lsh=64):
    bf = LocallySensitiveBloomFilter(dim=dim, m=m, k=k, bits_per_lsh=bits_per_lsh)
    f = h5py.File(dataset_path, "r")
    demo_idx = [a.decode('utf-8') for a in f['mask/20_percent'][:]]

    total_samples = 0

    # Adding samples
    print("Constructing Bloom filter...")
    for idx in tqdm.tqdm(demo_idx):
        actions = f[f'data/{idx}/actions'][:]
        obs = [f[f'data/{idx}/obs/{modality}'][:] for modality in ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]]
        state_actions = np.concatenate(obs + [actions], axis=1)

        for v in state_actions:
            bf.insert(v)
            total_samples += 1
    
    print(total_samples, "total samples added")

    f.close()

    return bf

def false_negative_test(dataset_path, bloom_filter, n_perturbations, T, seed=2025):
    f = h5py.File(dataset_path, "r")
    rng = np.random.default_rng(seed=seed)
    demo_idx = [a.decode('utf-8') for a in f['mask/20_percent'][:]]

    # Testing samples
    print("Testing perturbed samples")
    total_samples = 0
    false_negatives = 0
    for idx in tqdm.tqdm(demo_idx):
        actions = f[f'data/{idx}/actions'][:]
        obs = [f[f'data/{idx}/obs/{modality}'][:] for modality in ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]]
        state_actions = np.concatenate(obs + [actions], axis=1)
        for _ in range(n_perturbations):
            perturbation = rng.uniform(low=-T, high=T)
            state_actions_perturbed = state_actions + perturbation

            for v in state_actions_perturbed:
                if not bloom_filter.query(v):
                    false_negatives += 1
                total_samples += 1
    print(f"False Negative Rate: {false_negatives/total_samples} ({false_negatives}/{total_samples})")

    f.close()

def false_positive_test(bloom_filter, n_random, seed=2025):
    # Definitely out of range
    rng = np.random.default_rng(seed=seed)

    print("Testing random dataset")
    total_samples = 0
    false_positives = 0
    random_dataset = rng.uniform(low=2, high=4, size=(n_random, bloom_filter.dim))

    for v in random_dataset:
        if bloom_filter.query(v):
            false_positives += 1
        total_samples += 1
    fp_rate = false_positives / total_samples
    print(f"False Positive Rate: {fp_rate} ({false_positives}/{total_samples})")
    return fp_rate


def main(args):
    dim = get_dataset_dim(args.dataset_path)
    bf = create_bloom_filter(args.dataset_path, dim=dim, m=2**15)

    os.mkdir('results', exist_ok=True)

    for T in [0.1, 0.01, 0.001, 0.0001]:
        print("T=",T)
        false_negative_test(args.dataset_path, bf, args.n_perturbations, T=T)

    plt.figure()
    m_power_range = [6,9,12,15]

    for k in [2,4,6,8,10]:
        fp_rates = []
        for m_power in m_power_range:
            bf = create_bloom_filter(args.dataset_path, dim=dim, m=2**m_power, k=k)
            fp = false_positive_test(bf, args.n_random)
            fp_rates.append(fp)
        plt.plot(m_power_range, fp_rates, label=f"k={k}")
    
    task_name = args.dataset_path.split("/")[-1].split("_full.hdf5")[0]
    
    plt.legend()
    plt.xlabel("Bitarray Size (as a power of 2)")
    plt.xticks(m_power_range, m_power_range)
    plt.ylabel("False Positive Rate")
    plt.title(f"False Positive Rate Analysis on \"{task_name}\" Task")
    plt.savefig(f"results/fp_{task_name}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset to test membership against
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str
    )

    # How many small perturbations to make to each sample
    parser.add_argument(
        "--n_perturbations",
        type=int,
        default=1
    )

    # Size of random dataset
    parser.add_argument(
        "--n_random",
        type=int,
        default=10000
    )

    args = parser.parse_args()
    main(args)