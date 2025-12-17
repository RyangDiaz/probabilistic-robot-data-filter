import h5py
import numpy as np
import argparse
import random

# If no 20_percent segment is available in dataset, create one

def add_segment(path, percent=20, seed=2025):
    random.seed(2025)
    f = h5py.File(path, "r+")
    if f'{percent}_percent' not in f['mask']:
        demo_list = list(f['data'].keys())
        num_demos = len(demo_list)
        num_masked = int(num_demos * percent * 0.01)
        random.shuffle(demo_list)
        masked_demos = np.array(demo_list[:num_masked]).astype('S')

        f.create_dataset(f"mask/{percent}_percent", data=masked_demos)

        print(f"Masked {num_masked} demos.")

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str
    )

    parser.add_argument(
        "--percent",
        type=int,
        default=20
    )

    args = parser.parse_args()
    add_segment(args.dataset_path, args.percent)
