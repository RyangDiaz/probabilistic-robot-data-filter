# NOTE: You may want to change the dataset paths to match where your datasets are located
# To use these commands as is, rename each of the downloaded datasets to "[TASK]_full.hdf5"

# =========================================== #
# ========== Bloom Filter Analysis ========== #
# =========================================== #

# False Negative and False Positive Analysis (Table 1 and Figure 2)
python probabilistic_filter/test_bloom_filter.py --dataset_path robomimic/datasets/lift/ph/lift_full.hdf5 --n_perturbations 5 --n_random 10000
python probabilistic_filter/test_bloom_filter.py --dataset_path robomimic/datasets/can/ph/can_full.hdf5 --n_perturbations 5 --n_random 10000
python probabilistic_filter/test_bloom_filter.py --dataset_path robomimic/datasets/square/ph/square_full.hdf5 --n_perturbations 5 --n_random 10000

# =========================================== #
# ======= Policy Performance Analysis ======= #
# =========================================== #

# Filter full datasets with Bloom Filter
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/lift/ph/lift_full.hdf5 --filter_type BloomFilter
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/can/ph/can_full.hdf5 --filter_type BloomFilter
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/square/ph/square_full.hdf5 --filter_type BloomFilter

# Filter full datasets with RACE Sketch
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/lift/ph/lift_full.hdf5 --filter_type RaceSketch
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/can/ph/can_full.hdf5 --filter_type RaceSketch
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/square/ph/square_full.hdf5 --filter_type RaceSketch

# Filter full datasets with Random Filter
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/lift/ph/lift_full.hdf5 --filter_type Random
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/can/ph/can_full.hdf5 --filter_type Random
python probabilistic_filter/filter_demos.py --dataset_path robomimic/datasets/square/ph/square_full.hdf5 --filter_type Random

# Generate configs and run policy training
python robomimic/robomimic/scripts/hyperparam_helper.py --config config/base.json --script config/experiment/run_training.sh
bash config/experiment/run_training0.sh

# Evaluate trained policies (Table 2)
# Put trained policy you want to evaluate below:
PATH_TO_TRAINED_POLICY = robomimic/trained_models/demo-filter_task_.../2025.../models/model_epoch_..._.pth
python robomimic/robomimic/scripts/run_trained_agent.py --agent robomimic/trained_models/${PATH_TO_TRAINED_POLICY} --n_rollouts 50 --horizon 400 --seed 2025
