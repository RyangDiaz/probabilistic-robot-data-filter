# Efficient On-the-Fly Robot Dataset Curation with Probabilistic Data Structures

Ryan Diaz - rd88

This repository contains code for the COMP 580 final project related to the use of probabilistic data structures for filtering incoming robot demonstrations to use in imitation learning.

### Installation

First, clone the [Robomimic](https://github.com/ARISE-Initiative/robomimic/tree/master) repository:

```
git clone https://github.com/ARISE-Initiative/robomimic.git
```

To configure the environment to run the code, setup the Conda environment and install required dependencies by running the commands below:

```
conda env create -f environment.yml
conda activate project580
cd robomimic
pip install -e .
cd ..
```

### Running the Code
To download the datasets used in this project, use the `robomimic/robomimic/scripts/download_datasets.py` script (we use the `lift`, `can`, and `square` datasets).

See `run_experiments.sh` for commands to run in order to reproduce each of the plots in the report. For policy evaluation, you will have to locate the specific policy weight file for evaluation (it should end in `.pth`).