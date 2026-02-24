### Prereqs

- Python 3.12
- Conda

### Install

```bash
conda create -y -p ./conda_env python=3.12
conda activate ./conda_env

conda install -y -c conda-forge pybullet

pip install numpy scipy torch matplotlib pandas scikit-learn
```


### Usage

Variance Levels:
  low    — block pretty much always in the same spot, same with arm position
  medium — more variance in block position and arm start, slightly noisy path
  high   — more variance in block position and arm start, more noisy path

Usage:
  python data_collection.py --variability [low/medium/high] --iterations [some_number] --steps [some_step_count] --output data_[level].csv