# Quick Start Guide
## Installation
1. Create a new conda environment
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

2. Install the package
```bash
pip install -e .
```

## Running experiments
### Baseline experiments
```bash
python3 train_baseline_sac.py --env_name <env_name>
```
The list of baseline environments are `['BaselineCompLift-v1', 'BaselineCompStack-v1', 'BaselineCompNut-v1', 'BaselineCompPickPlaceCan-v1']`

### Compositional RL experiments
```bash
python3 train_comp_sac.py --env_name <env_name>
```
The list of compositional environments are `['CompLift-v1', 'CompStack-v1', 'CompNut-v1', 'CompPickPlaceCan-v1']`

### Compositional RL with skill reuse/recycle
```bash
python3 train_reuse.py --env_name CompStack-v1
python3 train_reuse.py --env_name CompPickPlaceCan-v1 --keep_learning
```

## Visualize a trained compositional policy
```bash
python3 eval_comp_sac.py
```