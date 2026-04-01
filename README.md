# Comparing Temporal Logic Reward Shaping Approaches in Reinforcement Learning

This repository contains research code comparing reward shaping approaches for teaching RL agents to satisfy formal specifications on MiniGrid environments.

## Overview

This project evaluates three reinforcement learning training methods on complex sequential decision-making tasks:

1. **Baseline**: Vanilla PPO with sparse task rewards
2. **LTL**: Linear Temporal Logic-based reward shaping using discrete automaton states
3. **TLTL**: Truncated Linear Temporal Logic with robustness-based continuous reward shaping

The key finding: **TLTL dramatically outperforms both baseline and LTL** on complex ordered-subtask problems (e.g., DoorKey: 94.8% vs 0.6% success rate).

All scripts are located in the `src/` directory.

## Supported Environments

- **Empty**: Navigate to goal position
- **LavaGap**: Reach goal while avoiding lava
- **DoorKey**: Collect key, open door, reach goal (ordered subtasks)

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- MiniGrid
- NumPy, Pandas, Matplotlib, SciPy

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (from src directory)
cd src
pip install -r requirements.txt
cd ..
```

## Usage

### Running the Main Experiment

Train 30 seeds across all 9 configurations (270 models total):

```bash
cd src
python train_all_seeds.py
cd ..
```

Models are saved to `models/seeds30/` and training logs to `results/seeds30/` in the project root.

### Analysis

Analyze multi-seed results with learning curves and evaluation metrics:

```bash
cd src
python analyze_seeds.py
cd ..
```

Generates plots in `plots/` and comprehensive metrics in `results/seed_evaluation_metrics.csv`.

### Legacy Scripts

Individual training scripts for each method are also available:

```bash
cd src
python train_baseline.py   # Baseline PPO without wrappers
python train_ltl.py        # LTL wrapper training
python train_tltl.py       # TLTL wrapper training
python evaluate.py         # Evaluate baseline/LTL/TLTL models
python plot_results.py     # Generate training curve plots
python extract_per_run_stats.py  # Extract per-run statistics
cd ..
```

**Important**: All scripts must be run from the `src/` directory. Results are saved to the project root directories (`results/`, `models/`, `plots/`).

## Key Results

### DoorKey Environment (Ordered Subtasks)

| Method   | Success Rate | Avg Episodes   |
| -------- | ------------ | -------------- |
| Baseline | 0.6%         | 8x slower      |
| LTL      | 0.6%         | Similar        |
| **TLTL** | **94.8%**    | **~2x faster** |

### Core Insights

- **Baseline** struggles with ordered subtasks; sparse reward signal provides insufficient guidance
- **LTL** provides automaton-based feedback but discreteness limits learning on complex tasks
- **TLTL** enables dense, continuous guidance via robustness metrics, dramatically accelerating learning

## License

MIT License
