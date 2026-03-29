# Comparing Temporal Logic Reward Shaping Approaches in Reinforcement Learning

This repository contains research code comparing reward shaping approaches for teaching RL agents to satisfy formal specifications on MiniGrid environments.

## Overview

This project evaluates three reinforcement learning training methods on complex sequential decision-making tasks:

1. **Baseline**: Vanilla PPO with sparse task rewards
2. **LTL**: Linear Temporal Logic-based reward shaping using discrete automaton states
3. **TLTL**: Truncated Linear Temporal Logic with robustness-based continuous reward shaping

The key finding: **TLTL dramatically outperforms both baseline and LTL** on complex ordered-subtask problems (e.g., DoorKey: 94.8% vs 0.6% success rate).

## Project Structure

```
├── ltl_wrappers.py           # LTL wrapper implementations (infinite-horizon)
├── tltl_wrappers.py          # TLTL wrapper implementations (finite-horizon)
├── train_baseline.py         # Single-run baseline training
├── train_ltl.py              # Single-run LTL training
├── train_tltl.py             # Single-run TLTL training
├── train_all_seeds.py        # Main training script (30 seeds × 9 configurations)
├── evaluate.py               # Multi-seed evaluation and metrics
├── analyze_seeds.py          # Statistical analysis and visualization
├── extract_per_run_stats.py  # Extract statistics from results
├── plot_results.py           # Training metrics plotting utilities
└── results_log.md            # Experimental results documentation
```

## Supported Environments

- **Empty**: Navigate to goal position
- **LavaGaps**: Reach goal while avoiding lava
- **DoorKey**: Collect key, open door, reach goal (ordered subtasks)

All tasks use the MiniGrid 8x8 environment variant.

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

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Main Experiment

Train 30 seeds across all 9 configurations (270 models total):

```bash
python train_all_seeds.py
```

Models are saved to `models/seeds30/` and training logs to `results/seeds30/`.

### Analysis

Analyze multi-seed results with learning curves and evaluation metrics:

```bash
python analyze_seeds.py
```

Generates plots in `plots/` and comprehensive metrics in `results/seed_evaluation_metrics.csv`.

### Legacy Scripts

Individual training scripts for each method are also available:

```bash
python train_baseline.py   # Baseline PPO without wrappers
python train_ltl.py        # LTL wrapper training
python train_tltl.py       # TLTL wrapper training
python evaluate.py         # Evaluate baseline/LTL/TLTL models
```

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
