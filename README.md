# Project Soliter

Long-horizon continual learning validation through embodied survival simulation.

## Overview

Project Soliter tests homeostatic sleep-wake architectures in a complex, dynamic environment. An autonomous agent must survive across seasons by balancing multiple physiological needs while learning continuously without catastrophic forgetting.

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd soliter

# Install dependencies with uv
uv sync

# Verify installation
uv run python -c "import torch; import pygame; print('Setup complete!')"
```

## Quick Start
```bash
# Run with default configuration
uv run python scripts/train.py

# Run with custom config
uv run python scripts/train.py --config configs/experiments/long_run.yaml

# Evaluate a checkpoint
uv run python scripts/evaluate.py --checkpoint data/experiments/run_001/day_100.pt
```

## Project Structure

- `soliter/core/` - Neural network architectures (CfC, NCP)
- `soliter/agents/` - Agent implementations
- `soliter/environment/` - Simulation world
- `soliter/memory/` - Replay buffer, Fisher matrix
- `soliter/training/` - Training loops, sleep-wake cycle
- `soliter/visualization/` - PyGame rendering, metrics
- `configs/` - YAML configuration files
- `scripts/` - Training and evaluation scripts
- `notebooks/` - Analysis notebooks

## Research Goals

1. Validate Fisher Information saturation at ~20 days in embodied agents
2. Demonstrate stable learning across 3+ simulated years
3. Measure context integration (φ_seasonal)
4. Test for consciousness prerequisites

## Key Features

- **CfC Neural Networks**: Continuous-time dynamics with closed-form solutions
- **Sleep-Wake Cycle**: Homeostatic synaptic scaling + Fisher consolidation
- **Epistemic Pruning**: Two-stage buffer compression (uncertainty + Fisher)
- **Seasonal Conflict**: Summer vs Winter policies test catastrophic forgetting
- **Consciousness Metrics**: φ_seasonal, value emergence, self-model coherence

## License

MIT

## Citation

If you use this code, please cite:
```bibtex
@software{soliter2026,
  author = {Komarovsky, Stan},
  title = {Project Soliter: Embodied Continual Learning Testbed},
  year = {2026},
  url = {https://github.com/yourusername/soliter}
}
```

## References

- Komarovsky, S. (2025). Homeostatic Sleep-Wake Architecture for Continual Learning.
- Hasani, R., et al. (2022). Closed-form continuous-time neural networks.
- Tononi, G., et al. (2016). Integrated information theory.
