# Soliter: Long-horizon Continual Learning Validation

Soliter is an embodied survival simulation that validates long-horizon continual learning through biological drive systems and sleep-wake consolidation cycles. The project explores how agents can learn to survive in dynamic environments through continuous adaptation and memory consolidation.

## Overview

The Soliter agent is an embodied vehicle with four vital parameters that must be balanced to survive:
- **Energy**: Fuels movement and metabolism
- **Hydration**: Maintains cognitive function and cooling
- **Temperature**: Must be regulated to prevent hypothermia/hyperthermia
- **Wakefulness**: Meta-resource that can only be restored through sleep

The agent operates in a 2D continuous world with seasonal and diurnal cycles, requiring it to find resources (food, water, heat) to maintain homeostasis.

The system automatically detects and utilizes CUDA-capable GPUs when available, falling back to CPU if needed.

## Architecture

### Agent Components
- **CfC Brain**: Closed-form Continuous-time neural networks with NCP (Neural Circuit Policy) wiring
- **Biological Drive System**: Internal reward based on homeostatic regulation rather than external shaping
- **Gradient Sensors**: Directional sensors for resource detection (smell/heat gradients)
- **Sleep-Wake Cycle**: Consolidation and homeostatic scaling during sleep phases

### Learning System
- **PPO (Proximal Policy Optimization)**: Stable policy learning during wake periods
- **Epistemic Replay Buffer**: Surprise-gated experience replay with pruning
- **Fisher Information Matrix**: Elastic Weight Consolidation for preventing catastrophic forgetting
- **Homeostatic Scaling**: Prevents neural saturation during sleep

## Features

- **Embodied Survival**: Agent must actively seek resources to maintain vital signs
- **Biological Drive System**: Internal reward based on drive reduction rather than external shaping
- **Seasonal Environment**: Dynamic world with day/night and seasonal cycles
- **Sleep-Wake Consolidation**: Memory consolidation and synaptic scaling during sleep
- **Gradient Navigation**: Agents can detect and navigate toward resources
- **Epistemic Pruning**: Removal of predictable experiences to focus learning on novel events

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd soliter

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

### Quick Start
```bash
# Run a basic training session
python main.py --cycles 10

# Run with custom parameters
python scripts/train_soliter.py --cycles 50 --cpu --output-dir my_experiments
```

### Configuration
The system uses a comprehensive configuration system in `configs/default.yaml`. You can modify:
- Agent vitals and decay rates
- World parameters (size, seasonal cycles)
- Training hyperparameters
- Sleep-wake cycle parameters
- Memory and consolidation settings

## Project Structure

```
soliter/
├── soliter/                 # Core modules
│   ├── agents/              # Agent implementations
│   │   └── soliter_agent.py # Main agent with vitals
│   ├── core/                # Core systems
│   │   ├── cfc_network.py   # CfC neural networks
│   │   ├── drive_system.py  # Biological drive system
│   │   └── ncp_wiring.py    # Neural Circuit Policy wiring
│   ├── environment/         # Environment simulation
│   ├── memory/              # Replay and memory systems
│   ├── training/            # Training algorithms
│   │   └── sleep_wake.py    # Sleep-wake trainer
│   └── utils/               # Utility functions
├── scripts/                 # Training and analysis scripts
├── configs/                 # Configuration files
├── experiments/             # Training logs and checkpoints
├── notebooks/               # Analysis notebooks
└── tests/                   # Unit and integration tests
```

## Key Concepts

### Biological Drive System
Instead of traditional shaped rewards, Soliter uses internal homeostatic drives:
- Hunger drive: Based on energy deficit
- Thirst drive: Based on hydration deficit
- Cold drive: Based on temperature deviation
- Curiosity drive: Based on sensory monotony

### Sleep-Wake Consolidation
- **Wake**: Exploration, experience collection, learning
- **Sleep**: Memory consolidation, synaptic scaling, forgetting of predictable experiences

### Gradient Navigation
Agents can detect gradients of resources (food, water, heat) to navigate toward them, simulating smell or heat detection.

## Research Goals

- Validate long-horizon continual learning in embodied agents
- Demonstrate biological drive systems as superior to external reward shaping
- Explore the role of sleep in continual learning and memory consolidation
- Investigate emergence of survival behaviors in complex environments

## Results

The agent demonstrates:
- Self-motivated resource seeking behavior
- Adaptation to seasonal and daily environmental changes
- Effective memory consolidation during sleep
- Prevention of catastrophic forgetting through EWC
- Emergent survival strategies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Closed-form Continuous-time (CfC) neural networks
- Inspired by C. elegans neural circuit patterns (NCP wiring)
- Incorporates principles from biological sleep and memory consolidation research
