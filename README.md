# Dual-Process Learning Architecture

Code accompanying the manuscript: **Dopamine ramps as a normative consequence of dual process control** 

This repository implements a dual-process learning model where model-based values are used to train model-free values. The model generates ramping RPEs during reward approach, akin to dopamine ramps during spatial navigation and related tasks.

## Overview

The codebase simulates reinforcement learning agents in track and grid environments, reproducing results in the following studies:

- **Guru et al. (2020)** – Dopamine ramp dynamics during learning at short and long timescales
- **Kim et al. (2020)** – RPE-like dopamine responses to unexpected state transitions
- **Mikhael et al. (2022)** – Dopamine ramp dynamics under state-uncertainty manipulations
- **Krausz et al. (2023)** – Rapid, global updates to dopamine ramps by reward outcomes

## Installation

```bash
git clone https://github.com/[username]/dopamine_ramps.git
cd dopamine_ramps
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

## Usage

Activate the virtual environment, then run experiments by editing the flags in `dopamine_ramps.py`:

```python
RUN_DEMO = True              # Basic demonstration
RUN_DUAL_ARCHITECTURES = False
RUN_GURU = False             # Guru et al. simulations
RUN_KIM = False              # Kim et al. simulations
RUN_KRAUSZ = False           # Krausz et al. simulations
RUN_MIKHAEL = False          # Mikhael et al. simulations
```

Then execute:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python dopamine_ramps.py
```

Figures are saved to the `figs/` directory.

## Project Structure

```
dopamine_ramps/
├── agents.py          # RL agents (TD, Dual-process)
├── environments.py    # Linear track and grid environments
├── episodes.py        # Episode runners for different paradigms
├── experiments.py     # Simulations of published studies
├── plotting.py        # Visualisation functions
├── config.py          # Global hyperparameters
├── dopamine_ramps.py  # Main entry point
└── figs/              # Output figures
```

## Citation

Forthcoming.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Guru, A., et al. (2020). Ramping activity in midbrain dopamine neurons signifies the use of a cognitive map. *bioRxiv*. DOI: 10.1101/2020.05.21.108886
- Kim, H. R., et al. (2020). A unified framework for dopamine signals across timescales. *Cell*. DOI: 10.1016/j.cell.2020.11.013
- Mikhael, J. G., et al. (2021). The role of state uncertainty in the dynamics of dopamine *Current Biology*. DOI: 10.1016/j.cub.2022.01.025
- Krausz, T., et al. (2023). Expectation-modulated dopamine signals. *Neuron*. DOI: 10.1016/j.neuron.2023.07.017

