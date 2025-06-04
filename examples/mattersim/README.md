# MatterSim: Deep Learning Atomistic Model

This directory contains the implementation of MatterSim model in PaddleScience, based on the paper ["MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures"](https://arxiv.org/abs/2405.04967) and the [reference implementation](https://github.com/microsoft/mattersim).

## Overview

MatterSim is a deep learning model for atomistic simulations, capable of accurately predicting materials properties across a wide range of elements, temperatures, and pressures. It can serve as:

1. A machine learning force field for atomistic simulations
2. A model for predicting ground-state material structures and energetics
3. A tool for simulating material behavior under realistic temperatures and pressures
4. A platform for computing materials' lattice dynamics, mechanical and thermodynamic properties

The model achieves near-first-principles accuracy while being significantly faster to run, making it valuable for materials research and design.

## Model Architecture

The MatterSim model is based on the M3GNet architecture with enhancements for handling diverse material systems and operating conditions. Key components include:

1. Graph neural network representation of atomic structures
2. Message passing layers for inter-atomic interactions
3. Specialized layers for energy, force, and stress predictions
4. Support for temperature and pressure conditioning

## Features

Our implementation in PaddleScience includes:

- Core MatterSim model architecture using PaddlePaddle
- Inference functionality for energy, force, and stress predictions
- Fine-tuning capabilities for custom datasets
- Integration with ASE (Atomic Simulation Environment) for easy use in workflows
- Temperature and pressure-dependent property calculations

## Usage

Detailed usage instructions and examples are provided in the included Jupyter notebooks.

## Citation

If you use this implementation in your research, please cite both the original MatterSim paper and PaddleScience:

```
@article{mattersim2024,
  title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
  author={[Authors from the original paper]},
  journal={arXiv preprint arXiv:2405.04967},
  year={2024}
}
```

## Reference

- Original paper: [arXiv:2405.04967](https://arxiv.org/abs/2405.04967)
- Reference implementation: [microsoft/mattersim](https://github.com/microsoft/mattersim)
