# TurbDiff: Generative Modeling for 3D Flow Simulation

This is an implementation of the TurbDiff model as described in the paper "From Zero to Turbulence: Generative Modeling for 3D Flow Simulation" (ICLR 2024) by Marten Lienen, David Lüdke, Jan Hansen-Palmus, and Stephan Günnemann.

## Overview

TurbDiff is a denoising diffusion probabilistic model (DDPM) designed for generating realistic 3D turbulent flow fields. Unlike traditional autoregressive approaches, TurbDiff directly learns the manifold of all possible turbulent flow states without relying on any initial flow state.

## Model Architecture

The model architecture consists of:

1. A 3D U-Net backbone with attention mechanisms
2. Specialized conditioning for boundary conditions and geometry
3. A diffusion process based on the DDPM framework
4. Custom components for handling turbulent flow characteristics

## Usage

Please refer to the `train.py` and `infer.py` scripts for training and inference examples.

## Citation

```
@inproceedings{lienen2024zero,
  title = {From {{Zero}} to {{Turbulence}}: {{Generative Modeling}} for {{3D Flow Simulation}}},
  author = {Lienen, Marten and L{\"u}dke, David and {Hansen-Palmus}, Jan and G{\"u}nnemann, Stephan},
  booktitle = {International {{Conference}} on {{Learning Representations}}},
  year = {2024},
}
```

## References

- Original implementation: [https://github.com/martenlienen/generative-turbulence](https://github.com/martenlienen/generative-turbulence)
- Paper: [https://arxiv.org/abs/2306.01776](https://arxiv.org/abs/2306.01776)
