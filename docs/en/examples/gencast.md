# GenCast

Before starting evaluation, please obtain relevant data on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast) and place it under the path configured in the `gencast.yaml` file.

- Download all files under the directory `dm_graphcast/gencast/stats` and place them in the `./data/stats/` directory.
- Download any or all files under the directory `dm_graphcast/gencast/dataset` (e.g., source-era5_date-2019-03-29_res-1.0_levels-13_steps-12.nc) and place them in the `./data/dataset/` directory.

=== "Model Training Command"

    ``` sh
    # Set path to PaddleScience/jointContribution folder
    cd PaddleScience/jointContribution
    export PYTHONPATH=$PWD:$PYTHONPATH
    # Run training script
    python run_gencast.py mode=train
    ```

=== "Model Evaluation Command"

    ``` sh
    # Set path to PaddleScience/jointContribution folder
    cd PaddleScience/jointContribution
    export PYTHONPATH=$PWD:$PYTHONPATH
    # Download model parameters
    cd gencast/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/gencast/gencast_params_GenCast-1p0deg-Mini-_2019.pdparams -P ./data/params/
    # Run evaluation script
    python run_gencast.py mode=eval
    ```

## 1. Background Introduction

Weather forecasts are inherently uncertain, so predicting the range of possible weather scenarios is critical for many important decisions, from warning the public about dangerous weather to planning renewable energy use. Here, we introduce GenCast, a probabilistic weather model that outperforms the world's top medium-range weather forecast - the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast ENS in skill and speed. Unlike traditional methods based on Numerical Weather Prediction (NWP), GenCast is a Machine Learning Weather Prediction (MLWP) method trained on decades of reanalysis data. GenCast can generate a random 15-day global forecast ensemble in 8 minutes, with a 12-hour step, 0.25-degree latitude-longitude resolution, covering over 80 surface and atmospheric variables. Among the 1320 targets we evaluated, GenCast outperformed ENS on 97.4% and was better at predicting extreme weather, tropical cyclones, and wind power generation. This work helps open the next chapter of operational weather forecasting, enabling important weather-dependent decisions to be made with greater accuracy and efficiency.

## 2. Model Principle

Here, we introduce a probabilistic weather model - GenCast, which generates global 15-day ensemble forecasts at 0.25° resolution, achieving higher accuracy than the top operational ensemble system ECMWF's ENS for the first time. Generating a single 15-day GenCast forecast on a cloud TPUv5 device takes about 8 minutes, and multiple forecast ensembles can be generated in parallel.

GenCast models the conditional probability distribution $p(X^{t+1} | X^t, X^{t-1})$ of the future weather state $X^{t+1}$, which is conditioned on the current and previous weather states. A forecast trajectory $X^{1:T}$ of length $T$ is modeled by conditioning on the initial and previous states $(X^0, X^{-1})$ and factoring the joint distribution of consecutive states:

$$
p(X^{1:T} | X^0, X^{-1}) = \prod_{t=0}^{T-1} p(X^{t+1} | X^t, X^{t-1})
$$

Each state is derived through autoregressive sampling.

The representation of the global weather state $X$ includes 6 surface variables and 6 atmospheric variables on 13 vertical pressure levels, distributed on a 0.25° latitude-longitude grid (see Table B1 for details). The forecast duration is 15 days, and the interval between consecutive steps $t$ and $t+1$ is 12 hours, so $T = 30$.

GenCast is implemented as a conditional diffusion model, a generative machine learning model used to generate new samples from a given data distribution, which powers many recent advances in natural image, sound, and video modeling and is known as "Generative AI". Diffusion models operate through a process of iterative refinement. The future atmospheric state $X^{t+1}$ is produced by iteratively refining a candidate state $Z_0^{t+1}$, which is initialized purely from noise and conditioned on the previous two atmospheric states $(X^t, X^{t-1})$. The blue box in the figure shows how the first forecast step is generated from initial conditions, and how the entire trajectory $X^{1:T}$ is generated autoregressively. Since each time step in the forecast is initialized with noise (i.e., $Z_0^{t+1}$), the process can be repeated with different noise samples to generate an ensemble of trajectories.

<figure markdown>
  ![gencast.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/gencast/gencast.png){ loading=lazy }
</figure>

At each stage of the iterative refinement process, GenCast applies a neural network architecture consisting of an encoder, a processor, and a decoder. The encoder component maps the input $Z_n^{t+1}$ and conditions $(X^t, X^{t-1})$ from the original latitude-longitude grid to an internal learned representation on a six-times refined icosahedral mesh. The processor component is a Graph Transformer where each node attends to its k-hop neighbors on the internal mesh. The decoder component maps the internal mesh representation back to $Z_{n+1}^{t+1}$, which is defined on the latitude-longitude grid.

GenCast is trained on 40 years of ERA5 reanalysis data, spanning from 1979 to 2018, using a standard diffusion model denoising objective. Importantly, although GenCast is trained directly only on single-step prediction tasks, it can generate 15-day ensemble forecasts through autoregressive unrolling.

## 3. Model Construction

### 3.1 Environment Dependencies

* paddlepaddle
* matpoltlib (for image plotting)
* pickle (for storing and loading graph templates)
* xarray (for loading .nc data)
* trimesh (for making mesh data)
* scipy (for sparse matrix operations during spherical harmonic transform)
* math (for mathematical calculations during spherical harmonic transform)

### 3.2 Model Related File Description

- **xarray_tree.py**: An implementation of tree.map_structure suitable for xarray.

- **denoiser.py**: GenCast denoiser for one-step prediction.

- **dpm_solver_plus_plus_2s.py**: Sampler using DPM-Solver++ 2S in [1].

- **gencast.py**: Combines GenCast model architecture with sampler, encapsulated as a denoiser to generate predictions.

- **samplers_base.py**: Defines the interface for samplers.

- **samplers_utils.py**: Utility methods for samplers.

- **sparse_transformer.py**: Generic sparse transformer acting on TypedGraph, where inputs and outputs are flat vectors of features per node and edge. `predictor.py` uses one of these for the mesh Graph Neural Network (GNN).

- **spherical_harmonic.py**: Spherical harmonic basis evaluation and differential operators.

- **main.py**: Evaluation and visualization script.

[1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models, https://arxiv.org/abs/2211.01095

## 4. Result Display

The figure below shows the ground truth results, prediction results, and errors for 2-meter temperature.

<figure markdown>
  ![gencast_2m_t.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/gencast/gencast_2m_t.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Ground truth results ("targets"), prediction results ("prediction") and errors ("diff")</figcaption>
</figure>

It can be seen that the model prediction results are basically consistent with the real results.

## 4. References

* [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)
* [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)
* [GenCast Github Address](https://github.com/deepmind/graphcast)
* [dinosaur Github Address](https://github.com/neuralgcm/dinosaur)
