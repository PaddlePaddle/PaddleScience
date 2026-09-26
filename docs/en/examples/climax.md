# ClimaX

Before starting training and evaluation, please download the ClimateBench v1.0 dataset from [Zenodo record 7064308](https://zenodo.org/record/7064308): [train_val.tar.gz](https://zenodo.org/record/7064308/files/train_val.tar.gz?download=1) (839 MB) and [test.tar.gz](https://zenodo.org/record/7064308/files/test.tar.gz?download=1) (74 MB), and extract both archives under `examples/climax/data/climatebench/`.

Then regrid both partitions to the 5.625 degree grid used by this case:

```sh
cd examples/climax
python regrid_climatebench.py data/climatebench/train_val --save_path data/climatebench/5.625deg/train_val --ddeg_out 5.625
python regrid_climatebench.py data/climatebench/test --save_path data/climatebench/5.625deg/test --ddeg_out 5.625
```

The regrid script additionally uses `xesmf`, which is installed with `conda install -c conda-forge xesmf`. The remaining extra dependencies (`click`, `netCDF4`, `numpy`, `scipy`, `xarray`) are installed with `pip install -r examples/climax/requirements.txt`.

Because the test partition concatenates the historical period with `ssp245`, `inputs_historical.nc` and `outputs_historical.nc` from the regridded `train_val` directory must also be symlinked or copied into `data/climatebench/5.625deg/test/`.

Before starting training, please prepare the pretrained weights. The released ClimaX checkpoint is hosted on HuggingFace ([tungnd/climax](https://huggingface.co/tungnd/climax)); its `5.625deg.ckpt` file (412 MB) matches the grid used by this case. The `.ckpt` file must be converted to a `.pdparams` file before use. The converted checkpoint wraps the parameters as `{"state_dict": ...}` with keys prefixed by `net.`, and is loaded through `TRAIN.init_state_path`.

=== "Model Training Command"

    ``` sh
    cd examples/climax
    python main.py mode=train
    # engage the CINN compiler (to_static + CINN)
    CLIMAX_USE_CINN=1 python main.py mode=train
    ```

=== "Model Evaluation Command"

    ``` sh
    cd examples/climax
    python main.py mode=eval EVAL.pretrained_model_path="./exps/climate_projection/best_model.pdparams"
    ```

## 1. Background Introduction

Climate projection studies how key climate variables respond to greenhouse-gas and aerosol emission scenarios. Scenario simulations with full earth-system models are computationally expensive, and every additional scenario requires a separate model run, which motivates data-driven emulators that learn the mapping from forcing inputs to climate responses from existing simulations.

ClimaX (arXiv 2301.10343) is a foundation model for weather and climate built on the Vision Transformer (ViT). Unlike conventional deep learning weather models whose input and output channels are fixed, ClimaX embeds an arbitrary set of input variables, aggregates the per-variable token sequences into a single sequence, and feeds it into a standard ViT encoder, so one pretrained backbone can be fine-tuned to different downstream weather and climate tasks.

This case applies ClimaX to its climate-projection downstream task on the ClimateBench benchmark: on a 5.625 degree grid, the model emulates the surface air temperature (`tas`) response to greenhouse-gas and aerosol forcings, taking four variables `CO2`, `SO2`, `CH4` and `BC` as inputs.

## 2. Model Principle

This chapter briefly introduces the model principle of ClimaX.

The case uses the climate-projection variant `ClimaXClimateBench`, whose forward pipeline consists of the following stages:

| Stage                        | Description                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-variable patch embedding | Each input variable is tokenized by its own `PatchEmbed` (all variables share one set of weights when `parallel_patch_embed=True`)           |
| Variable aggregation         | Per-variable token sequences are pooled into a single sequence by cross-attention against a learned query                                    |
| ViT backbone                 | Standard transformer encoder blocks (`embed_dim` / `depth` / `num_heads`) with 2D sincos positional embedding and lead-time conditioning     |
| Temporal aggregation         | A time positional embedding and a cross-attention module pool the input history window (`time_history` steps) into a single token per sample |
| Head                         | A linear head regresses the output grid from the pooled token                                                                                |

### 2.1 Per-Variable Patch Embedding

The input of the climate-projection task is a `[B, T, Vi, H, W]` tensor, i.e. `T` consecutive time steps of `Vi` variables on an `H x W` grid. Each time step is flattened into the batch dimension, every variable is tokenized by its own `PatchEmbed`, a learned variable embedding is added to the tokens of the corresponding variable, the per-variable sequences are aggregated by `aggregate_variables` (see 2.2), and a 2D sincos positional embedding is added:

```py linenums="137" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:137:154
--8<--
```

### 2.2 Variable Aggregation

The per-variable token sequences are pooled into one sequence by a cross-attention layer, where a learned query attends over the tokens of all variables:

```py linenums="220" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py:220:231
--8<--
```

### 2.3 ViT Encoder with Temporal and Lead-Time Conditioning

The aggregated sequence is unflattened back to `(B, T)` time steps, a time positional embedding is added, the lead time is embedded by a linear layer and added to every token, and the sequence is encoded by the ViT transformer blocks:

```py linenums="155" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:155:165
--8<--
```

### 2.4 Temporal Aggregation and Prediction Head

The encoded time steps are averaged over the time dimension, and a cross-attention layer with a learned time query pools the result into a single token per sample:

```py linenums="166" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:166:169
--8<--
```

Finally, a linear head (`Linear(embed_dim, H * W)`) maps the pooled token to the predicted `tas` map:

```py linenums="191" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:191:194
--8<--
```

## 3. Model Training

### 3.1 Dataset Introduction

The case uses the ClimateBench v1.0 dataset, a benchmark for climate projection emulation built on CMIP6 scenario simulations. After regridding to the 5.625 degree grid (32 x 64 latitude-longitude cells), each input field is a `T x Vi x H x W` matrix, where `Vi = 4` input variables (`CO2`, `SO2`, `CH4`, `BC`), and the target `tas` is an `H x W` map.

The training and validation partitions are built from six simulations (`ssp126`, `ssp370`, `ssp585`, `historical`, `hist-GHG`, `hist-aer`): non-historical simulations are first concatenated with the `historical` period along the time axis, then a sliding window of `history=10` consecutive time steps forms each input sample, and the target is the field at the end point of the window. All samples are randomly split into training and validation subsets with `train_ratio=0.9` under the configured seed (`seed=42`), which yields 793 training samples and 89 validation samples in this case (`batch_size=1`). The test partition is built from the unseen `ssp245` simulation and keeps the last 21 samples.

Input normalization statistics are fitted on the training partition only and shared with the validation and test partitions, so the three partitions see a consistent input distribution.

### 3.2 Model Training

#### 3.2.1 Model Construction

This case is implemented based on the `ClimaXClimateBench` model, configured from the `MODEL` section of `conf/climax.yaml`: `img_size=[32, 64]` (the 5.625 degree grid), `time_history=10`, `patch_size=2`, `embed_dim=1024`, `depth=8`, `num_heads=16`, and `freeze_encoder=true` (the transformer blocks are frozen except their normalization parameters). Expressed in PaddleScience code as follows:

```py linenums="226" title="examples/climax/main.py"
--8<--
examples/climax/main.py:226:234
--8<--
```

The initial weights are loaded before the solver is built: `TRAIN.init_state_path` strictly loads a full initial state (backbone and heads; the `{"state_dict": ...}` wrapper and the `net.` key prefix are unwrapped automatically) and takes precedence over `TRAIN.pretrained_path`, which leniently loads a pre-trained MAE checkpoint after cleaning (positional-embedding interpolation, `channel`-to-`var` renaming and dropping of mismatched keys).

#### 3.2.2 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct a supervised constraint. Before defining the constraint, you need to first specify the training dataloader configuration, whose `dataset` entry names the dataset class `ClimateBenchDataset` built in PaddleScience together with its constructor arguments taken from the `DATA` section of the config file:

```py linenums="182" title="examples/climax/main.py"
--8<--
examples/climax/main.py:182:192
--8<--
```

The training loss is a plain mean squared error on the normalized fields, wrapped by `ppsci.loss.FunctionalLoss`. The code for defining supervised constraint is as follows:

```py linenums="194" title="examples/climax/main.py"
--8<--
examples/climax/main.py:194:200
--8<--
```

#### 3.2.3 Validator Construction

In this case, the training process will use the validation set to evaluate the training status of the current model at a certain training epoch interval, and `SupervisedValidator` needs to be used to construct the validator. The validator loss is the latitude-weighted MSE `w_mse`, and the metrics are the latitude-weighted family `w_mse`, `w_rmse` and `w_nrmse` (the latter combining the spread and grid nrmse variants), where `w_rmse` and `w_nrmse` are computed after denormalization. The code for defining the validator is as follows:

```py linenums="66" title="examples/climax/main.py"
--8<--
examples/climax/main.py:66:81
--8<--
```

The validation and test datasets reuse the normalization statistics fitted on the training partition (via `set_normalize`), and two validators are registered: `Val` for evaluation during training and `Test` for `mode=eval` and the post-training evaluation. The code for assembling them is as follows:

```py linenums="205" title="examples/climax/main.py"
--8<--
examples/climax/main.py:205:224
--8<--
```

`Val` is placed last in the validator dict so that the best-model metric tracked by the solver follows `Val/w_mse`.

#### 3.2.4 Learning Rate and Optimizer Construction

The learning rate follows a step-unit linear-warmup cosine schedule, `LinearWarmupCosineAnnealingLR`: `learning_rate=5e-4`, `warmup_epochs=60`, `eta_min=1e-8`. Both `warmup_epochs` and `max_epochs` count optimizer steps (793 per epoch), and the schedule restarts SGDR-style once `max_epochs=600` steps are exceeded. The optimizer uses `AdamW` (`beta_1=0.9`, `beta_2=0.999`) with two parameter groups: parameters named `var_embed` / `pos_embed` / `time_pos_embed` get no weight decay, the remaining trainable parameters get `weight_decay=1e-5`, and frozen parameters (excluded from gradient updates when `freeze_encoder=true`) are left out. The code for defining the optimizer is as follows:

```py linenums="154" title="examples/climax/main.py"
--8<--
examples/climax/main.py:154:177
--8<--
```

The code for initializing the learning rate scheduler and the optimizer is as follows:

```py linenums="236" title="examples/climax/main.py"
--8<--
examples/climax/main.py:236:240
--8<--
```

#### 3.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training. `to_static` is controlled by the `CLIMAX_USE_CINN` environment variable. After training finishes, the model is evaluated on both the validation and the test set:

```py linenums="242" title="examples/climax/main.py"
--8<--
examples/climax/main.py:242:262
--8<--
```

#### 3.2.6 Evaluation During Training

By setting the `eval_during_train` and `eval_freq` parameters in `ppsci.solver.Solver`, the model is evaluated on the validators once per epoch (`eval_freq=1`) during training, and the model parameters with the best effect on the validation set are automatically saved.

```py linenums="251" title="examples/climax/main.py"
--8<--
examples/climax/main.py:251:252
--8<--
```

### 3.3 Evaluation Model

#### 3.3.1 Validator Construction

For standalone evaluation (`mode=eval`), the test dataset is constructed in the same way as in training, reusing the normalization statistics fitted on the training partition, and a `Test` validator is built over it:

```py linenums="266" title="examples/climax/main.py"
--8<--
examples/climax/main.py:266:276
--8<--
```

#### 3.3.2 Load Model and Evaluate

The evaluation checkpoint given by `EVAL.pretrained_model_path` is loaded strictly before the solver is built (the `{"state_dict": ...}` wrapper and the `net.` key prefix are unwrapped automatically), then `ppsci.solver.Solver` is instantiated and evaluation starts:

```py linenums="278" title="examples/climax/main.py"
--8<--
examples/climax/main.py:278:297
--8<--
```

## 4. Complete Code

Dataset Interface:

```py linenums="1" title="ppsci/data/dataset/climatebench_dataset.py"
--8<--
ppsci/data/dataset/climatebench_dataset.py
--8<--
```

Model Structure (ClimaX backbone):

```py linenums="1" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py
--8<--
```

Model Structure (climate projection variant):

```py linenums="1" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py
--8<--
```

Model Training:

```py linenums="1" title="examples/climax/main.py"
--8<--
examples/climax/main.py
--8<--
```

Configuration File:

```py linenums="1" title="examples/climax/conf/climax.yaml"
--8<--
examples/climax/conf/climax.yaml
--8<--
```

## 5. Result Display

The numerical behavior of this implementation was verified against the torch implementation of ClimaX:

- Forward pass: with the same weights and a fixed input, the output of this implementation matches the torch reference with a mean absolute difference of 2.24e-08 (eval-mode forward on GPU; the dynamic-graph run gives 2.19e-08).
- Training: a 5-epoch run from the same initial weights (seed, FP32, single GPU, batch_size=1) reaches final test metrics within 7.3% of the torch baseline across the five latitude-weighted metrics.

The latitude-weighted test metrics of the 5-epoch run (best checkpoint) are:

| Test metric (5-epoch run, best checkpoint) | Torch baseline | This implementation |
| ------------------------------------------ | -------------- | ------------------- |
| w_mse                                      | 0.13900        | 0.13394             |
| w_rmse                                     | 0.36871        | 0.36180             |
| w_nrmse_tas                                | 0.34081        | 0.34945             |
| w_nrmses_tas                               | 0.17788        | 0.17455             |
| w_nrmseg_tas                               | 0.03258        | 0.03498             |
