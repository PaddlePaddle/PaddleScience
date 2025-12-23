# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GAOT主程序 - 统一的训练/评估/导出入口
输入: JSON/YAML配置文件, NetCDF数据集 | 输出: 训练模型, 评估结果 | 地位: 项目主入口
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

"""
GAOT (Geometry-Aware Operator Transformer)
Reproducing the GAOT Poisson-Gauss benchmark results with PaddleScience framework.

Reference: https://github.com/Shizheng-Wen/GAOT
Paper: "Geometry Aware Operator Transformer as an Efficient and Accurate 
        Neural Surrogate for PDEs on Arbitrary Domains" (NeurIPS 2025)
"""


import argparse
import json
import os
from os import path as osp
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import ppsci
from ppsci.utils import logger

if TYPE_CHECKING:
    pass


# ============================================================================
# JSON Configuration Loading
# ============================================================================


def load_json_config(json_path: str) -> SimpleNamespace:
    """
    Load configuration from JSON file.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        SimpleNamespace object with nested dict converted to attributes
    """
    with open(json_path, "r") as f:
        config_dict = json.load(f)

    def dict_to_namespace(d):
        """Recursively convert dict to SimpleNamespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    return dict_to_namespace(config_dict)


# ============================================================================
# Custom Loss Functions
# ============================================================================


def train_mse_func(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Training MSE loss function."""
    return {"mse": F.mse_loss(output_dict["u"], label_dict["u"])}


def eval_relative_l1_median_func(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    metadata: Optional[Dict] = None,
    *args,
) -> Dict[str, paddle.Tensor]:
    """
    Evaluation metric: relative L1 median error (matching GAOT paper).

    This uses the correct L1+median method with chunk grouping,
    matching the PyTorch implementation exactly.

    Args:
        output_dict: Model predictions {"u": tensor [B, N, U]}
        label_dict: Ground truth {"u": tensor [B, N, U]}
        metadata: Dataset metadata with normalization stats and chunk info

    Returns:
        Dictionary with "Rel_L1_Median" metric
    """
    pred = output_dict["u"]  # [B, N, U]
    true = label_dict["u"]  # [B, N, U]

    # If metadata not provided, use simplified L1 error
    if metadata is None:
        abs_error = paddle.abs(pred - true).sum()
        abs_true = paddle.abs(true).sum()
        relative_error = abs_error / (abs_true + 1e-8)
        return {"Rel_L1_Median": relative_error}

    # Add time dimension if missing (required by compute_batch_errors)
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)  # [B, N, U] -> [B, 1, N, U]
        true = true.unsqueeze(1)

    # Use correct L1+median computation with chunk grouping
    errors = compute_batch_errors(true, pred, metadata)  # [B, num_chunks]

    # Compute median over batch, then mean over chunks
    final_metric = compute_final_metric(errors)

    return {"Rel_L1_Median": paddle.to_tensor(final_metric)}


# ============================================================================
# Custom Dataset (TODO: Implement full GAOT dataset)
# ============================================================================


class GAOTDataset(paddle.io.Dataset):
    """
    GAOT Dataset for loading NetCDF format PDE data.

    Expected data format:
    - x: coordinates [N, coord_dim] or [B, N, coord_dim] for variable coords
    - c: input conditions [B, N, c_dim] (optional)
    - u: output solution [B, N, u_dim]

    Args:
        data_path: Path to the NetCDF data file
        mode: 'train', 'val', or 'test'
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        sample_rate: Subsample rate for point clouds
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        train_size: int = 1024,
        val_size: int = 128,
        test_size: int = 256,
        sample_rate: float = 1.0,
    ):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sample_rate = sample_rate

        # Load data
        self._load_data()

    def _load_data(self):
        """Load data from NetCDF file."""
        try:
            import netCDF4 as nc

            dataset = nc.Dataset(self.data_path, "r")

            # Load coordinates
            if "x" in dataset.variables:
                x = np.array(dataset.variables["x"][:])
            else:
                raise KeyError("Dataset must contain 'x' (coordinates)")

            # Load input conditions (optional)
            if "c" in dataset.variables:
                c = np.array(dataset.variables["c"][:])
            else:
                c = None

            # Load output solution
            if "u" in dataset.variables:
                u = np.array(dataset.variables["u"][:])
            else:
                raise KeyError("Dataset must contain 'u' (solution)")

            dataset.close()

            # Handle data format: squeeze extra dimensions
            # Expected: x [B, N, 2] or [N, 2], c [B, N, C], u [B, N, U]
            # Poisson-Gauss format: x [1, 1, N, 2], c [B, 1, N, C], u [B, 1, N, U]

            # Squeeze x coordinates
            while x.ndim > 2 and x.shape[0] == 1:
                x = x.squeeze(0)  # Remove batch dim if size 1
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)  # [1, N, 2] -> [N, 2]

            # Squeeze c and u
            if c is not None:
                while c.ndim > 3 and (c.shape[1] == 1 or c.shape[0] == 1):
                    if c.shape[1] == 1:
                        c = c.squeeze(1)  # Remove time dim
                    elif c.shape[0] == 1 and c.ndim > 3:
                        c = c.squeeze(0)

            while u.ndim > 3 and (u.shape[1] == 1 or u.shape[0] == 1):
                if u.shape[1] == 1:
                    u = u.squeeze(1)  # Remove time dim
                elif u.shape[0] == 1 and u.ndim > 3:
                    u = u.squeeze(0)

            # Determine data splits
            total_samples = u.shape[0]

            if self.mode == "train":
                start_idx = 0
                end_idx = min(self.train_size, total_samples)
            elif self.mode == "val":
                start_idx = self.train_size
                end_idx = min(self.train_size + self.val_size, total_samples)
            else:  # test
                start_idx = self.train_size + self.val_size
                end_idx = min(
                    self.train_size + self.val_size + self.test_size, total_samples
                )

            # Check if coordinates are fixed or variable
            if x.ndim == 2:
                # Fixed coordinates: [N, coord_dim]
                self.x = x.astype(np.float32)
                self.is_variable_coords = False
            else:
                # Variable coordinates: [B, N, coord_dim]
                self.x = x[start_idx:end_idx].astype(np.float32)
                self.is_variable_coords = True

            self.c = c[start_idx:end_idx].astype(np.float32) if c is not None else None
            self.u = u[start_idx:end_idx].astype(np.float32)

            # Apply subsampling if needed
            if self.sample_rate < 1.0:
                self._subsample()

            # Compute metadata for evaluation metrics
            # This includes global statistics for normalization and chunk grouping
            u_dim = self.u.shape[-1]
            self.metadata = {
                "active_variables": list(
                    range(u_dim)
                ),  # All output variables are active
                "global_mean": self.u.mean(axis=(0, 1)).tolist(),  # Global mean [u_dim]
                "global_std": self.u.std(axis=(0, 1)).tolist(),  # Global std [u_dim]
                "chunked_variables": list(
                    range(u_dim)
                ),  # Each variable in its own chunk
            }

            logger.info(f"Loaded {self.mode} dataset: {len(self)} samples")
            logger.info(
                f"  Coordinates shape: {self.x.shape} (fixed={not self.is_variable_coords})"
            )
            logger.info(
                f"  Input shape: {self.c.shape if self.c is not None else 'None'}"
            )
            logger.info(f"  Output shape: {self.u.shape}")
            logger.info("  Dataset metadata computed:")
            logger.info(f"    Global mean: {self.metadata['global_mean']}")
            logger.info(f"    Global std: {self.metadata['global_std']}")

        except ImportError:
            logger.warning("netCDF4 not installed. Using dummy data for testing.")
            self._create_dummy_data()
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            import traceback

            traceback.print_exc()
            logger.warning("Using dummy data instead.")
            self._create_dummy_data()

    def _create_dummy_data(self):
        """Create dummy data for testing without actual dataset."""
        num_samples = {"train": 64, "val": 16, "test": 32}[self.mode]
        num_points = 1024
        coord_dim = 2
        c_dim = 1
        u_dim = 1

        # Create random dummy data
        self.x = np.random.randn(num_points, coord_dim).astype(np.float32)
        self.c = np.random.randn(num_samples, num_points, c_dim).astype(np.float32)
        self.u = np.random.randn(num_samples, num_points, u_dim).astype(np.float32)
        self.is_variable_coords = False

        # Create metadata for dummy data
        self.metadata = {
            "active_variables": [0],
            "global_mean": [0.0],
            "global_std": [1.0],
            "chunked_variables": [0],
        }

        logger.warning(f"Using dummy data: {num_samples} samples, {num_points} points")

    def _subsample(self):
        """Subsample point cloud."""
        if self.is_variable_coords:
            num_points = self.x.shape[1]
        else:
            num_points = self.x.shape[0]

        num_sampled = int(num_points * self.sample_rate)
        indices = np.random.choice(num_points, num_sampled, replace=False)
        indices = np.sort(indices)

        if self.is_variable_coords:
            self.x = self.x[:, indices, :]
        else:
            self.x = self.x[indices, :]

        if self.c is not None:
            self.c = self.c[:, indices, :]
        self.u = self.u[:, indices, :]

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        """Get a single sample."""
        if self.is_variable_coords:
            x = paddle.to_tensor(self.x[idx])
        else:
            x = paddle.to_tensor(self.x)

        u = paddle.to_tensor(self.u[idx])

        if self.c is not None:
            c = paddle.to_tensor(self.c[idx])
            return {"x": x, "c": c}, {"u": u}
        else:
            return {"x": x}, {"u": u}


# ============================================================================
# Import Complete GAOT Model
# ============================================================================

from gaot_layers import GAOT
from gaot_layers import GAOTConfig
from gaot_layers import MAGNOConfig
from gaot_layers import TransformerConfig
from gaot_layers.metrics import compute_batch_errors
from gaot_layers.metrics import compute_final_metric

# ============================================================================
# GAOT Model Wrapper for ppsci
# ============================================================================


class GAOTModel(nn.Layer):
    """
    Complete GAOT model wrapper for ppsci framework.

    Architecture:
    - MAGNO Encoder (multi-scale attentional graph neural operator)
    - Patch-based Vision Transformer Processor
    - MAGNO Decoder

    Args:
        input_keys: Input tensor keys
        output_keys: Output tensor keys
        coord_dim: Coordinate dimension (2 or 3)
        input_dim: Input feature dimension (conditions)
        output_dim: Output feature dimension (solution)
        latent_tokens_size: Latent grid size [H, W] for 2D or [H, W, D] for 3D

        # MAGNO config
        radius: Neighbor search radius
        scales: Multi-scale factors
        use_attention: Whether to use attention in AGNO
        use_geoembed: Whether to use geometric embedding
        lifting_channels: Lifting layer output channels
        hidden_size: Hidden dimension for MLP layers
        mlp_layers: Number of MLP layers

        # Transformer config
        patch_size: Patch size for vision transformer
        num_transformer_layers: Number of transformer layers
        num_heads: Number of attention heads
        positional_embedding: Position encoding type ('absolute' or 'rope')
    """

    input_keys: Tuple[str, ...]
    output_keys: Tuple[str, ...]

    def __init__(
        self,
        input_keys: Tuple[str, ...] = ("x", "c"),
        output_keys: Tuple[str, ...] = ("u",),
        coord_dim: int = 2,
        input_dim: int = 1,
        output_dim: int = 1,
        latent_tokens_size: List[int] = None,
        # MAGNO config
        radius: float = 0.033,
        scales: List[float] = None,
        use_attention: bool = True,
        use_geoembed: bool = True,
        lifting_channels: int = 64,
        hidden_size: int = 128,
        mlp_layers: int = 3,
        # Transformer config
        patch_size: int = 8,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        positional_embedding: str = "absolute",
    ):
        super().__init__()

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.coord_dim = coord_dim

        # Set default values
        if latent_tokens_size is None:
            latent_tokens_size = [32, 32] if coord_dim == 2 else [16, 16, 16]

        if scales is None:
            scales = [1.0, 0.5, 0.25]

        # Create MAGNO config
        magno_config = MAGNOConfig(
            coord_dim=coord_dim,
            radius=radius,
            hidden_size=hidden_size,
            mlp_layers=mlp_layers,
            lifting_channels=lifting_channels,
            scales=scales,
            use_attention=use_attention,
            use_geoembed=use_geoembed,
            transform_type="linear",
            attention_type="cosine",
        )

        # Create Transformer config
        transformer_config = TransformerConfig(
            patch_size=patch_size,
            hidden_size=lifting_channels,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            positional_embedding=positional_embedding,
            ffn_multiplier=4,
        )

        # Create GAOT config
        gaot_config = GAOTConfig(
            input_size=input_dim,
            output_size=output_dim,
            coord_dim=coord_dim,
            latent_tokens_size=latent_tokens_size,
            magno=magno_config,
            transformer=transformer_config,
        )

        # Initialize complete GAOT model
        self.gaot = GAOT(config=gaot_config)

        # Generate latent grid coordinates
        self.latent_tokens_coord = self._generate_latent_grid(
            latent_tokens_size, coord_dim
        )

    def _generate_latent_grid(self, size: List[int], coord_dim: int) -> paddle.Tensor:
        """Generate uniform latent grid coordinates."""
        if coord_dim == 2:
            H, W = size
            h = paddle.linspace(0, 1, H, dtype="float32")
            w = paddle.linspace(0, 1, W, dtype="float32")
            grid_h, grid_w = paddle.meshgrid(h, w)
            coords = paddle.stack([grid_h.flatten(), grid_w.flatten()], axis=-1)
        else:  # 3D
            H, W, D = size
            h = paddle.linspace(0, 1, H, dtype="float32")
            w = paddle.linspace(0, 1, W, dtype="float32")
            d = paddle.linspace(0, 1, D, dtype="float32")
            grid_h, grid_w, grid_d = paddle.meshgrid(h, w, d)
            coords = paddle.stack(
                [grid_h.flatten(), grid_w.flatten(), grid_d.flatten()], axis=-1
            )

        return coords

    def forward(self, input_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward pass.

        Args:
            input_dict: Dictionary containing:
                - 'x': coordinates [B, N, coord_dim] or [N, coord_dim] for fx mode
                - 'c': conditions [B, N, c_dim] (optional)

        Returns:
            Dictionary containing:
                - 'u': predicted solution [B, N, u_dim]
        """
        x_coord = input_dict["x"]  # Coordinates

        # Prepare input features
        if "c" in input_dict and input_dict["c"] is not None:
            pndata = input_dict["c"]  # [B, N, c_dim]
        else:
            # If no conditions, use dummy input
            if x_coord.ndim == 3:
                batch_size, num_points = x_coord.shape[:2]
            else:
                num_points = x_coord.shape[0]
                # Create dummy batch dimension
                x_coord = x_coord.unsqueeze(0)
                batch_size = 1

            pndata = paddle.zeros([batch_size, num_points, 1], dtype=x_coord.dtype)

        # Forward through GAOT
        output = self.gaot(
            latent_tokens_coord=self.latent_tokens_coord,
            xcoord=x_coord,
            pndata=pndata,
        )

        return {self.output_keys[0]: output}


# ============================================================================
# Training and Evaluation Functions
# ============================================================================


def train(cfg: SimpleNamespace):
    """Training function."""
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.setup.seed)

    # Initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    logger.info("=" * 60)
    logger.info("GAOT Training")
    logger.info("=" * 60)

    # Build complete GAOT model with JSON config mapping
    model = GAOTModel(
        input_keys=("x", "c"),
        output_keys=("u",),
        coord_dim=cfg.model.args.magno.coord_dim,
        input_dim=1,
        output_dim=1,
        latent_tokens_size=cfg.model.latent_tokens_size,
        # MAGNO config
        radius=cfg.model.args.magno.radius,
        scales=[1.0, 0.5, 0.25],  # Default scales, not in JSON
        use_attention=True,
        use_geoembed=True,
        lifting_channels=cfg.model.args.magno.lifting_channels,
        hidden_size=cfg.model.args.magno.hidden_size,
        mlp_layers=cfg.model.args.magno.mlp_layers,
        # Transformer config
        patch_size=cfg.model.args.transformer.patch_size,
        num_transformer_layers=3,  # Default value
        num_heads=8,  # Default value
        positional_embedding="absolute",
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataset path from JSON config
    data_path = os.path.join(cfg.dataset.base_path, f"{cfg.dataset.name}.nc")

    # Build dataset and dataloader
    train_dataset = GAOTDataset(
        data_path=data_path,
        mode="train",
        train_size=cfg.dataset.train_size,
        val_size=cfg.dataset.val_size,
        test_size=cfg.dataset.test_size,
        sample_rate=1.0,
    )

    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "x": train_dataset.x if not train_dataset.is_variable_coords else None
            },
            "label": {"u": train_dataset.u},
        },
        "batch_size": cfg.dataset.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": cfg.dataset.shuffle,
        },
        "num_workers": cfg.dataset.num_workers,
    }

    # Build constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # Build optimizer with JSON config
    lr = cfg.optimizer.args.lr
    weight_decay = cfg.optimizer.args.weight_decay
    epochs = cfg.optimizer.args.epoch

    lr_scheduler = ppsci.optimizer.lr_scheduler.CosineAnnealingDecay(
        epochs=epochs,
        learning_rate=lr,
        eta_min=1e-6,
    )()
    optimizer = ppsci.optimizer.AdamW(
        learning_rate=lr_scheduler,
        weight_decay=weight_decay,
    )(model)

    # Build validator
    val_dataset = GAOTDataset(
        data_path=data_path,
        mode="val",
        train_size=cfg.dataset.train_size,
        val_size=cfg.dataset.val_size,
        test_size=cfg.dataset.test_size,
        sample_rate=1.0,
    )

    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "x": val_dataset.x if not val_dataset.is_variable_coords else None
            },
            "label": {"u": val_dataset.u},
        },
        "batch_size": cfg.dataset.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    # Get metadata from validation dataset for accurate evaluation
    eval_metadata = val_dataset.metadata

    # Create evaluation function with metadata
    def eval_func_with_metadata(output_dict, label_dict, *args):
        return eval_relative_l1_median_func(
            output_dict, label_dict, eval_metadata, *args
        )

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"u": lambda out: out["u"]},
        metric={
            "Rel_L1_Median": ppsci.metric.FunctionalMetric(eval_func_with_metadata)
        },
        name="Val",
    )
    validator = {sup_validator.name: sup_validator}

    # Initialize solver with JSON config
    eval_freq = cfg.optimizer.args.eval_every_eps

    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        epochs=epochs,
        iters_per_epoch=None,
        save_freq=eval_freq,
        eval_during_train=True,
        eval_freq=eval_freq,
        validator=validator,
        eval_with_no_grad=True,
    )

    # Train
    solver.train()

    logger.info("Training completed!")


def evaluate(cfg: SimpleNamespace):
    """Evaluation function."""
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.setup.seed)

    # Initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    logger.info("=" * 60)
    logger.info("GAOT Evaluation")
    logger.info("=" * 60)

    # Build complete GAOT model with JSON config mapping
    model = GAOTModel(
        input_keys=("x", "c"),
        output_keys=("u",),
        coord_dim=cfg.model.args.magno.coord_dim,
        input_dim=1,
        output_dim=1,
        latent_tokens_size=cfg.model.latent_tokens_size,
        # MAGNO config
        radius=cfg.model.args.magno.radius,
        scales=[1.0, 0.5, 0.25],
        use_attention=True,
        use_geoembed=True,
        lifting_channels=cfg.model.args.magno.lifting_channels,
        hidden_size=cfg.model.args.magno.hidden_size,
        mlp_layers=cfg.model.args.magno.mlp_layers,
        # Transformer config
        patch_size=cfg.model.args.transformer.patch_size,
        num_transformer_layers=3,
        num_heads=8,
        positional_embedding="absolute",
    )

    # Build dataset path from JSON config
    data_path = os.path.join(cfg.dataset.base_path, f"{cfg.dataset.name}.nc")

    # Build test dataset
    test_dataset = GAOTDataset(
        data_path=data_path,
        mode="test",
        train_size=cfg.dataset.train_size,
        val_size=cfg.dataset.val_size,
        test_size=cfg.dataset.test_size,
        sample_rate=1.0,
    )

    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "x": test_dataset.x if not test_dataset.is_variable_coords else None
            },
            "label": {"u": test_dataset.u},
        },
        "batch_size": cfg.dataset.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    # Get metadata from test dataset for accurate evaluation
    eval_metadata = test_dataset.metadata

    # Create evaluation function with metadata
    def eval_func_with_metadata(output_dict, label_dict, *args):
        return eval_relative_l1_median_func(
            output_dict, label_dict, eval_metadata, *args
        )

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"u": lambda out: out["u"]},
        metric={
            "Rel_L1_Median": ppsci.metric.FunctionalMetric(eval_func_with_metadata)
        },
        name="Test",
    )
    validator = {sup_validator.name: sup_validator}

    # Initialize solver
    pretrained_path = cfg.path.ckpt_path if hasattr(cfg, "path") else None

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.setup.seed,
        validator=validator,
        pretrained_model_path=pretrained_path,
        eval_with_no_grad=True,
    )

    # Evaluate
    solver.eval()

    logger.info("Evaluation completed!")


def export(cfg: SimpleNamespace):
    """Export model for deployment."""
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.setup.seed)

    # Initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "export.log"), "info")

    logger.info("=" * 60)
    logger.info("GAOT Model Export")
    logger.info("=" * 60)

    # Build complete GAOT model with JSON config mapping
    model = GAOTModel(
        input_keys=("x", "c"),
        output_keys=("u",),
        coord_dim=cfg.model.args.magno.coord_dim,
        input_dim=1,
        output_dim=1,
        latent_tokens_size=cfg.model.latent_tokens_size,
        # MAGNO config
        radius=cfg.model.args.magno.radius,
        scales=[1.0, 0.5, 0.25],
        use_attention=True,
        use_geoembed=True,
        lifting_channels=cfg.model.args.magno.lifting_channels,
        hidden_size=cfg.model.args.magno.hidden_size,
        mlp_layers=cfg.model.args.magno.mlp_layers,
        # Transformer config
        patch_size=cfg.model.args.transformer.patch_size,
        num_transformer_layers=3,
        num_heads=8,
        positional_embedding="absolute",
    )

    # Initialize solver for export
    pretrained_path = cfg.path.ckpt_path if hasattr(cfg, "path") else None
    export_path = (
        cfg.path.ckpt_path.replace(".pt", "_exported")
        if hasattr(cfg, "path")
        else "./exported_model"
    )

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.setup.seed,
        pretrained_model_path=pretrained_path,
    )

    # Export
    from paddle.static import InputSpec

    input_spec = [
        {
            "x": InputSpec(
                [None, None, cfg.model.args.magno.coord_dim], "float32", name="x"
            ),
            "c": InputSpec([None, None, 1], "float32", name="c"),
        }
    ]
    solver.export(input_spec, export_path)

    logger.info(f"Model exported to: {export_path}")


def inference(cfg: SimpleNamespace):
    """Inference with exported model."""
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.setup.seed)

    # Initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "infer.log"), "info")

    logger.info("=" * 60)
    logger.info("GAOT Inference")
    logger.info("=" * 60)

    # TODO: Implement inference with exported model
    logger.warning("Inference mode not fully implemented yet.")
    logger.info("Please use 'eval' mode for model evaluation.")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GAOT Training with JSON Configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/work/GAOT/config/examples/time_indep/poisson_gauss.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "export", "infer"],
        help="Execution mode",
    )

    args = parser.parse_args()

    # Load JSON configuration
    logger.info(f"Loading configuration from: {args.config}")
    cfg = load_json_config(args.config)
    cfg.mode = args.mode

    # Add output_dir based on JSON config
    cfg.output_dir = os.path.dirname(cfg.path.ckpt_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Execution mode: {cfg.mode}")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)


if __name__ == "__main__":
    main()
