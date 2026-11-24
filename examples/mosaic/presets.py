# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Released Mosaic configurations, matching the pinned inference.py presets."""

SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
PRESSURE_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
VARIABLES = SURFACE_VARIABLES + [
    f"{name}_{level}" for name in PRESSURE_VARIABLES for level in LEVELS
]
STATIC_VARIABLES = ["geopotential_at_surface", "land_sea_mask", "soil_type"]


def model_config(variant):
    if variant not in ("era5", "hres"):
        raise ValueError(f"Unknown Mosaic variant {variant}")
    stages = [
        dict(
            nside=64,
            dim=768,
            num_heads=12,
            block_attn_size=1024,
            sparse_block_size=128,
            sparse_block_count=24,
            encoder_depth=4,
            decoder_depth=2,
            mlp_ratio=4.0,
            gqa_ratio=4,
        ),
        dict(
            nside=32,
            dim=1024,
            num_heads=16,
            block_attn_size=1024,
            sparse_block_size=128,
            sparse_block_count=12,
            encoder_depth=4,
            decoder_depth=2,
            mlp_ratio=4.0,
            gqa_ratio=4,
        ),
    ]
    bottleneck = dict(
        nside=16,
        dim=1280,
        num_heads=20,
        block_attn_size=1024,
        sparse_block_size=128,
        sparse_block_count=4,
        depth=2,
        mlp_ratio=4.0,
        gqa_ratio=4,
    )
    return dict(
        dim=768,
        num_heads=12,
        k_neighbors=24 if variant == "era5" else 20,
        qk_norm=False,
        rope=True,
        rope_theta=10000,
        sparse_every=1,
        variables=VARIABLES,
        static_variables=STATIC_VARIABLES,
        qkv_compress_ratio=1,
        cg_stage_cfgs=stages,
        bottleneck_cfg=bottleneck,
        num_history_steps=2 if variant == "era5" else 4,
        noise_dim=32,
        rmsnorm_elementwise_affine=False,
    )
