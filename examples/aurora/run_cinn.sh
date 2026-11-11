#!/usr/bin/env bash
set -euo pipefail

export FLAGS_prim_all=true
export FLAGS_use_cinn=true
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_allocator_strategy=auto_growth

exec "${PYTHON:-python}" "$(dirname -- "${BASH_SOURCE[0]}")/run_cinn.py" "$@"
