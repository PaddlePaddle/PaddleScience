#!/usr/bin/env bash
set -euo pipefail

export NVIDIA_TF32_OVERRIDE=0
export FLAGS_allocator_strategy=auto_growth

exec "${PYTHON:-python}" "$(dirname -- "${BASH_SOURCE[0]}")/run_cinn.py" "$@"
