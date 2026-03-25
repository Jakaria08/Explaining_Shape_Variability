#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pass through all CLI args to main.py
# Example:
#   ./run.sh --optuna_stage stage1 --n_trials 100 --parallel_mode model --model_parallel_gpus 0,1,2
python main.py "$@"
