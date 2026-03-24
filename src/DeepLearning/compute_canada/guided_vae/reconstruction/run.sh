#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=22:00:00
#SBATCH --account=def-uofavis-ab

set -euo pipefail

module load cuda cudnn
source ~/ENV/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

python main_gpu0_latent8.py > "$LOG_DIR/gpu0_latent8.log" 2>&1 &
PID0=$!
python main_gpu1_latent12.py > "$LOG_DIR/gpu1_latent12.log" 2>&1 &
PID1=$!
python main_gpu2_latent16.py > "$LOG_DIR/gpu2_latent16.log" 2>&1 &
PID2=$!

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?
wait "$PID2" || STATUS=$?

exit "$STATUS"
