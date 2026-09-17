#!/bin/bash
#SBATCH -J mixtral-c006-expanded
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o output/mixtral_hotcold_c006/expanded_bpd/logs/%x-%A_%a.out
#SBATCH -e output/mixtral_hotcold_c006/expanded_bpd/logs/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PIM_TRACE_SCALE_REPEATS=0
py=/lustre/home/2501111916/anaconda3/bin/python
"$py" commands/run_mixtral_c006.py "$@"
