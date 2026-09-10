#!/bin/bash
#SBATCH -J dense-fast-export
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o results/dense_cal_export.%j.out
#SBATCH -e results/dense_cal_export.%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd /lustre/home/2501111916/workspace/Het-Infer/dops-camc-work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
/lustre/home/2501111916/anaconda3/bin/python commands/export_hetinfer_dense_calibration_suite.py "$@"
