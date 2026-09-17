#!/bin/bash
#SBATCH -J hotcold-smoke
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o output/mixtral_hotcold_c006/smoke-%j.out
#SBATCH -e output/mixtral_hotcold_c006/smoke-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PIM_TRACE_SCALE_REPEATS=0
root=/lustre/home/2501111916/workspace/Het-Infer/hetinfer-simplified-20260917
py=/lustre/home/2501111916/anaconda3/bin/python
cd "$root/het-infer"
PYTHONPATH=src "$py" -m pytest -q tests
cd "$root/dops"
PYTHONPATH=src "$py" -m pytest -q tests
run="$PWD/output/mixtral_hotcold_c006/smoke-$SLURM_JOB_ID"
/usr/bin/time -v "$py" commands/export_mixtral_hotcold_fast.py --batch 1 --horizon 2 --output "$run"
cd "$root/het-infer"
PYTHONPATH=src /usr/bin/time -v "$py" scripts/run_mixtral_hotcold_fast.py --bundle "$run/bundle.json" \
 --output "results/mixtral_hotcold_c006/smoke-$SLURM_JOB_ID" \
 --routes "results/mixtral_hotcold_c006/routes" \
 --policies R0 R2 --arms S0 S1 S2 --seeds 11 --warmup 0 --detail
"$py" - "results/mixtral_hotcold_c006/smoke-$SLURM_JOB_ID" <<'PY_CHECK'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = json.loads((root/'summary.json').read_text())
for row in rows:
    p = row['bifocal_parameters']
    assert (p['SCHED_JOINT_LK_H'], p['SCHED_JOINT_LK_GAMMA'], p['SCHED_JOINT_LK_CONSIST_LAMBDA'], p['SCHED_DECODE_AMORT_ALPHA']) == (2,.4,.5,0)
    assert math.isclose(row['preload_s']+row['prefill_s']+row['decode_s'], row['startup_inclusive_s'], rel_tol=1e-10)
    assert row['prefix_scheduler'] == ('dops_default' if row['arm'] == 'S0' else 'camc')
    for capacity in row['capacity_budget'].values():
        assert capacity['reserved_for_other_bytes'] >= 0
print('MIXTRAL_C006_SMOKE_OK', len(rows), flush=True)
PY_CHECK
