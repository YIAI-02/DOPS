#!/bin/bash
# Run from deployed dops: MATRIX CPUS WALLTIME CONCURRENCY.
# Main-matrix resources are chosen after this preflight completes.
set -euo pipefail
matrix=$(realpath "$1")
cpus=$2
walltime=$3
concurrency=$4
py=/lustre/home/2501111916/anaconda3/bin/python
count=$("$py" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["preflight_workloads"]))' "$matrix")
mkdir -p output/mixtral_hotcold_c006/expanded_bpd/logs
verify=$(sbatch --parsable --cpus-per-task=1 --time=00:15:00 --job-name=c006-verify \
  commands/run_mixtral_c006.slurm verify --matrix "$matrix")
job=$(sbatch --parsable --dependency="afterok:$verify" --kill-on-invalid-dep=yes \
  --array="0-$((count-1))%$concurrency" --cpus-per-task="$cpus" \
  --time="$walltime" --job-name=c006-preflight commands/run_mixtral_c006.slurm preflight --matrix "$matrix")
printf '{"verification_job":"%s","preflight_job":"%s","cpus_per_task":%s,"concurrency":%s,"walltime":"%s","formal_submitted":false}\n' \
  "$verify" "$job" "$cpus" "$concurrency" "$walltime" | tee output/mixtral_hotcold_c006/expanded_bpd/preflight_jobs.json
