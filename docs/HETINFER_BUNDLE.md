# Het-Infer fast-mode export

DOPS `evaluate --config CONFIG --hetinfer-bundle-out bundle.json` captures the completed Bifocal schedules and writes one Het-Infer input file. It uses analytical NPU/PIM fast timing; it does not run tensors or generate tokens.

`src/hetinfer_capture.py` captures native placement, per-device service, directional movement, KV and TP collective semantics. `src/hetinfer_experiment_export.py` builds the final input in memory. No intermediate prior, network, profile, or tensor-binding files are generated.

The file contains:

- Graph/workload identity, model dimensions, device domains, weight capacities, and Bifocal parameters.
- A weight catalog with byte sizes and per-device host load costs.
- `expert_service_buckets`, shared by phase and operator family, indexed by routed expert token count.
- `movements`, with tensor, source, destination, byte count, layout, and duration in seconds.
- `networks`, one prefill or decode round each. A network holds its workload, domain capabilities, default order, and operator records.
- Each operator holds dependencies, role, layer/shard, legal/default/native devices, placement group, weight/KV identity, service times, inputs, and collective context.

A MoE expert remains one placement group. Its default device minimizes the sum of its operators' service times; `native_device` retains the original per-operator DOPS placement. Each network retains its own domain capabilities because these depend on phase and workload.

The runtime reads only the single bundle. Complete historical inputs can be converted with Het-Infer `scripts/convert_legacy_mixtral_bundle.py`; otherwise re-export with the current command.

## Experiment entrypoints

- `commands/export_mixtral_hotcold_fast.py`: Mixtral FFN-half TP=2 export with candidate parameters.
- `commands/export_hetinfer_dense_calibration_suite.py`: configured Qwen dense cases.
- `commands/run_mixtral_c006.py`: C006 matrix export, preflight, replay, verification, and summarization.
- `configs/mixtral_c006_expanded_matrix.json`: the approved workload matrix copied from the HPC deployment. Pass it with `--matrix`; run DOPS and Het-Infer from sibling `dops/` and `het-infer/` directories.

The old parameter-search entrypoint has been retired. Execute export, replay, and tests inside a Slurm compute allocation on HPC; submit jobs from the login node.

## Checks

```sh
MIXTRAL_MATRIX="$PWD/configs/mixtral_c006_expanded_matrix.json" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
```

The tests cover capture neutrality, TP collective resources and residencies, KV service/movement separation, whole-expert projection, integer token costs, matrix task coverage, export resume, and export reuse.
