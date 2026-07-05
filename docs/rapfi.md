# Rapfi Integration

This repository can call a Rapfi-compatible Gomocup engine through a thin Python subprocess wrapper.

Rapfi itself is not vendored here. Install or build Rapfi separately, then point the wrapper at the executable:

```bash
export RAPFI_PATH=/path/to/rapfi
export RAPFI_CWD=/path/to/rapfi/workdir  # optional, useful when config/weights are colocated
python scripts/rapfi_move.py board.txt --side black
```

The wrapper sends a Gomocup-style `START 15` command, then uses `BOARD` records with zero-based `x,y,player`
coordinates and parses the engine's `x,y` response.

Intended use:

- offline oracle for generating `best_moves` and `good_moves`;
- fixed opponent experiments;
- local validation of tactical puzzle families.

Kaggle Benchmarks tasks should remain Python-first and should prefer precomputed JSONL labels for reproducibility.
Calling Rapfi online from a task is possible only when the runtime has the executable and any required weights/config.

## Licensing

Rapfi is GPL-3.0. If you distribute Rapfi binaries, modified Rapfi sources, or a combined distribution, follow the
GPL-3.0 obligations for that distribution. This repository's wrapper does not include Rapfi binaries or weights.

