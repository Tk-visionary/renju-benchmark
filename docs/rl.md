# Rapfi RL Research Track

This track is separate from the Kaggle Benchmark tasks. The goal is to build a compact Renju model that can first beat
weak Rapfi settings, then climb toward stronger Rapfi configurations.

## Stages

1. **Rapfi environment**
   - Keep Rapfi optional and local.
   - Use `RAPFI_PATH` or `--rapfi-path` to call a Rapfi-compatible engine through the Gomocup protocol.
2. **Rapfi imitation**
   - Generate reachable positions.
   - Ask Rapfi for best moves.
   - Train a policy/value network on `board -> rapfi_best`.
3. **Neural-guided tactics**
   - Use legal masks, immediate win detection, blocks, and forbidden masks.
   - Let the network rank tactical candidates instead of selecting from all 225 points blindly.
4. **Self-play and sparring**
   - Add self-play after imitation is stable.
   - Evaluate against Rapfi 10ms, 30ms, 100ms, then stronger settings.

## Data Collection

Set up Rapfi locally first. This clones Rapfi under ignored `external/` and attempts an ARM64 NEON or x64 AVX2 build:

```bash
python scripts/setup_rapfi.py
export RAPFI_PATH=/absolute/path/printed/by/setup
export RAPFI_CWD=/absolute/runtime/dir/printed/by/setup
```

Rapfi is GPL-3.0; this repository does not vendor its source, binaries, networks, or generated build artifacts.
The setup script clones and builds Rapfi under ignored `external/`, then copies the executable, `config.toml`, and
network files into ignored `external/rapfi-runtime`.

```bash
export RAPFI_PATH=/path/to/rapfi
python scripts/rl_collect_rapfi.py --count 1000 --output data/generated/rl/rapfi_1k.jsonl
```

For weak-Rapfi curriculum data, cap the engine budget:

```bash
python scripts/rl_collect_rapfi.py \
  --count 1000 \
  --output data/generated/rl/rapfi_1k.jsonl \
  --rapfi-path external/rapfi-runtime/pbrain-rapfi \
  --rapfi-cwd external/rapfi-runtime \
  --timeout-turn-ms 100 \
  --max-node 1000
```

For a quick local smoke test, keep positions shallow:

```bash
python scripts/rl_collect_rapfi.py \
  --count 16 \
  --output data/generated/rl/rapfi_16_shallow.jsonl \
  --seed 13 \
  --min-plies 0 \
  --max-plies 8 \
  --rapfi-path external/rapfi-runtime/pbrain-rapfi \
  --rapfi-cwd external/rapfi-runtime \
  --timeout-turn-ms 50 \
  --max-node 300 \
  --max-depth 2
```

Current integration note: the built Rapfi runtime has been verified for empty-board and intermediate-position
best-move annotation. Rapfi may emit the best move as a PV in `MESSAGE Depth ... | H5 ...` rather than a raw `x,y`
line, and the Python wrapper handles both formats.

Each row contains:

```json
{
  "board": "...",
  "side": "black",
  "rapfi_best": "H8",
  "policy_index": 112,
  "source": "rapfi_best_move"
}
```

## Imitation Training

PyTorch is optional for the repository, but required for training:

```bash
python scripts/rl_train_imitation.py \
  --input data/generated/rl/rapfi_1k.jsonl \
  --output data/generated/rl/policy_value.pt \
  --epochs 3
```

Measure imitation accuracy:

```bash
python scripts/rl_eval_imitation.py \
  --checkpoint data/generated/rl/policy_value.pt \
  --input data/generated/rl/rapfi_1k.jsonl \
  --top-k 5
```

Smoke-test a trained checkpoint:

```bash
python scripts/rl_policy_move.py data/generated/rl/policy_value.pt board.txt --side black
python scripts/rl_policy_move.py data/generated/rl/policy_value.pt board.txt --side black --tactical
```

The first model is intentionally small:

- 15x15 planes
- black stones
- white stones
- side to move
- legal mask
- last move
- small residual CNN
- 225-way policy head
- scalar value head

## Evaluation Against Rapfi

The first evaluation target is a baseline move function versus weak Rapfi settings:

```bash
python scripts/rl_evaluate_vs_rapfi.py --games 20 --max-plies 120
```

Evaluate a trained checkpoint instead of the heuristic baseline:

```bash
python scripts/rl_evaluate_vs_rapfi.py \
  --checkpoint data/generated/rl/policy_value.pt \
  --tactical \
  --games 20 \
  --max-plies 120 \
  --rapfi-path external/rapfi-runtime/pbrain-rapfi \
  --rapfi-cwd external/rapfi-runtime \
  --timeout-turn-ms 100 \
  --max-node 1000
```

`rl_evaluate_vs_rapfi.py` uses a fresh Rapfi process for each Rapfi move by default. This is slower, but it avoids
stale protocol output in early experiments. `--reuse-rapfi-process` is available for faster runs once protocol handling
is stable for the chosen setting.

`--tactical` makes the policy/value model choose among tactical candidates: immediate wins, immediate blocks, and
nearby legal points. It prioritizes double threats that create multiple next-turn wins, then single winning threats, and
filters out moves that allow an immediate winning reply when there is a safer candidate. This is the default direction
for beating weak Rapfi; raw policy-only moves are mainly useful for debugging imitation quality.

If `--tactical` is used without `--checkpoint`, the evaluator runs the same role-prioritized tactical search as a
deterministic baseline. Use this to compare the learned policy against the hand-written tactical floor.

## Experiment Runner

Use `scripts/rl_run_experiment.py` to run the current MVP loop and keep artifacts together:

```bash
python scripts/rl_run_experiment.py \
  --run-dir data/generated/rl/runs/smoke-001 \
  --count 16 \
  --seed 13 \
  --min-plies 0 \
  --max-plies 8 \
  --epochs 3 \
  --eval-games 2 \
  --eval-max-plies 12
```

The run directory contains:

- `config.json`
- `rapfi_examples.jsonl`
- `policy_value.pt`
- `metrics.json`

`metrics.json` records imitation accuracy, WDL score, illegal rate, side scores, and per-game move logs. This is the
main artifact for tracking progress toward the first target: score above 0.55 against weak Rapfi.

The evaluator prints a JSON report with aggregate WDL metrics and per-game logs:

```json
{
  "summary": {
    "games": 20,
    "score": 0.55,
    "win_rate": 0.4,
    "draw_rate": 0.3,
    "loss_rate": 0.3,
    "illegal_rate": 0.0,
    "black_score": 0.6,
    "white_score": 0.5
  },
  "games": []
}
```

Use `--games-only` if you need the previous raw list format.

For local Rapfi settings, configure the engine through its own config or use a low move timeout:

```bash
python scripts/rl_evaluate_vs_rapfi.py --games 20 --move-timeout 0.05
```

## Near-Term Goal

The first concrete target is not full-strength Rapfi. It is:

- 200 alternating-side games
- Renju rules
- weak Rapfi setting, such as 10ms per move
- win rate above 55%

After that, raise the opponent budget gradually.
