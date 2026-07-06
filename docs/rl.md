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

You can also collect fast tactical-floor labels without starting Rapfi:

```bash
python scripts/rl_collect_tactical.py \
  --count 1000 \
  --output data/generated/rl/tactical_1k.jsonl \
  --seed 13 \
  --min-plies 0 \
  --max-plies 12
```

These rows use the same `policy_index` training schema and can be mixed with Rapfi labels. They are useful for
pretraining immediate wins, blocks, double threats, and safety before slower Rapfi imitation.

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

## HRM-Style Model

The experimental `--model-type hrm` model keeps low-level and high-level latent feature maps and updates them for a
fixed number of recurrent cycles before producing the policy/value heads. It is intended as a Renju adaptation of the
HRM idea: latent board-state refinement instead of one-shot policy prediction.

```bash
python scripts/rl_train_imitation.py \
  --input data/generated/rl/tactical_1k.jsonl \
  --output data/generated/rl/hrm_policy_value.pt \
  --model-type hrm \
  --channels 32 \
  --hrm-cycles 4 \
  --hrm-low-steps 2 \
  --epochs 3
```

The first comparison target is HRM versus the default ResNet on tactical labels, then Rapfi imitation labels. HRM should
be treated as a policy/value and tactical-feature model; it still needs tactical filtering or search for play.

## Rapfi-Free HRM Self-Play

You can start from a very weak randomly initialized HRM and grow it only from masked-legal self-play. Move selection
always applies the strict legal policy mask before sampling, so occupied, off-board, and black-forbidden moves are not
chosen by the player even when the network weights are random:

```bash
python scripts/rl_init_random_hrm.py \
  --output data/generated/rl/selfplay/hrm_random.pt \
  --seed 123 \
  --channels 16 \
  --hrm-cycles 1 \
  --hrm-low-steps 1

python scripts/rl_collect_self_play.py \
  --checkpoint data/generated/rl/selfplay/hrm_random.pt \
  --output data/generated/rl/selfplay/iter0.jsonl \
  --games 16 \
  --seed 123 \
  --channels 16 \
  --hrm-cycles 1 \
  --hrm-low-steps 1 \
  --temperature 1.5 \
  --epsilon 0.25

python scripts/rl_train_self_play.py \
  --init-checkpoint data/generated/rl/selfplay/hrm_random.pt \
  --input data/generated/rl/selfplay/iter0.jsonl \
  --output data/generated/rl/selfplay/hrm_iter1.pt \
  --epochs 3 \
  --channels 16 \
  --hrm-cycles 1 \
  --hrm-low-steps 1
```

This is intentionally weak at first. The goal is to establish a closed loop: random HRM policy, legal self-play rows
with final outcome values, HRM policy/value update, then next self-play generation. Tactical labels or symbolic rules can
be mixed in later, but Rapfi is not required for the loop.

Initial local timing on an 8-channel, 1-cycle, 1-low-step HRM:

- 4 games x 20 plies: 80 positions, self-play collection about 1.3 seconds, 1 training epoch about 1.8 seconds.
- 32 games x 20 plies: 640 positions, self-play collection about 4.1 seconds, 1 training epoch about 4.1 seconds.
- Batched collection with `--parallel-games 32` reduces 32 games x 20 plies from about 4.3 seconds to 3.7 seconds.
- Batched collection with `--parallel-games 32` reduces 128 games x 20 plies from about 13.5 seconds to 11.6 seconds.
- 128 games x 20 plies with batched collection plus 1 training epoch is about 27.8 seconds total.

The self-play collector keeps one model process in memory and reseeds the sampler per game. Early experiments should
stay with small HRM settings and short `--max-plies`; once the loop shows learning signal, the next speed target is the
strict legality path. The Python rule engine is good for correctness, but larger self-play runs should move hot legality
checks behind a compiled extension or a bitboard-backed implementation while keeping the same Python API.

## Symbolic Rule Learning

The non-neural route is to learn weights over interpretable tactical rules. This keeps the engine debuggable while still
allowing Rapfi/tactical labels to improve move ordering:

```bash
python scripts/rl_fit_symbolic.py \
  --input data/generated/rl/tactical_1k.jsonl \
  --output data/generated/rl/symbolic_weights.json \
  --epochs 5

python scripts/rl_eval_symbolic.py \
  --input data/generated/rl/tactical_1k.jsonl \
  --weights data/generated/rl/symbolic_weights.json
```

The first implementation scores tactical roles such as immediate win, block, force-win, threat, unsafe, center bias,
edge distance, self/opponent run length, and open ends. This is the starting point for a learned rule-based engine:
feature weights, rule ordering, candidate thresholds, and search limits can be optimized without training a neural
network.

For a benchmark-derived sanity check, generate next-move records and convert them into imitation examples:

```bash
python scripts/generate_puzzles.py \
  --seed 92 \
  --count-per-family 50 \
  --output data/generated/rl/symbolic_benchmark_records_50pf.jsonl

python scripts/rl_convert_next_move_records.py \
  --input data/generated/rl/symbolic_benchmark_records_50pf.jsonl \
  --output data/generated/rl/symbolic_benchmark_next_50pf.jsonl
```

On a 300/100 split of that 400-example next-move set, the symbolic learner improved the held-out test slice from
top-1 `0.64` / top-5 `0.88` to top-1 `0.78` / top-5 `0.96` with `--candidate-limit 32`,
`--force-reply-limit 4`, and `--threat-forbidden-depth 0`. The depth-0 setting is a fast diagnostic mode; use the
default depth for stricter tactical labels.

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
Use `--force-reply-limit` to trade speed for deeper checks of opponent double-threat replies.

If `--tactical` is used without `--checkpoint`, the evaluator runs the same role-prioritized tactical search as a
deterministic baseline. Use this to compare the learned policy against the hand-written tactical floor.

Compare a checkpoint against that tactical floor without starting Rapfi:

```bash
python scripts/rl_evaluate_vs_tactical.py \
  --checkpoint data/generated/rl/policy_value.pt \
  --games 20 \
  --max-plies 120
```

## Experiment Runner

Use `scripts/rl_run_experiment.py` to run the current MVP loop and keep artifacts together:

```bash
python scripts/rl_run_experiment.py \
  --run-dir data/generated/rl/runs/smoke-001 \
  --count 16 \
  --tactical-count 64 \
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

`metrics.json` records imitation accuracy, tactical-baseline WDL, Rapfi WDL, illegal rate, side scores, and per-game
move logs. This is the main artifact for tracking progress toward the first target: score above 0.55 against weak Rapfi.

Compare multiple runs:

```bash
python scripts/rl_summarize_runs.py data/generated/rl/runs
python scripts/rl_summarize_runs.py data/generated/rl/runs --min-rapfi-score 0.55
```

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
