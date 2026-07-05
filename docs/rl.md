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

```bash
export RAPFI_PATH=/path/to/rapfi
python scripts/rl_collect_rapfi.py --count 1000 --output data/generated/rl/rapfi_1k.jsonl
```

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

