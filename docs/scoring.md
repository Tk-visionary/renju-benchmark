# Scoring

Renju Benchmark uses deterministic JSONL labels for official scoring.

## Rule Classification

The model returns one JSON object:

```json
{"class": "forbidden"}
```

Score is `1.0` when the extracted class exactly matches `expected`, otherwise `0.0`.
Parse failures score `0.0`.

## Next Move

The model returns one JSON object:

```json
{"move": "H8"}
```

Move scoring:

- `1.00`: move is in `best_moves`, or it wins when no explicit best set is provided
- `0.80`: move is in `good_moves`
- `0.75`: move is in `blocking_moves`, creates an immediate threat, or wins outside the explicit best set
- `0.50`: legal neutral move that avoids immediate tactical collapse
- `0.20`: legal move that leaves an immediate opponent win or loses to the heuristic reply
- `0.00`: parse failure, occupied point, off-board move, black forbidden move, or move in `forbidden_moves`

Coordinates such as `P16` and `A0` are parsed as coordinates, then scored as off-board moves rather than parser
failures. Non-coordinate output is a parser failure.

## Metrics

`scripts/evaluate_records.py` reports:

- overall score
- track scores, such as `rule_classification` and `next_move`
- namespaced tag scores, such as `tag:forbidden`
- track/tag scores, such as `rule_classification/tag:forbidden`
- difficulty scores, such as `difficulty:hard`
- result-type rates, such as `next_move/result:off_board_rate`

