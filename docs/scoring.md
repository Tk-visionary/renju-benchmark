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
- family scores, such as `family:noisy_exact_five`
- difficulty scores, such as `difficulty:hard`
- result-type rates, such as `next_move/result:off_board_rate`

Difficulty labels are assigned by the generator:

- `easy`: exact five, occupied, off-board
- `medium`: overline basics, must-block, tempting occupied
- `hard`: double-four, double-three, color contrast, exact-five exception
- `expert`: noisy board variants and future mixed traps

Noisy records are generated with unrelated stones away from the main tactical line. In noisy next-move records,
`best_moves` contains the intended labeled answers, but the evaluator still gives partial credit to winning moves
outside the explicit best set.

Each record also has a `family` field for dataset management. Family metrics avoid parsing IDs when comparing
families such as `double_three`, `noisy_black_overline`, or `exact_five_exception_overline`.

## Rule Mode

Rule-classification records are evaluated with their expected labels. Next-move records are scored with a mode chosen
from the record:

- explicit `mode`, when present;
- `strict` when tags include `strict` or `double_three`;
- otherwise `fast`.

FAST mode intentionally skips recursive double-three detection. STRICT mode should be used for double-three and other
strict forbidden-move cases.

Result-type rates are track-specific, for example `next_move/result:black_forbidden_rate`. This avoids mixing
rule-classification results such as `correct` with next-move results such as `legal` or `off_board`.
