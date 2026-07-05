# Renju Benchmark Design

## Sources

- RIF International Rules of Renju: https://www.renju.net/rifrules/
- RIF Qualification/Rating System: https://www.renju.net/qsystem/
- Japanese introductory pamphlet: https://renjusha.net/wp-content/uploads/2022/10/A4pamph.pdf

## Rule interpretation

The engine targets 15x15 Renju. Black starts. A move is normally a stone placement.
Passes, clocks, claims, record keeping, and organizer procedures are out of scope for automated LLM evaluation.

Win and draw rules included:

- exactly five in a row wins for either side;
- white overline also wins;
- black overline, double-four, and forbidden double-three lose unless the move simultaneously makes exactly five;
- full board is a draw.

Opening rules are intentionally excluded from initial benchmarks. Benchmarks start from specified positions,
which makes model comparisons reproducible and avoids mixing opening-rule memorization with tactical strength.

## Benchmark tracks

1. Rule classification:
   Given a board, side, and coordinate, classify the move as legal, win, forbidden, occupied, or off-board.
   This track should use strict labels generated offline.

2. Next move:
   Given a board and side, return one coordinate. Score:
   - 0.00 for illegal/forbidden moves;
   - 0.20 for legal but tactically losing moves;
   - 0.50 for legal moves that avoid immediate loss;
   - 0.75 for moves that block a critical threat or create an immediate threat;
   - 1.00 for known best or winning moves.
   Match-scale next-move evaluation should use fast mode and JSONL labels where possible.

3. Match play:
   Run models against fixed bots or each other with deterministic seeds. Aggregate win/draw/loss,
   illegal move rate, forbidden move rate, and average move count.

Recommended reporting metrics:

- win_rate
- draw_rate
- illegal_move_rate
- forbidden_move_rate
- occupied_move_rate
- off_board_rate
- avg_move_count
- parse_failure_rate
- rule_classification_accuracy
- tactical_best_move_accuracy
- must_block_accuracy

## Rating

RIF currently uses Whole-History Rating with Bradley-Terry likelihood, temporal smoothing, `w^2=19.3`,
and two virtual draws against a 1900-rated opponent for each player. That is more complex than needed for
small benchmark runs.

This package ships an Elo-style table initialized at 1900 with `K=32`, matching the spirit of the older
RIF appendix and common chess/shogi implementations. If match logs become large and dated, a WHR backend
can be added behind the same `RatingTable.record_game()` interface.

## Strict and fast modes

`RenjuGame` defaults to `RuleMode.STRICT`, which is intended for rule-classification and puzzle scoring.
For high-volume match simulations, use `RuleMode.FAST`. Fast mode still checks black overline and
double-four, but skips double-three validation. `RuleMode.PUZZLE` currently maps to strict depth and is reserved
for slower correctness-first validation.

The rule engine stores boards as flat 225-character strings. It avoids board-wide scans for forbidden move checks
and evaluates only the four lines crossing the candidate move plus nearby extension points. Straight-four analysis
is cached by local line strings instead of whole-board keys, so different boards with the same local pattern can
reuse results. Strict double-three checks still recurse for RIF 9.3-style validation, but `black_forbidden_report()`
is cached because repeated local positions occur frequently during move generation and bot search.

The game engine also checks virtual wins/forbidden moves before mutating the board, so a normal move copies the
board at most once. `Board.neighbor_empty_points()` provides local candidate generation for match simulation.

## Dataset split

Public examples should stay small and human-readable. Larger validation and hidden sets should be generated from
`scripts/generate_puzzles.py` with different seeds and validated with `scripts/validate_puzzles.py`.

JSONL records should declare their track:

- `track="rule_classification"` with `move` and `expected`
- `track="next_move"` with `best_moves`, `good_moves`, `blocking_moves`, and `forbidden_moves`

Rule-classification coordinates use a relaxed parser so off-board labels such as `P16`, `A0`, and `Z9` can be
tested. Next-move parsing also accepts off-board coordinates and scores them as illegal moves rather than parse
failures.

Recommended files:

- `data/public_examples.jsonl`
- `data/generated/validation_public.jsonl`
- `data/hidden_test.jsonl` on the benchmark host only

## Future optimization

The next major engine optimization is a mutable match-only `GameState` with candidate-set delta updates, or a
bitmask representation using precomputed radius-2 neighbor masks. That is intentionally separate from the immutable
`Board` used for puzzle labels and cache keys.

## Offline Evaluation

`scripts/evaluate_records.py` scores prediction JSONL files by record id and reports category averages by track and
tag. Prediction rows should look like:

```json
{"id": "rule_black_overline_seed42_00001", "response": "{\"class\":\"forbidden\"}"}
{"id": "next_exact_five_seed42_00001", "response": "{\"move\":\"J8\"}"}
```

This script is intended for local analysis and benchmark development; Kaggle Benchmarks tasks can still return a
single per-record score.
