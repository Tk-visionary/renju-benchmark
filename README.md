# Renju Benchmark

Kaggle Benchmarks向けの連珠ベンチ試作です。

## What is included

- `renju_benchmark/rules.py`: 15x15連珠ルールエンジン
- `renju_benchmark/rating.py`: Elo風の対戦レーティング集計
- `renju_benchmark/tasks.py`: 採点・応答解析ヘルパー
- `renju_benchmark/kaggle_tasks.py`: Kaggle Benchmarks用タスクエントリポイント
- `renju_benchmark/agents.py`: 固定bot
- `data/public_examples.jsonl`: 公開サンプル
- `data/puzzles.jsonl`: 開発用の小さな問題セット
- `scripts/generate_puzzles.py`: seed指定の問題生成器
- `scripts/validate_puzzles.py`: JSONL問題の検証
- `scripts/evaluate_records.py`: 予測JSONLのカテゴリ別集計
- `scripts/summarize_records.py`: JSONL問題セットの構成集計
- `renju_benchmark/rapfi.py`: Rapfi/Gomocup互換エンジン用のPython wrapper
- `docs/rapfi.md`: Rapfi連携メモ
- `docs/design.md`: ルール・評価設計メモ
- `docs/scoring.md`: 採点仕様
- `docs/kaggle.md`: Kaggle Benchmarks連携メモ
- `tests/`: ルールとレーティングの基本テスト

## Rule scope

RIF International Rulesをベースに、盤面サイズ、勝敗、黒の禁手、引き分けを扱います。
正式競技の開局規定はベンチでは局面指定で回避し、任意局面からの次の一手・対局評価を対象にします。

## Rule modes

- `RuleMode.FAST`: 対局・大量評価用。黒の長連と四々は見るが、三々は省略。
- `RuleMode.STRICT`: ルール分類・通常パズル用。三々も再帰的に検証。
- `RuleMode.PUZZLE`: 現状は `STRICT` と同じ深さ。将来の追加検証用。

## Quick check

```bash
git clone https://github.com/Tk-visionary/renju-benchmark.git
cd renju-benchmark
python -m pip install -e ".[dev]"
python -m pytest -q
python -m renju_benchmark.demo
python scripts/validate_puzzles.py data/puzzles.jsonl
python scripts/generate_puzzles.py --seed 42 --count-per-family 10 --output data/generated/validation_public.jsonl
python scripts/validate_puzzles.py data/generated/validation_public.jsonl
python scripts/summarize_records.py data/generated/validation_public.jsonl
# predictions.jsonl is a model-output file with {"id": ..., "response": ...} rows.
# python scripts/evaluate_records.py data/generated/validation_public.jsonl predictions.jsonl
```

## Kaggle Benchmarks usage

Kaggle Notebook上で `kaggle_benchmarks` が使える環境なら、`renju_benchmark.kaggle_tasks`
をimportして `renju_next_move` / `renju_rule_classification` 系の `@kbench.task` エントリポイントを登録できます。
Kaggle向けの薄い入口は `renju_benchmark/kaggle_tasks.py` に置き、再利用可能な採点ロジックは
`renju_benchmark/tasks.py` に分離しています。

`renju_next_move` は大量評価では `mode="fast"`、禁手・例外を含む精密問題では `mode="strict"` を使います。
`renju_rule_classification` は strict 相当の少数精密問題に向けています。next-moveの採点は `best_moves`,
`good_moves`, `blocking_moves`, `forbidden_moves` を使えます。Scoring details are documented in
`docs/scoring.md`; Kaggle integration details are documented in `docs/kaggle.md`.

## Rapfi integration

Rapfi-compatible Gomocup engines can be used as optional offline oracles through `renju_benchmark.rapfi`.
The repository does not vendor Rapfi binaries or weights.

```bash
export RAPFI_PATH=/path/to/rapfi
python scripts/rapfi_move.py board.txt --side black
python scripts/rapfi_annotate.py data/generated/validation_public.jsonl data/generated/rapfi_annotations.jsonl
```

Use this primarily for dataset generation, fixed-opponent experiments, and local tactical validation. Kaggle
Benchmark tasks should prefer precomputed JSONL labels for reproducibility.

Rapfi is intentionally optional and not part of the official Kaggle score path. The official benchmark tasks are
Python-only and score against deterministic JSONL labels. Rapfi is best treated as a candidate proposer and local
analysis tool, not as the sole source of labels.
