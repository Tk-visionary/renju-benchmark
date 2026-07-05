# Kaggle Benchmarks Usage

Kaggle Benchmarks tasks are Python functions decorated with `@kbench.task`. The first argument is the LLM under test;
additional arguments are task inputs. Kaggle's docs also show `.evaluate()` for running a task across a dataset.

This repository keeps reusable parsing and scoring logic in `renju_benchmark.tasks` and Kaggle-decorated entrypoints
in `renju_benchmark.kaggle_tasks`.

For Kaggle CLI uploads, use the self-contained files under `kaggle_tasks/`. These files intentionally do not import
the local `renju_benchmark` package, because `kaggle b t push ... -f task.py` uploads the task file as the executable
artifact. The push files embed small public evaluation records and score against deterministic labels.

## Tasks

- `renju_next_move`
- `renju_next_move_record`
- `renju_rule_classification`
- `renju_rule_classification_record`

The `*_record` variants accept one JSON string. They are the most schema-stable option if Kaggle's task input schema
does not handle list arguments exactly as expected.

## Input Schema

### `renju_next_move`

- `board_text`: 15 lines of 15 characters using `.`, `X`, `O`
- `side`: `black` or `white`
- `best_moves`: list of coordinates, e.g. `["H8"]`
- `good_moves`: list of coordinates
- `blocking_moves`: list of coordinates
- `forbidden_moves`: list of coordinates
- `mode`: `fast`, `strict`, or `puzzle`

### `renju_rule_classification`

- `board_text`
- `side`
- `move`
- `expected`: `legal`, `win`, `forbidden`, `occupied`, or `off_board`

## Model Output

Ask models to return JSON only:

```json
{"move": "H8"}
```

```json
{"class": "legal"}
```

The parsers tolerate some non-JSON fallback text for development, but official prompts request exact JSON.

## Official Scoring

Rapfi is not used in official scoring. Official Kaggle tasks should score against deterministic JSONL labels using
the Python rule engine and scoring functions.

## Smoke Test In A Kaggle Notebook

```python
import json
import renju_benchmark.kaggle_tasks

record = {
    "board": "...............\n" * 7 + ".....XXXX......\n" + "...............\n" * 7,
    "side": "black",
    "best_moves": ["E8", "J8"],
    "good_moves": [],
    "blocking_moves": [],
    "forbidden_moves": [],
    "mode": "fast",
}
json.dumps(record)
```

After importing `renju_benchmark.kaggle_tasks` in a Kaggle Benchmark notebook, the decorated task functions should be
registered by `kaggle_benchmarks`.

## Push-Ready Task Files

The files below are formatted as Jupytext-style Python notebooks with `# %%` cells, import `kaggle_benchmarks as
kbench`, define exactly one `@kbench.task`, include a return type annotation, and call `.evaluate(...)` at the end.

- `kaggle_tasks/renju_next_move_public.py`
- `kaggle_tasks/renju_rule_classification_public.py`

Suggested local validation and push flow:

```bash
kaggle b init -y
python kaggle_tasks/renju_next_move_public.py
ls -1 *.run.json
kaggle b t push renju-next-move-public -f kaggle_tasks/renju_next_move_public.py --wait

python kaggle_tasks/renju_rule_classification_public.py
ls -1 *.run.json
kaggle b t push renju-rule-classification-public -f kaggle_tasks/renju_rule_classification_public.py --wait
```

`python kaggle_tasks/...` requires the `kaggle_benchmarks` SDK in the active local Python environment. If your local
environment only has the Kaggle CLI, `python ...` can fail with `ModuleNotFoundError: No module named
'kaggle_benchmarks'`; in that case, the `kaggle b t push ... --wait` path still runs in Kaggle's benchmark environment.

If credentials expire, refresh them with:

```bash
kaggle b auth -y
```

The push files are intentionally small public smoke tasks. Larger public/hidden sets should be generated and attached
as Kaggle datasets later, then read inside the same task-file pattern.
