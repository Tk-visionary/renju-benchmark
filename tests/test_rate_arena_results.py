from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.rate_arena_results import iter_result_dicts


def test_iter_result_dicts_finds_nested_kaggle_results(tmp_path: Path) -> None:
    payload = {
        "runs": [
            {
                "result": {
                    "match_id": "m1",
                    "black": "model-a",
                    "white": "model-b",
                    "result": "black_win",
                    "candidate_score": 1.0,
                }
            }
        ]
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    rows = list(iter_result_dicts(path))
    assert rows == [payload["runs"][0]["result"]]


def test_rate_arena_results_outputs_leaderboard(tmp_path: Path) -> None:
    payload = {
        "result": {
            "match_id": "m1",
            "black": "model-a",
            "white": "model-b",
            "result": "black_win",
            "candidate_score": 1.0,
        }
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    repo = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        [sys.executable, "scripts/rate_arena_results.py", str(path)],
        cwd=repo,
        text=True,
    )
    report = json.loads(output)
    assert report["leaderboard"][0]["model"] == "model-a"
    assert report["leaderboard"][0]["rating"] > 1900
