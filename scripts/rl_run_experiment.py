from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def capture_json(command: list[str]) -> dict:
    print("+", " ".join(str(part) for part in command), flush=True)
    output = subprocess.check_output(command, text=True)
    print(output, end="" if output.endswith("\n") else "\n")
    return json.loads(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Rapfi-imitation experiment end to end.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-plies", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--resblocks", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--eval-max-plies", type=int, default=12)
    parser.add_argument("--tactical-games", type=int, default=2)
    parser.add_argument("--tactical-max-plies", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--rapfi-path", type=Path, default=Path("external/rapfi-runtime/pbrain-rapfi"))
    parser.add_argument("--rapfi-cwd", type=Path, default=Path("external/rapfi-runtime"))
    parser.add_argument("--move-timeout", type=float, default=10.0)
    parser.add_argument("--timeout-turn-ms", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-node", type=int, default=300)
    parser.add_argument("--tactical", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = run_dir / "rapfi_examples.jsonl"
    checkpoint = run_dir / "policy_value.pt"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "run_dir": str(run_dir),
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
    })
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    common_rapfi = [
        "--rapfi-path", str(args.rapfi_path),
        "--rapfi-cwd", str(args.rapfi_cwd),
        "--move-timeout", str(args.move_timeout),
        "--timeout-turn-ms", str(args.timeout_turn_ms),
        "--max-depth", str(args.max_depth),
        "--max-node", str(args.max_node),
    ]

    run([
        sys.executable,
        "scripts/rl_collect_rapfi.py",
        "--count", str(args.count),
        "--output", str(dataset),
        "--seed", str(args.seed),
        "--min-plies", str(args.min_plies),
        "--max-plies", str(args.max_plies),
        *common_rapfi,
    ])
    run([
        sys.executable,
        "scripts/rl_train_imitation.py",
        "--input", str(dataset),
        "--output", str(checkpoint),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--channels", str(args.channels),
        "--resblocks", str(args.resblocks),
    ])
    imitation = capture_json([
        sys.executable,
        "scripts/rl_eval_imitation.py",
        "--checkpoint", str(checkpoint),
        "--input", str(dataset),
        "--top-k", str(args.top_k),
    ])
    tactical_match = capture_json([
        sys.executable,
        "scripts/rl_evaluate_vs_tactical.py",
        "--checkpoint", str(checkpoint),
        "--games", str(args.tactical_games),
        "--max-plies", str(args.tactical_max_plies),
        "--candidate-limit", str(args.candidate_limit),
    ])
    eval_command = [
        sys.executable,
        "scripts/rl_evaluate_vs_rapfi.py",
        "--checkpoint", str(checkpoint),
        "--games", str(args.eval_games),
        "--max-plies", str(args.eval_max_plies),
        *common_rapfi,
    ]
    if args.tactical:
        eval_command.append("--tactical")
    match = capture_json(eval_command)

    metrics = {
        "config": config,
        "imitation": imitation,
        "tactical_match": tactical_match["summary"],
        "tactical_games": tactical_match["games"],
        "match": match["summary"],
        "games": match["games"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
