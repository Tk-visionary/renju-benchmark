from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import Board, parse_coord
from renju_benchmark.rl.datasets import load_examples
from renju_benchmark.rl.inference import PolicyValueAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy checkpoint against Rapfi imitation labels.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = PolicyValueAgent(args.checkpoint, device=args.device)
    examples = load_examples(args.input)
    top1 = 0
    topk = 0
    ranks = []
    for example in examples:
        board = Board.from_text(example["board"])
        target = parse_coord(best_move_label(example))
        rank = agent.score_move_rank(board, example["side"], target, top_k=args.top_k)
        if rank == 1:
            top1 += 1
        if rank is not None:
            topk += 1
            ranks.append(rank)

    total = len(examples)
    output = {
        "examples": total,
        "top1_accuracy": top1 / total if total else 0.0,
        f"top{args.top_k}_accuracy": topk / total if total else 0.0,
        "mean_rank_in_top_k": sum(ranks) / len(ranks) if ranks else None,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def best_move_label(example: dict) -> str:
    for key in ("best_move", "rapfi_best", "tactical_best"):
        value = example.get(key)
        if isinstance(value, str):
            return value
    raise KeyError("example has no best_move, rapfi_best, or tactical_best label")


if __name__ == "__main__":
    main()
