from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import Board, parse_coord
from renju_benchmark.rl.board_encoding import side_to_color
from renju_benchmark.rl.datasets import load_examples
from renju_benchmark.rl.symbolic import best_move_label, load_weights, rank_symbolic_moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate symbolic rule weights against move labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--force-reply-limit", type=int, default=16)
    args = parser.parse_args()

    weights = load_weights(args.weights) if args.weights else None
    examples = load_examples(args.input)
    top1 = 0
    topk = 0
    covered = 0
    ranks = []
    for example in examples:
        board = Board.from_text(example["board"])
        color = side_to_color(example["side"])
        target = parse_coord(best_move_label(example))
        ranked = rank_symbolic_moves(
            board,
            color,
            weights=weights,
            limit=args.candidate_limit,
            force_reply_limit=args.force_reply_limit,
        )
        ranked_moves = [item.move for item in ranked]
        if target not in ranked_moves:
            continue
        covered += 1
        rank = ranked_moves.index(target) + 1
        if rank == 1:
            top1 += 1
        if rank <= args.top_k:
            topk += 1
            ranks.append(rank)

    total = len(examples)
    print(json.dumps({
        "examples": total,
        "candidate_coverage": covered / total if total else 0.0,
        "top1_accuracy": top1 / total if total else 0.0,
        f"top{args.top_k}_accuracy": topk / total if total else 0.0,
        "mean_rank_in_top_k": sum(ranks) / len(ranks) if ranks else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
