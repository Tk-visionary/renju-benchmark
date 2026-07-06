from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.datasets import load_examples
from renju_benchmark.rl.symbolic import fit_pairwise_weights, save_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit interpretable symbolic Renju rule weights from move labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--force-reply-limit", type=int, default=16)
    parser.add_argument("--threat-forbidden-depth", type=int, default=2)
    args = parser.parse_args()

    examples = load_examples(args.input)
    weights = fit_pairwise_weights(
        examples,
        epochs=args.epochs,
        learning_rate=args.lr,
        margin=args.margin,
        limit=args.candidate_limit,
        force_reply_limit=args.force_reply_limit,
        threat_forbidden_depth=args.threat_forbidden_depth,
    )
    save_weights(weights, args.output)
    print(json.dumps({"examples": len(examples), "output": str(args.output), "weights": weights}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
