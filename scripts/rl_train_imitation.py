from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.datasets import encoded_training_row, load_examples
from renju_benchmark.rl.policy_value_net import ModelConfig, build_model, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small policy/value network on Rapfi best moves.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--resblocks", type=int, default=6)
    args = parser.parse_args()

    torch = require_torch()
    examples = load_examples(args.input)
    rows = [encoded_training_row(example) for example in examples]
    x = torch.tensor([row["planes"] for row in rows], dtype=torch.float32)
    y = torch.tensor([row["policy_index"] for row in rows], dtype=torch.long)

    model = build_model(ModelConfig(channels=args.channels, residual_blocks=args.resblocks))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(x.shape[0])
        total_loss = 0.0
        for start in range(0, x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits, _value = model(x[indices])
            loss = loss_fn(logits, y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        print(f"epoch={epoch + 1} loss={total_loss / max(1, x.shape[0]):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "channels": args.channels,
                "resblocks": args.resblocks,
                "input_channels": x.shape[1],
            },
        },
        args.output,
    )
    print(f"wrote checkpoint to {args.output}")


if __name__ == "__main__":
    main()
