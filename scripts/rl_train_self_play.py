from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.datasets import encoded_training_row, load_examples
from renju_benchmark.rl.policy_value_net import ModelConfig, build_model, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HRM policy/value network from masked self-play rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hrm-cycles", type=int, default=1)
    parser.add_argument("--hrm-low-steps", type=int, default=1)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    args = parser.parse_args()

    torch = require_torch()
    examples = load_examples(args.input)
    rows = [encoded_training_row(example) for example in examples]
    x = torch.tensor([row["planes"] for row in rows], dtype=torch.float32)
    y_policy = torch.tensor([row["policy_index"] for row in rows], dtype=torch.long)
    y_value = torch.tensor([float(example.get("value", 0.0)) for example in examples], dtype=torch.float32)

    config = ModelConfig(
        model_type="hrm",
        channels=args.channels,
        hrm_cycles=args.hrm_cycles,
        hrm_low_steps=args.hrm_low_steps,
    )
    model = build_model(config)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(x.shape[0])
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        for start in range(0, x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits, value = model(x[indices])
            policy_loss = policy_loss_fn(logits, y_policy[indices])
            value_loss = value_loss_fn(value, y_value[indices])
            loss = policy_loss + args.value_loss_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = len(indices)
            total_loss += float(loss.detach()) * batch_size
            total_policy += float(policy_loss.detach()) * batch_size
            total_value += float(value_loss.detach()) * batch_size
        denom = max(1, x.shape[0])
        print(
            f"epoch={epoch + 1} loss={total_loss / denom:.4f} "
            f"policy={total_policy / denom:.4f} value={total_value / denom:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "model_type": "hrm",
                "channels": args.channels,
                "input_channels": x.shape[1],
                "hrm_cycles": args.hrm_cycles,
                "hrm_low_steps": args.hrm_low_steps,
            },
        },
        args.output,
    )
    print(f"wrote checkpoint to {args.output}")


if __name__ == "__main__":
    main()
