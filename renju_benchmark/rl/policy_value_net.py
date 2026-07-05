from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.rules import BOARD_SIZE
from renju_benchmark.rl.board_encoding import CHANNELS


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 64
    residual_blocks: int = 6
    input_channels: int = len(CHANNELS)
    board_size: int = BOARD_SIZE


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install
        raise RuntimeError("PyTorch is required for RL training. Install torch to use this module.") from exc
    return torch


def build_model(config: ModelConfig | None = None):
    torch = require_torch()
    nn = torch.nn
    cfg = config or ModelConfig()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.net(x))

    class PolicyValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.input_channels, cfg.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cfg.channels),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(*[ResidualBlock(cfg.channels) for _ in range(cfg.residual_blocks)])
            self.policy = nn.Sequential(
                nn.Conv2d(cfg.channels, 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2 * cfg.board_size * cfg.board_size, cfg.board_size * cfg.board_size),
            )
            self.value = nn.Sequential(
                nn.Conv2d(cfg.channels, 1, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(cfg.board_size * cfg.board_size, cfg.channels),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.channels, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            features = self.body(self.stem(x))
            return self.policy(features), self.value(features).squeeze(-1)

    return PolicyValueNet()

