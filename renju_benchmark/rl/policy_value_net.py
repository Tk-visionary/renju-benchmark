from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.rules import BOARD_SIZE
from renju_benchmark.rl.board_encoding import CHANNELS


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "resnet"
    channels: int = 64
    residual_blocks: int = 6
    input_channels: int = len(CHANNELS)
    board_size: int = BOARD_SIZE
    hrm_cycles: int = 4
    hrm_low_steps: int = 2


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

    if cfg.model_type == "hrm":
        return _build_hrm_model(torch, nn, cfg)
    if cfg.model_type != "resnet":
        raise ValueError(f"unknown model_type: {cfg.model_type}")

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


def _build_hrm_model(torch, nn, cfg: ModelConfig):
    class LowBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cfg.channels * 3, cfg.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cfg.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1, groups=cfg.channels, bias=False),
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, z_low, z_high, x_embed):
            return self.relu(z_low + self.net(torch.cat([z_low, z_high, x_embed], dim=1)))

    class HighBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local = nn.Sequential(
                nn.Conv2d(cfg.channels * 2, cfg.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.channels),
                nn.ReLU(inplace=True),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.channels,
                nhead=_attention_heads(cfg.channels),
                dim_feedforward=cfg.channels * 2,
                batch_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.global_update = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.norm = nn.BatchNorm2d(cfg.channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, z_high, z_low):
            local = self.local(torch.cat([z_high, z_low], dim=1))
            batch, channels, height, width = local.shape
            tokens = local.flatten(2).transpose(1, 2)
            updated = self.global_update(tokens).transpose(1, 2).reshape(batch, channels, height, width)
            return self.relu(z_high + self.norm(updated))

    class HRMPolicyValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Sequential(
                nn.Conv2d(cfg.input_channels, cfg.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cfg.channels),
                nn.ReLU(inplace=True),
            )
            self.low = LowBlock()
            self.high = HighBlock()
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
            x_embed = self.embed(x)
            z_low = torch.zeros_like(x_embed)
            z_high = torch.zeros_like(x_embed)
            for _cycle in range(cfg.hrm_cycles):
                for _step in range(cfg.hrm_low_steps):
                    z_low = self.low(z_low, z_high, x_embed)
                z_high = self.high(z_high, z_low)
            return self.policy(z_high), self.value(z_high).squeeze(-1)

    return HRMPolicyValueNet()


def _attention_heads(channels: int) -> int:
    for heads in (8, 4, 2):
        if channels % heads == 0:
            return heads
    return 1
