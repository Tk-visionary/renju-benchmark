from __future__ import annotations

from pathlib import Path

from renju_benchmark.rules import Board
from renju_benchmark.rl.board_encoding import encode_position, index_to_move
from renju_benchmark.rl.policy_value_net import ModelConfig, build_model, require_torch


class PolicyValueAgent:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        torch = require_torch()
        payload = torch.load(checkpoint, map_location=device)
        config_payload = payload.get("config", {})
        config = ModelConfig(
            channels=int(config_payload.get("channels", 64)),
            residual_blocks=int(config_payload.get("resblocks", config_payload.get("residual_blocks", 6))),
        )
        self.torch = torch
        self.device = device
        self.model = build_model(config).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def move(self, board: Board, side: str) -> tuple[int, int]:
        encoded = encode_position(board, side)
        x = self.torch.tensor([encoded.planes], dtype=self.torch.float32, device=self.device)
        mask = self.torch.tensor(encoded.legal_policy_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _value = self.model(x)
            logits = logits[0].masked_fill(~mask, -1.0e9)
            index = int(self.torch.argmax(logits).item())
        return index_to_move(index)
