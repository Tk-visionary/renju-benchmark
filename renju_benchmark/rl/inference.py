from __future__ import annotations

from pathlib import Path

from renju_benchmark.rules import Board
from renju_benchmark.rl.board_encoding import encode_position, index_to_move, move_to_index
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
        return index_to_move(self.rank_moves(board, side, top_k=1)[0])

    def rank_moves(self, board: Board, side: str, top_k: int = 10) -> list[int]:
        encoded = encode_position(board, side)
        x = self.torch.tensor([encoded.planes], dtype=self.torch.float32, device=self.device)
        mask = self.torch.tensor(encoded.legal_policy_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _value = self.model(x)
            logits = logits[0].masked_fill(~mask, -1.0e9)
            k = min(top_k, int(mask.sum().item()))
            return [int(index) for index in self.torch.topk(logits, k=k).indices.tolist()]

    def score_move_rank(self, board: Board, side: str, move: tuple[int, int], top_k: int = 10) -> int | None:
        target = move_to_index(move)
        for rank, index in enumerate(self.rank_moves(board, side, top_k=top_k), start=1):
            if index == target:
                return rank
        return None
