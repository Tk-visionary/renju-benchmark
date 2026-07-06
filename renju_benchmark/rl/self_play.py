from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from renju_benchmark.rules import BLACK, BOARD_SIZE, WHITE, Board, MoveResult, RenjuGame, RuleMode, format_coord
from renju_benchmark.rl.board_encoding import encode_position, move_to_index, side_to_color
from renju_benchmark.rl.policy_value_net import ModelConfig, build_model, require_torch
from renju_benchmark.rl.rapfi_env import winner_after_move_result


@dataclass(frozen=True)
class SelfPlayConfig:
    model_type: str = "hrm"
    channels: int = 16
    residual_blocks: int = 2
    hrm_cycles: int = 1
    hrm_low_steps: int = 1
    temperature: float = 1.0
    epsilon: float = 0.1
    max_plies: int = 120
    forbidden_depth: int = 2
    seed: int = 0


def model_config(config: SelfPlayConfig) -> ModelConfig:
    return ModelConfig(
        model_type=config.model_type,
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        hrm_cycles=config.hrm_cycles,
        hrm_low_steps=config.hrm_low_steps,
    )


def random_checkpoint(path: Path, config: SelfPlayConfig) -> None:
    torch = require_torch()
    torch.manual_seed(config.seed)
    model = build_model(model_config(config))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "model_type": config.model_type,
            "channels": config.channels,
            "resblocks": config.residual_blocks,
            "input_channels": model_config(config).input_channels,
            "hrm_cycles": config.hrm_cycles,
            "hrm_low_steps": config.hrm_low_steps,
            "seed": config.seed,
        },
    }, path)


class MaskedPolicyPlayer:
    def __init__(
        self,
        checkpoint: Path | None,
        config: SelfPlayConfig,
        device: str = "cpu",
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.device = device
        self.config = config
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.model = build_model(model_config(config)).to(device)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=device)
            self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def select_move(self, board: Board, color: str) -> tuple[int, int]:
        encoded = encode_position(
            board,
            "black" if color == BLACK else "white",
            forbidden_depth=self.config.forbidden_depth,
        )
        legal_indices = [index for index, value in enumerate(encoded.legal_policy_mask) if value]
        if not legal_indices:
            raise ValueError("no legal moves")
        if self.rng.random() < self.config.epsilon:
            return divmod(self.rng.choice(legal_indices), BOARD_SIZE)

        x = self.torch.tensor([encoded.planes], dtype=self.torch.float32, device=self.device)
        mask = self.torch.tensor(encoded.legal_policy_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _value = self.model(x)
            logits = logits[0].masked_fill(~mask, -1.0e9)
            if self.config.temperature <= 0:
                index = int(self.torch.argmax(logits).item())
            else:
                probs = self.torch.softmax(logits / self.config.temperature, dim=0)
                index = int(self.torch.multinomial(probs, num_samples=1).item())
        return divmod(index, BOARD_SIZE)


def play_self_game(player: MaskedPolicyPlayer, mode: RuleMode | str | int = RuleMode.STRICT) -> list[dict]:
    game = RenjuGame.new(mode=mode)
    positions = []
    final_result = MoveResult.OK
    final_mover = BLACK
    reached_limit = True
    for _ply in range(player.config.max_plies):
        mover = game.turn
        board_before = game.board
        row, col = player.select_move(board_before, mover)
        result = game.play(row, col)
        positions.append({
            "board": board_before.to_text(),
            "side": "black" if mover == BLACK else "white",
            "best_move": format_coord(row, col),
            "policy_index": move_to_index((row, col)),
            "source": "masked_hrm_self_play",
        })
        final_result = result
        final_mover = mover
        if result != MoveResult.OK:
            reached_limit = False
            break

    winner = winner_after_move_result(final_result, final_mover)
    result_value = "max_plies" if reached_limit and final_result == MoveResult.OK else final_result.value
    for position in positions:
        color = side_to_color(position["side"])
        if winner == color:
            value = 1.0
        elif winner in {BLACK, WHITE}:
            value = -1.0
        else:
            value = 0.0
        position["value"] = value
        position["game_result"] = result_value
        position["winner"] = "black" if winner == BLACK else "white" if winner == WHITE else None
    return positions


def collect_self_play(
    checkpoint: Path | None,
    output: Path,
    games: int,
    config: SelfPlayConfig,
    device: str = "cpu",
) -> list[dict]:
    rows = []
    for game_index in range(games):
        game_config = SelfPlayConfig(**{**config.__dict__, "seed": config.seed + game_index})
        player = MaskedPolicyPlayer(checkpoint, game_config, device=device)
        rows.extend(play_self_game(player, mode=game_config.forbidden_depth))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows
