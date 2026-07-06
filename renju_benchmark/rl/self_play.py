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
    max_plies: int = BOARD_SIZE * BOARD_SIZE
    forbidden_depth: int = 2
    local_legal_only: bool = True
    parallel_games: int = 16
    seed: int = 0


@dataclass
class _ActiveGame:
    game: RenjuGame
    rng: random.Random
    positions: list[dict]
    final_result: MoveResult = MoveResult.OK
    final_mover: str = BLACK
    reached_limit: bool = True


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
        return self.select_moves([(board, color)], [self.rng])[0]

    def select_moves(self, states: list[tuple[Board, str]], rngs: list[random.Random]) -> list[tuple[int, int]]:
        encoded_positions = [
            encode_position(
                board,
                "black" if color == BLACK else "white",
                forbidden_depth=self.config.forbidden_depth,
                local_only=self.config.local_legal_only,
            )
            for board, color in states
        ]
        legal_indices_by_state = [
            [index for index, value in enumerate(encoded.legal_policy_mask) if value]
            for encoded in encoded_positions
        ]
        for legal_indices in legal_indices_by_state:
            if not legal_indices:
                raise ValueError("no legal moves")

        random_choices: dict[int, tuple[int, int]] = {}
        model_rows = []
        model_masks = []
        model_state_indices = []
        for state_index, (encoded, legal_indices, rng) in enumerate(zip(encoded_positions, legal_indices_by_state, rngs)):
            if rng.random() < self.config.epsilon:
                random_choices[state_index] = divmod(rng.choice(legal_indices), BOARD_SIZE)
            else:
                model_rows.append(encoded.planes)
                model_masks.append(encoded.legal_policy_mask)
                model_state_indices.append(state_index)

        result: list[tuple[int, int] | None] = [None] * len(states)
        for state_index, move in random_choices.items():
            result[state_index] = move
        if model_rows:
            x = self.torch.tensor(model_rows, dtype=self.torch.float32, device=self.device)
            mask = self.torch.tensor(model_masks, dtype=self.torch.bool, device=self.device)
            with self.torch.no_grad():
                logits, _value = self.model(x)
                logits = logits.masked_fill(~mask, -1.0e9)
                if self.config.temperature <= 0:
                    indices = self.torch.argmax(logits, dim=1).tolist()
                else:
                    probs = self.torch.softmax(logits / self.config.temperature, dim=1)
                    indices = self.torch.multinomial(probs, num_samples=1).squeeze(1).tolist()
            for state_index, index in zip(model_state_indices, indices):
                result[state_index] = divmod(int(index), BOARD_SIZE)
        if any(move is None for move in result):
            raise RuntimeError("batched move selection left an unset move")
        return [move for move in result if move is not None]


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

    return _finalize_positions(positions, final_result, final_mover, reached_limit)


def collect_self_play(
    checkpoint: Path | None,
    output: Path,
    games: int,
    config: SelfPlayConfig,
    device: str = "cpu",
) -> list[dict]:
    rows = []
    player = MaskedPolicyPlayer(checkpoint, config, device=device)
    for start in range(0, games, config.parallel_games):
        batch_size = min(config.parallel_games, games - start)
        active_games = [
            _ActiveGame(
                game=RenjuGame.new(mode=config.forbidden_depth),
                rng=random.Random(config.seed + start + offset),
                positions=[],
            )
            for offset in range(batch_size)
        ]
        for _ply in range(config.max_plies):
            live_games = [item for item in active_games if item.final_result == MoveResult.OK]
            if not live_games:
                break
            states = [(item.game.board, item.game.turn) for item in live_games]
            rngs = [item.rng for item in live_games]
            moves = player.select_moves(states, rngs)
            for item, (row, col) in zip(live_games, moves):
                mover = item.game.turn
                board_before = item.game.board
                result = item.game.play(row, col)
                item.positions.append({
                    "board": board_before.to_text(),
                    "side": "black" if mover == BLACK else "white",
                    "best_move": format_coord(row, col),
                    "policy_index": move_to_index((row, col)),
                    "source": "masked_hrm_self_play",
                })
                item.final_result = result
                item.final_mover = mover
                if result != MoveResult.OK:
                    item.reached_limit = False
        for item in active_games:
            rows.extend(_finalize_positions(
                item.positions,
                item.final_result,
                item.final_mover,
                item.reached_limit,
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def _finalize_positions(
    positions: list[dict],
    final_result: MoveResult,
    final_mover: str,
    reached_limit: bool,
) -> list[dict]:
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
