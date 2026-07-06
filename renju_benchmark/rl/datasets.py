from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from renju_benchmark.agents import heuristic_move, random_move
from renju_benchmark.rapfi import RapfiConfig, RapfiEngine
from renju_benchmark.rules import BLACK, Board, MoveResult, RenjuGame, RuleMode, format_coord
from renju_benchmark.rl.board_encoding import coord_to_index, encode_position


@dataclass(frozen=True)
class RapfiExample:
    board: str
    side: str
    rapfi_best: str
    policy_index: int
    source: str


def random_reachable_position(
    rng: random.Random,
    plies: int,
    mode: RuleMode | str = RuleMode.FAST,
) -> tuple[Board, str, list[tuple[int, int, str]]]:
    game = RenjuGame.new(mode=mode)
    history: list[tuple[int, int, str]] = []
    for ply in range(plies):
        color = game.turn
        if game.turn == BLACK:
            row, col = heuristic_move(game.board, game.turn) if ply % 3 == 0 else random_move(game.board, game.turn)
        else:
            row, col = random_move(game.board, game.turn)
        result = game.play(row, col)
        history.append((row, col, color))
        if result != MoveResult.OK:
            break
    return game.board, game.turn, history


def collect_rapfi_examples(
    count: int,
    output: Path,
    config: RapfiConfig | None = None,
    seed: int = 0,
    min_plies: int = 4,
    max_plies: int = 30,
) -> None:
    rng = random.Random(seed)
    with RapfiEngine(config or RapfiConfig.from_env()) as rapfi:
        rows = []
        for _ in range(count):
            board, turn, history = random_reachable_position(rng, rng.randint(min_plies, max_plies))
            move = rapfi.best_move_from_history(history, turn)
            coord = format_coord(*move)
            example = RapfiExample(
                board=board.to_text(),
                side="black" if turn == BLACK else "white",
                rapfi_best=coord,
                policy_index=coord_to_index(coord),
                source="rapfi_best_move",
            )
            rows.append(json.dumps(asdict(example)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encoded_training_row(example: dict) -> dict:
    board = Board.from_text(example["board"])
    encoded = encode_position(board, example["side"])
    return {
        "planes": encoded.planes,
        "legal_policy_mask": encoded.legal_policy_mask,
        "policy_index": int(example["policy_index"]),
    }
