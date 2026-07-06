from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from renju_benchmark.agents import heuristic_move, random_move
from renju_benchmark.rapfi import RapfiConfig, RapfiEngine, RapfiError
from renju_benchmark.rules import BLACK, Board, MoveResult, RenjuGame, RuleMode, format_coord
from renju_benchmark.rl.board_encoding import coord_to_index, encode_position


@dataclass(frozen=True)
class RapfiExample:
    board: str
    side: str
    rapfi_best: str
    policy_index: int
    source: str
    best_move: str | None = None


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
    fresh_rapfi_per_position: bool = True,
) -> None:
    rng = random.Random(seed)
    rapfi_config = config or RapfiConfig.from_env()
    rows = []
    attempts = 0
    max_attempts = max(count * 10, count)
    rapfi_context = None if fresh_rapfi_per_position else RapfiEngine(rapfi_config)
    try:
        if rapfi_context is not None:
            rapfi_context.start()
        while len(rows) < count and attempts < max_attempts:
            attempts += 1
            board, turn, history = random_reachable_position(rng, rng.randint(min_plies, max_plies))
            try:
                if fresh_rapfi_per_position:
                    with RapfiEngine(rapfi_config) as rapfi:
                        move = rapfi.best_move_from_history(history, turn)
                else:
                    assert rapfi_context is not None
                    move = rapfi_context.best_move_from_history(history, turn)
            except RapfiError:
                continue
            coord = format_coord(*move)
            example = RapfiExample(
                board=board.to_text(),
                side="black" if turn == BLACK else "white",
                rapfi_best=coord,
                policy_index=coord_to_index(coord),
                source="rapfi_best_move",
                best_move=coord,
            )
            rows.append(json.dumps(asdict(example)))
    finally:
        if rapfi_context is not None:
            rapfi_context.close()
    if len(rows) < count:
        raise RapfiError(f"collected {len(rows)} examples after {attempts} attempts; requested {count}")
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
