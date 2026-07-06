from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from renju_benchmark.rules import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, WHITE, Board
from renju_benchmark.rl.search import tactical_candidates_with_roles

DEFAULT_WEIGHTS = {
    "role:win": 1000.0,
    "role:block": 500.0,
    "role:force_win": 250.0,
    "role:threat": 100.0,
    "role:safe": 1.0,
    "role:unsafe": -500.0,
    "center": 2.0,
    "edge_distance": 0.1,
    "self_max_run": 0.0,
    "self_open_ends": 0.0,
    "opponent_max_run": 0.0,
    "opponent_open_ends": 0.0,
}


@dataclass(frozen=True)
class SymbolicMove:
    move: tuple[int, int]
    role: str
    features: dict[str, float]
    score: float


def move_features(board: Board, move: tuple[int, int], color: str, role: str) -> dict[str, float]:
    row, col = move
    center = (BOARD_SIZE - 1) / 2.0
    center_distance = abs(row - center) + abs(col - center)
    edge_distance = min(row, col, BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col)
    opponent = WHITE if color == BLACK else BLACK
    self_run, self_open_ends = _line_profile(board, move, color)
    opponent_run, opponent_open_ends = _line_profile(board, move, opponent)
    return {
        f"role:{role}": 1.0,
        "center": -center_distance / (BOARD_SIZE - 1),
        "edge_distance": edge_distance / center,
        "self_max_run": self_run / 5.0,
        "self_open_ends": self_open_ends / 2.0,
        "opponent_max_run": opponent_run / 5.0,
        "opponent_open_ends": opponent_open_ends / 2.0,
    }


def _line_profile(board: Board, move: tuple[int, int], color: str) -> tuple[int, int]:
    row, col = move
    best_run = 1
    best_open_ends = 0
    for dx, dy in DIRECTIONS:
        run = 1
        open_ends = 0
        for sign in (-1, 1):
            step = 1
            while True:
                r = row + sign * dy * step
                c = col + sign * dx * step
                if not board.in_bounds(r, c) or board.get(r, c) != color:
                    break
                run += 1
                step += 1
            end_r = row + sign * dy * step
            end_c = col + sign * dx * step
            if board.in_bounds(end_r, end_c) and board.get(end_r, end_c) == EMPTY:
                open_ends += 1
        if (run, open_ends) > (best_run, best_open_ends):
            best_run = run
            best_open_ends = open_ends
    return best_run, best_open_ends


def score_features(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def rank_symbolic_moves(
    board: Board,
    color: str,
    weights: dict[str, float] | None = None,
    limit: int = 32,
    force_reply_limit: int = 16,
    threat_forbidden_depth: int = 2,
) -> list[SymbolicMove]:
    active_weights = weights or DEFAULT_WEIGHTS
    items = tactical_candidates_with_roles(
        board,
        color,
        limit=limit,
        force_reply_limit=force_reply_limit,
        threat_forbidden_depth=threat_forbidden_depth,
    )
    ranked = []
    for item in items:
        features = move_features(board, item.move, color, item.role)
        ranked.append(SymbolicMove(
            move=item.move,
            role=item.role,
            features=features,
            score=score_features(features, active_weights),
        ))
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def symbolic_move(
    board: Board,
    color: str,
    weights: dict[str, float] | None = None,
    limit: int = 32,
    force_reply_limit: int = 16,
    threat_forbidden_depth: int = 2,
) -> tuple[int, int]:
    ranked = rank_symbolic_moves(
        board,
        color,
        weights=weights,
        limit=limit,
        force_reply_limit=force_reply_limit,
        threat_forbidden_depth=threat_forbidden_depth,
    )
    if not ranked:
        raise ValueError("no symbolic candidates")
    return ranked[0].move


def fit_pairwise_weights(
    examples: list[dict],
    epochs: int = 5,
    learning_rate: float = 1.0,
    margin: float = 1.0,
    limit: int = 32,
    force_reply_limit: int = 16,
    threat_forbidden_depth: int = 2,
) -> dict[str, float]:
    from renju_benchmark.rules import parse_coord
    from renju_benchmark.rl.board_encoding import side_to_color

    weights = dict(DEFAULT_WEIGHTS)
    for _epoch in range(epochs):
        for example in examples:
            board = Board.from_text(example["board"])
            color = side_to_color(example["side"])
            target = parse_coord(best_move_label(example))
            ranked = rank_symbolic_moves(
                board,
                color,
                weights=weights,
                limit=limit,
                force_reply_limit=force_reply_limit,
                threat_forbidden_depth=threat_forbidden_depth,
            )
            if not ranked:
                continue
            target_item = next((item for item in ranked if item.move == target), None)
            if target_item is None:
                continue
            predicted = ranked[0]
            if predicted.move == target:
                continue
            if target_item.score <= predicted.score + margin:
                _add_scaled(weights, target_item.features, learning_rate)
                _add_scaled(weights, predicted.features, -learning_rate)
    return weights


def _add_scaled(weights: dict[str, float], features: dict[str, float], scale: float) -> None:
    for name, value in features.items():
        weights[name] = weights.get(name, 0.0) + scale * value


def best_move_label(example: dict) -> str:
    for key in ("best_move", "rapfi_best", "tactical_best"):
        value = example.get(key)
        if isinstance(value, str):
            return value
    raise KeyError("example has no best_move, rapfi_best, or tactical_best label")


def save_weights(weights: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n")


def load_weights(path: Path) -> dict[str, float]:
    return {str(key): float(value) for key, value in json.loads(path.read_text()).items()}
