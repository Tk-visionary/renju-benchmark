from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.agents import heuristic_move
from renju_benchmark.rapfi import RapfiConfig, RapfiEngine
from renju_benchmark.rules import BLACK, WHITE, MoveResult, RenjuGame, RuleMode, format_coord


@dataclass(frozen=True)
class GameResult:
    result: str
    moves: list[str]
    winner: str | None


def opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


def result_winner(result: MoveResult) -> str | None:
    if result == MoveResult.BLACK_WIN:
        return BLACK
    if result in (MoveResult.WHITE_WIN, MoveResult.BLACK_FORBIDDEN):
        return WHITE
    if result == MoveResult.ILLEGAL_OFF_BOARD or result == MoveResult.ILLEGAL_OCCUPIED:
        return None
    return None


def play_vs_rapfi(
    model_move_fn,
    model_color: str = BLACK,
    config: RapfiConfig | None = None,
    max_plies: int = 120,
    mode: RuleMode | str = RuleMode.STRICT,
) -> GameResult:
    game = RenjuGame.new(mode=mode)
    moves: list[str] = []
    with RapfiEngine(config or RapfiConfig.from_env()) as rapfi:
        for _ in range(max_plies):
            if game.turn == model_color:
                row, col = model_move_fn(game.board, game.turn)
            else:
                row, col = rapfi.best_move(game.board, game.turn)
            coord = format_coord(row, col) if game.board.in_bounds(row, col) else f"{row},{col}"
            result = game.play(row, col)
            moves.append(coord)
            if result != MoveResult.OK:
                return GameResult(result=result.value, moves=moves, winner=result_winner(result))
    return GameResult(result="max_plies", moves=moves, winner=None)


def play_heuristic_vs_rapfi(
    model_color: str = BLACK,
    config: RapfiConfig | None = None,
    max_plies: int = 120,
) -> GameResult:
    return play_vs_rapfi(
        lambda board, color: heuristic_move(board, color),
        model_color=model_color,
        config=config,
        max_plies=max_plies,
    )
