from __future__ import annotations

from .agents import heuristic_move, move_to_text
from .rating import RatingTable
from .rules import BLACK, WHITE, MoveResult, RenjuGame


def main() -> None:
    table = RatingTable()
    for game_id in range(4):
        game = RenjuGame.new()
        game.forbidden_depth = 0
        for _ in range(80):
            move = heuristic_move(game.board, game.turn)
            result = game.play(*move)
            if result != MoveResult.OK:
                break
        if game.winner == BLACK:
            result_key = "black_win"
        elif game.winner == WHITE:
            result_key = "white_win"
        else:
            result_key = "draw"
        table.record_game("heuristic_black", "heuristic_white", result_key)
        print(f"game {game_id + 1}: {result_key}, last={move_to_text(move)}")
    print(table.leaderboard())


if __name__ == "__main__":
    main()
