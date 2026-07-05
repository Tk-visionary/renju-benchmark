from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloRating:
    default_rating: float = 1900.0
    k_factor: float = 32.0

    def expected_score(self, player_rating: float, opponent_rating: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((opponent_rating - player_rating) / 400.0))

    def update_pair(
        self,
        player_rating: float,
        opponent_rating: float,
        score: float,
    ) -> tuple[float, float]:
        expected = self.expected_score(player_rating, opponent_rating)
        delta = self.k_factor * (score - expected)
        return player_rating + delta, opponent_rating - delta


class RatingTable:
    def __init__(self, default_rating: float = 1900.0, k_factor: float = 32.0) -> None:
        self.system = EloRating(default_rating=default_rating, k_factor=k_factor)
        self.ratings: dict[str, float] = {}
        self.games: dict[str, int] = {}

    def rating(self, player: str) -> float:
        return self.ratings.get(player, self.system.default_rating)

    def record_game(self, black: str, white: str, result: str) -> None:
        if result == "black_win":
            black_score = 1.0
        elif result == "white_win":
            black_score = 0.0
        elif result == "draw":
            black_score = 0.5
        else:
            raise ValueError(f"unknown result: {result}")

        black_rating, white_rating = self.system.update_pair(
            self.rating(black), self.rating(white), black_score
        )
        self.ratings[black] = black_rating
        self.ratings[white] = white_rating
        self.games[black] = self.games.get(black, 0) + 1
        self.games[white] = self.games.get(white, 0) + 1

    def leaderboard(self) -> list[tuple[str, float, int]]:
        rows = [(name, rating, self.games.get(name, 0)) for name, rating in self.ratings.items()]
        return sorted(rows, key=lambda item: item[1], reverse=True)

