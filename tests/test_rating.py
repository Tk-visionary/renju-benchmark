from renju_benchmark.rating import RatingTable


def test_rating_updates_winner_up() -> None:
    table = RatingTable()
    table.record_game("black", "white", "black_win")
    assert table.rating("black") > 1900
    assert table.rating("white") < 1900


def test_leaderboard_sorted() -> None:
    table = RatingTable()
    table.record_game("a", "b", "black_win")
    table.record_game("a", "c", "black_win")
    rows = table.leaderboard()
    assert rows[0][0] == "a"

