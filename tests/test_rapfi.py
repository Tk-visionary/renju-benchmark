import pytest

from renju_benchmark.rapfi import RapfiError, _parse_engine_move


def test_parse_engine_move() -> None:
    assert _parse_engine_move("7,8") == (8, 7)


def test_parse_engine_move_rejects_off_board() -> None:
    with pytest.raises(RapfiError):
        _parse_engine_move("15,15")

