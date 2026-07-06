import pytest

from renju_benchmark.rapfi import RapfiConfig, RapfiError, _looks_like_engine_move, _parse_engine_move


def test_parse_engine_move() -> None:
    assert _parse_engine_move("7,8") == (8, 7)


def test_parse_engine_move_rejects_off_board() -> None:
    with pytest.raises(RapfiError):
        _parse_engine_move("15,15")


def test_looks_like_engine_move() -> None:
    assert _looks_like_engine_move("7,8")
    assert not _looks_like_engine_move("MESSAGE loading")


def test_rapfi_config_reads_weak_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPFI_PATH", "rapfi")
    monkeypatch.setenv("RAPFI_TIMEOUT_TURN_MS", "100")
    monkeypatch.setenv("RAPFI_MAX_DEPTH", "2")
    monkeypatch.setenv("RAPFI_MAX_NODE", "1000")
    config = RapfiConfig.from_env()

    assert config.timeout_turn_ms == 100
    assert config.max_depth == 2
    assert config.max_node == 1000
    assert config.rule == 4
