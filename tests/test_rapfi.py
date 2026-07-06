import pytest

from renju_benchmark.rapfi import RapfiConfig, RapfiEngine, RapfiError, _looks_like_engine_move, _parse_engine_move
from renju_benchmark.rules import BLACK, WHITE


def test_parse_engine_move() -> None:
    assert _parse_engine_move("7,8") == (8, 7)
    assert _parse_engine_move("MESSAGE Depth 2-3 | Eval -415 | Time 0ms | H5 G8") == (4, 7)


def test_parse_engine_move_rejects_off_board() -> None:
    with pytest.raises(RapfiError):
        _parse_engine_move("15,15")


def test_looks_like_engine_move() -> None:
    assert _looks_like_engine_move("7,8")
    assert _looks_like_engine_move("MESSAGE Depth 2-3 | Eval -415 | Time 0ms | H5 G8")
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


def test_board_protocol_uses_absolute_stone_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RapfiEngine(RapfiConfig("rapfi"))
    engine.process = object()
    sent: list[str] = []

    monkeypatch.setattr(engine, "_drain_pending_output", lambda timeout=0.0: None)
    monkeypatch.setattr(engine, "_send", sent.append)
    monkeypatch.setattr(engine, "_read_move_responses", lambda timeout=None: ["2,2"])

    assert engine.best_move_from_history([(7, 7, BLACK), (7, 8, WHITE)], WHITE) == (2, 2)
    assert sent[:4] == ["BOARD", "7,7,1", "8,7,2", "DONE"]
