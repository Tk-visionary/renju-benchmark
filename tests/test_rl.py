from __future__ import annotations

import json

import pytest

from renju_benchmark.rules import BLACK, WHITE, Board, parse_coord
from renju_benchmark.rl.board_encoding import (
    BOARD_SIZE,
    CHANNELS,
    coord_to_index,
    encode_position,
    index_to_coord,
    move_to_index,
)
from renju_benchmark.rl.datasets import encoded_training_row
from renju_benchmark.rl.policy_value_net import ModelConfig, build_model
from renju_benchmark.rl.rapfi_env import play_match, score_for_model, summarize_game_rows, winner_after_move_result
from renju_benchmark.rl.search import (
    opponent_force_win_replies,
    tactical_candidates,
    tactical_candidates_with_roles,
    tactical_heuristic_move,
    winning_threat_count,
)
from renju_benchmark.rl.self_play import (
    MaskedPolicyPlayer,
    SelfPlayConfig,
    collect_self_play,
    play_self_game,
    random_checkpoint,
)
from renju_benchmark.rl.symbolic import fit_pairwise_weights, rank_symbolic_moves, symbolic_move


def board_from_points(points: list[tuple[str, str]]) -> Board:
    rows = [list("." * BOARD_SIZE) for _ in range(BOARD_SIZE)]
    for coord, color in points:
        row, col = parse_coord(coord)
        rows[row][col] = color
    return Board.from_text("\n".join("".join(row) for row in rows))


def test_encode_position_has_expected_planes_and_mask() -> None:
    board = board_from_points([("H8", BLACK)])
    encoded = encode_position(board, "black", last_move=parse_coord("H8"))

    assert len(encoded.planes) == len(CHANNELS)
    assert len(encoded.planes[0]) == BOARD_SIZE
    assert encoded.planes[0][7][7] == 1.0
    assert encoded.planes[2][0][0] == 1.0
    assert encoded.planes[4][7][7] == 1.0
    assert encoded.legal_policy_mask[move_to_index(parse_coord("H8"))] == 0.0
    assert sum(encoded.legal_policy_mask) > 0


def test_coord_index_roundtrip() -> None:
    assert index_to_coord(coord_to_index("J8")) == "J8"


def test_encoded_training_row_uses_policy_index() -> None:
    board = board_from_points([("H8", BLACK)])
    row = encoded_training_row({
        "board": board.to_text(),
        "side": "white",
        "policy_index": coord_to_index("J8"),
    })

    assert row["policy_index"] == coord_to_index("J8")
    assert len(row["planes"]) == len(CHANNELS)


def test_tactical_candidates_prioritize_immediate_win() -> None:
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK)])
    candidates = tactical_candidates(board, BLACK)

    assert parse_coord("E8") in candidates
    assert parse_coord("J8") in candidates


def test_symbolic_move_prioritizes_immediate_win() -> None:
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK)])
    assert symbolic_move(board, BLACK) in {parse_coord("E8"), parse_coord("J8")}


def test_symbolic_pairwise_fit_updates_weights() -> None:
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK)])
    examples = [{
        "board": board.to_text(),
        "side": "black",
        "best_move": "E8",
        "policy_index": coord_to_index("E8"),
    }]
    weights = fit_pairwise_weights(examples, epochs=1, learning_rate=0.5)
    ranked = rank_symbolic_moves(board, BLACK, weights=weights)

    assert ranked
    assert isinstance(weights["center"], float)


def test_tactical_candidates_with_roles_marks_single_block_safe() -> None:
    board = board_from_points([
        ("E8", BLACK),
        ("F8", WHITE),
        ("G8", WHITE),
        ("H8", WHITE),
        ("I8", WHITE),
    ])
    roles = {item.move: item.role for item in tactical_candidates_with_roles(board, BLACK)}

    assert roles[parse_coord("J8")] == "block"


def test_tactical_candidates_with_roles_keeps_open_four_blocks() -> None:
    board = board_from_points([
        ("F8", WHITE),
        ("G8", WHITE),
        ("H8", WHITE),
        ("I8", WHITE),
    ])
    roles = {item.move: item.role for item in tactical_candidates_with_roles(board, BLACK)}

    assert roles[parse_coord("E8")] == "block"
    assert roles[parse_coord("J8")] == "block"


def test_tactical_heuristic_blocks_rapfi_loss_diagonal() -> None:
    board = board_from_points([
        ("H8", BLACK),
        ("H7", BLACK),
        ("G8", BLACK),
        ("H9", BLACK),
        ("H10", BLACK),
        ("I9", BLACK),
        ("I7", BLACK),
        ("G7", WHITE),
        ("I8", WHITE),
        ("F9", WHITE),
        ("H6", WHITE),
        ("H11", WHITE),
        ("F8", WHITE),
        ("I5", WHITE),
    ])

    assert tactical_heuristic_move(board, BLACK) in {parse_coord("E9"), parse_coord("J4")}


def test_tactical_heuristic_avoids_allowing_rapfi_double_threat() -> None:
    board = board_from_points([
        ("H8", BLACK),
        ("H7", BLACK),
        ("G8", BLACK),
        ("H9", BLACK),
        ("H10", BLACK),
        ("I9", BLACK),
        ("G7", WHITE),
        ("I8", WHITE),
        ("F9", WHITE),
        ("H6", WHITE),
        ("H11", WHITE),
        ("F8", WHITE),
    ])
    move = tactical_heuristic_move(board, BLACK, force_reply_limit=64)

    assert parse_coord("I5") in opponent_force_win_replies(board, BLACK, parse_coord("I7"))
    assert parse_coord("I5") not in opponent_force_win_replies(board, BLACK, move)


def test_tactical_candidates_with_roles_marks_winning_threat() -> None:
    board = board_from_points([
        ("F8", BLACK),
        ("G8", BLACK),
        ("H8", BLACK),
        ("D8", WHITE),
        ("J8", WHITE),
    ])
    roles = {item.move: item.role for item in tactical_candidates_with_roles(board, BLACK)}

    assert roles[parse_coord("E8")] == "threat"


def test_tactical_candidates_with_roles_marks_double_threat_as_force_win() -> None:
    board = board_from_points([
        ("F8", BLACK),
        ("G8", BLACK),
        ("H6", BLACK),
        ("H7", BLACK),
        ("H9", BLACK),
    ])
    move = parse_coord("H8")
    roles = {item.move: item.role for item in tactical_candidates_with_roles(board, BLACK)}

    assert winning_threat_count(board, BLACK, move) >= 2
    assert roles[move] == "force_win"


def test_policy_value_model_config_is_plain_dataclass() -> None:
    config = ModelConfig(channels=32, residual_blocks=2)
    assert config.channels == 32
    assert config.residual_blocks == 2


def test_build_hrm_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")

    config = ModelConfig(model_type="hrm", channels=8, hrm_cycles=1, hrm_low_steps=1)
    model = build_model(config)
    logits, value = model(torch.zeros(2, len(CHANNELS), BOARD_SIZE, BOARD_SIZE))

    assert logits.shape == (2, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (2,)


def test_build_model_reports_missing_torch_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="PyTorch is required"):
        build_model()


def test_policy_value_agent_checkpoint_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from renju_benchmark.rl.inference import PolicyValueAgent

    checkpoint = tmp_path / "model.pt"
    config = ModelConfig(channels=8, residual_blocks=1)
    model = build_model(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"channels": config.channels, "resblocks": config.residual_blocks},
        },
        checkpoint,
    )

    agent = PolicyValueAgent(checkpoint)
    move = agent.move(Board.empty(), "black")
    assert Board.empty().in_bounds(*move)
    assert agent.rank_moves(Board.empty(), "black", top_k=3)


def test_policy_value_agent_hrm_checkpoint_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from renju_benchmark.rl.inference import PolicyValueAgent

    checkpoint = tmp_path / "hrm.pt"
    config = ModelConfig(model_type="hrm", channels=8, hrm_cycles=1, hrm_low_steps=1)
    model = build_model(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "model_type": config.model_type,
                "channels": config.channels,
                "input_channels": config.input_channels,
                "hrm_cycles": config.hrm_cycles,
                "hrm_low_steps": config.hrm_low_steps,
            },
        },
        checkpoint,
    )

    agent = PolicyValueAgent(checkpoint)
    assert Board.empty().in_bounds(*agent.move(Board.empty(), "black"))


def test_masked_hrm_self_play_never_returns_illegal(tmp_path) -> None:
    pytest.importorskip("torch")

    checkpoint = tmp_path / "random_hrm.pt"
    config = SelfPlayConfig(channels=8, hrm_cycles=1, hrm_low_steps=1, max_plies=8, seed=123)
    random_checkpoint(checkpoint, config)
    player = MaskedPolicyPlayer(checkpoint, config)
    rows = play_self_game(player)

    assert rows
    assert {row["game_result"] for row in rows}.isdisjoint({"illegal_occupied", "illegal_off_board", "black_forbidden"})
    assert all(-1.0 <= row["value"] <= 1.0 for row in rows)


def test_batched_masked_hrm_self_play_never_returns_illegal(tmp_path) -> None:
    pytest.importorskip("torch")

    checkpoint = tmp_path / "random_hrm.pt"
    output = tmp_path / "selfplay.jsonl"
    config = SelfPlayConfig(
        channels=8,
        hrm_cycles=1,
        hrm_low_steps=1,
        max_plies=6,
        seed=123,
        parallel_games=4,
    )
    random_checkpoint(checkpoint, config)
    rows = collect_self_play(checkpoint, output, games=4, config=config)

    assert len(rows) == 24
    assert output.exists()
    assert {row["game_result"] for row in rows}.isdisjoint({"illegal_occupied", "illegal_off_board", "black_forbidden"})


def test_policy_value_agent_tactical_move_prioritizes_win(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from renju_benchmark.rl.inference import PolicyValueAgent

    checkpoint = tmp_path / "model.pt"
    config = ModelConfig(channels=8, residual_blocks=1)
    model = build_model(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"channels": config.channels, "resblocks": config.residual_blocks},
        },
        checkpoint,
    )
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK)])
    agent = PolicyValueAgent(checkpoint)
    assert agent.tactical_move(board, "black") in {parse_coord("E8"), parse_coord("J8")}


def test_policy_value_agent_tactical_move_prioritizes_force_win_group(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from renju_benchmark.rl.inference import PolicyValueAgent

    checkpoint = tmp_path / "model.pt"
    config = ModelConfig(channels=8, residual_blocks=1)
    model = build_model(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"channels": config.channels, "resblocks": config.residual_blocks},
        },
        checkpoint,
    )
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK)])
    agent = PolicyValueAgent(checkpoint)
    roles = {item.move: item.role for item in tactical_candidates_with_roles(board, BLACK)}
    assert roles[agent.tactical_move(board, "black")] == "force_win"


def test_tactical_heuristic_move_uses_role_priority() -> None:
    board = board_from_points([
        ("F8", BLACK),
        ("G8", BLACK),
        ("H6", BLACK),
        ("H7", BLACK),
        ("H9", BLACK),
    ])

    assert tactical_heuristic_move(board, BLACK) == parse_coord("H8")


def test_rapfi_eval_agent_name_for_tactical_baseline() -> None:
    from scripts.rl_evaluate_vs_rapfi import agent_name

    assert agent_name(has_checkpoint=False, tactical=True) == "tactical_heuristic"
    assert agent_name(has_checkpoint=False, tactical=False) == "heuristic"


def test_play_match_reports_winner() -> None:
    black_moves = [parse_coord("F8"), parse_coord("G8"), parse_coord("H8"), parse_coord("I8"), parse_coord("J8")]
    white_moves = [parse_coord("A1"), parse_coord("A2"), parse_coord("A3"), parse_coord("A4")]

    def scripted_black(_board, _side):
        return black_moves.pop(0)

    def scripted_white(_board, _side):
        return white_moves.pop(0)

    result = play_match(scripted_black, scripted_white, max_plies=9)
    assert result.result == "black_win"
    assert result.winner == BLACK


def test_tactical_eval_script_imports() -> None:
    import scripts.rl_evaluate_vs_tactical as module

    assert callable(module.main)


def test_tactical_collect_script_imports() -> None:
    import scripts.rl_collect_tactical as module

    assert callable(module.main)


def test_imitation_eval_accepts_source_neutral_best_move() -> None:
    from scripts.rl_eval_imitation import best_move_label

    assert best_move_label({"best_move": "H8"}) == "H8"
    assert best_move_label({"rapfi_best": "I8"}) == "I8"
    assert best_move_label({"tactical_best": "J8"}) == "J8"


def test_rl_summarize_runs_reads_metrics(tmp_path) -> None:
    from scripts.rl_summarize_runs import summarize_metrics

    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({
            "config": {"run_dir": "run-a", "count": 4, "tactical_count": 8, "epochs": 1},
            "imitation": {"top1_accuracy": 0.25, "top5_accuracy": 0.75},
            "tactical_match": {"score": 0.5, "win_rate": 0.0, "illegal_rate": 0.0, "games": 2},
            "match": {"score": 0.25, "win_rate": 0.0, "illegal_rate": 0.0, "games": 2},
        })
    )

    row = summarize_metrics(metrics)
    assert row["run"] == "run-a"
    assert row["tactical_count"] == 8
    assert row["top5_accuracy"] == 0.75
    assert row["rapfi_score"] == 0.25


def test_winner_after_illegal_move_is_opponent() -> None:
    from renju_benchmark.rules import MoveResult, WHITE

    assert winner_after_move_result(MoveResult.ILLEGAL_OCCUPIED, BLACK) == WHITE


def test_summarize_game_rows_reports_wdl_and_side_scores() -> None:
    rows = [
        {"model_color": "black", "winner": BLACK, "result": "black_win"},
        {"model_color": "white", "winner": BLACK, "result": "black_win"},
        {"model_color": "black", "winner": None, "result": "max_plies"},
        {"model_color": "white", "winner": BLACK, "result": "illegal_occupied"},
    ]

    assert score_for_model(rows[0]) == 1.0
    assert score_for_model(rows[1]) == 0.0
    summary = summarize_game_rows(rows)
    assert summary["games"] == 4
    assert summary["wins"] == 1
    assert summary["draws"] == 1
    assert summary["losses"] == 2
    assert summary["illegal_rate"] == 0.25
    assert summary["black_score"] == 0.75
    assert summary["white_score"] == 0.0
