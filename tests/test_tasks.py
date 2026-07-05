from renju_benchmark.rules import BLACK, WHITE, Board, parse_coord
from renju_benchmark.tasks import (
    _immediate_winning_moves,
    extract_first_coord,
    extract_label,
    parse_coord_relaxed,
    score_candidate_move,
)


def board_from_points(points: list[tuple[str, str]]) -> Board:
    rows = [list("." * 15) for _ in range(15)]
    for coord, color in points:
        row, col = parse_coord(coord)
        rows[row][col] = color
    return Board.from_text("\n".join("".join(row) for row in rows))


def test_score_illegal_move_zero() -> None:
    board = board_from_points([("H8", BLACK)])
    assert score_candidate_move(board, WHITE, parse_coord("H8"), set()) == 0.0


def test_score_known_best_move_one() -> None:
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK)])
    assert score_candidate_move(board, BLACK, parse_coord("J8"), {parse_coord("J8")}) == 1.0


def test_score_must_block_gets_threat_credit() -> None:
    board = board_from_points([
        ("E8", BLACK),
        ("F8", WHITE), ("G8", WHITE), ("H8", WHITE), ("I8", WHITE),
    ])
    assert score_candidate_move(board, BLACK, parse_coord("J8"), set()) == 0.75


def test_white_overline_counts_as_immediate_win() -> None:
    board = board_from_points([
        ("F8", WHITE), ("G8", WHITE), ("H8", WHITE), ("I8", WHITE), ("J8", WHITE),
    ])
    wins = _immediate_winning_moves(board, WHITE)
    assert parse_coord("E8") in wins
    assert parse_coord("K8") in wins


def test_extract_label_prefers_json() -> None:
    assert extract_label('{"class":"off-board"}') == "off_board"


def test_extract_label_does_not_accept_negated_expected_word() -> None:
    assert extract_label("not forbidden; legal") == "legal"


def test_extract_first_coord_prefers_json() -> None:
    assert extract_first_coord('{"move":"K8"} explanation H8') == parse_coord("K8")


def test_extract_first_coord_accepts_off_board_json() -> None:
    assert extract_first_coord('{"move":"P16"}') == (15, 15)


def test_parse_coord_relaxed_accepts_a0() -> None:
    assert parse_coord_relaxed("A0") == (-1, 0)


def test_score_off_board_move_zero() -> None:
    board = Board.empty()
    assert score_candidate_move(board, BLACK, parse_coord_relaxed("P16"), set()) == 0.0


def test_score_good_and_blocking_moves() -> None:
    board = Board.empty()
    good = {parse_coord("H8")}
    blocking = {parse_coord("I8")}
    assert score_candidate_move(board, BLACK, parse_coord("H8"), set(), good_moves=good) == 0.80
    assert score_candidate_move(board, BLACK, parse_coord("I8"), set(), blocking_moves=blocking) == 0.75
