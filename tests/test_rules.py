from renju_benchmark.rules import BLACK, WHITE, Board, MoveResult, RenjuGame, RuleMode, black_forbidden_report, parse_coord


def empty_rows() -> list[list[str]]:
    return [list("." * 15) for _ in range(15)]


def board_from_points(points: list[tuple[str, str]]) -> Board:
    rows = empty_rows()
    for coord, color in points:
        row, col = parse_coord(coord)
        rows[row][col] = color
    return Board.from_text("\n".join("".join(row) for row in rows))


def test_black_exact_five_wins_even_if_extensions_exist() -> None:
    board = board_from_points([("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK)])
    game = RenjuGame(board=board, turn=BLACK)
    assert game.play(*parse_coord("J8")) == MoveResult.BLACK_WIN


def test_white_overline_wins() -> None:
    board = board_from_points([
        ("F8", WHITE), ("G8", WHITE), ("H8", WHITE), ("I8", WHITE), ("J8", WHITE)
    ])
    game = RenjuGame(board=board, turn=WHITE)
    assert game.play(*parse_coord("K8")) == MoveResult.WHITE_WIN


def test_white_edge_five_does_not_hang() -> None:
    board = board_from_points([
        ("K15", WHITE), ("L15", WHITE), ("M15", WHITE), ("N15", WHITE),
    ])
    game = RenjuGame(board=board, turn=WHITE)
    assert game.play(*parse_coord("O15")) == MoveResult.WHITE_WIN


def test_black_overline_is_forbidden() -> None:
    board = board_from_points([
        ("F8", BLACK), ("G8", BLACK), ("H8", BLACK), ("I8", BLACK), ("J8", BLACK)
    ])
    report = black_forbidden_report(board, *parse_coord("K8"))
    assert report.forbidden
    assert report.overline


def test_black_double_four_is_forbidden() -> None:
    board = board_from_points([
        ("E8", BLACK), ("F8", BLACK), ("G8", BLACK),
        ("H5", BLACK), ("H6", BLACK), ("H7", BLACK),
    ])
    report = black_forbidden_report(board, *parse_coord("H8"))
    assert report.forbidden
    assert report.double_four
    assert report.fours >= 2


def test_black_double_three_is_forbidden_in_strict_mode() -> None:
    board = board_from_points([
        ("F8", BLACK), ("G8", BLACK),
        ("H6", BLACK), ("H7", BLACK),
    ])
    report = black_forbidden_report(board, *parse_coord("H8"), _depth=1)
    assert report.forbidden
    assert report.double_three
    assert report.threes >= 2


def test_fast_mode_skips_double_three() -> None:
    board = board_from_points([
        ("F8", BLACK), ("G8", BLACK),
        ("H6", BLACK), ("H7", BLACK),
    ])
    strict = black_forbidden_report(board, *parse_coord("H8"), _depth=2)
    fast = black_forbidden_report(board, *parse_coord("H8"), _depth=0)
    assert strict.double_three
    assert not fast.double_three


def test_named_rule_mode_sets_depth() -> None:
    fast_game = RenjuGame.new(RuleMode.FAST)
    strict_game = RenjuGame.new(RuleMode.STRICT)
    assert fast_game.forbidden_depth == 0
    assert strict_game.forbidden_depth == 2


def test_occupied_move_is_illegal() -> None:
    board = board_from_points([("H8", BLACK)])
    game = RenjuGame(board=board, turn=WHITE)
    assert game.play(*parse_coord("H8")) == MoveResult.ILLEGAL_OCCUPIED
