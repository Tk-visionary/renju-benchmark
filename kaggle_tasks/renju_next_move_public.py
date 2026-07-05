# %%
import json
import re

import kaggle_benchmarks as kbench
import pandas as pd


# %%
RECORDS = [
    {
        "record_json": json.dumps({
            "id": "black_exact_five",
            "side": "black",
            "board": "\n".join([
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                ".....XXXX......",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]),
            "best_moves": ["E8", "J8"],
            "good_moves": [],
            "blocking_moves": [],
            "forbidden_moves": [],
            "mode": "fast",
        })
    },
    {
        "record_json": json.dumps({
            "id": "white_overline_win",
            "side": "white",
            "board": "\n".join([
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                ".....OOOOO.....",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]),
            "best_moves": ["E8", "K8"],
            "good_moves": [],
            "blocking_moves": [],
            "forbidden_moves": [],
            "mode": "fast",
        })
    },
    {
        "record_json": json.dumps({
            "id": "black_double_four_forbidden",
            "side": "black",
            "board": "\n".join([
                "...............",
                "...............",
                "...............",
                "...............",
                ".......X.......",
                ".......X.......",
                ".......X.......",
                "....XXX........",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]),
            "best_moves": [],
            "good_moves": [],
            "blocking_moves": [],
            "forbidden_moves": ["H8"],
            "mode": "strict",
        })
    },
]

EVALUATION_DATA = pd.DataFrame(RECORDS)
COORD_RE = re.compile(r"\b([A-Za-z])([0-9]{1,2})\b")


# %%
def extract_first_coord(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("move"), str):
        return normalize_coord(payload["move"])

    match = COORD_RE.search(text)
    if match:
        return normalize_coord(match.group(0))
    return None


def normalize_coord(coord: str) -> str:
    match = COORD_RE.fullmatch(coord.strip())
    if not match:
        return coord.strip().upper()
    return f"{match.group(1).upper()}{int(match.group(2))}"


def prompt_for_record(record: dict) -> str:
    return (
        "You are playing Renju on a 15x15 board. Coordinates are A1 through O15. "
        "X is black and O is white. Black loses by forbidden overline, double-four, "
        "or forbidden double-three unless black simultaneously makes exactly five. "
        f"It is {record['side']}'s turn. Return exactly one JSON object like {{\"move\":\"H8\"}}.\n\n"
        f"{record['board']}"
    )


def score_move(move: str | None, record: dict) -> float:
    if move is None:
        return 0.0
    if move in {normalize_coord(coord) for coord in record.get("forbidden_moves", [])}:
        return 0.0
    if move in {normalize_coord(coord) for coord in record.get("best_moves", [])}:
        return 1.0
    if move in {normalize_coord(coord) for coord in record.get("good_moves", [])}:
        return 0.80
    if move in {normalize_coord(coord) for coord in record.get("blocking_moves", [])}:
        return 0.75
    return 0.0


# %%
@kbench.task(name="renju-next-move-public")
def renju_next_move_public(llm, record_json: str) -> float:
    record = json.loads(record_json)
    response = llm.prompt(prompt_for_record(record))
    return score_move(extract_first_coord(response), record)


# %%
renju_next_move_public.evaluate(
    llm=[kbench.llm],
    evaluation_data=EVALUATION_DATA,
)
