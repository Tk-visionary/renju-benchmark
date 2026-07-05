# %%
import json
import re

import kaggle_benchmarks as kbench
import pandas as pd


# %%
EMPTY_BOARD = "\n".join(["..............."] * 15)

RECORDS = [
    {
        "record_json": json.dumps({
            "id": "rule_black_exact_five",
            "side": "black",
            "move": "J8",
            "expected": "win",
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
        })
    },
    {
        "record_json": json.dumps({
            "id": "rule_black_overline_forbidden",
            "side": "black",
            "move": "K8",
            "expected": "forbidden",
            "board": "\n".join([
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                ".....XXXXX.....",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]),
        })
    },
    {
        "record_json": json.dumps({
            "id": "rule_occupied",
            "side": "white",
            "move": "H8",
            "expected": "occupied",
            "board": "\n".join([
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                ".......X.......",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]),
        })
    },
    {
        "record_json": json.dumps({
            "id": "rule_off_board",
            "side": "black",
            "move": "P16",
            "expected": "off_board",
            "board": EMPTY_BOARD,
        })
    },
]

EVALUATION_DATA = pd.DataFrame(RECORDS)
LABELS = {"legal", "win", "forbidden", "occupied", "off_board"}


# %%
def extract_label(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("class"), str):
        label = normalize_label(payload["class"])
        if label in LABELS:
            return label

    normalized = text.strip().lower().replace("-", "_")
    previous = ""
    for token in re.findall(r"[a-z_]+", normalized):
        if token in LABELS:
            if previous in {"not", "non", "no"}:
                previous = token
                continue
            return token
        previous = token
    return None


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_")


def prompt_for_record(record: dict) -> str:
    return (
        "Classify this Renju move as one of: legal, win, forbidden, occupied, off_board. "
        "Use RIF rules: white overline wins, black overline/double-four/forbidden double-three lose "
        "unless black simultaneously makes exactly five.\n"
        f"Side: {record['side']}\nMove: {record['move']}\nBoard:\n{record['board']}\n"
        "Return exactly one JSON object like {\"class\":\"legal\"}."
    )


# %%
@kbench.task(name="renju-rule-classification-public")
def renju_rule_classification_public(llm, record_json: str) -> bool:
    record = json.loads(record_json)
    response = llm.prompt(prompt_for_record(record))
    return extract_label(response) == normalize_label(record["expected"])


# %%
renju_rule_classification_public.evaluate(
    llm=[kbench.llm],
    evaluation_data=EVALUATION_DATA,
)
