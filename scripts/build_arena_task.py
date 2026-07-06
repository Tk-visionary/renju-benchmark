from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_OPENING_BOARD = "\n".join([
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
])

DEFAULT_TEMPLATE = Path("kaggle_tasks/renju_model_arena_public.py")
EVALUATION_DATA_RE = re.compile(
    r"EVALUATION_DATA\s*=\s*pd\.DataFrame\(\[.*?\]\)\n",
    re.DOTALL,
)
TASK_NAME_RE = re.compile(r'@kbench\.task\(name="[^"]+"\)')


def slug_part(model: str) -> str:
    value = model.split("/")[-1].replace("@", "-")
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if args.pair:
        pairs = args.pair
    else:
        if len(args.model) < 2:
            raise SystemExit("Pass at least two --model values, or one or more --pair BLACK WHITE values.")
        pairs = [
            (black, white)
            for black in args.model
            for white in args.model
            if black != white
        ]

    for black, white in pairs:
        for game_index in range(args.games_per_pair):
            rows.append({
                "match_id": f"{slug_part(black)}-vs-{slug_part(white)}-{game_index:03d}",
                "black_model": black,
                "white_model": white,
                "board_text": args.board_text,
                "max_plies": args.max_plies,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained Kaggle model-arena task file.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="renju-model-arena-public")
    parser.add_argument("--model", action="append", default=[], help="Model slug for ordered round-robin generation.")
    parser.add_argument("--pair", action="append", nargs=2, metavar=("BLACK_MODEL", "WHITE_MODEL"))
    parser.add_argument("--games-per-pair", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--board-text", default=DEFAULT_OPENING_BOARD)
    args = parser.parse_args()

    rows = build_rows(args)
    source = args.template.read_text()
    replacement = (
        "EVALUATION_ROWS = "
        + json.dumps(rows, indent=4, sort_keys=True)
        + "\nEVALUATION_DATA = pd.DataFrame(EVALUATION_ROWS)\n"
    )
    source, count = EVALUATION_DATA_RE.subn(lambda _match: replacement, source, count=1)
    if count != 1:
        raise SystemExit(f"Could not replace EVALUATION_DATA in {args.template}")
    source = TASK_NAME_RE.sub(f'@kbench.task(name="{args.task_name}")', source, count=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    print(f"wrote {args.output} with {len(rows)} games for task {args.task_name}")


if __name__ == "__main__":
    main()
