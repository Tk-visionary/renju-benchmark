from __future__ import annotations

import importlib
import sys
import types


class DummyLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def prompt(self, text: str) -> str:
        self.prompts.append(text)
        return self.response


def import_with_fake_kbench():
    fake = types.ModuleType("kaggle_benchmarks")

    def task(name: str):
        def decorator(fn):
            fn._kaggle_task_name = name
            return fn

        return decorator

    fake.task = task
    sys.modules["kaggle_benchmarks"] = fake
    sys.modules.pop("renju_benchmark.kaggle_tasks", None)
    return importlib.import_module("renju_benchmark.kaggle_tasks")


def test_kaggle_tasks_register_with_kbench_decorator() -> None:
    module = import_with_fake_kbench()

    assert module.renju_next_move._kaggle_task_name == "renju_next_move"
    assert module.renju_next_move_record._kaggle_task_name == "renju_next_move_record"
    assert module.renju_rule_classification._kaggle_task_name == "renju_rule_classification"
    assert module.renju_rule_classification_record._kaggle_task_name == "renju_rule_classification_record"


def test_kaggle_next_move_scores_json_response() -> None:
    module = import_with_fake_kbench()
    board_text = "\n".join([
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
    ])

    assert module.score_next_move_response('{"move":"J8"}', board_text, "black", ["J8"], mode="fast") == 1.0
    assert module.renju_next_move(DummyLlm('{"move":"J8"}'), board_text, "black", ["J8"], mode="fast") == 1.0


def test_kaggle_rule_classification_scores_json_response() -> None:
    module = import_with_fake_kbench()
    assert module.score_rule_response('{"class":"off_board"}', "off_board") is True
