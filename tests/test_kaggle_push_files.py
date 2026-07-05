from __future__ import annotations

import ast
from pathlib import Path


TASK_FILES = (
    Path("kaggle_tasks/renju_next_move_public.py"),
    Path("kaggle_tasks/renju_rule_classification_public.py"),
    Path("kaggle_tasks/renju_model_arena_public.py"),
)


def test_kaggle_push_files_have_notebook_cells_and_evaluate_call() -> None:
    repo = Path(__file__).resolve().parents[1]
    for rel_path in TASK_FILES:
        text = (repo / rel_path).read_text()
        assert "# %%" in text
        assert "import kaggle_benchmarks as kbench" in text
        assert "@kbench.task(" in text
        assert ".evaluate(" in text
        assert ".run(" not in text


def test_kaggle_push_files_define_one_annotated_task() -> None:
    repo = Path(__file__).resolve().parents[1]
    for rel_path in TASK_FILES:
        tree = ast.parse((repo / rel_path).read_text())
        task_functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "task"
                for decorator in node.decorator_list
            )
        ]
        assert len(task_functions) == 1
        task_fn = task_functions[0]
        assert task_fn.args.args[0].arg == "llm"
        assert task_fn.returns is not None
