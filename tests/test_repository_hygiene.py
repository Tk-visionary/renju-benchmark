from __future__ import annotations

import subprocess
from pathlib import Path


LOCAL_PATH_MARKERS = (
    "/" + "Users" + "/",
    "kawahara" + "futoshishi",
    "file" + "://",
    "/" + "private" + "/",
    "/" + "var" + "/" + "folders" + "/",
    "C:" + "\\" + "Users" + "\\",
)


def test_tracked_files_do_not_contain_local_paths() -> None:
    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    offenders: list[str] = []
    for rel_path in tracked:
        path = repo / rel_path
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for marker in LOCAL_PATH_MARKERS:
            if marker in text:
                offenders.append(f"{rel_path}: {marker}")

    assert offenders == []
