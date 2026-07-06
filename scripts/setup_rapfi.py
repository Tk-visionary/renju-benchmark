from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

RAPFI_REPO = "https://github.com/dhbloo/rapfi.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def default_build_dir(repo: Path) -> Path:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return repo / "Rapfi" / "build" / "arm64-clang-NEON"
    return repo / "Rapfi" / "build" / "x64-clang-AVX2"


def configure_command(repo: Path, build_dir: Path) -> list[str]:
    machine = platform.machine().lower()
    source = str((repo / "Rapfi").resolve())
    command = [
        "cmake",
        source,
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if machine in {"arm64", "aarch64"}:
        command.extend(["-DUSE_NEON=ON", "-DUSE_NEON_DOTPROD=OFF"])
    else:
        command.extend([
            "-DUSE_SSE=ON",
            "-DUSE_AVX2=ON",
            "-DUSE_AVX512=OFF",
            "-DUSE_BMI2=OFF",
            "-DUSE_VNNI=OFF",
        ])
    return command


def find_executable(repo: Path) -> Path | None:
    candidates = [
        path
        for path in repo.rglob("*")
        if path.is_file() and os.access(path, os.X_OK) and "rapfi" in path.name.lower()
    ]
    return sorted(candidates, key=lambda path: len(path.parts))[0] if candidates else None


def prepare_runtime(repo: Path, executable: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_executable = runtime_dir / executable.name
    shutil.copy2(executable, runtime_executable)

    networks = repo / "Networks"
    files = [
        networks / "config-example" / "config.toml",
        networks / "classical" / "model210901.bin",
        networks / "mix9svq" / "mix9svqfreestyle_bsmix.bin.lz4",
        networks / "mix9svq" / "mix9svqstandard_bs15.bin.lz4",
        networks / "mix9svq" / "mix9svqrenju_bs15_black.bin.lz4",
        networks / "mix9svq" / "mix9svqrenju_bs15_white.bin.lz4",
    ]
    for source in files:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, runtime_dir / source.name)
    runtime_executable.chmod(runtime_executable.stat().st_mode | 0o111)
    return runtime_executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and build Rapfi outside the tracked repository files.")
    parser.add_argument("--repo-dir", type=Path, default=Path("external/rapfi"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("external/rapfi-runtime"))
    parser.add_argument("--url", default=RAPFI_REPO)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    repo = args.repo_dir
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--recursive", args.url, str(repo)])
    else:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo)

    if not args.skip_build:
        build_dir = default_build_dir(repo)
        build_dir.mkdir(parents=True, exist_ok=True)
        run(configure_command(repo, build_dir), cwd=build_dir)
        run(["cmake", "--build", "."], cwd=build_dir)

    executable = find_executable(repo)
    if executable is None:
        print("Rapfi executable was not found. Inspect the build directory manually.")
        return
    runtime_executable = prepare_runtime(repo, executable, args.runtime_dir)
    print(f"Rapfi executable: {runtime_executable.resolve()}")
    print(f"Rapfi runtime cwd: {args.runtime_dir.resolve()}")
    print(f"export RAPFI_PATH={runtime_executable.resolve()}")
    print(f"export RAPFI_CWD={args.runtime_dir.resolve()}")


if __name__ == "__main__":
    main()
