from __future__ import annotations

import os
import select
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .rules import BLACK, WHITE, BOARD_SIZE, Board


class RapfiError(RuntimeError):
    pass


@dataclass(frozen=True)
class RapfiConfig:
    executable: str | Path
    cwd: str | Path | None = None
    startup_timeout: float = 5.0
    move_timeout: float = 10.0
    timeout_turn_ms: int | None = None
    max_depth: int | None = None
    max_node: int | None = None
    rule: int | None = 4

    @classmethod
    def from_env(cls) -> "RapfiConfig":
        executable = os.environ.get("RAPFI_PATH")
        if not executable:
            raise RapfiError("RAPFI_PATH is not set")
        return cls(
            executable=executable,
            cwd=os.environ.get("RAPFI_CWD") or None,
            move_timeout=float(os.environ.get("RAPFI_MOVE_TIMEOUT", "10.0")),
            timeout_turn_ms=_optional_int_env("RAPFI_TIMEOUT_TURN_MS"),
            max_depth=_optional_int_env("RAPFI_MAX_DEPTH"),
            max_node=_optional_int_env("RAPFI_MAX_NODE"),
            rule=_optional_int_env("RAPFI_RULE", default=4),
        )


class RapfiEngine:
    """Minimal Gomocup-protocol wrapper for Rapfi-compatible engines."""

    def __init__(self, config: RapfiConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "RapfiEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        executable = Path(self.config.executable).resolve()
        cwd = Path(self.config.cwd).resolve() if self.config.cwd is not None else None
        self.process = subprocess.Popen(
            [str(executable)],
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send(f"START {BOARD_SIZE}")
        response = self._read_until(lambda line: line.upper() in {"OK", "OK."})
        if response.upper() not in {"OK", "OK."}:
            raise RapfiError(f"Rapfi START failed: {response}")
        if self.config.rule is not None:
            self._send(f"INFO rule {self.config.rule}")
        if self.config.timeout_turn_ms is not None:
            self._send(f"INFO timeout_turn {self.config.timeout_turn_ms}")
        if self.config.max_depth is not None:
            self._send(f"INFO max_depth {self.config.max_depth}")
        if self.config.max_node is not None:
            self._send(f"INFO max_node {self.config.max_node}")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            self._send("END")
        except RapfiError:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=2)
        finally:
            self.process = None

    def best_move(self, board: Board, color: str) -> tuple[int, int]:
        stones = [(row, col, stone) for row, col, stone in _iter_stones(board)]
        return self.best_move_from_history(stones, color)

    def best_move_from_history(self, history: list[tuple[int, int, str]], color: str) -> tuple[int, int]:
        if self.process is None:
            self.start()
        occupied = {(row, col) for row, col, _stone in history}
        self._drain_pending_output()
        self._send("BOARD")
        for row, col, stone in history:
            player = 1 if stone == BLACK else 2
            self._send(f"{col},{row},{player}")
        self._send("DONE")
        responses = []
        last_error: RapfiError | None = None
        for _attempt in range(4):
            try:
                responses.extend(self._read_move_responses(timeout=None if _attempt == 0 else 0.1))
            except RapfiError:
                if responses:
                    break
                raise
            for response in responses:
                try:
                    move = _parse_engine_move(response)
                except RapfiError as exc:
                    last_error = exc
                    continue
                if move not in occupied:
                    self._drain_pending_output(timeout=0.02)
                    return move
        if last_error is not None:
            raise last_error
        raise RapfiError(f"Rapfi returned no legal move candidates: {responses!r}")

    def _send(self, command: str) -> None:
        stdin = self._stdin()
        stdin.write(command + "\n")
        stdin.flush()

    def _readline(self, timeout: float | None = None) -> str:
        stdout = self._stdout()
        wait = self.config.move_timeout if timeout is None else timeout
        ready, _, _ = select.select([stdout], [], [], wait)
        if not ready:
            raise RapfiError(f"Rapfi did not respond within {wait:.1f}s")
        line = stdout.readline()
        if line == "":
            raise RapfiError("Rapfi process ended unexpectedly")
        return line.strip()

    def _read_until(self, predicate, timeout: float | None = None) -> str:
        while True:
            line = self._readline(timeout)
            if predicate(line):
                return line
            if line.upper().startswith("ERROR"):
                raise RapfiError(line)

    def _read_move_responses(self, timeout: float | None = None) -> list[str]:
        candidates: list[str] = []
        while True:
            try:
                line = self._readline(timeout=timeout)
            except RapfiError:
                if candidates:
                    return candidates
                raise
            if line.upper().startswith("ERROR"):
                raise RapfiError(line)
            if _looks_like_raw_engine_move(line):
                return [line, *candidates]
            if _looks_like_engine_move(line):
                candidates.extend(_extract_move_tokens(line, raise_on_missing=False))

    def _stdin(self) -> TextIO:
        if self.process is None or self.process.stdin is None:
            raise RapfiError("Rapfi process is not running")
        return self.process.stdin

    def _stdout(self) -> TextIO:
        if self.process is None or self.process.stdout is None:
            raise RapfiError("Rapfi process is not running")
        return self.process.stdout

    def _drain_pending_output(self, timeout: float = 0.0) -> None:
        stdout = self._stdout()
        while True:
            ready, _, _ = select.select([stdout], [], [], timeout)
            if not ready:
                return
            if stdout.readline() == "":
                return
            timeout = 0.0


def _iter_stones(board: Board):
    for index, stone in enumerate(board.cells):
        if stone in (BLACK, WHITE):
            row, col = divmod(index, BOARD_SIZE)
            yield row, col, stone


def _parse_engine_move(response: str) -> tuple[int, int]:
    first = _extract_move_token(response)
    parts = first.split(",")
    if len(parts) >= 2:
        try:
            col = int(parts[0])
            row = int(parts[1])
        except ValueError as exc:
            raise RapfiError(f"could not parse Rapfi move: {response!r}") from exc
    else:
        col = ord(first[0].lower()) - ord("a")
        try:
            row = int(first[1:]) - 1
        except ValueError as exc:
            raise RapfiError(f"could not parse Rapfi move: {response!r}") from exc
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise RapfiError(f"Rapfi returned off-board move: {response!r}")
    return row, col


def _looks_like_engine_move(response: str) -> bool:
    first = _extract_move_token(response, raise_on_missing=False)
    if not first:
        return False
    parts = first.split(",")
    if len(parts) >= 2:
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            return False
        return True
    if len(first) < 2 or not ("A" <= first[0].upper() <= "O"):
        return False
    return first[1:].isdigit()


def _looks_like_raw_engine_move(response: str) -> bool:
    first = response.split()[0] if response.split() else ""
    parts = first.split(",")
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _extract_move_token(response: str, raise_on_missing: bool = True) -> str:
    tokens = _extract_move_tokens(response, raise_on_missing=raise_on_missing)
    return tokens[0] if tokens else ""


def _extract_move_tokens(response: str, raise_on_missing: bool = True) -> list[str]:
    tokens = response.split()
    if not tokens:
        if raise_on_missing:
            raise RapfiError(f"could not parse Rapfi move: {response!r}")
        return []
    first = tokens[0]
    if "," in first:
        return [first]
    if first.upper() == "MESSAGE" and "|" in tokens:
        pipe = len(tokens) - 1 - tokens[::-1].index("|")
        if pipe + 1 < len(tokens):
            return tokens[pipe + 1 :]
    return [first]


def rapfi_move(board: Board, color: str, config: RapfiConfig | None = None) -> tuple[int, int]:
    with RapfiEngine(config or RapfiConfig.from_env()) as engine:
        return engine.best_move(board, color)


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)
