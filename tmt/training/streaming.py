from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch


@dataclass
class ByteBatch:
    current: torch.Tensor
    next: torch.Tensor
    end: torch.Tensor
    new_document: torch.Tensor


def find_input_files(pattern: str | Sequence[str]) -> list[Path]:
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files: list[Path] = []
    for item in patterns:
        files.extend(Path().glob(item))
    return sorted(path for path in files if path.is_file())


class BytePairReader:
    def __init__(self, files: Sequence[Path], *, repeat: bool = True):
        if not files:
            raise ValueError("BytePairReader requires at least one input file")
        self.files = list(files)
        self.repeat = repeat
        self.file_index = 0
        self.data = b""
        self.position = 0
        self.current_file_index: int | None = None
        self._new_document_next = True

    def __iter__(self) -> "BytePairReader":
        return self

    def __next__(self) -> tuple[int, int, bool, bool]:
        while self.position >= max(0, len(self.data) - 1):
            self._load_next_document()
        current = self.data[self.position]
        next_byte = self.data[self.position + 1]
        end = self.position == len(self.data) - 2
        new_document = self._new_document_next
        self._new_document_next = False
        self.position += 1
        return current, next_byte, end, new_document

    def _load_next_document(self) -> None:
        while True:
            if self.file_index >= len(self.files):
                if not self.repeat:
                    raise StopIteration
                self.file_index = 0
            path = self.files[self.file_index]
            self.current_file_index = self.file_index
            self.file_index += 1
            data = path.read_bytes()
            if len(data) >= 2:
                self.data = data
                self.position = 0
                self._new_document_next = True
                return

    def state_dict(self) -> dict[str, object]:
        return {
            "files": [str(path.resolve()) for path in self.files],
            "repeat": self.repeat,
            "file_index": self.file_index,
            "current_file_index": self.current_file_index,
            "position": self.position,
            "new_document_next": self._new_document_next,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected_files = [str(path.resolve()) for path in self.files]
        if state.get("files") != expected_files:
            raise ValueError("checkpoint input files do not match the current data files")
        if bool(state.get("repeat")) != self.repeat:
            raise ValueError("checkpoint repeat setting does not match the current stream config")

        file_index = int(state["file_index"])  # type: ignore[arg-type]
        current_file_index = state.get("current_file_index")
        current_file_index = None if current_file_index is None else int(current_file_index)
        if not 0 <= file_index <= len(self.files):
            raise ValueError("checkpoint file index is out of range")
        if current_file_index is not None and not 0 <= current_file_index < len(self.files):
            raise ValueError("checkpoint current file index is out of range")

        self.file_index = file_index
        self.current_file_index = current_file_index
        self.data = b"" if current_file_index is None else self.files[current_file_index].read_bytes()
        self.position = int(state["position"])  # type: ignore[arg-type]
        if not 0 <= self.position <= len(self.data):
            raise ValueError("checkpoint byte position is out of range")
        self._new_document_next = bool(state["new_document_next"])


class IndependentByteBatcher:
    def __init__(self, files: Sequence[Path], batch_size: int, *, repeat: bool = True):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        files = list(files)
        if not files:
            raise ValueError("IndependentByteBatcher requires at least one input file")
        self.readers: list[BytePairReader] = []
        for index in range(batch_size):
            assigned = files[index::batch_size] or files
            self.readers.append(BytePairReader(assigned, repeat=repeat))

    def __iter__(self) -> "IndependentByteBatcher":
        return self

    def __next__(self) -> ByteBatch:
        rows = [next(reader) for reader in self.readers]
        current, next_byte, end, new_document = zip(*rows)
        return ByteBatch(
            current=torch.tensor(current, dtype=torch.long),
            next=torch.tensor(next_byte, dtype=torch.long),
            end=torch.tensor(end, dtype=torch.float32),
            new_document=torch.tensor(new_document, dtype=torch.bool),
        )

    def state_dict(self) -> dict[str, object]:
        return {"readers": [reader.state_dict() for reader in self.readers]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        reader_states = state.get("readers")
        if not isinstance(reader_states, list) or len(reader_states) != len(self.readers):
            raise ValueError("checkpoint reader count does not match the current batch size")
        for reader, reader_state in zip(self.readers, reader_states):
            if not isinstance(reader_state, dict):
                raise ValueError("invalid reader state in checkpoint")
            reader.load_state_dict(reader_state)
