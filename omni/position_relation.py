"""Exact memory-mapped canonical relation for position-owned native rows."""
from __future__ import annotations

from collections import Counter
import json
import mmap
from pathlib import Path
import struct
from typing import Mapping, Sequence


SCHEMA = "unison-packed-position-relation/v1"
RECORD = struct.Struct("<6I")


class PackedPositionRelation:
    """Read exact prefix marginals from sorted fixed-width canonical records."""

    def __init__(self, path: str | Path, receipt_path: str | Path | None = None):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        size = self.path.stat().st_size
        if size <= 0 or size % RECORD.size:
            self._handle.close()
            raise RuntimeError("packed position relation has invalid byte length")
        self.row_count = size // RECORD.size
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.receipt = None
        if receipt_path is not None:
            self.receipt = json.loads(Path(receipt_path).read_text())
            if (self.receipt.get("schema") != SCHEMA + "/receipt"
                    or self.receipt.get("status") != "sealed"
                    or self.receipt.get("packed_bytes") != size
                    or self.receipt.get("unique_canonical_entries") != self.row_count):
                self.close()
                raise RuntimeError("packed position relation receipt mismatch")

    def close(self) -> None:
        held_map = getattr(self, "_map", None)
        if held_map is not None:
            held_map.close()
            self._map = None
        held_handle = getattr(self, "_handle", None)
        if held_handle is not None:
            held_handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _record(self, index: int) -> tuple[int, int, int, int, int, int]:
        return RECORD.unpack_from(self._map, index * RECORD.size)

    def _boundary(self, prefix: Sequence[int], upper: bool) -> int:
        if not 1 <= len(prefix) <= 5:
            raise ValueError("canonical prefix must contain one to five fields")
        prefix = tuple(int(value) for value in prefix)
        left, right = 0, self.row_count
        width = len(prefix)
        while left < right:
            middle = (left + right) // 2
            held = self._record(middle)[:width]
            if held < prefix or (upper and held == prefix):
                left = middle + 1
            else:
                right = middle
        return left

    def range(self, prefix: Sequence[int]) -> tuple[int, int]:
        return self._boundary(prefix, False), self._boundary(prefix, True)

    def counts(self, prefix: Sequence[int]) -> dict[int, int]:
        """Marginalise unfixed canonical fields onto exact next-token counts."""
        start, end = self.range(prefix)
        counts: Counter[int] = Counter()
        for index in range(start, end):
            _, _, _, _, next_id, count = self._record(index)
            counts[next_id] += count
        return dict(counts)

    def value_counts(self, relative_position: int, key_id: int) -> dict[int, int]:
        return self.counts((relative_position, key_id))

    def semantic2_counts(self, last_id: int, relative_position: int,
                         key_id: int) -> dict[int, int]:
        return self.counts((relative_position, key_id, last_id))

    def semantic3_counts(self, previous_id: int, last_id: int,
                         relative_position: int, key_id: int) -> dict[int, int]:
        return self.counts((relative_position, key_id, last_id, previous_id))

    def identity(self) -> Mapping:
        return {
            "schema": SCHEMA,
            "path": str(self.path),
            "rows": self.row_count,
            "sha256": None if self.receipt is None
            else self.receipt.get("packed_sha256"),
        }
