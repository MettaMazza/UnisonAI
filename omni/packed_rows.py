"""Exact memory-mapped count rows for Unison's native transformer.

Python dictionaries expand the sealed v4 artifact from 2.92 GB on disk to
roughly 59 GB resident.  This module changes only representation: each address
is stored once in a packed data file and located through a collision-verified
open-address index.  The index capacity is the smallest power of two strictly
larger than the number of rows, so every insertion is total and no count row is
pruned, approximated, or capped.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import pickle
import struct
import time
from typing import Mapping


PACKED_SCHEMA = "unison-packed-exact-rows/v1"
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_PAIR = struct.Struct("<IQ")
_MASK64 = (1 << 64) - 1
_EMPTY = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _key_tuple(key, key_size: int) -> tuple[int, ...]:
    values = (key,) if key_size == 1 and isinstance(key, int) else tuple(key)
    if len(values) != key_size:
        raise ValueError(f"expected {key_size}-part key, got {values!r}")
    for value in values:
        if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"packed key component outside uint32: {value!r}")
    return values


def _hash_key(values: tuple[int, ...]) -> int:
    """Stable 64-bit tuple hash; collisions are resolved and key-verified."""
    held = 0xCBF29CE484222325
    for value in values:
        held ^= value
        held = (held * 0x100000001B3) & _MASK64
        held ^= held >> 32
    return held


def _capacity(row_count: int) -> int:
    capacity = 1
    while capacity <= row_count:
        capacity <<= 1
    return capacity


def build_packed_table(directory: Path, name: str, mapping: Mapping,
                       key_size: int, scalar: bool = False) -> dict:
    """Write one exact mapping as a packed data file and verified hash index."""
    directory.mkdir(parents=True, exist_ok=True)
    row_count = len(mapping)
    capacity = _capacity(row_count)
    index_path = directory / f"{name}.index"
    data_path = directory / f"{name}.data"
    key_struct = struct.Struct("<" + "I" * key_size)

    with index_path.open("w+b") as index_handle:
        index_handle.truncate(capacity * _U64.size)
        index = mmap.mmap(index_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            with data_path.open("wb") as data:
                next_report = time.monotonic() + 60
                for ordinal, (raw_key, raw_value) in enumerate(mapping.items(), 1):
                    key = _key_tuple(raw_key, key_size)
                    offset = data.tell()
                    row_bytes = bytearray(key_struct.pack(*key))
                    if scalar:
                        value = int(raw_value)
                        if value < 0 or value > _MASK64:
                            raise ValueError(f"packed scalar outside uint64: {value}")
                        row_bytes.extend(_U64.pack(value))
                    else:
                        items = sorted((int(token_id), int(count))
                                       for token_id, count in raw_value.items()
                                       if count > 0)
                        row_bytes.extend(_U32.pack(len(items)))
                        for token_id, count in items:
                            if not 0 <= token_id <= 0xFFFFFFFF:
                                raise ValueError("packed token id outside uint32")
                            if not 0 <= count <= _MASK64:
                                raise ValueError("packed count outside uint64")
                            row_bytes.extend(_PAIR.pack(token_id, count))
                    data.write(row_bytes)

                    slot = _hash_key(key) & (capacity - 1)
                    while _U64.unpack_from(index, slot * _U64.size)[0] != _EMPTY:
                        slot = (slot + 1) & (capacity - 1)
                    # Zero is the empty sentinel; data offsets are stored plus one.
                    _U64.pack_into(index, slot * _U64.size, offset + 1)

                    now = time.monotonic()
                    if now >= next_report:
                        print(f"packing {name}: {ordinal:,}/{row_count:,} rows", flush=True)
                        next_report = now + 60
            index.flush()
        finally:
            index.close()

    return {
        "name": name,
        "rows": row_count,
        "capacity": capacity,
        "key_size": key_size,
        "scalar": bool(scalar),
        "index_bytes": index_path.stat().st_size,
        "data_bytes": data_path.stat().st_size,
        "index_sha256": _sha256(index_path),
        "data_sha256": _sha256(data_path),
    }


class PackedTable:
    """Read-only mapping facade over one exact memory-mapped packed table."""

    def __init__(self, directory: Path, specification: dict):
        self.directory = Path(directory)
        self.specification = dict(specification)
        self.name = self.specification["name"]
        self.key_size = int(self.specification["key_size"])
        self.scalar = bool(self.specification["scalar"])
        self.capacity = int(self.specification["capacity"])
        self.row_count = int(self.specification["rows"])
        if self.capacity != _capacity(self.row_count):
            raise RuntimeError(f"packed {self.name} index capacity is not minimal")
        self._key_struct = struct.Struct("<" + "I" * self.key_size)
        index_path = self.directory / f"{self.name}.index"
        data_path = self.directory / f"{self.name}.data"
        if (index_path.stat().st_size != self.specification["index_bytes"]
                or data_path.stat().st_size != self.specification["data_bytes"]):
            raise RuntimeError(f"packed {self.name} file-size mismatch")
        self._index_handle = index_path.open("rb")
        self._data_handle = data_path.open("rb")
        self._index = mmap.mmap(self._index_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._data = mmap.mmap(self._data_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.row_count

    def get(self, raw_key, default=None):
        key = _key_tuple(raw_key, self.key_size)
        slot = _hash_key(key) & (self.capacity - 1)
        for _ in range(self.capacity):
            held = _U64.unpack_from(self._index, slot * _U64.size)[0]
            if held == _EMPTY:
                return default
            offset = held - 1
            found = self._key_struct.unpack_from(self._data, offset)
            if found == key:
                cursor = offset + self._key_struct.size
                if self.scalar:
                    return _U64.unpack_from(self._data, cursor)[0]
                item_count = _U32.unpack_from(self._data, cursor)[0]
                cursor += _U32.size
                row = {}
                for _ in range(item_count):
                    token_id = _U32.unpack_from(self._data, cursor)[0]
                    cursor += _U32.size
                    count = _U64.unpack_from(self._data, cursor)[0]
                    cursor += _U64.size
                    row[token_id] = count
                return row
            slot = (slot + 1) & (self.capacity - 1)
        raise RuntimeError(f"packed {self.name} lookup exhausted its exact index")

    def close(self) -> None:
        for value in (getattr(self, "_index", None), getattr(self, "_data", None)):
            if value is not None:
                value.close()
        for value in (getattr(self, "_index_handle", None),
                      getattr(self, "_data_handle", None)):
            if value is not None:
                value.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


TABLES = {
    "semantic_ffn3": (3, False),
    "semantic_ffn": (2, False),
    "qk": (2, True),
    "values": (1, False),
    "ffn3": (2, False),
    "ffn2": (1, False),
    "profiles": (1, False),
}


def pack_record(record: dict, directory: Path, artifact_sha256: str,
                consume: bool = False) -> dict:
    """Create the exact packed serving view of a counted transformer record."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tables = []
    source = record if consume else dict(record)
    metadata = {
        key: value for key, value in source.items()
        if key not in TABLES and key != "profile_index"
    }
    metadata["artifact_sha256"] = artifact_sha256
    metadata["packed_schema"] = PACKED_SCHEMA
    metadata["packed_counts"] = {
        name: len(source[name]) for name in TABLES
    }
    if consume:
        source.pop("profile_index", None)
    for name, (key_size, scalar) in TABLES.items():
        mapping = source.pop(name) if consume else source[name]
        specification = build_packed_table(
            directory, name, mapping, key_size=key_size, scalar=scalar)
        tables.append(specification)
        if consume:
            del mapping
    with (directory / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": PACKED_SCHEMA,
        "artifact_sha256": artifact_sha256,
        "tables": tables,
        "metadata_bytes": (directory / "metadata.pkl").stat().st_size,
        "metadata_sha256": _sha256(directory / "metadata.pkl"),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_packed_record(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != PACKED_SCHEMA:
        raise RuntimeError("packed native-transformer manifest schema mismatch")
    metadata_path = directory / "metadata.pkl"
    if (metadata_path.stat().st_size != manifest.get("metadata_bytes")
            or _sha256(metadata_path) != manifest.get("metadata_sha256")):
        raise RuntimeError("packed native-transformer metadata mismatch")
    with metadata_path.open("rb") as handle:
        record = pickle.load(handle)
    if (record.get("packed_schema") != PACKED_SCHEMA
            or record.get("artifact_sha256") != manifest.get("artifact_sha256")):
        raise RuntimeError("packed native-transformer provenance mismatch")
    for specification in manifest["tables"]:
        record[specification["name"]] = PackedTable(directory, specification)
    return record


def verify_packed_files(directory: Path) -> dict:
    """Hash every packed file without loading count rows into Python objects."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    failures = []
    for specification in manifest.get("tables", []):
        for suffix in ("index", "data"):
            path = directory / f"{specification['name']}.{suffix}"
            expected = specification[f"{suffix}_sha256"]
            if _sha256(path) != expected:
                failures.append(str(path.name))
    if _sha256(directory / "metadata.pkl") != manifest.get("metadata_sha256"):
        failures.append("metadata.pkl")
    return {"schema": PACKED_SCHEMA + "/verification",
            "status": "verified" if not failures else "failed",
            "failures": failures, "tables": len(manifest.get("tables", []))}
