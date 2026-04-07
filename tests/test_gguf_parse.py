"""Tests for the GGUF metadata parser."""

import struct
from pathlib import Path

import pytest

from gguf_parse import (
    GGUF_MAGIC,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GGUFParser,
    GGML_TYPE_NAMES,
)


def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_kv_string(f, key: str, value: str):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    _write_string(f, value)


def _write_kv_uint32(f, key: str, value: int):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def _write_tensor_info(f, name: str, dims: tuple, type_id: int, offset: int):
    _write_string(f, name)
    f.write(struct.pack("<I", len(dims)))
    for d in dims:
        f.write(struct.pack("<Q", d))
    f.write(struct.pack("<I", type_id))
    f.write(struct.pack("<Q", offset))


def _create_minimal_gguf(path: Path, metadata: dict, tensors: list[dict]):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        for key, (type_id, value) in metadata.items():
            if type_id == GGUF_TYPE_STRING:
                _write_kv_string(f, key, value)
            elif type_id == GGUF_TYPE_UINT32:
                _write_kv_uint32(f, key, value)

        for t in tensors:
            _write_tensor_info(f, t["name"], t["dims"], t["type_id"], t["offset"])


class TestGGUFParser:
    def test_parse_minimal_gguf(self, tmp_path):
        path = tmp_path / "test.gguf"
        metadata = {
            "general.architecture": (GGUF_TYPE_STRING, "qwen3"),
            "general.name": (GGUF_TYPE_STRING, "test-model"),
        }
        tensors = [
            {"name": "token_embd.weight", "dims": (4096, 32000), "type_id": 1, "offset": 0},
        ]
        _create_minimal_gguf(path, metadata, tensors)

        meta = GGUFParser(path).parse()

        assert meta.version == 3
        assert meta.n_tensors == 1
        assert meta.architecture == "qwen3"
        assert meta.name == "test-model"
        assert len(meta.tensors) == 1
        assert meta.tensors[0].name == "token_embd.weight"
        assert meta.tensors[0].dims == (4096, 32000)
        assert meta.tensors[0].type_name == "F16"

    def test_parameter_count(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_minimal_gguf(path, {"general.architecture": (GGUF_TYPE_STRING, "llama")}, [
            {"name": "w1", "dims": (100, 200), "type_id": 0, "offset": 0},
            {"name": "w2", "dims": (50, 50, 4), "type_id": 0, "offset": 0},
        ])
        assert GGUFParser(path).parse().parameter_count == (100 * 200) + (50 * 50 * 4)

    def test_quantization_type_counts(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_minimal_gguf(path, {"general.architecture": (GGUF_TYPE_STRING, "qwen3")}, [
            {"name": "attn.wq", "dims": (4096, 4096), "type_id": 28, "offset": 0},
            {"name": "attn.wk", "dims": (4096, 1024), "type_id": 28, "offset": 0},
            {"name": "norm.weight", "dims": (4096,), "type_id": 1, "offset": 0},
        ])
        qtypes = GGUFParser(path).parse().quantization_types
        assert qtypes.get("Q1_0") == 2
        assert qtypes.get("F16") == 1

    def test_file_size_gb(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_minimal_gguf(path, {"general.architecture": (GGUF_TYPE_STRING, "test")}, [])
        meta = GGUFParser(path).parse()
        assert meta.file_size == path.stat().st_size
        assert meta.file_size_gb == meta.file_size / (1000 ** 3)

    def test_invalid_magic_raises(self, tmp_path):
        path = tmp_path / "bad.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0xDEADBEEF))
            f.write(b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            GGUFParser(path).parse()

    def test_multiple_metadata_types(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_minimal_gguf(path, {
            "general.architecture": (GGUF_TYPE_STRING, "qwen3"),
            "qwen3.block_count": (GGUF_TYPE_UINT32, 36),
        }, [])
        meta = GGUFParser(path).parse()
        assert meta.metadata["general.architecture"] == "qwen3"
        assert meta.metadata["qwen3.block_count"] == 36

    def test_tensor_type_name_mapping(self):
        assert GGML_TYPE_NAMES[0] == "F32"
        assert GGML_TYPE_NAMES[1] == "F16"
        assert GGML_TYPE_NAMES[8] == "Q8_0"
        assert GGML_TYPE_NAMES[15] == "Q4_K_M"
        assert GGML_TYPE_NAMES[28] == "Q1_0"

    def test_unknown_tensor_type(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_minimal_gguf(path, {"general.architecture": (GGUF_TYPE_STRING, "test")}, [
            {"name": "weird", "dims": (10,), "type_id": 255, "offset": 0},
        ])
        assert GGUFParser(path).parse().tensors[0].type_name == "unknown(255)"
