"""Minimal GGUF metadata parser.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import math
import struct
from pathlib import Path
from dataclasses import dataclass

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16",
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS",
    28: "Q1_0",
    41: "Q1_0_g128",
}


@dataclass
class GGUFTensorInfo:
    name: str
    dims: tuple
    type_id: int
    type_name: str
    offset: int


@dataclass
class GGUFMetadata:
    version: int
    n_tensors: int
    metadata: dict
    tensors: list
    file_size: int
    file_path: str

    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    @property
    def name(self) -> str:
        return self.metadata.get("general.name", "unknown")

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(t.dims) for t in self.tensors)

    @property
    def quantization_types(self) -> dict:
        counts = {}
        for t in self.tensors:
            counts[t.type_name] = counts.get(t.type_name, 0) + 1
        return counts

    @property
    def file_size_gb(self) -> float:
        return self.file_size / (1000 ** 3)  # GB, not GiB (matching paper convention)


class GGUFParser:
    """Parses GGUF file headers without loading weight data."""

    _READERS = {
        GGUF_TYPE_UINT8:   ("<B", 1),
        GGUF_TYPE_INT8:    ("<b", 1),
        GGUF_TYPE_UINT16:  ("<H", 2),
        GGUF_TYPE_INT16:   ("<h", 2),
        GGUF_TYPE_UINT32:  ("<I", 4),
        GGUF_TYPE_INT32:   ("<i", 4),
        GGUF_TYPE_FLOAT32: ("<f", 4),
        GGUF_TYPE_UINT64:  ("<Q", 8),
        GGUF_TYPE_INT64:   ("<q", 8),
        GGUF_TYPE_FLOAT64: ("<d", 8),
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = None

    def parse(self) -> GGUFMetadata:
        file_size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            self._f = f
            magic = self._read_u32()
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Not a GGUF file: magic=0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
                )
            version = self._read_u32()
            n_tensors = self._read_u64()
            n_metadata = self._read_u64()

            metadata = {}
            for _ in range(n_metadata):
                key = self._read_string()
                type_id = self._read_u32()
                metadata[key] = self._read_typed_value(type_id)

            tensors = [self._read_tensor_info() for _ in range(n_tensors)]

        return GGUFMetadata(
            version=version,
            n_tensors=int(n_tensors),
            metadata=metadata,
            tensors=tensors,
            file_size=file_size,
            file_path=str(self.path),
        )

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._f.read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack("<Q", self._f.read(8))[0]

    def _read_string(self) -> str:
        length = self._read_u64()
        return self._f.read(length).decode("utf-8")

    def _read_typed_value(self, type_id: int):
        if type_id == GGUF_TYPE_BOOL:
            return struct.unpack("<B", self._f.read(1))[0] != 0
        if type_id == GGUF_TYPE_STRING:
            return self._read_string()
        if type_id == GGUF_TYPE_ARRAY:
            elem_type = self._read_u32()
            count = self._read_u64()
            return [self._read_typed_value(elem_type) for _ in range(count)]
        spec = self._READERS.get(type_id)
        if spec is None:
            raise ValueError(f"Unknown GGUF value type: {type_id}")
        fmt, size = spec
        return struct.unpack(fmt, self._f.read(size))[0]

    def _read_tensor_info(self) -> GGUFTensorInfo:
        name = self._read_string()
        n_dims = self._read_u32()
        dims = tuple(self._read_u64() for _ in range(n_dims))
        type_id = self._read_u32()
        offset = self._read_u64()
        return GGUFTensorInfo(
            name=name,
            dims=dims,
            type_id=type_id,
            type_name=GGML_TYPE_NAMES.get(type_id, f"unknown({type_id})"),
            offset=offset,
        )
