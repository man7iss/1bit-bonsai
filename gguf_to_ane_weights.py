"""Extract Q1_0_g128 weights from GGUF and convert to FP32 for ANE training pipeline.

The ANE dynamic pipeline expects FP32 weight files in a specific layout.
This script reads a Bonsai GGUF, dequantizes Q1_0_g128 tensors to FP32,
and writes a binary checkpoint that the ANE training code can load.

Weight layout per layer (matching ANE training's LayerWeights struct):
    Wq[Q_DIM * DIM] + Wk[KV_DIM * DIM] + Wv[KV_DIM * DIM] + Wo[DIM * Q_DIM]
    + W1[HIDDEN * DIM] + W2[DIM * HIDDEN] + W3[HIDDEN * DIM]
    + rms_att[DIM] + rms_ffn[DIM]
"""

import struct
import sys
from pathlib import Path

from gguf_parse import GGUFParser
from weight_convert import Q1WeightBlock, BITS_BYTES, GROUP_BYTES


def dequantize_q1_tensor(data: bytes) -> list[float]:
    """Dequantize Q1_0_g128 raw data to FP32 values."""
    n_groups = len(data) // GROUP_BYTES
    values = []
    for g in range(n_groups):
        offset = g * GROUP_BYTES
        bits = data[offset:offset + BITS_BYTES]
        scale = struct.unpack("<e", data[offset + BITS_BYTES:offset + GROUP_BYTES])[0]
        block = Q1WeightBlock(bits=bits, scale=scale)
        values.extend(block.to_float())
    return values


def extract_weights(gguf_path: str, output_path: str):
    """Extract and dequantize all weights from a Bonsai GGUF file."""
    meta = GGUFParser(gguf_path).parse()
    print(f"Model: {meta.name}, {meta.parameter_count:,} params")
    print(f"Architecture: {meta.architecture}")
    print(f"Tensors: {meta.n_tensors}")

    n_layers = meta.metadata.get(f"{meta.architecture}.block_count", 0)
    dim = meta.metadata.get(f"{meta.architecture}.embedding_length", 0)
    print(f"Layers: {n_layers}, DIM: {dim}")

    # Read raw tensor data
    path = Path(gguf_path)
    file_size = path.stat().st_size

    # Find data start offset (after header + metadata + tensor info)
    # Re-parse to get the file position after header
    with open(path, "rb") as f:
        # Skip to the end of the header to find alignment padding
        # GGUF aligns tensor data to 32-byte boundaries
        f.seek(0)
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_metadata = struct.unpack("<Q", f.read(8))[0]

        # Skip metadata
        for _ in range(n_metadata):
            key_len = struct.unpack("<Q", f.read(8))[0]
            f.read(key_len)  # key
            type_id = struct.unpack("<I", f.read(4))[0]
            _skip_value(f, type_id)

        # Read tensor info to get offsets
        tensor_infos = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            n_elements = 1
            for d in dims:
                n_elements *= d
            tensor_infos.append({
                "name": name, "dims": dims, "type_id": type_id,
                "offset": offset, "n_elements": n_elements,
            })

        # Data starts after alignment
        header_end = f.tell()
        alignment = 32
        data_start = (header_end + alignment - 1) // alignment * alignment

        print(f"Header end: {header_end}, Data start: {data_start}")
        print(f"Extracting {len(tensor_infos)} tensors...")

        # Extract each tensor
        tensors = {}
        for ti in tensor_infos:
            abs_offset = data_start + ti["offset"]
            f.seek(abs_offset)

            if ti["type_id"] == 41:  # Q1_0_g128
                n_groups = (ti["n_elements"] + 127) // 128
                raw = f.read(n_groups * GROUP_BYTES)
                values = dequantize_q1_tensor(raw)
                tensors[ti["name"]] = values[:ti["n_elements"]]
            elif ti["type_id"] == 0:  # F32
                raw = f.read(ti["n_elements"] * 4)
                values = list(struct.unpack(f"<{ti['n_elements']}f", raw))
                tensors[ti["name"]] = values

        print(f"Extracted {len(tensors)} tensors")

    # Write as flat binary (FP32)
    total_floats = sum(len(v) for v in tensors.values())
    print(f"Total values: {total_floats:,}")
    print(f"Output size: {total_floats * 4 / 1e9:.2f} GB")

    with open(output_path, "wb") as f:
        # Header: n_tensors, then (name_len, name, n_elements) for each
        f.write(struct.pack("<I", len(tensors)))
        for name, values in tensors.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", len(values)))

        # Data: all values in order
        for name, values in tensors.items():
            f.write(struct.pack(f"<{len(values)}f", *values))

    print(f"Written to {output_path}")


def _skip_value(f, type_id):
    """Skip a GGUF typed value in the file."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if type_id == 8:  # string
        slen = struct.unpack("<Q", f.read(8))[0]
        f.read(slen)
    elif type_id == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            _skip_value(f, elem_type)
    else:
        f.read(sizes.get(type_id, 4))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.bin>")
        sys.exit(1)
    extract_weights(sys.argv[1], sys.argv[2])
