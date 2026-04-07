"""Convert Q1_0_g128 weights to int8 ANE weight blob format.

Q1_0_g128 (Bonsai): 128 packed sign bits (16 bytes) + 1 FP16 scale (2 bytes) per group.
    w_i = scale * (2 * bit_i - 1), bit_i in {0,1} -> w_i in {-scale, +scale}
    Effective: 1.125 bits/weight.

ANE int8 (constexpr_affine_dequantize): int8 value (-1 or +1) per weight,
    per-group FP16 scale, zero_point=0.
    w = scale * int8_val
"""

import struct
from dataclasses import dataclass

GROUP_SIZE = 128
BITS_BYTES = GROUP_SIZE // 8
SCALE_BYTES = 2
GROUP_BYTES = BITS_BYTES + SCALE_BYTES  # 18


@dataclass
class Q1WeightBlock:
    """A single group of 128 Q1_0_g128 weights."""
    bits: bytes  # 16 bytes, packed little-endian
    scale: float

    def to_int8(self) -> list[int]:
        """bit=0 -> -1, bit=1 -> +1."""
        values = []
        for byte_val in self.bits:
            for bit_pos in range(8):
                values.append(1 if (byte_val >> bit_pos) & 1 else -1)
        return values

    def to_float(self) -> list[float]:
        return [self.scale * v for v in self.to_int8()]


def parse_q1_blocks(data: bytes) -> list[Q1WeightBlock]:
    """Parse raw Q1_0_g128 data into blocks of 18 bytes each."""
    if len(data) % GROUP_BYTES != 0:
        raise ValueError(
            f"Data length {len(data)} is not a multiple of {GROUP_BYTES}"
        )
    blocks = []
    for offset in range(0, len(data), GROUP_BYTES):
        bits = data[offset:offset + BITS_BYTES]
        scale = struct.unpack("<e", data[offset + BITS_BYTES:offset + GROUP_BYTES])[0]
        blocks.append(Q1WeightBlock(bits=bits, scale=scale))
    return blocks


def convert_q1_to_ane_int8(data: bytes) -> tuple[bytes, list[float]]:
    """Convert Q1_0_g128 data to (int8_weights, per_group_scales)."""
    blocks = parse_q1_blocks(data)
    int8_values = bytearray()
    scales = []
    for block in blocks:
        for v in block.to_int8():
            int8_values.append(v & 0xFF)
        scales.append(block.scale)
    return bytes(int8_values), scales


def validate_int8_reconstruction(
    original_data: bytes,
    int8_weights: bytes,
    scales: list[float],
    tolerance: float = 1e-3,
) -> bool:
    """Verify int8 weights + scales reconstruct original Q1 weights."""
    blocks = parse_q1_blocks(original_data)
    offset = 0
    for group_idx, block in enumerate(blocks):
        scale = scales[group_idx]
        for expected in block.to_float():
            int8_val = struct.unpack("b", int8_weights[offset:offset + 1])[0]
            if abs(scale * int8_val - expected) > tolerance:
                return False
            offset += 1
    return True


def compute_ane_blob_size(n_weights: int) -> dict:
    """Compute memory requirements for Q1_0_g128 vs ANE int8."""
    n_groups = (n_weights + GROUP_SIZE - 1) // GROUP_SIZE
    q1_size = n_groups * GROUP_BYTES
    ane_total = n_weights + n_groups * 2  # 1 byte/weight + FP16 scale/group

    return {
        "n_weights": n_weights,
        "n_groups": n_groups,
        "q1_size_bytes": q1_size,
        "q1_bits_per_weight": q1_size * 8 / n_weights if n_weights else 0,
        "ane_int8_size_bytes": n_weights,
        "ane_scale_size_bytes": n_groups * 2,
        "ane_total_bytes": ane_total,
        "ane_bits_per_weight": ane_total * 8 / n_weights if n_weights else 0,
        "expansion_ratio": ane_total / q1_size if q1_size else 0,
    }


def _float_to_fp16(value: float) -> bytes:
    return struct.pack("<e", value)
