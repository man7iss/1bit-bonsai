"""Tests for Q1_0_g128 to ANE int8 weight conversion."""

import struct

import pytest

from weight_convert import (
    GROUP_SIZE,
    GROUP_BYTES,
    BITS_BYTES,
    Q1WeightBlock,
    parse_q1_blocks,
    convert_q1_to_ane_int8,
    validate_int8_reconstruction,
    compute_ane_blob_size,
    _float_to_fp16,
)


def _make_q1_group(bits: list[int], scale: float) -> bytes:
    assert len(bits) == GROUP_SIZE
    packed = bytearray(BITS_BYTES)
    for i, b in enumerate(bits):
        if b:
            packed[i // 8] |= (1 << (i % 8))
    return bytes(packed) + _float_to_fp16(scale)


class TestFP16Conversion:
    def test_roundtrip(self):
        for val in [0.5, -1.25, 0.0]:
            decoded = struct.unpack("<e", _float_to_fp16(val))[0]
            assert abs(decoded - val) < 1e-4

    def test_size(self):
        assert len(_float_to_fp16(1.0)) == 2


class TestQ1WeightBlock:
    def test_all_zeros_to_int8(self):
        values = Q1WeightBlock(bits=b"\x00" * BITS_BYTES, scale=1.0).to_int8()
        assert len(values) == GROUP_SIZE
        assert all(v == -1 for v in values)

    def test_all_ones_to_int8(self):
        values = Q1WeightBlock(bits=b"\xff" * BITS_BYTES, scale=1.0).to_int8()
        assert all(v == 1 for v in values)

    def test_alternating_bits(self):
        # 0xAA = 10101010: odd bit positions set
        values = Q1WeightBlock(bits=b"\xaa" * BITS_BYTES, scale=1.0).to_int8()
        for i in range(GROUP_SIZE):
            expected = 1 if (i % 8) % 2 == 1 else -1
            assert values[i] == expected

    def test_to_float_with_scale(self):
        floats = Q1WeightBlock(bits=b"\x00" * BITS_BYTES, scale=2.5).to_float()
        assert all(abs(v - (-2.5)) < 1e-4 for v in floats)

    def test_to_float_mixed(self):
        bits = b"\x01" + b"\x00" * (BITS_BYTES - 1)
        floats = Q1WeightBlock(bits=bits, scale=3.0).to_float()
        assert abs(floats[0] - 3.0) < 1e-4
        assert abs(floats[1] - (-3.0)) < 1e-4


class TestParseQ1Blocks:
    def test_single_group(self):
        blocks = parse_q1_blocks(_make_q1_group([1] * 64 + [0] * 64, 1.5))
        assert len(blocks) == 1
        assert abs(blocks[0].scale - 1.5) < 0.01

    def test_two_groups(self):
        data = _make_q1_group([0] * GROUP_SIZE, 1.0) + _make_q1_group([1] * GROUP_SIZE, 2.0)
        blocks = parse_q1_blocks(data)
        assert len(blocks) == 2
        assert abs(blocks[0].scale - 1.0) < 0.01
        assert abs(blocks[1].scale - 2.0) < 0.01

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="not a multiple"):
            parse_q1_blocks(b"\x00" * 17)

    def test_group_bytes_constant(self):
        assert GROUP_BYTES == 18


class TestConvertQ1ToAneInt8:
    def test_output_length(self):
        int8_weights, scales = convert_q1_to_ane_int8(_make_q1_group([0] * GROUP_SIZE, 1.0))
        assert len(int8_weights) == GROUP_SIZE
        assert len(scales) == 1

    def test_all_zeros_produce_negative_ones(self):
        int8_weights, _ = convert_q1_to_ane_int8(_make_q1_group([0] * GROUP_SIZE, 1.0))
        for i in range(GROUP_SIZE):
            assert struct.unpack("b", int8_weights[i:i + 1])[0] == -1

    def test_all_ones_produce_positive_ones(self):
        int8_weights, _ = convert_q1_to_ane_int8(_make_q1_group([1] * GROUP_SIZE, 1.0))
        for i in range(GROUP_SIZE):
            assert struct.unpack("b", int8_weights[i:i + 1])[0] == 1

    def test_scales_extracted(self):
        data = _make_q1_group([0] * GROUP_SIZE, 0.5) + _make_q1_group([1] * GROUP_SIZE, 1.75)
        _, scales = convert_q1_to_ane_int8(data)
        assert abs(scales[0] - 0.5) < 0.01
        assert abs(scales[1] - 1.75) < 0.01

    def test_two_groups_output_length(self):
        int8_weights, scales = convert_q1_to_ane_int8(_make_q1_group([0] * GROUP_SIZE, 1.0) * 2)
        assert len(int8_weights) == GROUP_SIZE * 2
        assert len(scales) == 2


class TestValidateReconstruction:
    def test_perfect_reconstruction(self):
        data = _make_q1_group([1, 0] * 64, 1.5)
        int8_weights, scales = convert_q1_to_ane_int8(data)
        assert validate_int8_reconstruction(data, int8_weights, scales)

    def test_multiple_groups(self):
        data = _make_q1_group([0] * GROUP_SIZE, 0.25) + _make_q1_group([1] * GROUP_SIZE, 3.0)
        int8_weights, scales = convert_q1_to_ane_int8(data)
        assert validate_int8_reconstruction(data, int8_weights, scales)

    def test_corrupted_weights_fail(self):
        data = _make_q1_group([1] * GROUP_SIZE, 1.0)
        int8_weights, scales = convert_q1_to_ane_int8(data)
        corrupted = bytearray(int8_weights)
        corrupted[0] = 0
        assert not validate_int8_reconstruction(data, bytes(corrupted), scales)


class TestComputeAneBlobSize:
    def test_single_group(self):
        info = compute_ane_blob_size(128)
        assert info["n_groups"] == 1
        assert info["q1_size_bytes"] == 18
        assert info["ane_int8_size_bytes"] == 128
        assert info["ane_total_bytes"] == 130
        assert info["q1_bits_per_weight"] == 1.125

    def test_expansion_ratio(self):
        assert 7.0 < compute_ane_blob_size(128)["expansion_ratio"] < 7.5

    def test_8b_model_sizes(self):
        info = compute_ane_blob_size(8_190_000_000)
        q1_gb = info["q1_size_bytes"] / 1e9
        ane_gb = info["ane_total_bytes"] / 1e9
        assert 1.1 < q1_gb < 1.2
        assert 8.0 < ane_gb < 8.5
        assert abs(info["q1_bits_per_weight"] - 1.125) < 0.001
        assert abs(info["ane_bits_per_weight"] - 8.125) < 0.01

    def test_zero_weights(self):
        info = compute_ane_blob_size(0)
        assert info["q1_bits_per_weight"] == 0
        assert info["expansion_ratio"] == 0
