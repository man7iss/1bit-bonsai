"""Convert Bonsai GGUF (Q1_0_g128) to FP16 safetensors for ANE-LM.

Dequantizes 1-bit weights to FP16, copies F32 tensors as-is,
writes a single safetensors file + config.json compatible with ANE-LM.
"""

import json
import struct
import sys
from pathlib import Path

from gguf_parse import GGUFParser
from weight_convert import Q1WeightBlock, BITS_BYTES, GROUP_BYTES

# GGUF tensor name -> HuggingFace safetensors name mapping for Qwen3
GGUF_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

def gguf_layer_to_hf(name: str) -> str:
    """Convert 'blk.N.attn_q.weight' -> 'model.layers.N.self_attn.q_proj.weight'."""
    if name in GGUF_TO_HF:
        return GGUF_TO_HF[name]
    parts = name.split(".")
    if parts[0] == "blk":
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        mapping = {
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        hf_rest = mapping.get(rest, rest)
        return f"model.layers.{layer_idx}.{hf_rest}"
    return name


def dequantize_q1_to_fp16(data: bytes, n_elements: int) -> bytes:
    """Dequantize Q1_0_g128 to IEEE FP16 bytes."""
    n_groups = len(data) // GROUP_BYTES
    fp16_values = bytearray()
    count = 0
    for g in range(n_groups):
        offset = g * GROUP_BYTES
        bits = data[offset:offset + BITS_BYTES]
        scale = struct.unpack("<e", data[offset + BITS_BYTES:offset + GROUP_BYTES])[0]
        block = Q1WeightBlock(bits=bits, scale=scale)
        for v in block.to_float():
            if count >= n_elements:
                break
            fp16_values.extend(struct.pack("<e", v))
            count += 1
    return bytes(fp16_values)


def convert(gguf_path: str, output_dir: str):
    meta = GGUFParser(gguf_path).parse()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arch = meta.architecture
    n_layers = meta.metadata.get(f"{arch}.block_count", 0)
    dim = meta.metadata.get(f"{arch}.embedding_length", 0)
    hidden = meta.metadata.get(f"{arch}.feed_forward_length", 0)
    heads = meta.metadata.get(f"{arch}.attention.head_count", 0)
    kv_heads = meta.metadata.get(f"{arch}.attention.head_count_kv", 0)
    head_dim = dim // heads if heads else 0
    vocab = meta.metadata.get(f"{arch}.vocab_size",
            meta.metadata.get("tokenizer.ggml.tokens", []))
    if isinstance(vocab, list):
        vocab = len(vocab)
    ctx = meta.metadata.get(f"{arch}.context_length", 0)
    rope_theta = meta.metadata.get(f"{arch}.rope.freq_base", 1000000.0)

    print(f"Model: {meta.name}")
    print(f"  dim={dim} hidden={hidden} heads={heads} kv_heads={kv_heads} hd={head_dim}")
    print(f"  layers={n_layers} vocab={vocab} ctx={ctx}")

    # Write config.json
    config = {
        "model_type": "qwen3",
        "hidden_size": dim,
        "intermediate_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "max_position_embeddings": ctx,
        "rope_theta": rope_theta,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float16",
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote config.json")

    # Parse GGUF and extract tensors
    path = Path(gguf_path)
    with open(path, "rb") as f:
        f.seek(0)
        struct.unpack("<I", f.read(4))  # magic
        struct.unpack("<I", f.read(4))  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_metadata = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_metadata):
            key_len = struct.unpack("<Q", f.read(8))[0]
            f.read(key_len)
            type_id = struct.unpack("<I", f.read(4))[0]
            _skip_value(f, type_id)

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

        header_end = f.tell()
        data_start = (header_end + 31) // 32 * 32

        # Build safetensors
        tensors_data = {}
        for ti in tensor_infos:
            abs_offset = data_start + ti["offset"]
            f.seek(abs_offset)
            hf_name = gguf_layer_to_hf(ti["name"])

            if ti["type_id"] == 41:  # Q1_0_g128
                n_groups = (ti["n_elements"] + 127) // 128
                raw = f.read(n_groups * GROUP_BYTES)
                fp16_bytes = dequantize_q1_to_fp16(raw, ti["n_elements"])
                tensors_data[hf_name] = {
                    "dtype": "F16",
                    "shape": list(reversed(ti["dims"])),  # GGUF is row-major, safetensors expects [rows, cols]
                    "data": fp16_bytes,
                }
            elif ti["type_id"] == 0:  # F32
                raw = f.read(ti["n_elements"] * 4)
                # Convert F32 to BF16 for ANE-LM compatibility
                fp32_vals = struct.unpack(f"<{ti['n_elements']}f", raw)
                fp16_bytes = b"".join(struct.pack("<e", v) for v in fp32_vals)
                shape = list(reversed(ti["dims"])) if len(ti["dims"]) > 1 else ti["dims"]
                tensors_data[hf_name] = {
                    "dtype": "F16",
                    "shape": shape,
                    "data": fp16_bytes,
                }

            print(f"  {ti['name']:40s} -> {hf_name:50s} [{ti['n_elements']:>12,}] type={ti['type_id']}")

    # Write safetensors file
    write_safetensors(out / "model.safetensors", tensors_data)
    total_bytes = sum(len(t["data"]) for t in tensors_data.values())
    print(f"\nWrote {len(tensors_data)} tensors, {total_bytes / 1e9:.2f} GB to {out / 'model.safetensors'}")


def write_safetensors(path: Path, tensors: dict):
    """Write a safetensors file. Format: 8-byte header size + JSON header + tensor data."""
    header = {}
    offset = 0
    for name, info in tensors.items():
        size = len(info["data"])
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name in tensors:
            f.write(tensors[name]["data"])


def _skip_value(f, type_id):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if type_id == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        f.read(slen)
    elif type_id == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            _skip_value(f, elem_type)
    else:
        f.read(sizes.get(type_id, 4))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
