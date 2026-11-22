#!/usr/bin/env python3
"""Convert Thinking Machines Lab's Inkling 2 Mini mtp.safetensors (DSpark drafter) to a Q8_0 GGUF.

Input:  thinkingmachines/Inkling-2-Mini mtp.safetensors (607 MiB BF16, 54 tensors)
Output: Inkling-2-Mini-DSpark-Q8_0.gguf (~335 MiB, Q8_0 matmul weights + F32 norms/biases)

Pre-computes at conversion time:
  - dspark.fuse_alpha [4, 4] F32: row-wise float64 softmax of fuse_weight
  - dspark.markov.rnn_w_sp [768, 512] F32, rnn_w_h [768, 2304] Q8_0, rnn_w_f [768, 2304] Q8_0
    split from markov_head.rnn_proj.weight [768, 5120]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    hdr_len = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + hdr_len])
    body = memoryview(data)[8 + hdr_len :]
    out: dict[str, np.ndarray] = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "BF16", f"unexpected dtype {meta['dtype']} for {name}"
        s, e = meta["data_offsets"]
        u16 = np.frombuffer(body[s:e], dtype=np.uint16).reshape(meta["shape"])
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        out[name] = f32
    return out


def quantize_q8_0(arr: np.ndarray) -> bytes:
    """GGML Q8_0: 34 bytes per 32 elements = f16 scale (2B) + 32 x int8 (32B)."""
    assert arr.ndim == 2 and arr.shape[1] % 32 == 0, f"bad Q8_0 shape {arr.shape}"
    rows, cols = arr.shape
    blocks = arr.astype(np.float32, copy=False).reshape(rows * (cols // 32), 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = (amax / 127.0).astype(np.float32)
    scale_f16 = scale.astype(np.float16)
    d = scale_f16.astype(np.float32)
    inv_d = np.where(d > 0.0, 1.0 / d, 0.0).astype(np.float32)
    qs = np.clip(np.rint(blocks * inv_d[:, None]), -128, 127).astype(np.int8)
    # Interleave 2-byte f16 scale + 32-byte int8 codes per block.
    nblk = blocks.shape[0]
    packed = np.empty((nblk, 34), dtype=np.uint8)
    packed[:, :2] = scale_f16.view(np.uint8).reshape(nblk, 2)
    packed[:, 2:] = qs.view(np.uint8)
    return packed.tobytes()


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(out_path: Path, kv: list[tuple[str, int, bytes]], tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kv))
    for k, vtype, vbytes in kv:
        buf += pack_str(k)
        buf += struct.pack("<I", vtype)
        buf += vbytes

    # Tensor infos with offsets relative to start of aligned data section.
    offset = 0
    aligned_blobs: list[bytes] = []
    for name, shape, gtype, raw in tensors:
        # GGUF stores dimensions in reverse (ne0 = innermost / cols, ne1 = rows).
        dims = list(reversed(shape))
        buf += pack_str(name)
        buf += struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", gtype, offset)
        pad = (-len(raw)) % GGUF_ALIGNMENT
        aligned_blobs.append(raw + (b"\x00" * pad))
        offset += len(raw) + pad

    hdr_pad = (-len(buf)) % GGUF_ALIGNMENT
    buf += b"\x00" * hdr_pad

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(buf)
        for blob in aligned_blobs:
            f.write(blob)
    tmp_path.replace(out_path)


def convert(src_path: Path, out_path: Path) -> None:
    st = load_safetensors(src_path)
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def add_f32(name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        tensors.append((name, tuple(a.shape), GGML_TYPE_F32, a.tobytes()))

    def add_q8(name: str, arr: np.ndarray) -> None:
        raw = quantize_q8_0(arr)
        tensors.append((name, tuple(arr.shape), GGML_TYPE_Q8_0, raw))

    # Pre-softmax fuse_weight [4, 4] in float64 along axis 1.
    fw = st["fuse_weight"].astype(np.float64)
    fw = fw - np.max(fw, axis=1, keepdims=True)
    alpha = np.exp(fw)
    alpha /= np.sum(alpha, axis=1, keepdims=True)
    add_f32("dspark.fuse_alpha", alpha.astype(np.float32))

    add_f32("dspark.mask_embed", st["mask_embed"])
    add_f32("dspark.norm.weight", st["norm.weight"])
    add_f32("dspark.markov.dflare_norm.weight", st["markov_head.dflare_norm.weight"])

    # Markov head: w1 [201024, 256] Q8_0, rnn_proj [768, 5120] split into sp/h/f, out [2304, 256] F32.
    add_q8("dspark.markov.w1.weight", st["markov_head.markov_w1.weight"])
    rnn_w = st["markov_head.rnn_proj.weight"]
    assert rnn_w.shape == (768, 5120), rnn_w.shape
    add_f32("dspark.markov.rnn_w_sp.weight", rnn_w[:, :512])
    add_q8("dspark.markov.rnn_w_h.weight", rnn_w[:, 512:2816])
    add_q8("dspark.markov.rnn_w_f.weight", rnn_w[:, 2816:5120])
    add_f32("dspark.markov.rnn_b", st["markov_head.rnn_proj.bias"])
    add_f32("dspark.markov.out.weight", st["markov_head.hidden_out.weight"])
    add_f32("dspark.markov.out.bias", st["markov_head.hidden_out.bias"])

    # 3 dense draft layers.
    for l in range(3):
        p = f"layers.{l}."
        b = f"dspark.blk.{l}."
        add_f32(b + "dflare_norm.weight", st[p + "dflare_norm.weight"])
        add_f32(b + "attn_norm.weight", st[p + "input_layernorm.weight"])
        add_f32(b + "ffn_norm.weight", st[p + "post_attention_layernorm.weight"])
        add_f32(b + "attn_q_norm.weight", st[p + "self_attn.q_norm.weight"])
        add_f32(b + "attn_k_norm.weight", st[p + "self_attn.k_norm.weight"])
        add_f32(b + "attn_k_ctx_norm.weight", st[p + "self_attn.k_norm_ctx.weight"])
        add_q8(b + "attn_q.weight", st[p + "self_attn.q_proj.weight"])
        add_q8(b + "attn_k.weight", st[p + "self_attn.k_proj.weight"])
        add_q8(b + "attn_v.weight", st[p + "self_attn.v_proj.weight"])
        add_q8(b + "attn_k_ctx.weight", st[p + "self_attn.k_proj_ctx.weight"])
        add_q8(b + "attn_v_ctx.weight", st[p + "self_attn.v_proj_ctx.weight"])
        add_q8(b + "attn_output.weight", st[p + "self_attn.o_proj.weight"])
        add_q8(b + "ffn_gate.weight", st[p + "mlp.gate_proj.weight"])
        add_q8(b + "ffn_up.weight", st[p + "mlp.up_proj.weight"])
        add_q8(b + "ffn_down.weight", st[p + "mlp.down_proj.weight"])

    kv = [
        ("general.architecture", GGUF_TYPE_STRING, pack_str("inkling2-dspark")),
        ("general.name", GGUF_TYPE_STRING, pack_str("Inkling-2-Mini-DSpark-Q8_0")),
        ("dspark.block_count", GGUF_TYPE_UINT32, struct.pack("<I", 3)),
        ("dspark.block_size", GGUF_TYPE_UINT32, struct.pack("<I", 8)),
        ("dspark.markov_rank", GGUF_TYPE_UINT32, struct.pack("<I", 256)),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(out_path, kv, tensors)
    total_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({len(tensors)} tensors, {total_mb:.1f} MiB)")
    for name, shape, gtype, raw in tensors:
        tname = "F32 " if gtype == GGML_TYPE_F32 else "Q8_0"
        print(f"  {name:40s} {tname} {str(list(shape)):18s} {len(raw):10d} B")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="Path to mtp.safetensors")
    ap.add_argument("out", type=Path, help="Output GGUF path")
    args = ap.parse_args()
    convert(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
