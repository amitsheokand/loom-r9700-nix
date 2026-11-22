#!/usr/bin/env python3
"""Build an RDNA4-enabled (dual-target gfx11 + gfx12) Inkling 2 Mini Loom model package.

Upstream loom-fun/unsloth-Inkling-2-Mini-loom already ships _rdna4 twins for all MoE and
GEMM kernels, leaving only Inkling's 4 custom flash2 attention kernels on RDNA3 (gfx11)
WMMA layouts:
  - g_ink_attention_flash2_ts2
  - g_ink_attention_flash2_partial
  - g_ink_attention_flash2_swa
  - g_ink_attention_flash2_partial_swa

This script:
  1. Copies the pinned upstream Loom package files into --out-dir.
  2. Tags those 4 attention kernels with target(@rdna3), generates matching target(@rdna4)
     twins with RDNA4 (gfx12) 16x16 WMMA fragment layouts (vector<8xf16>, rows 8*(l/16)+i,
     relative-bias base %lane_h8_i32 and step 1..7, direct fptrunc without xor-16 shuffles
     or vector.interleave), rewrites their kernel.launch sites into template.apply, and
     reorders encoding-typed kernels forward to stay under hrx-loom 0.2.0's 16-bit SSA
     encoding value ID limit (65,536).
  3. Updates model.json with the configured OpenAI model id, R9700 preset, and symlinks
     parameter/<gguf-file> -> --gguf-path.
  4. When --loom-compile is provided, compile-verifies all 8 attention kernels (4x gfx1201
     + 4x gfx1151) to AMDGPU HSACO at Nix build time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGETS = [
    "g_ink_attention_flash2_ts2",
    "g_ink_attention_flash2_partial",
    "g_ink_attention_flash2_swa",
    "g_ink_attention_flash2_partial_swa",
]

STEP_CONSTS = {
    "c2_i32": ("2", "1"),
    "c4_i32": ("4", "2"),
    "c6_i32": ("6", "3"),
    "c8_i32": ("8", "4"),
    "c10_i32": ("10", "5"),
    "c12_i32": ("12", "6"),
    "c14_i32": ("14", "7"),
}

LAUNCH = re.compile(
    r"^(\s*)kernel\.launch @(\w+)\[([^\]]*)\]\((.*)\) : \[([^\]]*)\]\((.*)\)\s*$"
)
ENC_LIMIT = 60000
ENC_BYTES = 4_000_000
_ENC_LINE = re.compile(r"encoding<layout>|view<[^>]*, %")
_VALUE_DEFS = re.compile(
    r"%[\w.$]+(?::(\d+))?(?=(?:, %[\w.$]+(?::\d+)?)* = )|%[\w.$]+(?=: )"
)


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _enc_depth(text: str) -> int:
    c = worst = pos = 0
    for line in text.split("\n"):
        pos += len(line.encode()) + 1
        enc = _ENC_LINE.search(line)
        if enc and pos > ENC_BYTES:
            return ENC_LIMIT + 1
        if enc:
            worst = max(worst, c)
        for m in _VALUE_DEFS.finditer(line):
            c += int(m.group(1)) if m.group(1) else 1
    return worst


def encodings_first(text: str) -> str:
    """Keep encoding-typed kernels well below hrx-loom 0.2.0's 2^16 SSA value ID cap."""
    if _enc_depth(text) <= ENC_LIMIT:
        return text
    parts = re.split(r"(?m)^(?=kernel\.def |func\.def |template\.|command\.def )", text)
    head, rest = parts[0], parts[1:]
    first_late = next(
        (i for i, p in enumerate(rest) if p.startswith(("template.", "command.def"))),
        len(rest),
    )
    funcs = [i for i in range(first_late) if rest[i].startswith("func.def")]
    enc = [
        i
        for i, p in enumerate(rest)
        if p.startswith("kernel.def") and bool(_ENC_LINE.search(p))
    ]
    moved = set(funcs) | set(enc)
    out = head + "".join(
        [rest[i] for i in funcs + enc]
        + [p for i, p in enumerate(rest) if i not in moved]
    )
    depth = _enc_depth(out)
    if depth > ENC_LIMIT:
        raise ValueError(
            f"encoding-typed lines still lie {depth} values deep after reordering (limit {ENC_LIMIT})"
        )
    return out


def port_kernel_to_rdna4(kname: str, text: str) -> str:
    # 0. Sanity-check that %c2_i32..%c14_i32 are ONLY used for rel-bias distance steps.
    for cname, (old_v, _new_v) in STEP_CONSTS.items():
        uses = [l.strip() for l in text.splitlines() if f"%{cname}" in l]
        assert uses, f"{kname}: missing %{cname}"
        assert (
            uses[0] == f"%{cname} = scalar.constant {old_v} : i32"
        ), f"{kname}: unexpected def {uses[0]}"
        for u in uses[1:]:
            assert "scalar.subi %d0_" in u, f"{kname}: unexpected use of %{cname}: {u}"

    # 1. Rename kernel def + tag target(@rdna4).
    text, c = re.subn(
        rf'^kernel\.def export\("{kname}"\) @{kname}\(',
        f'kernel.def target(@rdna4) export("{kname}_rdna4") @{kname}_rdna4(',
        text,
        count=1,
        flags=re.M,
    )
    assert c == 1, f"{kname}: failed to rename header"

    # 2. Rel-bias step constants: RDNA3 rows 2i+half (step 2) -> RDNA4 rows 8*half+i (step 1).
    for cname, (old_v, new_v) in STEP_CONSTS.items():
        text, c = re.subn(
            rf"^(  %{cname} = scalar\.constant ){old_v}( : i32)$",
            rf"\g<1>{new_v}\2",
            text,
            flags=re.M,
        )
        assert c == 1, f"{kname}: failed step const {cname}"

    # 3. Lane half * 8 constants right after %lane_half.
    text, c = re.subn(
        r"^(  %lane_half = index\.div %lane, %c16 : index\n)",
        r"\1  %r4_c8 = index.constant 8 : index\n"
        r"  %lane_h8_v = index.mul %lane_half, %r4_c8 : index\n"
        r"  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index\n"
        r"  %lane_h8_i32 = index.cast %lane_h8 : index to i32\n",
        text,
        flags=re.M,
    )
    assert c == 1, f"{kname}: failed %lane_half hook"

    # 4. Rel-bias base offset: %dbase* - %half32 -> %dbase* - %lane_h8_i32.
    text, c_rb = re.subn(
        r"(scalar\.subi %dbase\w+, )%half32( : i32)",
        r"\1%lane_h8_i32\2",
        text,
    )
    assert c_rb >= 2, f"{kname}: expected >=2 rel-bias %half32 replacements, got {c_rb}"

    # 5. Q^T rhs load: offset %c0 -> %lane_h8 (8 f32 per lane).
    text, c_q = re.subn(
        r"(%q_view\[%q_token, %head, %qc\d+, )%c0\]",
        r"\1%lane_h8]",
        text,
    )
    assert c_q == 8, f"{kname}: expected 8 Q loads, got {c_q}"

    # 6. %row_off: 8 * half + i (step 1).
    text, c_ro = re.subn(
        r"^  %row_off = vector\.iota %half32 step %two_i32 : vector<8xi32>\n",
        "  %one_row_i32 = scalar.constant 1 : i32\n"
        "  %row_off = vector.iota %lane_h8_i32 step %one_row_i32 : vector<8xi32>\n",
        text,
        flags=re.M,
    )
    assert c_ro == 1, f"{kname}: failed %row_off"

    # 7. P^T: S^T result layout already matches P^T rhs on RDNA4; remove xor-16 & interleave.
    text, c_il = re.subn(
        r"^ *%pi(\d+)_0((?:_pt)?) = vector\.extract .*?^ *%pt\1_f\2 = vector\.interleave.*?\n",
        "",
        text,
        flags=re.M | re.S,
    )
    assert c_il in (2, 4), f"{kname}: expected 2 or 4 interleave blocks, got {c_il}"
    text, c_fp = re.subn(
        r"(%pt(\d+)_h((?:_pt)?) = vector\.fptrunc )%pt\2_f\3 : vector<16xf32>",
        r"\1%p\2\3 : vector<8xf32>",
        text,
    )
    assert c_fp == c_il, f"{kname}: fptrunc count {c_fp} != interleave count {c_il}"
    assert "interleave" not in text, f"{kname}: leftover interleave"

    # 8. Masked V^T path: %iota16 starts at %lane_h8_i32 (vector<8xi32>).
    text, c_io = re.subn(
        r"%iota16 = vector\.iota %c0_i32 step %c1_i32m : vector<16xi32>",
        "%iota16 = vector.iota %lane_h8_i32 step %c1_i32m : vector<8xi32>",
        text,
    )
    assert c_io == 1, f"{kname}: failed %iota16"

    # 9. WMMA operand types: vector<16x...> -> vector<8x...> (leave K/V staging alone).
    operand = re.compile(
        r"vector\.fragment|vector\.mma|%zero16h|%iota16|%thr_v|%lo_v|%mhi|%mlo|"
        r"%m\d+_\d+(?:_pt)? = vector\.andi|%vs\d|%qraw|%qh\d+ = vector\.fptrunc|%pt\d+_h(?:_pt)? = "
    )
    text = "\n".join(
        re.sub(r"vector<16x(f16|f32|i32|i1)>", r"vector<8x\1>", ln)
        if operand.search(ln)
        else ln
        for ln in text.split("\n")
    )

    # 10. Partial tile epilogue: row = 16*d + 8*half + i.
    if "partial" in kname:
        text, c_cb = re.subn(
            r"(%col_base = index\.add %\w+, )%lane_half : index",
            r"\1%lane_h8 : index",
            text,
        )
        assert c_cb == 1, f"{kname}: failed %col_base"
        text, c_dc = re.subn(
            r"(%dc(\d+)_(\d+) = index\.constant )(\d+)( : index)",
            lambda m: f"{m.group(1)}{int(m.group(2)) * 16 + int(m.group(3))}{m.group(5)}",
            text,
        )
        assert c_dc == 64, f"{kname}: expected 64 %dc constants, got {c_dc}"

    return text


def port_llm_loom(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    found: set[str] = set()
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^kernel\.def export\("(\w+)"\) @(\w+)\(', line)
        if m and m.group(1) in TARGETS:
            kname = m.group(1)
            found.add(kname)
            j = i + 1
            while lines[j] != "}":
                j += 1
            orig_block = "\n".join(lines[i : j + 1])
            rdna3_block = orig_block.replace(
                "kernel.def ", "kernel.def target(@rdna3) ", 1
            )
            rdna4_block = port_kernel_to_rdna4(kname, orig_block)
            out.append(rdna3_block)
            out.append("")
            out.append(rdna4_block)
            i = j + 1
            continue
        out.append(line)
        i += 1
    assert found == set(TARGETS), f"Missing targets: {set(TARGETS) - found}"

    new_lines = "\n".join(out).splitlines()
    decls: list[str] = []
    sigs: dict[tuple[str, tuple[str, ...], tuple[str, ...]], str] = {}
    res: list[str] = []
    for line in new_lines:
        lm = LAUNCH.match(line)
        if lm and lm.group(2) in found:
            ind, k, grid, args, gty, aty = lm.groups()
            g, a = _split_top(grid), _split_top(args)
            gt, at = _split_top(gty), _split_top(aty)
            key = (k, tuple(gt), tuple(at))
            if key not in sigs:
                tname = f"sel_r4_{len(sigs) + 1}_{k}"
                sigs[key] = tname
                ps = [f"%g{idx}: {t}" for idx, t in enumerate(gt)] + [
                    f"%a{idx}: {t}" for idx, t in enumerate(at)
                ]
                gv = ", ".join(f"%g{idx}" for idx in range(len(gt)))
                av = ", ".join(f"%a{idx}" for idx in range(len(at)))
                decls.append(f"template.decl @{tname}({', '.join(ps)})")
                for tgt, kk in (("rdna3", k), ("rdna4", k + "_rdna4")):
                    decls.append(
                        f"template.def<@{tname}> target(@{tgt}) @{tname}_{tgt}({', '.join(ps)}) {{"
                    )
                    decls.append(
                        f"  kernel.launch @{kk}[{gv}]({av}) : [{', '.join(gt)}]({', '.join(at)})"
                    )
                    decls.append("  template.return\n}")
            res.append(
                f"{ind}template.apply<@{sigs[key]}>({', '.join(g + a)}) : ({', '.join(gt + at)})"
            )
        else:
            res.append(line)

    ins_at = next(
        idx
        for idx, l in enumerate(res)
        if l.startswith(("template.decl ", "command.def "))
    )
    res = res[:ins_at] + decls + [""] + res[ins_at:]
    final_text = "\n".join(res) + "\n"

    depth_before = _enc_depth(final_text)
    final_text = encodings_first(final_text)
    depth_after = _enc_depth(final_text)
    print(
        f"port-inkling-rdna4: rewrote {len(sigs)} launch templates; "
        f"enc_depth {depth_before} -> {depth_after} (limit {ENC_LIMIT})"
    )
    return final_text


_HEAD_LDS_ANCHOR = """  %c256o = index.constant 256 : offset
  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>"""

_HEAD_LDS_NEW = """  %c256o = index.constant 256 : offset
  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>
  %cmax_bytes = index.constant 6256 : offset
  %cmax_buf = buffer.alloca<workgroup> align(16) %cmax_bytes : buffer
  %cmax_v = buffer.view %cmax_buf[%zero] : buffer -> view<1564xf32>
  %surv_buf = buffer.alloca<workgroup> align(16) %cmax_bytes : buffer
  %surv_v = buffer.view %surv_buf[%zero] : buffer -> view<1564xi32>
  %scnt_bytes = index.constant 16 : offset
  %scnt_buf = buffer.alloca<workgroup> align(16) %scnt_bytes : buffer
  %scnt_v = buffer.view %scnt_buf[%zero] : buffer -> view<1xi32>"""

_HEAD_TOPK_OLD = """        %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x200064xf32>
        // k rounds: the largest logit strictly after the previous pick in (value desc, index asc) order.
        %last_v, %last_i = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pi = %c0_i32 : i32) -> (f32, i32) {
          %minus_one = scalar.constant -1 : i32
          %prev_i = scalar.addi %pi, %c0_i32 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_i = scf.select %first_round, %minus_one, %prev_i : i32
          %lv, %li = scf.for %v = [%lane to %vocab step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %bounded_v = index.assume %v [range(%v, 0, 200057)] : index
            %x = view.load %logit_view[%span, %bounded_v] : view<[%spans]x200064xf32> -> f32
            %v32 = index.cast %bounded_v : index to i32
            %below = scalar.cmpf olt, %x, %pv : f32
            %tie = scalar.cmpf oeq, %x, %pv : f32
            %after = scalar.cmpi sgt, %v32, %floor_i : i32
            %tie_after = scalar.andi %tie, %after : i1
            %eligible = scalar.ori %below, %tie_after : i1
            %greater = scalar.cmpf ogt, %x, %bv : f32
            %take = scalar.andi %eligible, %greater : i1
            %nv = scf.select %take, %x, %bv : f32
            %ni = scf.select %take, %v32, %bi : i32
            scf.yield %nv, %ni : f32, i32
          }"""

_HEAD_TOPK_NEW = """        %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x200064xf32>
        // Exact chunk-max pruning using %partials (1563 chunks of 128 logits already computed by head_mv).
        // Stage 1: cache chunk maxima in LDS and find the k-th largest chunk max (threshold T).
        %p_view_s = buffer.view %partials[%zero] : buffer -> view<[%spans]x1563x2xi32>
        scf.for %cc = [%lane to %chunks step %c256] {
          %cc_b = index.assume %cc [range(%cc, 0, 1562)] : index
          %cbits = view.load %p_view_s[%span, %cc_b, %c0] : view<[%spans]x1563x2xi32> -> i32
          %cval = scalar.bitcast %cbits : i32 to f32
          view.store %cval, %cmax_v[%cc_b] : f32, view<1564xf32>
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        %thresh, %thresh_ci = scf.for %cr = [%c0 to %k step %c1](%cpv = %pos_inf : f32, %cpi = %c0_i32 : i32) -> (f32, i32) {
          %cm1 = scalar.constant -1 : i32
          %cfirst = index.cmp eq, %cr, %c0 : index
          %cfloor = scf.select %cfirst, %cm1, %cpi : i32
          %clv, %cli = scf.for %cc = [%lane to %chunks step %c256](%cbv = %neg_inf : f32, %cbi = %big : i32) -> (f32, i32) {
            %cc_b = index.assume %cc [range(%cc, 0, 1562)] : index
            %cx = view.load %cmax_v[%cc_b] : view<1564xf32> -> f32
            %cc32 = index.cast %cc_b : index to i32
            %cbel = scalar.cmpf olt, %cx, %cpv : f32
            %ctie = scalar.cmpf oeq, %cx, %cpv : f32
            %caft = scalar.cmpi sgt, %cc32, %cfloor : i32
            %cta = scalar.andi %ctie, %caft : i1
            %cel = scalar.ori %cbel, %cta : i1
            %cgt = scalar.cmpf ogt, %cx, %cbv : f32
            %ctk = scalar.andi %cel, %cgt : i1
            %cnv = scf.select %ctk, %cx, %cbv : f32
            %cni = scf.select %ctk, %cc32, %cbi : i32
            scf.yield %cnv, %cni : f32, i32
          }
          %cbest = kernel.workgroup.reduce<maxnumf> %clv : f32
          %cisb = scalar.cmpf oeq, %clv, %cbest : f32
          %ccand = scf.select %cisb, %cli, %big : i32
          %cwin = kernel.workgroup.reduce<minsi> %ccand : i32
          scf.yield %cbest, %cwin : f32, i32
        }
        // Stage 2: collect all chunks with chunk_max >= thresh (normally <= k chunks, bit-exact even under ties).
        scf.if %first {
          %ns = scf.for %cc = [%c0 to %chunks step %c1](%cnt = %c0 : index) -> (index) {
            %cc_b = index.assume %cc [range(%cc, 0, 1562)] : index
            %cx = view.load %cmax_v[%cc_b] : view<1564xf32> -> f32
            %keep = scalar.cmpf oge, %cx, %thresh : f32
            %nxt = scf.if %keep -> (index) {
              %cnt_b = index.assume %cnt [range(%cnt, 0, 1562)] : index
              %cc32 = index.cast %cc_b : index to i32
              view.store %cc32, %surv_v[%cnt_b] : i32, view<1564xi32>
              %cnt1 = index.add %cnt, %c1 : index
              scf.yield %cnt1 : index
            } else {
              scf.yield %cnt : index
            }
            scf.yield %nxt : index
          }
          %ns32 = index.cast %ns : index to i32
          view.store %ns32, %scnt_v[%c0] : i32, view<1xi32>
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        %nsurv32 = view.load %scnt_v[%c0] : view<1xi32> -> i32
        %nsurv_b = scalar.assume %nsurv32 [range(%nsurv32, 1, 1563)] : i32
        %nsurv = index.cast %nsurv_b : i32 to index
        %c2_s = index.constant 2 : index
        %c128_s = index.constant 128 : index
        %half_s = index.div %lane, %c128_s : index
        %off_s = index.rem %lane, %c128_s : index
        // Stage 3: k rounds over ONLY the surviving chunks (coalesced 2x128 lanes/iter, L1-hot).
        %last_v, %last_i = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pi = %c0_i32 : i32) -> (f32, i32) {
          %minus_one = scalar.constant -1 : i32
          %prev_i = scalar.addi %pi, %c0_i32 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_i = scf.select %first_round, %minus_one, %prev_i : i32
          %lv, %li = scf.for %ci = [%c0 to %nsurv step %c2_s](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %cidx = index.add %ci, %half_s : index
            %in_s = index.cmp ult, %cidx, %nsurv : index
            %nv, %ni = scf.if %in_s -> (f32, i32) {
              %cidx_b = index.assume %cidx [range(%cidx, 0, 1562)] : index
              %chk32 = view.load %surv_v[%cidx_b] : view<1564xi32> -> i32
              %chk_b = scalar.assume %chk32 [range(%chk32, 0, 1562)] : i32
              %chk = index.cast %chk_b : i32 to index
              %vbase = index.mul %chk, %c128_s : index
              %v = index.add %vbase, %off_s : index
              %in_v = index.cmp ult, %v, %vocab : index
              %rv, %ri = scf.if %in_v -> (f32, i32) {
                %bounded_v = index.assume %v [range(%v, 0, 200057)] : index
                %x = view.load %logit_view[%span, %bounded_v] : view<[%spans]x200064xf32> -> f32
                %v32 = index.cast %bounded_v : index to i32
                %below = scalar.cmpf olt, %x, %pv : f32
                %tie = scalar.cmpf oeq, %x, %pv : f32
                %after = scalar.cmpi sgt, %v32, %floor_i : i32
                %tie_after = scalar.andi %tie, %after : i1
                %eligible = scalar.ori %below, %tie_after : i1
                %greater = scalar.cmpf ogt, %x, %bv : f32
                %take = scalar.andi %eligible, %greater : i1
                %nv2 = scf.select %take, %x, %bv : f32
                %ni2 = scf.select %take, %v32, %bi : i32
                scf.yield %nv2, %ni2 : f32, i32
              } else {
                scf.yield %bv, %bi : f32, i32
              }
              scf.yield %rv, %ri : f32, i32
            } else {
              scf.yield %bv, %bi : f32, i32
            }
            scf.yield %nv, %ni : f32, i32
          }"""


def patch_head_select(text: str) -> str:
    """Replace g_head_select's 64x full-vocab scan (4.83 ms/tok on 1 CU) with bit-exact chunk-max pruning.

    Every head_mv kernel already writes the exact max of each 128-token chunk into %partials[span, c, 0].
    Any chunk whose max is strictly below the k-th largest chunk max cannot contain any global top-k
    logit, cutting scanned logits from 200,058 to <= k*128 (24.4x less work, L1-hot, 100% bit-exact).
    """
    assert text.count(_HEAD_LDS_ANCHOR) == 1, f"g_head_select LDS anchor count={text.count(_HEAD_LDS_ANCHOR)}"
    assert text.count(_HEAD_TOPK_OLD) == 1, f"g_head_select top-k block count={text.count(_HEAD_TOPK_OLD)}"
    text = text.replace(_HEAD_LDS_ANCHOR, _HEAD_LDS_NEW, 1)
    text = text.replace(_HEAD_TOPK_OLD, _HEAD_TOPK_NEW, 1)
    return text


def patch_control_defaults(text: str, temperature: float, top_p: float) -> str:
    """Set Inkling 2 Mini's default sampling in @request_defaults when clients omit temperature.

    Upstream control.loom leaves @request_defaults at temperature=0.0 (greedy), which causes
    deterministic tool-call repetition loops in agent harnesses that omit temperature. Explicit request temperature/top_p (including 0.0) still overrides these.
    """
    pat = re.compile(
        r"(func\.def public @request_defaults\(%options: buffer\) -> \(buffer\) \{\n.*?"
        r"  %temperature = scalar\.constant )0\.0( : f32\n"
        r"  %top_p = scalar\.constant )1\.0( : f32\n)",
        flags=re.S,
    )
    out, count = pat.subn(rf"\g<1>{temperature!r}\g<2>{top_p!r}\g<3>", text)
    assert count == 1, f"control.loom: expected 1 @request_defaults match, got {count}"
    return out


_FP8_KV_HELPERS = """
// IEEE FP8 E4M3FN KV cache helpers (unscaled: Inkling K is L2-clipped <= 22.63, V is RMS-normed = 1.0).
// Cuts full-attention KV cache from 32 KiB/tok (F16) to 16 KiB/tok (FP8 E4M3FN), freeing 4 GiB at 256K.
func.def inline @g_e4m3fn_to_f16_v4(%b: vector<4xi8>) -> (vector<4xf16>) {
  %u = vector.extui %b : vector<4xi8> to vector<4xi16>
  %m80 = vector.splati 128 : vector<4xi16>
  %m7f = vector.splati 127 : vector<4xi16>
  %c8 = vector.splati 8 : vector<4xi16>
  %c7 = vector.splati 7 : vector<4xi16>
  %c64 = vector.splati 64 : vector<4xi16>
  %sign0 = vector.andi %u, %m80 : vector<4xi16>
  %sign = vector.shli %sign0, %c8 : vector<4xi16>
  %mag = vector.andi %u, %m7f : vector<4xi16>
  %mag_p = vector.addi %mag, %c64 : vector<4xi16>
  %norm0 = vector.shli %mag_p, %c7 : vector<4xi16>
  %norm = vector.ori %norm0, %sign : vector<4xi16>
  %sub_f0 = vector.uitofp %mag : vector<4xi16> to vector<4xf16>
  %sub_sc = vector.splatf 0.001953125 : vector<4xf16>
  %sub_f = vector.mulf %sub_f0, %sub_sc : vector<4xf16>
  %sub0 = vector.bitcast %sub_f : vector<4xf16> to vector<4xi16>
  %sub = vector.ori %sub0, %sign : vector<4xi16>
  %is_sub = vector.cmpi ult, %mag, %c8 : vector<4xi16> -> vector<4xi1>
  %bits = vector.select %is_sub, %sub, %norm : vector<4xi16>
  %out = vector.bitcast %bits : vector<4xi16> to vector<4xf16>
  func.return %out : vector<4xf16>
}

func.def inline @g_e4m3fn_to_f16_v16(%b: vector<16xi8>) -> (vector<16xf16>) {
  %u = vector.extui %b : vector<16xi8> to vector<16xi16>
  %m80 = vector.splati 128 : vector<16xi16>
  %m7f = vector.splati 127 : vector<16xi16>
  %c8 = vector.splati 8 : vector<16xi16>
  %c7 = vector.splati 7 : vector<16xi16>
  %c64 = vector.splati 64 : vector<16xi16>
  %sign0 = vector.andi %u, %m80 : vector<16xi16>
  %sign = vector.shli %sign0, %c8 : vector<16xi16>
  %mag = vector.andi %u, %m7f : vector<16xi16>
  %mag_p = vector.addi %mag, %c64 : vector<16xi16>
  %norm0 = vector.shli %mag_p, %c7 : vector<16xi16>
  %norm = vector.ori %norm0, %sign : vector<16xi16>
  %sub_f0 = vector.uitofp %mag : vector<16xi16> to vector<16xf16>
  %sub_sc = vector.splatf 0.001953125 : vector<16xf16>
  %sub_f = vector.mulf %sub_f0, %sub_sc : vector<16xf16>
  %sub0 = vector.bitcast %sub_f : vector<16xf16> to vector<16xi16>
  %sub = vector.ori %sub0, %sign : vector<16xi16>
  %is_sub = vector.cmpi ult, %mag, %c8 : vector<16xi16> -> vector<16xi1>
  %bits = vector.select %is_sub, %sub, %norm : vector<16xi16>
  %out = vector.bitcast %bits : vector<16xi16> to vector<16xf16>
  func.return %out : vector<16xf16>
}

func.def inline @g_f32_to_e4m3fn_v4(%x: vector<4xf32>) -> (vector<4xi8>) {
  %lo = vector.splatf -448.0 : vector<4xf32>
  %hi = vector.splatf 448.0 : vector<4xf32>
  %x1 = vector.maxnumf %x, %lo : vector<4xf32>
  %xc = vector.minnumf %x1, %hi : vector<4xf32>
  %h16 = vector.fptrunc %xc : vector<4xf32> to vector<4xf16>
  %h = vector.bitcast %h16 : vector<4xf16> to vector<4xi16>
  %c8 = vector.splati 8 : vector<4xi16>
  %m80 = vector.splati 128 : vector<4xi16>
  %m7fff = vector.splati 32767 : vector<4xi16>
  %c2400 = vector.splati 9216 : vector<4xi16>
  %c2000 = vector.splati 8192 : vector<4xi16>
  %c7 = vector.splati 7 : vector<4xi16>
  %c1 = vector.splati 1 : vector<4xi16>
  %c63 = vector.splati 63 : vector<4xi16>
  %c7e = vector.splati 126 : vector<4xi16>
  %sign0 = vector.shrui %h, %c8 : vector<4xi16>
  %sign8 = vector.andi %sign0, %m80 : vector<4xi16>
  %abs16 = vector.andi %h, %m7fff : vector<4xi16>
  %sh = vector.subi %abs16, %c2000 : vector<4xi16>
  %lsb0 = vector.shrui %sh, %c7 : vector<4xi16>
  %lsb = vector.andi %lsb0, %c1 : vector<4xi16>
  %r0 = vector.addi %sh, %c63 : vector<4xi16>
  %r1 = vector.addi %r0, %lsb : vector<4xi16>
  %nmag0 = vector.shrui %r1, %c7 : vector<4xi16>
  %nmag_gt = vector.cmpi ugt, %nmag0, %c7e : vector<4xi16> -> vector<4xi1>
  %nmag = vector.select %nmag_gt, %c7e, %nmag0 : vector<4xi16>
  %ax = vector.absf %xc : vector<4xf32>
  %sc512 = vector.splatf 512.0 : vector<4xf32>
  %scaled = vector.mulf %ax, %sc512 : vector<4xf32>
  %rnd = vector.roundf %scaled : vector<4xf32>
  %smag32 = vector.fptosi %rnd : vector<4xf32> to vector<4xi32>
  %smag0 = vector.trunci %smag32 : vector<4xi32> to vector<4xi16>
  %smag_gt = vector.cmpi ugt, %smag0, %c8 : vector<4xi16> -> vector<4xi1>
  %smag = vector.select %smag_gt, %c8, %smag0 : vector<4xi16>
  %is_sub = vector.cmpi ult, %abs16, %c2400 : vector<4xi16> -> vector<4xi1>
  %mag8 = vector.select %is_sub, %smag, %nmag : vector<4xi16>
  %u8_16 = vector.ori %sign8, %mag8 : vector<4xi16>
  %out = vector.trunci %u8_16 : vector<4xi16> to vector<4xi8>
  func.return %out : vector<4xi8>
}
"""

_FP8_WRITERS = {
    "g_ink_sconv1_dyn_kv_prep_store",
    "g_ink_kv_prep_store",
    "g_ink_kv_prep_store_lv",
}
_FP8_SPLIT_READERS = {
    "g_ink_attn_split": 2,
    "g_ink_attn_direct": 1,
    "g16_ink_attn_split_listed": 1,
}
_FP8_FLASH2_READERS = {
    "g_ink_attention_flash2_ts2": 2,
    "g_ink_attention_flash2_ts2_rdna4": 2,
    "g_ink_attention_flash2_partial": 6,
    "g_ink_attention_flash2_partial_rdna4": 6,
}


def patch_storage_fp8_kv(text: str) -> str:
    """Halve full-attention paged KV slot & block bytes in storage.loom (F16 32 KiB/tok -> FP8 16 KiB/tok)."""
    old_pos = "  %kv_position_bytes = scalar.constant 2048 : i64\n"
    new_pos = "  %kv_position_bytes = scalar.constant 1024 : i64\n"
    old_blk = "  %block_bytes = scalar.constant 32768 : i64\n"
    new_blk = "  %block_bytes = scalar.constant 16384 : i64\n"
    assert text.count(old_pos) == 1, f"storage.loom: kv_position_bytes count={text.count(old_pos)}"
    assert text.count(old_blk) == 1, f"storage.loom: block_bytes count={text.count(old_blk)}"
    return text.replace(old_pos, new_pos, 1).replace(old_blk, new_blk, 1)


def patch_llm_fp8_kv(text: str) -> str:
    """Convert the 16 full-attention paged KV planes in llm.loom from F16 to unscaled FP8 E4M3FN.

    Only touches @g_kv_offset and the 10 full-attention KV writer/reader kernels; SWA ring buffers
    stay F16. Runs encodings_first() afterwards to preserve the SSA value ID limit.
    """
    parts = re.split(r"(?m)^(?=kernel\.def |func\.def |template\.|command\.def )", text)
    sym_re = re.compile(r"^(?:kernel|func)\.def\b[^\n]*@([A-Za-z0-9_]+)\(")
    write_re = re.compile(
        r"^(?P<ind>[ \t]*)%(?P<v16>[kv]16) = vector\.fptrunc %(?P<src>\w+) : vector<4xf32> to vector<4xf16>\n"
        r"(?P=ind)%(?P<buf>[kv]c) = buffer\.view %arena\[%(?P<off>[kv]o)\] : buffer -> view<8x32x4xf16>\n"
        r"(?P=ind)vector\.store %(?P=v16), %(?P=buf)\[%kh, %lane, %c0\] : vector<4xf16>, view<8x32x4xf16>\n",
        flags=re.M,
    )
    read4_re = re.compile(
        r"^(?P<ind>[ \t]*)%(?P<buf>[kv]c) = buffer\.view %arena\[%(?P<off>[kv]o)\] : buffer -> view<8x32x4xf16>\n"
        r"(?P=ind)%(?P<dst>[kv]h16) = vector\.load %(?P=buf)\[%kh, %lane, %c0\] : view<8x32x4xf16> -> vector<4xf16>\n",
        flags=re.M,
    )
    read16_re = re.compile(
        r"^(?P<ind>[ \t]*)%k_cache_(?P<tag>\w+) = buffer\.view %arena\[%k_off_(?P=tag)\] : buffer -> view<8x128xf16>\n"
        r"(?P=ind)%v_cache_(?P=tag) = buffer\.view %arena\[%v_off_(?P=tag)\] : buffer -> view<8x128xf16>\n"
        r"(?P=ind)%kh_(?P=tag) = vector\.load %k_cache_(?P=tag)\[%kv, %st_dim_(?P<dim>\w+)\] : view<8x128xf16> -> vector<16xf16>\n"
        r"(?P=ind)%vh_(?P=tag) = vector\.load %v_cache_(?P=tag)\[%kv, %st_dim_(?P=dim)\] : view<8x128xf16> -> vector<16xf16>\n",
        flags=re.M,
    )

    seen: set[str] = set()
    for i in range(1, len(parts)):
        m = sym_re.match(parts[i])
        if not m:
            continue
        name = m.group(1)
        blk = parts[i]
        if name == "g_kv_offset":
            assert blk.count("%position_bytes = index.constant 2048 : offset") == 1
            blk = blk.replace(
                "%position_bytes = index.constant 2048 : offset",
                "%position_bytes = index.constant 1024 : offset",
                1,
            )
            blk = blk + _FP8_KV_HELPERS
            seen.add(name)
        elif name in _FP8_WRITERS:
            assert blk.count("%pos_bytes = index.constant 2048 : offset") == 1, name
            blk = blk.replace(
                "%pos_bytes = index.constant 2048 : offset",
                "%pos_bytes = index.constant 1024 : offset",
                1,
            )
            blk, n = write_re.subn(
                r"\g<ind>%\g<v16>_fp8 = func.call @g_f32_to_e4m3fn_v4(%\g<src>) : (vector<4xf32>) -> (vector<4xi8>)\n"
                r"\g<ind>%\g<buf> = buffer.view %arena[%\g<off>] : buffer -> view<8x32x4xi8>\n"
                r"\g<ind>vector.store %\g<v16>_fp8, %\g<buf>[%kh, %lane, %c0] : vector<4xi8>, view<8x32x4xi8>\n",
                blk,
            )
            assert n == 2, f"{name}: expected 2 FP8 KV writes (K+V), got {n}"
            seen.add(name)
        elif name in _FP8_SPLIT_READERS:
            expected_pairs = _FP8_SPLIT_READERS[name]
            assert blk.count("%pos_bytes = index.constant 2048 : offset") == 1, name
            blk = blk.replace(
                "%pos_bytes = index.constant 2048 : offset",
                "%pos_bytes = index.constant 1024 : offset",
                1,
            )
            blk, n = read4_re.subn(
                r"\g<ind>%\g<buf> = buffer.view %arena[%\g<off>] : buffer -> view<8x32x4xi8>\n"
                r"\g<ind>%\g<dst>_fp8 = vector.load %\g<buf>[%kh, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>\n"
                r"\g<ind>%\g<dst> = func.call @g_e4m3fn_to_f16_v4(%\g<dst>_fp8) : (vector<4xi8>) -> (vector<4xf16>)\n",
                blk,
            )
            assert n == 2 * expected_pairs, f"{name}: expected {2 * expected_pairs} FP8 KV loads, got {n}"
            seen.add(name)
        elif name in _FP8_FLASH2_READERS:
            expected_pairs = _FP8_FLASH2_READERS[name]
            assert blk.count("%pos_bytes = index.constant 2048 : offset") == 1, name
            blk = blk.replace(
                "%pos_bytes = index.constant 2048 : offset",
                "%pos_bytes = index.constant 1024 : offset",
                1,
            )
            blk, n = read16_re.subn(
                r"\g<ind>%k_cache_\g<tag> = buffer.view %arena[%k_off_\g<tag>] : buffer -> view<8x128xi8>\n"
                r"\g<ind>%v_cache_\g<tag> = buffer.view %arena[%v_off_\g<tag>] : buffer -> view<8x128xi8>\n"
                r"\g<ind>%kh8_\g<tag> = vector.load %k_cache_\g<tag>[%kv, %st_dim_\g<dim>] : view<8x128xi8> -> vector<16xi8>\n"
                r"\g<ind>%vh8_\g<tag> = vector.load %v_cache_\g<tag>[%kv, %st_dim_\g<dim>] : view<8x128xi8> -> vector<16xi8>\n"
                r"\g<ind>%kh_\g<tag> = func.call @g_e4m3fn_to_f16_v16(%kh8_\g<tag>) : (vector<16xi8>) -> (vector<16xf16>)\n"
                r"\g<ind>%vh_\g<tag> = func.call @g_e4m3fn_to_f16_v16(%vh8_\g<tag>) : (vector<16xi8>) -> (vector<16xf16>)\n",
                blk,
            )
            assert n == expected_pairs, f"{name}: expected {expected_pairs} flash2 FP8 KV load pairs, got {n}"
            seen.add(name)
        parts[i] = blk

    expected_all = {"g_kv_offset"} | _FP8_WRITERS | set(_FP8_SPLIT_READERS) | set(_FP8_FLASH2_READERS)
    assert seen == expected_all, f"FP8 KV missed symbols: {sorted(expected_all - seen)}"
    return encodings_first("".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--gguf-file", required=True)
    ap.add_argument("--gguf-path", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--rows", required=True, type=int)
    ap.add_argument("--context-capacity", required=True, type=int)
    ap.add_argument("--prefill-capacity", default=2048, type=int)
    ap.add_argument("--max-tokens", required=True, type=int)
    ap.add_argument("--default-temperature", default=1.0, type=float)
    ap.add_argument("--default-top-p", default=0.95, type=float)
    ap.add_argument("--dspark-gguf-file", default=None)
    ap.add_argument("--dspark-gguf-path", default=None)
    ap.add_argument("--draft-depth", default=0, type=int)
    ap.add_argument("--vision-gguf-file", default=None)
    ap.add_argument("--vision-gguf-path", default=None)
    ap.add_argument("--loom-compile", default=None)
    ap.add_argument("--loom-opt", default=None)
    args = ap.parse_args()

    use_dspark = bool(args.dspark_gguf_file)
    if use_dspark:
        assert args.dspark_gguf_path, "--dspark-gguf-path is required with --dspark-gguf-file"
        wd_path = Path(__file__).resolve().parent / "wire-inkling-dspark.py"
        spec = importlib.util.spec_from_file_location("wire_inkling_dspark", wd_path)
        assert spec and spec.loader
        wd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wd)
    else:
        wd = None

    use_vision = bool(args.vision_gguf_file)
    if use_vision:
        assert args.vision_gguf_path, "--vision-gguf-path is required with --vision-gguf-file"
        wv_path = Path(__file__).resolve().parent / "wire-inkling-vision.py"
        vspec = importlib.util.spec_from_file_location("wire_inkling_vision", wv_path)
        assert vspec and vspec.loader
        wv = importlib.util.module_from_spec(vspec)
        vspec.loader.exec_module(wv)
    else:
        wv = None

    out = args.out_dir
    (out / "program").mkdir(parents=True, exist_ok=True)
    (out / "parameter").mkdir(parents=True, exist_ok=True)

    for rel in (
        "llm_config.json",
        "tokenizer.json",
        "weights.loom",
        "program/prepare.loom",
        "program/service.loom",
    ):
        shutil.copyfile(args.src_dir / rel, out / rel)

    storage_src = (args.src_dir / "program/storage.loom").read_text(encoding="utf-8")
    storage_out = out / "program/storage.loom"
    if use_dspark:
        storage_text = wd.emit_storage_loom()
    else:
        storage_text = patch_storage_fp8_kv(storage_src)
    storage_out.write_text(storage_text, encoding="utf-8", newline="\n")

    ctrl_src = (args.src_dir / "program/control.loom").read_text(encoding="utf-8")
    ctrl_text = patch_control_defaults(ctrl_src, args.default_temperature, args.default_top_p)
    if use_dspark:
        ctrl_text = wd.patch_control_dspark(ctrl_text)
    if use_vision:
        ctrl_text = wv.patch_control_vision(ctrl_text)
    ctrl_out = out / "program/control.loom"
    ctrl_out.write_text(ctrl_text, encoding="utf-8", newline="\n")

    prep_out = out / "program/prepare.loom"
    head_weights_out = out / "head_weights.loom"
    if use_dspark:
        prep_out.write_text(wd.emit_prepare_loom(), encoding="utf-8", newline="\n")
        head_weights_out.write_text(wd.emit_head_weights_loom(), encoding="utf-8", newline="\n")

    vis_prep_out = out / "program/vision_prepare.loom"
    vis_stages_out = out / "program/vision_stages.loom"
    vis_ctrl_out = out / "program/vision_control.loom"
    vis_enc_out = out / "program/vision_encoder.loom"
    if use_vision:
        vis_prep_out.write_text(wv.emit_vision_prepare_loom(), encoding="utf-8", newline="\n")
        vis_stages_out.write_text(wv.emit_vision_stages_loom(), encoding="utf-8", newline="\n")
        vis_ctrl_out.write_text(wv.emit_vision_control_loom(), encoding="utf-8", newline="\n")
        vis_enc_out.write_text(wv.emit_vision_encoder_loom(), encoding="utf-8", newline="\n")

    llm_src = (args.src_dir / "program/llm.loom").read_text(encoding="utf-8")
    llm_text = patch_llm_fp8_kv(patch_head_select(port_llm_loom(llm_src)))
    if use_dspark:
        llm_text = wd.patch_llm_dspark(llm_text)
    if use_vision:
        llm_text = encodings_first(wv.patch_llm_vision(llm_text))
    llm_out = out / "program/llm.loom"
    llm_out.write_text(llm_text, encoding="utf-8", newline="\n")

    card = json.loads((args.src_dir / "model.json").read_text(encoding="utf-8"))
    card["model"]["id"] = args.model_id
    card["model"]["name"] = args.model_name
    exec_cfg = {
        "rows": args.rows,
        "context_capacity": args.context_capacity,
        "prefill_capacity": args.prefill_capacity,
        "max_tokens": args.max_tokens,
    }
    if use_dspark:
        prep_input = card["programs"]["source"]["inputs"][0]
        if "head_weights.loom" not in prep_input["data"]:
            prep_input["data"].insert(1, "head_weights.loom")
        card["variants"]["gguf"]["services"]["chat"]["parameters"]["draft"] = {
            "files": [
                {
                    "path": f"parameter/{args.dspark_gguf_file}",
                    "format": "gguf",
                }
            ]
        }
        if args.draft_depth > 0:
            exec_cfg["draft_depth"] = args.draft_depth
    if use_vision:
        src_inputs = card["programs"]["source"]["inputs"]
        for v_in in ("program/vision_prepare.loom", "program/vision_control.loom"):
            if v_in not in src_inputs:
                src_inputs.append(v_in)
        src_libs = card["programs"]["source"]["libraries"]
        for v_lib in ("program/vision_stages.loom", "program/vision_encoder.loom"):
            if v_lib not in src_libs:
                src_libs.append(v_lib)
        card["services"]["chat"]["components"] = {"image": {"version": 1}}
        card["variants"]["gguf"]["services"]["chat"]["parameters"]["vision"] = {
            "files": [
                {
                    "path": f"parameter/{args.vision_gguf_file}",
                    "format": "gguf",
                }
            ]
        }
    card.setdefault("presets", {})[args.preset] = {
        "service": "chat",
        "variant": "gguf",
        "programs": "source",
        "execution": exec_cfg,
    }
    (out / "model.json").write_text(
        json.dumps(card, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    os.symlink(args.gguf_path, out / "parameter" / args.gguf_file)
    if use_dspark:
        os.symlink(args.dspark_gguf_path, out / "parameter" / args.dspark_gguf_file)
    if use_vision:
        os.symlink(args.vision_gguf_path, out / "parameter" / args.vision_gguf_file)

    if args.loom_opt:
        opt_targets = [("control.loom", ctrl_out), ("storage.loom", storage_out)]
        if use_dspark:
            opt_targets += [("prepare.loom", prep_out), ("head_weights.loom", head_weights_out)]
        if use_vision:
            opt_targets += [
                ("vision_prepare.loom", vis_prep_out),
                ("vision_stages.loom", vis_stages_out),
                ("vision_control.loom", vis_ctrl_out),
                ("vision_encoder.loom", vis_enc_out),
            ]
        for label, path in opt_targets:
            proc = subprocess.run(
                [args.loom_opt, str(path), "--output=/dev/null"],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                sys.stderr.write(
                    f"FAIL verify {label}:\n{proc.stderr}\n{proc.stdout}\n"
                )
                return 1
        print(
            f"port-inkling-rdna4: {', '.join(lbl for lbl, _ in opt_targets)} verified "
            f"(FP8 E4M3FN KV, dspark={use_dspark}, vision={use_vision}, draft_depth={args.draft_depth}, "
            f"default temperature={args.default_temperature}, top_p={args.default_top_p})"
        )

    if args.loom_compile:
        checks = [(arch, sym) for k in TARGETS for arch, sym in (("gfx1201", k + "_rdna4"), ("gfx1151", k))]
        extras = [
            "g_embed",
            "g_dfl_prepare",
            "g_head_select",
            "g_ink_kv_prep_store",
            "g_ink_sconv1_dyn_kv_prep_store",
            "g_ink_attn_split",
            "g_ink_attn_direct",
            "g16_ink_attn_split_listed",
        ]
        if use_dspark:
            extras += [
                "g_prep_q8_0_q4_0",
                "g_spec_rows",
                "g_spec_accept",
                "g_ink_sconv_commit_batch_c1024",
                "g_ink_sconv_commit_batch_c2304",
                "g_ink_swa_save_ring",
                "g_ink_swa_restore_batch",
                "g_dspark_mix",
                "g_dspark_kv_store",
                "g_dspark_attention",
                "g_dspark_mv_q8_0_k9216",
                "g_dspark_markov_rnn",
                "g_dspark_markov_out",
                "g_dspark_head_q4_0",
                "g_dspark_argmax",
            ]
        for extra in extras:
            checks += [("gfx1201", extra), ("gfx1151", extra)]
        for arch, sym in checks:
            proc = subprocess.run(
                [
                    args.loom_compile,
                    str(llm_out),
                    f"--target=amdgpu:{arch}",
                    f"--root={sym}",
                    "--format=amdgpu-hsaco",
                    "--output=/dev/null",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                sys.stderr.write(
                    f"FAIL compile {arch} {sym}:\n{proc.stderr}\n{proc.stdout}\n"
                )
                return 1
            print(f"  compile {arch:7s} {sym:42s}: OK")
        if use_vision:
            v_checks = []
            for vk in (
                "v_ink_prep_pixels",
                "v_ink_patch_and_pos_1024",
                "v_norm_ln_1024",
                "v_rope_ink_axial_64x16",
                "v_attn_64x16",
                "v_bias_ls_residual_1024",
                "v_bias_gelu_erf_4096",
                "v_ew_gelu_erf",
                "v_merge2x2_drop_cls_1024",
            ):
                v_checks += [("gfx1201", vk), ("gfx1151", vk)]
            for vgemm in (
                "v_gemm_f16_store_1024x1024",
                "v_gemm_f16_store_1024x4096",
                "v_gemm_f16_store_4096x1024",
                "v_gemm_f16_store_4096x4096",
                "v_gemm_f16_store_4096x2304",
            ):
                v_checks += [("gfx1201", vgemm + "_rdna4"), ("gfx1151", vgemm)]
            for arch, sym in v_checks:
                proc = subprocess.run(
                    [
                        args.loom_compile,
                        str(vis_enc_out),
                        f"--target=amdgpu:{arch}",
                        f"--root={sym}",
                        "--format=amdgpu-hsaco",
                        "--output=/dev/null",
                    ],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    sys.stderr.write(
                        f"FAIL compile vision {arch} {sym}:\n{proc.stderr}\n{proc.stdout}\n"
                    )
                    return 1
                print(f"  compile {arch:7s} {sym:42s}: OK")
        print(f"port-inkling-rdna4: all {len(checks) + (len(v_checks) if use_vision else 0)} kernels compiled cleanly")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
