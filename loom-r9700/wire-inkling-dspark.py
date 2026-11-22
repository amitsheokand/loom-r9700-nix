#!/usr/bin/env python3
"""Wire Inkling 2 Mini's built-in DSpark speculative drafter into the RDNA4 Loom package.

Uses hrx-loom 0.2.0's native block-diffusion speculative epoch contract (proven by Flint-loom):
  - head_weights.loom: converts target output.weight Q8_0 -> Q4_0 at load time for fast proposal
    head matvecs (259 MB vs 492 MB, ~0.42 ms/draft token on R9700).
  - prepare.loom: registers llm_verify, llm_dflash_warm, and llm_dflash_propose stages alongside
    llm_advance, binding parameter scope "draft" (Inkling-2-Mini-DSpark-Q8_0.gguf).
  - storage.loom: allocates residual tap buffer (4 x 2304 f32/tok), committed/results buffers,
    and 6 FP8 E4M3FN KV planes for the 3 DSpark layers (1.50 GiB at 256K) + per-row Markov
    context tail (f_prev[row, 2304] f32).
  - control.loom: wraps @encode_epoch, @publish_epoch, @epoch, and @resize_state with Loom's
    speculative dispatch and c4 occupancy gating (drafting on c1; anchors-only on c4).
  - llm.loom:
      * taps visits 5, 23, 41 and final normed state in both llm_advance and llm_verify
      * defers SWA ring and sconv commits in llm_verify (2 <= t <= 64) until after g_spec_accept
        writes %committed so partial draft rejections never corrupt SWA or sconv history
      * implements DSpark warm (4-tap softmax fuse -> k_ctx/v_ctx FP8 KV + f_prev)
      * implements DSpark propose (3 dense layers with RoPE theta=1e6, SWA 1024 on layers 0/2,
        full on layer 1 -> recurrent rank-256 Markov head -> Q4_0 proposal head argmax)
"""

from __future__ import annotations

import re

# Target vocab / hidden constants for Inkling 2 Mini.
N_HEAD_BLOCKS = 201024 * (2304 // 32)  # 14,473,728 Q8_0/Q4_0 blocks for output.weight
HEAD_Q4_BYTES = N_HEAD_BLOCKS * 18     # 260,527,104 bytes


def emit_head_weights_loom() -> str:
    """Convert target output.weight Q8_0 -> Q4_0 at load time for DSpark proposal head."""
    return f"""// Weight policy (DSpark proposal head): stages output.weight Q8_0 -> Q4_0 at load time.
func.decl import("weights") @input(%source: i32, %key: buffer, %key_offset: i64, %key_length: i64, %offset: i64, %length: i64, %alignment: i64) -> (i64)
func.decl import("prepare") @stage(%root: buffer, %tag: i64) -> (i32)
func.decl import("prepare") @config_i64(%stage: i32, %key: buffer, %value: i64)

// "llm_prep_q8_0_q4_0", "llm.prep_blocks", "llm.prep_input", "output.weight"
global.rodata.def @root0 = bytes("6c6c6d5f707265705f71385f305f71345f30")
global.rodata.def @blocks_key = bytes("6c6c6d2e707265705f626c6f636b73")
global.rodata.def @input_key = bytes("6c6c6d2e707265705f696e707574")
global.rodata.def @k0 = bytes("6f75747075742e776569676874")

func.def @same(%key: buffer, %name: buffer) -> (i1) {{
  %a = buffer.length %key
  %b = buffer.length %name
  %sized = index.cmp eq, %a, %b : offset
  %result = scf.if %sized -> (i1) {{
    %o = index.constant 0 : offset
    %order = buffer.compare %key[%o], %name[%o], %b
    %z = scalar.constant 0 : i32
    %same = scalar.cmpi eq, %order, %z : i32
    scf.yield %same : i1
  }} else {{
    %no = scalar.constant false : i1
    scf.yield %no : i1
  }}
  func.return %result : i1
}}

func.def public @prepare_weight(%key: buffer) -> (i64) {{
  %zero = scalar.constant 0 : i64
  %name0 = global.load @k0 : buffer
  %is0 = func.call @same(%key, %name0) : (buffer, buffer) -> (i1)
  %bytes = scf.if %is0 -> (i64) {{
    %nb = scalar.constant {N_HEAD_BLOCKS} : i64
    %inb = scalar.constant 34 : i64
    %outb = scalar.constant 18 : i64
    %source = scalar.constant 0 : i32
    %align = scalar.constant 256 : i64
    %length = scalar.muli %nb, %inb : i64
    %key_bytes = buffer.length %key
    %key_length = index.cast %key_bytes : offset to i64
    %first = func.call @input(%source, %key, %zero, %key_length, %zero, %length, %align) : (i32, buffer, i64, i64, i64, i64, i64) -> (i64)
    %root0 = global.load @root0 : buffer
    %stage0 = func.call @stage(%root0, %zero) : (buffer, i64) -> (i32)
    %bk0 = global.load @blocks_key : buffer
    %ik0 = global.load @input_key : buffer
    func.call @config_i64(%stage0, %bk0, %nb) : (i32, buffer, i64)
    func.call @config_i64(%stage0, %ik0, %first) : (i32, buffer, i64)
    %prepared = scalar.muli %nb, %outb : i64
    scf.yield %prepared : i64
  }} else {{
    scf.yield %zero : i64
  }}
  func.return %bytes : i64
}}
"""


def emit_prepare_loom() -> str:
    """Speculative-capable prepare.loom for Inkling 2 Mini (DSpark draft_depth 0..7)."""
    return """config.def @llm.card = false : i1
config.decl @llm.logits : i1

func.decl import("input") @require(%condition: i32, %diagnostic: buffer)
func.decl import("json") @members(%json: buffer) -> (buffer, i32, i64)
func.decl import("prepare") @stage(%root: buffer, %tag: i64) -> (i32)
func.decl import("prepare") @config_i64(%stage: i32, %key: buffer, %value: i64)
func.decl import("prepare") @source(%directory: buffer, %path: buffer) -> (i32)
func.decl import("prepare") @parameter(%stage: i32, %binding: i32, %input: i32, %source: i32, %policy: buffer)
func.decl import("prepare") @source_scope(%scope: buffer) -> (i32)

// "model", "draft"
global.rodata.def @model_scope = bytes("6d6f64656c")
global.rodata.def @draft_scope = bytes("6472616674")
// "llm_advance", "llm_verify", "llm_dflash_warm", "llm_dflash_propose"
global.rodata.def @root = bytes("6c6c6d5f616476616e6365")
global.rodata.def @verify_root = bytes("6c6c6d5f766572696679")
global.rodata.def @warm_root = bytes("6c6c6d5f64666c6173685f7761726d")
global.rodata.def @propose_root = bytes("6c6c6d5f64666c6173685f70726f706f7365")
// config keys
global.rodata.def @context_key = bytes("6c6c6d2e636f6e74657874")
global.rodata.def @spans_key = bytes("6c6c6d2e7370616e73")
global.rodata.def @tokens_key = bytes("6c6c6d2e746f6b656e73")
global.rodata.def @pool_key = bytes("6c6c6d2e706f6f6c")
global.rodata.def @logit_slots_key = bytes("6c6c6d2e6c6f6769745f736c6f7473")
global.rodata.def @width_key = bytes("6c6c6d2e7769647468")
global.rodata.def @gated_key = bytes("6c6c6d2e6761746564")
global.rodata.def @epoch_spans_key = bytes("6c6c6d2e65706f63685f7370616e73")
global.rodata.def @epoch_tokens_key = bytes("6c6c6d2e65706f63685f746f6b656e73")
// "weights.loom", "head_weights.loom"
global.rodata.def @policy = bytes("776569676874732e6c6f6f6d")
global.rodata.def @head_policy_name = bytes("686561645f776569676874732e6c6f6f6d")
// "<|content_model_end_sampling|>"
global.rodata.def @terminal = bytes("3c7c636f6e74656e745f6d6f64656c5f656e645f73616d706c696e677c3e")
global.rodata.def @empty = bytes("")
global.rodata.def @unsupported = bytes("6c6c6d20726571756972657320706f6f6c65642065706f6368732c2064726166745f6465707468203c3d2037")

func.decl @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64, %drafting: i64) -> (buffer, buffer, buffer, buffer, buffer)

func.def public @prepare(%prefill: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64, %draft_depth: i32, %shapes: buffer, %shape_count: i64, %weights: buffer, %options: buffer) -> (i64, i32, buffer, buffer, buffer, buffer, buffer, buffer, buffer) {
  %c0_source = scalar.constant 0 : i32
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c0_i64 = scalar.constant 0 : i64
  %c128_i64 = scalar.constant 128 : i64
  %cctx_i64 = scalar.constant 1048576 : i64
  %pooled = scalar.cmpi ne, %pool, %c0_i64 : i64
  %packed = scalar.cmpi ne, %shape_count, %c0_i64 : i64
  %few_shapes = scalar.cmpi ule, %shape_count, %c128_i64 : i64
  %short_context = scalar.cmpi ule, %context, %cctx_i64 : i64
  %has_epochs = scalar.andi %pooled, %packed : i1
  %c7_i32 = scalar.constant 7 : i32
  %depth_ok = scalar.cmpi ule, %draft_depth, %c7_i32 : i32
  %mode_ok = scalar.andi %has_epochs, %depth_ok : i1
  %limits_ok = scalar.andi %few_shapes, %short_context : i1
  %supported = scalar.andi %mode_ok, %limits_ok : i1
  %supported32 = scalar.extui %supported : i1 to i32
  %error = global.load @unsupported : buffer
  func.call @require(%supported32, %error) : (i32, buffer)
  %empty = global.load @empty : buffer
  %policy = global.load @policy : buffer
  %card = config.get @llm.card : i1
  %checkpoint = scf.if %card -> (i32) {
    %scope = global.load @model_scope : buffer
    %scoped = func.call @source_scope(%scope) : (buffer) -> (i32)
    scf.yield %scoped : i32
  } else {
    %direct = func.call @source(%empty, %weights) : (buffer, buffer) -> (i32)
    scf.yield %direct : i32
  }
  %bounded_count = scalar.assume %shape_count [range(%shape_count, 1, 128)] : i64
  %count = index.cast %bounded_count : i64 to index
  %header_bytes = index.constant 128 : offset
  %control_bytes = index.constant 2048 : offset
  %bounded_rows = scalar.assume %rows [range(%rows, 1, 1048576)] : i64
  %row_count = index.cast %bounded_rows : i64 to index
  %record_bytes = index.constant 16 : offset
  %sampling_bytes = index.scale %row_count, %record_bytes : index, offset -> offset
  %state_bytes = index.add %control_bytes, %sampling_bytes : offset
  %state = buffer.alloca<private> align(8) %state_bytes : buffer
  buffer.fill %c0_i32, %state[%origin], %state_bytes : i32
  %stage_view = buffer.view %state[%header_bytes] : buffer -> view<[%count]xi32>
  %verify_at = index.constant 640 : offset
  %warm_at = index.constant 1152 : offset
  %verify_view = buffer.view %state[%verify_at] : buffer -> view<[%count]xi32>
  %warm_view = buffer.view %state[%warm_at] : buffer -> view<[%count]xi32>
  %drafting = scalar.cmpi ne, %draft_depth, %c0_i32 : i32
  %depth64 = scalar.extui %draft_depth : i32 to i64
  %c1_w = scalar.constant 1 : i64
  %width = scalar.addi %depth64, %c1_w : i64
  %verify_root = global.load @verify_root : buffer
  %warm_root = global.load @warm_root : buffer
  %width_key = global.load @width_key : buffer
  %gated_key = global.load @gated_key : buffer
  %gated_off = scalar.constant 0 : i64
  %gated_on = scalar.constant 1 : i64
  %root = global.load @root : buffer
  %context_key = global.load @context_key : buffer
  %spans_key = global.load @spans_key : buffer
  %tokens_key = global.load @tokens_key : buffer
  %pool_key = global.load @pool_key : buffer
  %logits_cfg = config.get @llm.logits : i1
  %logit_slots_key = global.load @logit_slots_key : buffer
  %shape_view = buffer.view %shapes[%origin] : buffer -> view<[%count]x2xi64>
  %maximum, %span_capacity = scf.for %i = [%c0 to %count step %c1](%previous = %c0_i64 : i64, %previous_spans = %c0_i64 : i64) -> (i64, i64) {
    %tokens = view.load %shape_view[%i, 0] : view<[%count]x2xi64> -> i64
    %spans = view.load %shape_view[%i, 1] : view<[%count]x2xi64> -> i64
    %tag = index.cast %i : index to i64
    %stage = func.call @stage(%root, %tag) : (buffer, i64) -> (i32)
    func.call @config_i64(%stage, %context_key, %context) : (i32, buffer, i64)
    func.call @config_i64(%stage, %pool_key, %pool) : (i32, buffer, i64)
    func.call @config_i64(%stage, %tokens_key, %tokens) : (i32, buffer, i64)
    func.call @config_i64(%stage, %spans_key, %spans) : (i32, buffer, i64)
    scf.if %logits_cfg {
      %logit_slots = scalar.minsi %spans, %rows : i64
      func.call @config_i64(%stage, %logit_slots_key, %logit_slots) : (i32, buffer, i64)
    }
    func.call @parameter(%stage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
    view.store %stage, %stage_view[%i] : i32, view<[%count]xi32>
    %larger = scalar.cmpi ugt, %tokens, %previous : i64
    %next = scf.select %larger, %tokens, %previous : i64
    %more_spans = scalar.cmpi ugt, %spans, %previous_spans : i64
    %next_spans = scf.select %more_spans, %spans, %previous_spans : i64
    scf.yield %next, %next_spans : i64, i64
  }
  %draft_scope = global.load @draft_scope : buffer
  %draft_source = func.call @source_scope(%draft_scope) : (buffer) -> (i32)
  %head_policy = global.load @head_policy_name : buffer
  scf.if %drafting {
    %propose_root = global.load @propose_root : buffer
    %epoch_spans_key = global.load @epoch_spans_key : buffer
    %epoch_tokens_key = global.load @epoch_tokens_key : buffer
    %span32 = scalar.trunci %span_capacity : i64 to i32
    %one32 = scalar.constant 1 : i32
    %bits32 = scalar.constant 32 : i32
    %below = scalar.subi %span32, %one32 : i32
    %leading = scalar.ctlzi %below : i32
    %last_level = scalar.subi %bits32, %leading : i32
    %levels_all = scalar.addi %last_level, %one32 : i32
    %level_cap = scalar.constant 11 : i32
    %levels32 = scalar.minui %levels_all, %level_cap : i32
    %levels_b = scalar.assume %levels32 [range(%levels32, 1, 14)] : i32
    %levels = index.cast %levels_b : i32 to index
    %c1_64 = scalar.constant 1 : i64
    %first_propose = scf.for %level = [%c0 to %levels step %c1](%first = %c0_i32 : i32) -> (i32) {
      %shift = index.cast %level : index to i64
      %cap = scalar.shli %c1_64, %shift : i64
      %pstage = func.call @stage(%propose_root, %shift) : (buffer, i64) -> (i32)
      func.call @config_i64(%pstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %pool_key, %pool) : (i32, buffer, i64)
      %block_full = scalar.constant 8 : i64
      %block_tokens = scalar.minui %block_full, %width : i64
      %cap_tokens = scalar.muli %cap, %block_tokens : i64
      func.call @config_i64(%pstage, %tokens_key, %cap_tokens) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %spans_key, %cap) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %width_key, %width) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_spans_key, %span_capacity) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_tokens_key, %maximum) : (i32, buffer, i64)
      %pstage_b0 = scalar.constant 0 : i32
      func.call @parameter(%pstage, %pstage_b0, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
      %pstage_b1 = scalar.constant 1 : i32
      func.call @parameter(%pstage, %pstage_b1, %c0_source, %draft_source, %policy) : (i32, i32, i32, i32, buffer)
      %pstage_b2 = scalar.constant 2 : i32
      func.call @parameter(%pstage, %pstage_b2, %c0_source, %checkpoint, %head_policy) : (i32, i32, i32, i32, buffer)
      %initial = index.cmp eq, %level, %c0 : index
      %next_first = scf.select %initial, %pstage, %first : i32
      scf.yield %next_first : i32
    }
    scf.for %i = [%c0 to %count step %c1] {
      %tokens = view.load %shape_view[%i, 0] : view<[%count]x2xi64> -> i64
      %spans = view.load %shape_view[%i, 1] : view<[%count]x2xi64> -> i64
      %tag = index.cast %i : index to i64
      %vstage = func.call @stage(%verify_root, %tag) : (buffer, i64) -> (i32)
      func.call @config_i64(%vstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %pool_key, %pool) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %tokens_key, %tokens) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %spans_key, %spans) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %width_key, %width) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %gated_key, %gated_off) : (i32, buffer, i64)
      func.call @parameter(%vstage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
      view.store %vstage, %verify_view[%i] : i32, view<[%count]xi32>
      %wstage = func.call @stage(%warm_root, %tag) : (buffer, i64) -> (i32)
      func.call @config_i64(%wstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %pool_key, %pool) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %tokens_key, %tokens) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %spans_key, %spans) : (i32, buffer, i64)
      %wstage_b0 = scalar.constant 0 : i32
      func.call @parameter(%wstage, %wstage_b0, %c0_source, %draft_source, %policy) : (i32, i32, i32, i32, buffer)
      view.store %wstage, %warm_view[%i] : i32, view<[%count]xi32>
      %same_shape = scalar.cmpi eq, %tokens, %spans : i64
      %gate_slots = scalar.constant 32 : i64
      %few_spans = scalar.cmpi ult, %spans, %gate_slots : i64
      %gate_ok = scalar.andi %same_shape, %few_spans : i1
      scf.if %gate_ok {
        %gate_at = index.constant 1700 : offset
        %gate_view = buffer.view %state[%gate_at] : buffer -> view<32xi32>
        %gate_slot = index.cast %spans : i64 to index
        %gate_i32 = index.cast %i : index to i32
        %gate_one = scalar.constant 1 : i32
        %gate_idx1 = scalar.addi %gate_i32, %gate_one : i32
        %gtag_base = scalar.constant 256 : i64
        %gtag = scalar.addi %tag, %gtag_base : i64
        %gvstage = func.call @stage(%verify_root, %gtag) : (buffer, i64) -> (i32)
        func.call @config_i64(%gvstage, %context_key, %context) : (i32, buffer, i64)
        func.call @config_i64(%gvstage, %pool_key, %pool) : (i32, buffer, i64)
        func.call @config_i64(%gvstage, %tokens_key, %tokens) : (i32, buffer, i64)
        func.call @config_i64(%gvstage, %spans_key, %spans) : (i32, buffer, i64)
        func.call @config_i64(%gvstage, %width_key, %width) : (i32, buffer, i64)
        func.call @config_i64(%gvstage, %gated_key, %gated_on) : (i32, buffer, i64)
        func.call @parameter(%gvstage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
        %gate_shift = scalar.constant 8 : i32
        %gv_high = scalar.shli %gvstage, %gate_shift : i32
        %gate_tag = scalar.ori %gate_idx1, %gv_high : i32
        view.store %gate_tag, %gate_view[%gate_slot] : i32, view<32xi32>
      }
    }
    %draft_at = index.constant 1664 : offset
    %draft_view = buffer.view %state[%draft_at] : buffer -> view<3xi32>
    view.store %first_propose, %draft_view[0] : i32, view<3xi32>
    view.store %levels32, %draft_view[1] : i32, view<3xi32>
    %kind32 = scalar.constant 2 : i32
    view.store %kind32, %draft_view[2] : i32, view<3xi32>
  }
  %smaller = scalar.cmpi ult, %maximum, %prefill : i64
  %effective = scf.select %smaller, %maximum, %prefill : i64
  %layout_origin = index.constant 88 : offset
  %layout = buffer.view %state[%layout_origin] : buffer -> view<5xi64>
  view.store %maximum, %layout[0] : i64, view<5xi64>
  view.store %span_capacity, %layout[1] : i64, view<5xi64>
  view.store %context, %layout[2] : i64, view<5xi64>
  view.store %pool, %layout[3] : i64, view<5xi64>
  view.store %pins, %layout[4] : i64, view<5xi64>
  %stage_count32 = index.cast %count : index to i32
  %drafting64 = scalar.extui %drafting : i1 to i64
  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins, %drafting64) : (i64, i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)
  %terminal = global.load @terminal : buffer
  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %draft, %geometry : i64, i32, buffer, buffer, buffer, buffer, buffer, buffer, buffer
}
"""


def emit_storage_loom() -> str:
    """Inkling FP8 KV storage + DSpark allocations (6 FP8 KV planes + per-row Markov f_prev tail)."""
    return """// Target KV: 16 planes x 1024 B/pos (FP8 E4M3FN, 16 KiB/tok).
// DSpark KV:  6 planes x 1024 B/pos (FP8 E4M3FN,  6 KiB/tok) + rows x 9216 B f_prev tail.
config.decl @llm.logits : i1

func.def @allocation(%records: buffer, %slot: index, %bytes: i64, %clear: i64) {
  %origin = index.constant 0 : offset
  %bounded = index.assume %slot [range(%slot, 0, 10)] : index
  %view = buffer.view %records[%origin] : buffer -> view<11x3xi64>
  %alignment = scalar.constant 256 : i64
  view.store %bytes, %view[%bounded, 0] : i64, view<11x3xi64>
  view.store %alignment, %view[%bounded, 1] : i64, view<11x3xi64>
  view.store %clear, %view[%bounded, 2] : i64, view<11x3xi64>
  func.return
}

func.def public @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64, %drafting: i64) -> (buffer, buffer, buffer, buffer, buffer) {
  %zero_flag = scalar.constant 0 : i64
  %on = scalar.cmpi ne, %drafting, %zero_flag : i64
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c0_i64 = scalar.constant 0 : i64
  %c1_i64 = scalar.constant 1 : i64
  %c4_i64 = scalar.constant 4 : i64
  %bounded_rows = scalar.assume %rows [range(%rows, 1, 2147483647)] : i64
  %row_count = index.cast %bounded_rows : i64 to index
  %slot_stride = scalar.constant 256 : i64
  %token_bytes = scalar.muli %maximum, %c4_i64 : i64
  %alignment_bias = scalar.constant 255 : i64
  %biased_token_bytes = scalar.addi %token_bytes, %alignment_bias : i64
  %token_slots = scalar.divui %biased_token_bytes, %slot_stride : i64
  %padded_token_bytes = scalar.muli %token_slots, %slot_stride : i64
  %progress_offset = scalar.addi %slot_stride, %padded_token_bytes : i64
  %recurrent_offset = scalar.addi %progress_offset, %slot_stride : i64
  %recurrent_bytes = scalar.constant 171606016 : i64
  %recurrent_padded = scalar.addi %recurrent_bytes, %alignment_bias : i64
  %recurrent_pages = scalar.divui %recurrent_padded, %slot_stride : i64
  %recurrent_region = scalar.muli %recurrent_pages, %slot_stride : i64
  %row_stride = scalar.addi %recurrent_offset, %recurrent_region : i64
  %pinned = scalar.cmpi ne, %pins, %c0_i64 : i64
  %twice_rows = scalar.addi %rows, %rows : i64
  %fork_slots = scalar.addi %twice_rows, %pins : i64
  %slots = scf.select %pinned, %fork_slots, %rows : i64
  %private_bytes = scalar.muli %slots, %row_stride : i64
  %kv_position_bytes = scalar.constant 1024 : i64
  %plane_count = scalar.constant 16 : i64
  %plane_bytes = scalar.muli %pool, %kv_position_bytes : i64
  %pool_bytes = scalar.muli %plane_bytes, %plane_count : i64
  %biased_pool = scalar.addi %pool_bytes, %alignment_bias : i64
  %pool_slots = scalar.divui %biased_pool, %slot_stride : i64
  %private_origin = scalar.muli %pool_slots, %slot_stride : i64
  %recurrent_origin = scalar.addi %private_origin, %recurrent_offset : i64
  %arena_bytes = scalar.addi %private_origin, %private_bytes : i64
  %c15_i64 = scalar.constant 15 : i64
  %c16_i64 = scalar.constant 16 : i64
  %padded = scalar.addi %context, %c15_i64 : i64
  %blocks = scalar.divui %padded, %c16_i64 : i64
  %map_entries = scalar.muli %rows, %blocks : i64
  %map_bytes = scalar.muli %map_entries, %c4_i64 : i64
  %origin_stride = scalar.constant 40 : i64
  %map_origin = scalar.muli %bounded_rows, %origin_stride : i64
  %table_bytes = scalar.addi %map_origin, %map_bytes : i64
  %allocation_bytes = index.constant 264 : offset
  %allocations = buffer.alloca<private> align(8) %allocation_bytes : buffer
  %allocation_count = index.constant 11 : index
  scf.for %i = [%c0 to %allocation_count step %c1] {
    func.call @allocation(%allocations, %i, %c0_i64, %c0_i64) : (buffer, index, i64, i64)
  }
  // 4 tapped hidden rows x 2304 x f32 = 36864 bytes/token (always allocated since llm_advance writes taps).
  %tap_row = scalar.constant 36864 : i64
  %tap_bytes = scalar.muli %maximum, %tap_row : i64
  %span_floor = scalar.muli %span_capacity, %c4_i64 : i64
  %residual_bytes = scalar.maxui %tap_bytes, %span_floor : i64
  %header_bytes = scalar.constant 16 : i64
  %record_bytes = scalar.constant 36 : i64
  %plan_bytes = scalar.muli %span_capacity, %record_bytes : i64
  %metadata_bytes = scalar.addi %header_bytes, %plan_bytes : i64
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c5 = index.constant 5 : index
  func.call @allocation(%allocations, %c0, %residual_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c1, %arena_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c2, %metadata_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c3, %table_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c4, %token_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c5, %span_floor, %c0_i64) : (buffer, index, i64, i64)

  %c6 = index.constant 6 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c10 = index.constant 10 : index
  %committed_stride = scalar.constant 40 : i64
  %committed_records = scalar.muli %span_capacity, %committed_stride : i64
  %committed_bytes = scalar.addi %committed_records, %header_bytes : i64
  %result_stride = scalar.constant 84 : i64
  %results_bytes = scalar.muli %span_capacity, %result_stride : i64
  %first_bank_stride = scalar.constant 40 : i64
  %feedback_split = scalar.muli %span_capacity, %first_bank_stride : i64
  // DSpark KV: 6 FP8 E4M3FN planes (1024 B/pos) + rows x 9216 B f_prev tail.
  %draft_planes = scalar.constant 6 : i64
  %draft_pos_bytes = scalar.constant 1024 : i64
  %draft_plane_bytes = scalar.muli %pool, %draft_pos_bytes : i64
  %draft_kv_bytes = scalar.muli %draft_plane_bytes, %draft_planes : i64
  %fprev_row_bytes = scalar.constant 9216 : i64
  %fprev_slots = scalar.constant 1024 : i64
  %fprev_max = scalar.maxui %rows, %fprev_slots : i64
  %fprev_bytes = scalar.muli %fprev_max, %fprev_row_bytes : i64
  %draft_bytes = scalar.addi %draft_kv_bytes, %fprev_bytes : i64
  %committed_on = scf.select %on, %committed_bytes, %c0_i64 : i64
  %results_on = scf.select %on, %results_bytes, %c0_i64 : i64
  %draft_on = scf.select %on, %draft_bytes, %c0_i64 : i64
  %table_on = scf.select %on, %table_bytes, %c0_i64 : i64
  %split_on = scf.select %on, %feedback_split, %c4_i64 : i64
  func.call @allocation(%allocations, %c6, %c0_i64, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c7, %committed_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c8, %results_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c9, %draft_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c10, %table_on, %c0_i64) : (buffer, index, i64, i64)

  %logits_on = config.get @llm.logits : i1
  scf.if %logits_on {
    %logit_region = scalar.constant 32768 : i64
    %logit_row = scalar.constant 800256 : i64
    %logit_slots = scalar.minsi %span_capacity, %rows : i64
    %logit_rows = scalar.muli %logit_slots, %logit_row : i64
    %logit_total = scalar.addi %logit_region, %logit_rows : i64
    func.call @allocation(%allocations, %c5, %logit_total, %c0_i64) : (buffer, index, i64, i64)
  }

  %view_stride = index.constant 64 : offset
  %view_bytes = index.scale %row_count, %view_stride : index, offset -> offset
  %views = buffer.alloca<private> align(8) %view_bytes : buffer
  %view_records = buffer.view %views[%origin] : buffer -> view<[%row_count]x4x2xi64>
  %origin_bytes = index.cast %map_origin : i64 to offset
  %origins = buffer.alloca<private> align(8) %origin_bytes : buffer
  buffer.fill %c0_i32, %origins[%origin], %origin_bytes : i32
  %origin_records = buffer.view %origins[%origin] : buffer -> view<[%row_count]x5xi64>
  %control_bytes = scalar.constant 12 : i64
  %progress_bytes = scalar.constant 32 : i64
  scf.for %row = [%c0 to %row_count step %c1] {
    %row64 = index.cast %row : index to i64
    %relative = scalar.muli %row64, %row_stride : i64
    %base = scalar.addi %private_origin, %relative : i64
    %tokens = scalar.addi %base, %slot_stride : i64
    %progress = scalar.addi %base, %progress_offset : i64
    view.store %base, %view_records[%row, 0, 0] : i64, view<[%row_count]x4x2xi64>
    view.store %control_bytes, %view_records[%row, 0, 1] : i64, view<[%row_count]x4x2xi64>
    view.store %c0_i64, %view_records[%row, 1, 0] : i64, view<[%row_count]x4x2xi64>
    view.store %c0_i64, %view_records[%row, 1, 1] : i64, view<[%row_count]x4x2xi64>
    view.store %tokens, %view_records[%row, 2, 0] : i64, view<[%row_count]x4x2xi64>
    view.store %token_bytes, %view_records[%row, 2, 1] : i64, view<[%row_count]x4x2xi64>
    view.store %progress, %view_records[%row, 3, 0] : i64, view<[%row_count]x4x2xi64>
    view.store %progress_bytes, %view_records[%row, 3, 1] : i64, view<[%row_count]x4x2xi64>
    view.store %c0_i64, %origin_records[%row, 2] : i64, view<[%row_count]x5xi64>
    %map_stride = scalar.muli %blocks, %c4_i64 : i64
    %map_offset = scalar.muli %row64, %map_stride : i64
    %map_base = scalar.addi %map_origin, %map_offset : i64
    view.store %map_base, %origin_records[%row, 4] : i64, view<[%row_count]x5xi64>
  }
  %geometry_one = index.constant 104 : offset
  %geometry_two = index.constant 144 : offset
  %geometry_bytes = scf.select %on, %geometry_two, %geometry_one : offset
  %geometry = buffer.alloca<private> align(8) %geometry_bytes : buffer
  %geometry_view = buffer.view %geometry[%origin] : buffer -> view<13xi64>
  view.store %c16_i64, %geometry_view[0] : i64, view<13xi64>
  view.store %map_origin, %geometry_view[1] : i64, view<13xi64>
  view.store %c0_i64, %geometry_view[2] : i64, view<13xi64>
  view.store %split_on, %geometry_view[3] : i64, view<13xi64>
  view.store %recurrent_origin, %geometry_view[4] : i64, view<13xi64>
  view.store %row_stride, %geometry_view[5] : i64, view<13xi64>
  view.store %recurrent_bytes, %geometry_view[6] : i64, view<13xi64>
  view.store %slots, %geometry_view[7] : i64, view<13xi64>
  view.store %c1_i64, %geometry_view[8] : i64, view<13xi64>
  view.store %c0_i64, %geometry_view[9] : i64, view<13xi64>
  view.store %plane_count, %geometry_view[10] : i64, view<13xi64>
  view.store %plane_bytes, %geometry_view[11] : i64, view<13xi64>
  %block_bytes = scalar.constant 16384 : i64
  view.store %block_bytes, %geometry_view[12] : i64, view<13xi64>
  %draft_origins = scf.if %on -> (buffer) {
    %region_at = index.constant 104 : offset
    %region = buffer.view %geometry[%region_at] : buffer -> view<5xi64>
    %c9_i64 = scalar.constant 9 : i64
    view.store %c9_i64, %region[0] : i64, view<5xi64>
    view.store %c0_i64, %region[1] : i64, view<5xi64>
    view.store %draft_planes, %region[2] : i64, view<5xi64>
    view.store %draft_plane_bytes, %region[3] : i64, view<5xi64>
    view.store %block_bytes, %region[4] : i64, view<5xi64>
    %copy = buffer.alloca<private> align(8) %origin_bytes : buffer
    buffer.copy %origins[%origin], %copy[%origin], %origin_bytes
    scf.yield %copy : buffer
  } else {
    %none_bytes = index.constant 0 : offset
    %none = buffer.alloca<private> align(1) %none_bytes : buffer
    scf.yield %none : buffer
  }
  func.return %allocations, %views, %origins, %draft_origins, %geometry : buffer, buffer, buffer, buffer, buffer
}
"""


_CONTROL_SPEC_IMPORTS = """func.decl import("runner") @execute_4(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer) -> (i64)
func.decl import("runner") @execute_5(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer) -> (i64)
func.decl import("runner") @execute_7(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer, %b5: hal.buffer, %b6: hal.buffer) -> (i64)
"""

_CONTROL_SPEC_WRAPPERS = """
func.def public @encode_epoch(%control_state: buffer, %plan: buffer, %inputs: buffer, %media_plan: buffer, %first_count: i32, %next_count: i32, %input_count: i32, %first_media_count: i32, %next_media_count: i32, %first_packet_capacity: i32, %next_packet_capacity: i32, %eos: i32, %metadata: buffer, %tokens: buffer) -> (i32, buffer, i64, i64) {
  %count, %state, %transfer_begin, %transfer_length = func.call @plain_encode_epoch(%control_state, %plan, %inputs, %media_plan, %first_count, %next_count, %input_count, %first_media_count, %next_media_count, %first_packet_capacity, %next_packet_capacity, %eos, %metadata, %tokens) : (buffer, buffer, buffer, buffer, i32, i32, i32, i32, i32, i32, i32, i32, buffer, buffer) -> (i32, buffer, i64, i64)
  %origin = index.constant 0 : offset
  %header_bytes = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %bounded_first = scalar.assume %first_count [range(%first_count, 0, 8192)] : i32
  %spans = index.cast %bounded_first : i32 to index
  %plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
  %requests = buffer.view %plan[%origin] : buffer -> view<[%plan_extent]x9xi32>
  %device = buffer.view %metadata[%header_bytes] : buffer -> view<[%spans]x9xi32>
  %media_total = scalar.addi %first_media_count, %next_media_count : i32
  %mixed = scalar.cmpi ne, %media_total, %c0_i32 : i32
  %one = index.constant 1 : index
  %eight = index.constant 8 : index
  %stride = scf.select %mixed, %eight, %one : index
  %token_bytes = buffer.length %tokens
  %token_words = index.cast %token_bytes : offset to index
  %four = index.constant 4 : index
  %words = index.div %token_words, %four : index
  %flat = buffer.view %tokens[%origin] : buffer -> view<[%words]xi32>
  %c1_i32 = scalar.constant 1 : i32
  %try = scalar.cmpi ugt, %first_count, %c1_i32 : i32
  %flag_value = scf.if %try -> (i32) {
    %nspec = scf.for %g = [%c0 to %spans step %c1](%acc = %c0_i32 : i32) -> (i32) {
      %gr = index.assume %g [lt(%g, %plan_extent)] : index
      %gcredit = view.load %requests[%gr, 7] : view<[%plan_extent]x9xi32> -> i32
      %gspec = scalar.cmpi ne, %gcredit, %c0_i32 : i32
      %gadd = scalar.extui %gspec : i1 to i32
      %gnext = scalar.addi %acc, %gadd : i32
      scf.yield %gnext : i32
    }
    %all_spec = scalar.cmpi eq, %nspec, %first_count : i32
    %gate_limit = scalar.constant 32 : i32
    %few_spans = scalar.cmpi ult, %first_count, %gate_limit : i32
    %gate_max = scalar.constant 31 : i32
    %gate_slot32 = scalar.minui %first_count, %gate_max : i32
    %gate_slot_b = scalar.assume %gate_slot32 [range(%gate_slot32, 0, 31)] : i32
    %gate_slot = index.cast %gate_slot_b : i32 to index
    %gate_at = index.constant 1700 : offset
    %gate_view = buffer.view %control_state[%gate_at] : buffer -> view<32xi32>
    %gate_raw = view.load %gate_view[%gate_slot] : view<32xi32> -> i32
    %gate_shape = scf.select %few_spans, %gate_raw, %c0_i32 : i32
    %has_shape = scalar.cmpi ne, %gate_shape, %c0_i32 : i32
    %gate_in = scalar.andi %all_spec, %has_shape : i1
    %flag_sel = scf.select %gate_in, %gate_shape, %c0_i32 : i32
    scf.yield %flag_sel : i32
  } else {
    scf.yield %c0_i32 : i32
  }
  %gate = scalar.cmpi ne, %flag_value, %c0_i32 : i32
  %flag_at = index.constant 1676 : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  view.store %flag_value, %flag_view[0] : i32, view<1xi32>
  %total = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0_i32 : i32) -> (i32) {
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %length_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length = scf.select %gate, %c1_i32, %length_host : i32
    scf.if %gate {
      view.store %length, %device[%s, 0] : i32, view<[%spans]x9xi32>
    }
    %flags = view.load %requests[%sr, 6] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
    %not_gate = scalar.cmpi eq, %flag_value, %c0_i32 : i32
    %mark_spec = scalar.andi %speculative, %not_gate : i1
    %credit_bits = scalar.shli %credit, %c8_i32 : i32
    %spec_bits = scalar.ori %credit_bits, %c4_i32 : i32
    %marked = scalar.ori %flags, %spec_bits : i32
    %device_flags = scf.select %mark_spec, %marked, %flags : i32
    view.store %cursor, %device[%s, 4] : i32, view<[%spans]x9xi32>
    view.store %device_flags, %device[%s, 6] : i32, view<[%spans]x9xi32>
    %next = scalar.addi %cursor, %length : i32
    scf.yield %next : i32
  }
  scf.for %r = [%c0 to %spans step %c1] {
    %last = index.sub %spans, %c1 : index
    %s = index.sub %last, %r : index
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %sd = index.assume %s [lt(%s, %spans)] : index
    %length32_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length32 = scf.select %gate, %c1_i32, %length32_host : i32
    %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
    %known32 = view.load %requests[%sr, 5] : view<[%plan_extent]x9xi32> -> i32
    %cursor32 = view.load %device[%sd, 4] : view<[%spans]x9xi32> -> i32
    %moves = scalar.cmpi ne, %begin32, %cursor32 : i32
    %length_b = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
    %known_b = scalar.assume %known32 [range(%known32, 0, 8192)] : i32
    %begin_b = scalar.assume %begin32 [range(%begin32, 0, 8191)] : i32
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %length = index.cast %length_b : i32 to index
    %known = index.cast %known_b : i32 to index
    %begin = index.cast %begin_b : i32 to index
    %cursor = index.cast %cursor_b : i32 to index
    scf.if %moves {
      scf.for %q = [%c0 to %known step %c1] {
        %kl = index.sub %known, %c1 : index
        %j = index.sub %kl, %q : index
        %src_t = index.add %begin, %j : index
        %dst_t = index.add %cursor, %j : index
        %src0 = index.mul %src_t, %stride : index
        %dst0 = index.mul %dst_t, %stride : index
        scf.for %w = [%c0 to %stride step %c1] {
          %si = index.add %src0, %w : index
          %di = index.add %dst0, %w : index
          %sb = index.assume %si [lt(%si, %words)] : index
          %db = index.assume %di [lt(%di, %words)] : index
          %v = view.load %flat[%sb] : view<[%words]xi32> -> i32
          view.store %v, %flat[%db] : i32, view<[%words]xi32>
        }
      }
    }
    scf.for %j = [%known to %length step %c1] {
      %dst_t = index.add %cursor, %j : index
      %dst0 = index.mul %dst_t, %stride : index
      scf.for %w = [%c0 to %stride step %c1] {
        %di = index.add %dst0, %w : index
        %db = index.assume %di [lt(%di, %words)] : index
        view.store %c0_i32, %flat[%db] : i32, view<[%words]xi32>
      }
    }
  }
  %total64 = scalar.extui %total : i32 to i64
  %stride_index = index.cast %stride : index to i64
  %word64 = scalar.constant 4 : i64
  %record_bytes = scalar.muli %stride_index, %word64 : i64
  %input_length = scalar.muli %total64, %record_bytes : i64
  %transfer_offset = index.constant 8 : offset
  %transfers = buffer.view %state[%transfer_offset] : buffer -> view<2x5xi64>
  view.store %input_length, %transfers[1, 4] : i64, view<2x5xi64>
  func.return %count, %state, %transfer_begin, %transfer_length : i32, buffer, i64, i64
}

func.def public @publish_epoch(%control_state: buffer, %plan: buffer, %media_plan: buffer, %count: i32, %first_media_count: i32, %next_media_count: i32, %verified: i32, %continues: i32, %next_speculative: buffer, %ordinary: buffer, %first_results: buffer, %next_results: buffer, %progress: buffer) {
  func.call @plain_publish_epoch(%control_state, %plan, %media_plan, %count, %first_media_count, %next_media_count, %verified, %continues, %next_speculative, %ordinary, %first_results, %next_results, %progress) : (buffer, buffer, buffer, i32, i32, i32, i32, i32, buffer, buffer, buffer, buffer, buffer)
  %c0_i32 = scalar.constant 0 : i32
  %verifies = scalar.cmpi ne, %verified, %c0_i32 : i32
  scf.if %verifies {
    %origin = index.constant 0 : offset
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %step = scalar.constant 2 : i32
    %bounded_count = scalar.assume %count [range(%count, 1, 2147483647)] : i32
    %length = index.cast %bounded_count : i32 to index
    %plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
    %requests = buffer.view %plan[%origin] : buffer -> view<[%plan_extent]x9xi32>
    %records = buffer.view %first_results[%origin] : buffer -> view<[%length]x10xi32>
    %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
    scf.for %i = [%c0 to %length step %c1] {
      %ir = index.assume %i [lt(%i, %plan_extent)] : index
      %consumed = view.load %records[%i, 0] : view<[%length]x10xi32> -> i32
      %outputs = view.load %records[%i, 1] : view<[%length]x10xi32> -> i32
      %credit = view.load %requests[%ir, 7] : view<[%plan_extent]x9xi32> -> i32
      %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
      %known = scf.select %speculative, %c0_i32, %consumed : i32
      %verification = scalar.extui %speculative : i1 to i32
      view.store %consumed, %published[%i, 0] : i32, view<[%length]x21xi32>
      view.store %known, %published[%i, 1] : i32, view<[%length]x21xi32>
      view.store %outputs, %published[%i, 2] : i32, view<[%length]x21xi32>
      view.store %verification, %published[%i, 3] : i32, view<[%length]x21xi32>
      scf.if %speculative {
        %source = view.load %requests[%ir, 2] : view<[%plan_extent]x9xi32> -> i32
        %advance = scalar.muli %consumed, %step : i32
        %next_source = scalar.addi %source, %advance : i32
        view.store %next_source, %published[%i, 4] : i32, view<[%length]x21xi32>
      }
      %bounded_outputs = scalar.assume %outputs [range(%outputs, 0, 8)] : i32
      %output_extent = index.cast %bounded_outputs : i32 to index
      %c2 = index.constant 2 : index
      %c5 = index.constant 5 : index
      scf.for %j = [%c0 to %output_extent step %c1] {
        %src = index.add %j, %c2 : index
        %dst = index.add %j, %c5 : index
        %src_b = index.assume %src [range(%src, 2, 9)] : index
        %dst_b = index.assume %dst [range(%dst, 5, 12)] : index
        %token = view.load %records[%i, %src_b] : view<[%length]x10xi32> -> i32
        view.store %token, %published[%i, %dst_b] : i32, view<[%length]x21xi32>
      }
    }
  }
  func.return
}

func.def public @epoch(%control_state: buffer, %shape: i32, %proposes: i32, %verifies: i32, %span_count: i32, %output_count: i32, %epoch_count: i32, %continuation_shape: i32, %continuation_proposes: i32, %first_media_count: i32, %next_media_count: i32, %residual: hal.buffer, %metadata: hal.buffer, %origins: hal.buffer, %state: hal.buffer, %tokens: hal.buffer, %outputs: hal.buffer, %carry: hal.buffer, %committed: hal.buffer, %results: hal.buffer, %draft_cache: hal.buffer, %draft_origins: hal.buffer, %next_results: hal.buffer) -> (i64) {
  %c0 = scalar.constant 0 : i32
  %c1 = scalar.constant 1 : i32
  %c0_i64 = scalar.constant 0 : i64
  %bounded_shape = scalar.assume %shape [range(%shape, 0, 127)] : i32
  %shape_index = index.cast %bounded_shape : i32 to index
  %advance_at = index.constant 128 : offset
  %verify_at = index.constant 640 : offset
  %warm_at = index.constant 1152 : offset
  %draft_at = index.constant 1664 : offset
  %advance_view = buffer.view %control_state[%advance_at] : buffer -> view<128xi32>
  %verify_view = buffer.view %control_state[%verify_at] : buffer -> view<128xi32>
  %warm_view = buffer.view %control_state[%warm_at] : buffer -> view<128xi32>
  %draft_view = buffer.view %control_state[%draft_at] : buffer -> view<3xi32>
  %propose_first = view.load %draft_view[0] : view<3xi32> -> i32
  %levels = view.load %draft_view[1] : view<3xi32> -> i32
  %kind = view.load %draft_view[2] : view<3xi32> -> i32
  %drafting = scalar.cmpi ne, %kind, %c0 : i32
  %propose = scalar.cmpi ne, %proposes, %c0 : i32
  %verify = scalar.cmpi ne, %verifies, %c0 : i32
  %flag_at = index.constant 1676 : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  %flag = view.load %flag_view[0] : view<1xi32> -> i32
  %gated = scalar.cmpi ne, %flag, %c0 : i32
  %flag_low_mask = scalar.constant 255 : i32
  %flag_low = scalar.andi %flag, %flag_low_mask : i32
  %flag_nz = scf.select %gated, %flag_low, %c1 : i32
  %flag_idx = scalar.subi %flag_nz, %c1 : i32
  %flag_shift = scalar.constant 8 : i32
  %gated_stage = scalar.shrui %flag, %flag_shift : i32
  %flag_b = scalar.assume %flag_idx [range(%flag_idx, 0, 127)] : i32
  %flag_shape = index.cast %flag_b : i32 to index
  %eff_index = scf.select %gated, %flag_shape, %shape_index : index
  %ungated = scalar.cmpi eq, %flag, %c0 : i32
  %propose_on = scalar.andi %propose, %ungated : i1
  scf.if %propose_on {
    %bits = scalar.constant 32 : i32
    %below = scalar.subi %span_count, %c1 : i32
    %leading = scalar.ctlzi %below : i32
    %level = scalar.subi %bits, %leading : i32
    %last = scalar.subi %levels, %c1 : i32
    %chosen = scalar.minui %level, %last : i32
    %pstage = scalar.addi %propose_first, %chosen : i32
    %proposal = func.call @execute_4(%pstage, %metadata, %tokens, %draft_origins, %draft_cache) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
  }
  %target = scf.if %verify -> (i64) {
    %vstage_table = view.load %verify_view[%eff_index] : view<128xi32> -> i32
    %vstage = scf.select %gated, %gated_stage, %vstage_table : i32
    %completion = func.call @execute_7(%vstage, %residual, %metadata, %origins, %state, %tokens, %committed, %results) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    %count64 = scalar.extui %span_count : i32 to i64
    %record = scalar.constant 40 : i64
    %length = scalar.muli %count64, %record : i64
    %feedback = func.call @feedback(%c1, %results, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
    scf.yield %completion : i64
  } else {
    %astage = view.load %advance_view[%eff_index] : view<128xi32> -> i32
    %completion = func.call @execute_6(%astage, %residual, %metadata, %origins, %state, %tokens, %outputs) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    %count64 = scalar.extui %span_count : i32 to i64
    %word = scalar.constant 4 : i64
    %length = scalar.muli %count64, %word : i64
    %feedback = func.call @feedback(%c0, %outputs, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
    scf.yield %completion : i64
  }
  %done = scf.if %drafting -> (i64) {
    %accepted = scf.if %verify -> (hal.buffer) {
      scf.yield %committed : hal.buffer
    } else {
      scf.yield %metadata : hal.buffer
    }
    %wstage = view.load %warm_view[%eff_index] : view<128xi32> -> i32
    %warm = func.call @execute_5(%wstage, %residual, %accepted, %draft_origins, %draft_cache, %tokens) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    scf.yield %warm : i64
  } else {
    scf.yield %target : i64
  }
  func.return %done : i64
}
"""


def patch_control_dspark(text: str) -> str:
    """Wrap Inkling's control.loom with speculative @encode_epoch, @publish_epoch, @epoch."""
    imp_anchor = 'func.decl import("runner") @execute_6(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer, %b5: hal.buffer) -> (i64)\n'
    assert text.count(imp_anchor) == 1
    text = text.replace(imp_anchor, imp_anchor + _CONTROL_SPEC_IMPORTS, 1)

    # Control state sampling table moves from offset 640 -> 2048 (matching prepare.loom).
    assert text.count(" = index.constant 640 : offset") == 3, "expected 3 sampling header offsets"
    text = text.replace(" = index.constant 640 : offset", " = index.constant 2048 : offset")

    # Rename plain encode_epoch / publish_epoch so speculative wrappers delegate to them.
    assert text.count("func.def public @encode_epoch(") == 1
    assert text.count("func.def public @publish_epoch(") == 1
    text = text.replace("func.def public @encode_epoch(", "func.def @plain_encode_epoch(", 1)
    text = text.replace("func.def public @publish_epoch(", "func.def @plain_publish_epoch(", 1)

    # Replace @epoch with speculative wrappers + @epoch.
    epoch_pat = re.compile(
        r"func\.def public @epoch\([^\n]+\n.*?\n\}\n",
        flags=re.S,
    )
    text, n = epoch_pat.subn(_CONTROL_SPEC_WRAPPERS.lstrip("\n") + "\n", text, count=1)
    assert n == 1, "expected 1 @epoch definition"

    # Update @llm_storage decl + call in @resize_state to pass %drafting64 from state[1672].
    old_decl = "func.decl @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64) -> (buffer, buffer, buffer, buffer, buffer)"
    new_decl = "func.decl @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64, %drafting: i64) -> (buffer, buffer, buffer, buffer, buffer)"
    assert text.count(old_decl) == 1
    text = text.replace(old_decl, new_decl, 1)

    old_call = "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins) : (i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)"
    new_call = (
        "  %kind_at = index.constant 1672 : offset\n"
        "  %kind_v = buffer.view %state[%kind_at] : buffer -> view<1xi32>\n"
        "  %kind32 = view.load %kind_v[0] : view<1xi32> -> i32\n"
        "  %drafting64 = scalar.extui %kind32 : i32 to i64\n"
        "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins, %drafting64) : (i64, i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)"
    )
    assert text.count(old_call) == 1
    text = text.replace(old_call, new_call, 1)
    return text


# ---------------------------------------------------------------------------
# DSpark GPU kernels + command.def generators for llm.loom
# ---------------------------------------------------------------------------

_DSPARK_CONFIG_DECLS = """
config.decl @llm.width : %value: index where [range(%value, 1, 8)]
config.decl @llm.gated : i64
config.decl @llm.epoch_spans : %value: index where [range(%value, 1, 8192)]
config.decl @llm.epoch_tokens : %value: index where [range(%value, 1, 8192)]
config.decl @llm.prep_blocks : i64
config.decl @llm.prep_input : i64
"""

_DSPARK_KERNELS = r"""
// ---- Speculative verification & DSpark drafter kernels ----

kernel.def export("g_prep_q8_0_q4_0") @g_prep_q8_0_q4_0(%blocks: index) {
  %c1 = index.constant 1 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rounded = index.add %blocks, %c63 : index
  %groups = index.div %rounded, %c64 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%blocks_arg: index, %in64: i64, %output: buffer, %input: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1073741824)] : index
  %in_bound = scalar.assume %in64 [range(%in64, 0, 9223372036854775807)] : i64
  %in_origin = index.cast %in_bound : i64 to offset
  %c64 = index.constant 64 : index
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %g = index.add %base, %lane : index
  %live = index.cmp ult, %g, %blocks : index
  scf.if %live {
    %ib = index.constant 34 : offset
    %two_i = index.constant 2 : offset
    %io = index.scale %g, %ib : index, offset -> offset
    %ia = index.add %in_origin, %io : offset
    %sd_view = buffer.view %input[%ia] : buffer -> view<1xf16>
    %sdh = view.load %sd_view[0] : view<1xf16> -> f16
    %sd = scalar.extf %sdh : f16 to f32
    %qa = index.add %ia, %two_i : offset
    %sq_view = buffer.view %input[%qa] : buffer -> view<32xi8>
    %sq = vector.load %sq_view[0] : view<32xi8> -> vector<32xi8>
    %sqf = vector.sitofp %sq : vector<32xi8> to vector<32xf32>
    %sdv = vector.splat %sd : vector<32xf32>
    %v = vector.mulf %sqf, %sdv : vector<32xf32>
    %zero_f = scalar.constant 0.0 : f32
    %ninf = scalar.constant -3.0e38 : f32
    %pinf = scalar.constant 3.0e38 : f32
    %cm8 = scalar.constant -8.0 : f32
    %one_f = scalar.constant 1.0 : f32
    %hi = vector.reduce<maxnumf> %v, %ninf : vector<32xf32>, f32
    %lo = vector.reduce<minnumf> %v, %pinf : vector<32xf32>, f32
    %nlo = scalar.negf %lo : f32
    %pos = scalar.cmpf oge, %hi, %nlo : f32
    %m = scf.select %pos, %hi, %lo : f32
    %d = scalar.divf %m, %cm8 : f32
    %nz = scalar.cmpf one, %d, %zero_f : f32
    %inv = scalar.divf %one_f, %d : f32
    %id = scf.select %nz, %inv, %zero_f : f32
    %idv = vector.splat %id : vector<32xf32>
    %off = vector.splatf 8.5 : vector<32xf32>
    %top = vector.splatf 15.0 : vector<32xf32>
    %sv = vector.fmaf %v, %idv, %off : vector<32xf32>
    %cl = vector.minnumf %sv, %top : vector<32xf32>
    %q = vector.fptosi %cl : vector<32xf32> to vector<32xi8>
    %ql = vector.slice %q[0] : vector<32xi8> -> vector<16xi8>
    %qh = vector.slice %q[16] : vector<32xi8> -> vector<16xi8>
    %four = vector.splati 4 : vector<16xi8>
    %qhs = vector.shli %qh, %four : vector<16xi8>
    %packed = vector.ori %ql, %qhs : vector<16xi8>
    %packed4 = vector.bitcast %packed : vector<16xi8> to vector<4xi32>
    %dh = scalar.fptrunc %d : f32 to f16
    %q_base = index.constant 0 : offset
    %d_base = index.constant 231579648 : offset
    %gb = index.assume %g [range(%g, 0, 14473727)] : index
    %c0_i = index.constant 0 : index
    %q_all = buffer.view %output[%q_base] : buffer -> view<14473728x4xi32>
    %d_all = buffer.view %output[%d_base] : buffer -> view<14473728xf16>
    vector.store %packed4, %q_all[%gb, %c0_i] : vector<4xi32>, view<14473728x4xi32>
    view.store %dh, %d_all[%gb] : f16, view<14473728xf16>
  }
  kernel.return
}

command.def public target(@device) @llm_prep_q8_0_q4_0() launch(%output: buffer, %input: buffer) {
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  kernel.launch @g_prep_q8_0_q4_0[%blocks](%blocks, %in64, %output, %input) : [index](index, i64, buffer, buffer)
  command.return
}

kernel.def export("g_spec_rows") @g_spec_rows() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%spans_arg: index, %rows_arg: index, %metadata: buffer, %rows_meta: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %out = buffer.view %rows_meta[%c16] : buffer -> view<[%rows]x9xi32>
  %c9 = index.constant 9 : index
  %total = scf.for %s = [%c0 to %count step %c1](%r = %c0 : index) -> (index) {
    %sb = index.assume %s [range(%s, 0, 2147483646), lt(%s, %spans)] : index
    %length32 = view.load %records[%sb, 0] : view<[%spans]x9xi32> -> i32
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %sel_bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %sel_bits, %c0_i32 : i32
    %spec_bits = scalar.andi %flags, %c4_i32 : i32
    %spec = scalar.cmpi ne, %spec_bits, %c0_i32 : i32
    %length_b = scalar.assume %length32 [range(%length32, 0, 8)] : i32
    %spec_rows32 = scf.select %spec, %length_b, %c1_i32 : i32
    %n32 = scf.select %selected, %spec_rows32, %c0_i32 : i32
    %n_b = scalar.assume %n32 [range(%n32, 0, 8)] : i32
    %n = index.cast %n_b : i32 to index
    %next = scf.for %j = [%c0 to %n step %c1](%rr = %r : index) -> (index) {
      %fits = index.cmp ult, %rr, %rows : index
      scf.if %fits {
        %rb = index.assume %rr [range(%rr, 0, 2147483646), lt(%rr, %rows)] : index
        scf.for %w = [%c0 to %c9 step %c1] {
          %wb = index.assume %w [range(%w, 0, 8)] : index
          %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
          view.store %v, %out[%rb, %wb] : i32, view<[%rows]x9xi32>
        }
        %j32 = index.cast %j : index to i32
        %len_j = scalar.addi %j32, %c1_i32 : i32
        %len_row = scf.select %spec, %len_j, %length32 : i32
        view.store %len_row, %out[%rb, 0] : i32, view<[%rows]x9xi32>
        view.store %c1_i32, %out[%rb, 6] : i32, view<[%rows]x9xi32>
      }
      %rn = index.add %rr, %c1 : index
      scf.yield %rn : index
    }
    scf.yield %next : index
  }
  %bounded_total = index.min %total, %rows : index
  %total32 = index.cast %bounded_total : index to i32
  %oheader = buffer.view %rows_meta[%zero] : buffer -> view<4xi32>
  view.store %total32, %oheader[0] : i32, view<4xi32>
  view.store %eos, %oheader[1] : i32, view<4xi32>
  view.store %c0_i32, %oheader[2] : i32, view<4xi32>
  view.store %c0_i32, %oheader[3] : i32, view<4xi32>
  kernel.return
}

kernel.def export("g_spec_accept") @g_spec_accept() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%spans_arg: index, %rows_arg: index, %tokens_arg: index, %metadata: buffer, %ids: buffer, %predictions: buffer, %results: buffer, %committed: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c10 = index.constant 10 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c255_i32 = scalar.constant 255 : i32
  %true = scalar.constant true : i1
  %false = scalar.constant false : i1
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1 : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %crecords = buffer.view %committed[%c16] : buffer -> view<[%spans]x9xi32>
  %pred = buffer.view %predictions[%zero] : buffer -> view<[%rows]xi32>
  %out = buffer.view %results[%zero] : buffer -> view<[%spans]x10xi32>
  %row_end = scf.for %s = [%c0 to %count step %c1](%r = %c0 : index) -> (index) {
    %sb = index.assume %s [range(%s, 0, 2147483646), lt(%s, %spans)] : index
    %length32 = view.load %records[%sb, 0] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%sb, 4] : view<[%spans]x9xi32> -> i32
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %sel_bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %sel_bits, %c0_i32 : i32
    %spec_bits = scalar.andi %flags, %c4_i32 : i32
    %spec = scalar.cmpi ne, %spec_bits, %c0_i32 : i32
    %credit_raw = scalar.shrui %flags, %c8_i32 : i32
    %credit32 = scalar.andi %credit_raw, %c255_i32 : i32
    scf.for %w = [%c2 to %c10 step %c1] {
      %wb = index.assume %w [range(%w, 2, 9)] : index
      view.store %c0_i32, %out[%sb, %wb] : i32, view<[%spans]x10xi32>
    }
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %cursor = index.cast %cursor_b : i32 to index
    %consumed, %emitted, %used = scf.if %spec -> (i32, i32, index) {
      %length_b = scalar.assume %length32 [range(%length32, 1, 8)] : i32
      %width = index.cast %length_b : i32 to index
      %credit_b = scalar.assume %credit32 [range(%credit32, 0, 255)] : i32
      %credit = index.cast %credit_b : i32 to index
      %last_draft = index.sub %width, %c1 : index
      %n, %live_end = scf.for %j = [%c0 to %width step %c1](%cnt = %c0 : index, %live = %true : i1) -> (index, i1) {
        %credited = index.cmp ult, %j, %credit : index
        %emit = scalar.andi %live, %credited : i1
        %rj = index.add %r, %j : index
        %row_ok = index.cmp ult, %rj, %rows : index
        %emit_ok = scalar.andi %emit, %row_ok : i1
        %nc, %nl = scf.if %emit_ok -> (index, i1) {
          %rb = index.assume %rj [range(%rj, 0, 2147483646), lt(%rj, %rows)] : index
          %p = view.load %pred[%rb] : view<[%rows]xi32> -> i32
          %slot = index.add %j, %c2 : index
          %slot_b = index.assume %slot [range(%slot, 2, 9)] : index
          view.store %p, %out[%sb, %slot_b] : i32, view<[%spans]x10xi32>
          %cnt1 = index.add %cnt, %c1 : index
          %has_draft = index.cmp ult, %j, %last_draft : index
          %match = scf.if %has_draft -> (i1) {
            %rel = index.add %j, %c1 : index
            %cand_i = index.add %cursor, %rel : index
            %cand_e = index.mul %cand_i, %id_stride : index
            %cand_b = index.assume %cand_e [lt(%cand_e, %n_ids)] : index
            %cand = view.load %ids_flat[%cand_b] : view<[%n_ids]xi32> -> i32
            %same = scalar.cmpi eq, %p, %cand : i32
            scf.yield %same : i1
          } else {
            scf.yield %false : i1
          }
          %not_eos = scalar.cmpi ne, %p, %eos : i32
          %go = scalar.andi %match, %not_eos : i1
          scf.yield %cnt1, %go : index, i1
        } else {
          scf.yield %cnt, %false : index, i1
        }
        scf.yield %nc, %nl : index, i1
      }
      %n32 = index.cast %n : index to i32
      scf.yield %n32, %n32, %width : i32, i32, index
    } else {
      %em, %us = scf.if %selected -> (i32, index) {
        %rok = index.cmp ult, %r, %rows : index
        scf.if %rok {
          %rb = index.assume %r [range(%r, 0, 2147483646), lt(%r, %rows)] : index
          %p = view.load %pred[%rb] : view<[%rows]xi32> -> i32
          view.store %p, %out[%sb, 2] : i32, view<[%spans]x10xi32>
        }
        scf.yield %c1_i32, %c1 : i32, index
      } else {
        scf.yield %c0_i32, %c0 : i32, index
      }
      scf.yield %length32, %em, %us : i32, i32, index
    }
    view.store %consumed, %out[%sb, 0] : i32, view<[%spans]x10xi32>
    view.store %emitted, %out[%sb, 1] : i32, view<[%spans]x10xi32>
    scf.for %w = [%c0 to %c9 step %c1] {
      %wb = index.assume %w [range(%w, 0, 8)] : index
      %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
      view.store %v, %crecords[%sb, %wb] : i32, view<[%spans]x9xi32>
    }
    view.store %consumed, %crecords[%sb, 0] : i32, view<[%spans]x9xi32>
    %rn = index.add %r, %used : index
    scf.yield %rn : index
  }
  %cheader = buffer.view %committed[%zero] : buffer -> view<4xi32>
  %count_c32 = index.cast %count : index to i32
  view.store %count_c32, %cheader[0] : i32, view<4xi32>
  view.store %eos, %cheader[1] : i32, view<4xi32>
  view.store %mixed_word, %cheader[2] : i32, view<4xi32>
  view.store %c0_i32, %cheader[3] : i32, view<4xi32>
  kernel.return
}

// Stage raw sconv/swa buffers during llm_verify (2 <= t <= 64) and batch-commit using %committed.
kernel.def export("g_ink_stage_f32") @g_ink_stage_f32(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %width_arg: index, %slot_arg: index, %src: buffer, %dst: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 1024, 2304)] : index
  %slot = index.assume %slot_arg [range(%slot_arg, 0, 95)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %sv = buffer.view %src[%zero] : buffer -> view<[%tokens]x[%width]xf32>
  %dv = buffer.view %dst[%zero] : buffer -> view<96x[%tokens]x[%width]xf32>
  scf.for %c = [%lane to %width step %c256] {
    %cb = index.assume %c [lt(%c, %width)] : index
    %v = view.load %sv[%tok, %cb] : view<[%tokens]x[%width]xf32> -> f32
    view.store %v, %dv[%slot, %tok, %cb] : f32, view<96x[%tokens]x[%width]xf32>
  }
  kernel.return
}

// Save existing SWA ring entries before g_ink_kv_prep_store_swa_ring overwrites them (for exact rollback on reject).
kernel.def export("g_ink_swa_save_ring") @g_ink_swa_save_ring(%spans: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %swa_idx_arg: index, %swa_off_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %saved: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %swa = index.assume %swa_idx_arg [range(%swa_idx_arg, 0, 39)] : index
  %swa_off_b = index.assume %swa_off_arg [range(%swa_off_arg, 0, 1073741824)] : index
  %swa_off = index.cast %swa_off_b : index to offset
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %kh_raw = kernel.subgroup.id : index
  %kh = index.assume %kh_raw [range(%kh_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cWIN = index.constant 1024 : index
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {
    %ring_rd, %ring_wr, %ring_fr = func.call @g_gdn_slots(%origins, %row, %start32, %swa_off) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %start_b = scalar.assume %start32 [range(%start32, 0, 1048575)] : i32
    %start = index.cast %start_b : i32 to index
    %plane_sz = index.constant 2097152 : offset
    %wr_v = index.add %ring_wr, %plane_sz : offset
    %sv = buffer.view %saved[%zero] : buffer -> view<40x[%tokens]x2x8x32x4xf16>
    %dkv = buffer.view %arena[%ring_wr] : buffer -> view<1024x8x32x4xf16>
    %dvv = buffer.view %arena[%wr_v] : buffer -> view<1024x8x32x4xf16>
    scf.for %idx = [%c0 to %length step %c1] {
      %tk_v = index.add %first, %idx : index
      %tk = index.assume %tk_v [lt(%tk_v, %tokens)] : index
      %kp_v = index.add %start, %idx : index
      %kp_b = index.assume %kp_v [range(%kp_v, 0, 1048575)] : index
      %rp_v = index.rem %kp_b, %cWIN : index
      %rp = index.assume %rp_v [range(%rp_v, 0, 1023)] : index
      %k4 = vector.load %dkv[%rp, %kh, %lane, %c0] : view<1024x8x32x4xf16> -> vector<4xf16>
      %v4 = vector.load %dvv[%rp, %kh, %lane, %c0] : view<1024x8x32x4xf16> -> vector<4xf16>
      vector.store %k4, %sv[%swa, %tk, %c0, %kh, %lane, %c0] : vector<4xf16>, view<40x[%tokens]x2x8x32x4xf16>
      vector.store %v4, %sv[%swa, %tk, %c1, %kh, %lane, %c0] : vector<4xf16>, view<40x[%tokens]x2x8x32x4xf16>
    }
  }
  kernel.return
}

// Batch-commit all 96 c1024 sconv streams (48 visits x {k, v}) from staged buffer using %committed.
kernel.def export("g_ink_sconv_commit_batch_c1024") @g_ink_sconv_commit_batch_c1024(%spans: index) {
  %c1 = index.constant 1 : index
  %c96 = index.constant 96 : index
  %c4 = index.constant 4 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c96, %c4) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %committed: buffer, %origins: buffer, %arena: buffer, %staged: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %slot_id = kernel.workgroup.id<y> : index
  %slot = index.assume %slot_id [range(%slot_id, 0, 95)] : index
  %cblk_id = kernel.workgroup.id<z> : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c256 = index.constant 256 : index
  %layer = index.div %slot, %c2 : index
  %is_v = index.rem %slot, %c2 : index
  %off_k = index.constant 0 : offset
  %off_v = index.constant 12288 : offset
  %is_v_b = index.cmp ne, %is_v, %c0 : index
  %chan_off = scf.select %is_v_b, %off_v, %off_k : offset
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%committed, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {
    %cbase = index.mul %cblk_id, %c256 : index
    %ch_raw = index.add %cbase, %lane : index
    %ch = index.assume %ch_raw [range(%ch_raw, 0, 1023)] : index
    %layer_stride = index.constant 79872 : offset
    %layer_bytes0 = index.scale %layer, %layer_stride : index, offset -> offset
    %layer_bytes = index.add %layer_bytes0, %chan_off : offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %old_v = buffer.view %arena[%reader] : buffer -> view<3x1024xf32>
    %new_v = buffer.view %arena[%writer] : buffer -> view<3x1024xf32>
    %cur_v = buffer.view %staged[%zero] : buffer -> view<96x[%tokens]x1024xf32>
    scf.for %h = [%c0 to %c3 step %c1] {
      %hb = index.assume %h [range(%h, 0, 2)] : index
      %need = index.sub %c3, %hb : index
      %from_cur = index.cmp uge, %length, %need : index
      %val = scf.if %from_cur -> (f32) {
        %e0 = index.add %first, %length : index
        %s0 = index.sub %e0, %need : index
        %sb = index.assume %s0 [lt(%s0, %tokens)] : index
        %v = view.load %cur_v[%slot, %sb, %ch] : view<96x[%tokens]x1024xf32> -> f32
        scf.yield %v : f32
      } else {
        %o = scf.if %fresh -> (f32) {
          scf.yield %zero_f : f32
        } else {
          %oi0 = index.add %hb, %length : index
          %oib = index.assume %oi0 [range(%oi0, 0, 2)] : index
          %v = view.load %old_v[%oib, %ch] : view<3x1024xf32> -> f32
          scf.yield %v : f32
        }
        scf.yield %o : f32
      }
      view.store %val, %new_v[%hb, %ch] : f32, view<3x1024xf32>
    }
  }
  kernel.return
}

// Batch-commit all 96 c2304 sconv streams (48 visits x {ap, fp}) from staged buffer using %committed.
kernel.def export("g_ink_sconv_commit_batch_c2304") @g_ink_sconv_commit_batch_c2304(%spans: index) {
  %c1 = index.constant 1 : index
  %c96 = index.constant 96 : index
  %c9 = index.constant 9 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c96, %c9) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %committed: buffer, %origins: buffer, %arena: buffer, %staged: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %slot_id = kernel.workgroup.id<y> : index
  %slot = index.assume %slot_id [range(%slot_id, 0, 95)] : index
  %cblk_id = kernel.workgroup.id<z> : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c256 = index.constant 256 : index
  %layer = index.div %slot, %c2 : index
  %is_fp = index.rem %slot, %c2 : index
  %off_ap = index.constant 24576 : offset
  %off_fp = index.constant 52224 : offset
  %is_fp_b = index.cmp ne, %is_fp, %c0 : index
  %chan_off = scf.select %is_fp_b, %off_fp, %off_ap : offset
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%committed, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {
    %cbase = index.mul %cblk_id, %c256 : index
    %ch_raw = index.add %cbase, %lane : index
    %ch = index.assume %ch_raw [range(%ch_raw, 0, 2303)] : index
    %layer_stride = index.constant 79872 : offset
    %layer_bytes0 = index.scale %layer, %layer_stride : index, offset -> offset
    %layer_bytes = index.add %layer_bytes0, %chan_off : offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %old_v = buffer.view %arena[%reader] : buffer -> view<3x2304xf32>
    %new_v = buffer.view %arena[%writer] : buffer -> view<3x2304xf32>
    %cur_v = buffer.view %staged[%zero] : buffer -> view<96x[%tokens]x2304xf32>
    scf.for %h = [%c0 to %c3 step %c1] {
      %hb = index.assume %h [range(%h, 0, 2)] : index
      %need = index.sub %c3, %hb : index
      %from_cur = index.cmp uge, %length, %need : index
      %val = scf.if %from_cur -> (f32) {
        %e0 = index.add %first, %length : index
        %s0 = index.sub %e0, %need : index
        %sb = index.assume %s0 [lt(%s0, %tokens)] : index
        %v = view.load %cur_v[%slot, %sb, %ch] : view<96x[%tokens]x2304xf32> -> f32
        scf.yield %v : f32
      } else {
        %o = scf.if %fresh -> (f32) {
          scf.yield %zero_f : f32
        } else {
          %oi0 = index.add %hb, %length : index
          %oib = index.assume %oi0 [range(%oi0, 0, 2)] : index
          %v = view.load %old_v[%oib, %ch] : view<3x2304xf32> -> f32
          scf.yield %v : f32
        }
        scf.yield %o : f32
      }
      view.store %val, %new_v[%hb, %ch] : f32, view<3x2304xf32>
    }
  }
  kernel.return
}

// Restore rejected SWA ring entries [consumed..length) across all 40 SWA rings (zero writes when all accepted).
kernel.def export("g_ink_swa_restore_batch") @g_ink_swa_restore_batch(%spans: index) {
  %c40 = index.constant 40 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c40, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %committed: buffer, %origins: buffer, %saved: buffer, %arena: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %swa_id = kernel.workgroup.id<y> : index
  %swa = index.assume %swa_id [range(%swa_id, 0, 39)] : index
  %kh_raw = kernel.subgroup.id : index
  %kh = index.assume %kh_raw [range(%kh_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cWIN = index.constant 1024 : index
  %m_act, %m_len, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %c_act, %c_len, %c_start32, %c_row, %c_first = func.call @g_gdn_span(%committed, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %has_rej = index.cmp ult, %c_len, %m_len : index
  %do_rest = scalar.andi %m_act, %has_rej : i1
  scf.if %do_rest {
    %swa_base = index.constant 3833856 : offset
    %swa_stride = index.constant 4194304 : offset
    %swa_add = index.scale %swa, %swa_stride : index, offset -> offset
    %swa_off = index.add %swa_base, %swa_add : offset
    %ring_rd, %ring_wr, %ring_fr = func.call @g_gdn_slots(%origins, %row, %start32, %swa_off) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %start_b = scalar.assume %start32 [range(%start32, 0, 1048575)] : i32
    %start = index.cast %start_b : i32 to index
    %plane_sz = index.constant 2097152 : offset
    %wr_v = index.add %ring_wr, %plane_sz : offset
    %sv = buffer.view %saved[%zero] : buffer -> view<40x[%tokens]x2x8x32x4xf16>
    %dkv = buffer.view %arena[%ring_wr] : buffer -> view<1024x8x32x4xf16>
    %dvv = buffer.view %arena[%wr_v] : buffer -> view<1024x8x32x4xf16>
    scf.for %idx = [%c_len to %m_len step %c1] {
      %tk_v = index.add %first, %idx : index
      %tk = index.assume %tk_v [lt(%tk_v, %tokens)] : index
      %kp_v = index.add %start, %idx : index
      %kp_b = index.assume %kp_v [range(%kp_v, 0, 1048575)] : index
      %rp_v = index.rem %kp_b, %cWIN : index
      %rp = index.assume %rp_v [range(%rp_v, 0, 1023)] : index
      %k4 = vector.load %sv[%swa, %tk, %c0, %kh, %lane, %c0] : view<40x[%tokens]x2x8x32x4xf16> -> vector<4xf16>
      %v4 = vector.load %sv[%swa, %tk, %c1, %kh, %lane, %c0] : view<40x[%tokens]x2x8x32x4xf16> -> vector<4xf16>
      vector.store %k4, %dkv[%rp, %kh, %lane, %c0] : vector<4xf16>, view<1024x8x32x4xf16>
      vector.store %v4, %dvv[%rp, %kh, %lane, %c0] : vector<4xf16>, view<1024x8x32x4xf16>
    }
  }
  kernel.return
}

// Target feature tap: copies [t, 2304] f32 -> features[t, slot, 2304].
kernel.def export("g_dspark_tap") @g_dspark_tap(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %slot_arg: index, %src: buffer, %features: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %slot = index.assume %slot_arg [range(%slot_arg, 0, 3)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %c2304 = index.constant 2304 : index
  %sv = buffer.view %src[%zero] : buffer -> view<[%tokens]x2304xf32>
  %dv = buffer.view %features[%zero] : buffer -> view<[%tokens]x4x2304xf32>
  scf.for %c = [%lane to %c2304 step %c256] {
    %cb = index.assume %c [range(%c, 0, 2303)] : index
    %v = view.load %sv[%tok, %cb] : view<[%tokens]x2304xf32> -> f32
    view.store %v, %dv[%tok, %slot, %cb] : f32, view<[%tokens]x4x2304xf32>
  }
  kernel.return
}

// DSpark 4-tap fuse + RMSNorm: computes mix_0..2 [t, 2304] and writes f_prev (mix_3) for each span's last token.
kernel.def export("g_dspark_mix") @g_dspark_mix(%tokens: index) {
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c4, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %pool_arg: index, %features: buffer, %alpha: buffer, %n0: buffer, %n1: buffer, %n2: buffer, %n3: buffer, %metadata: buffer, %info: buffer, %mix0: buffer, %mix1: buffer, %mix2: buffer, %dcache: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %m_id = kernel.workgroup.id<y> : index
  %m = index.assume %m_id [range(%m_id, 0, 3)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c256 = index.constant 256 : index
  %c2304 = index.constant 2304 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %inv_d = scalar.constant 0.00043402777777777775 : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %iv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %iv[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  scf.if %live {
    %av = buffer.view %alpha[%zero] : buffer -> view<4x4xf32>
    %a0 = view.load %av[%m, 0] : view<4x4xf32> -> f32
    %a1 = view.load %av[%m, 1] : view<4x4xf32> -> f32
    %a2 = view.load %av[%m, 2] : view<4x4xf32> -> f32
    %a3 = view.load %av[%m, 3] : view<4x4xf32> -> f32
    %fv = buffer.view %features[%zero] : buffer -> view<[%tokens]x4x2304xf32>
    %sq = scf.for %c = [%lane to %c2304 step %c256](%acc = %zero_f : f32) -> (f32) {
      %cb = index.assume %c [range(%c, 0, 2303)] : index
      %f0 = view.load %fv[%tok, 0, %cb] : view<[%tokens]x4x2304xf32> -> f32
      %f1 = view.load %fv[%tok, 1, %cb] : view<[%tokens]x4x2304xf32> -> f32
      %f2 = view.load %fv[%tok, 2, %cb] : view<[%tokens]x4x2304xf32> -> f32
      %f3 = view.load %fv[%tok, 3, %cb] : view<[%tokens]x4x2304xf32> -> f32
      %s0 = scalar.mulf %a0, %f0 : f32
      %s1 = scalar.fmaf %a1, %f1, %s0 : f32
      %s2 = scalar.fmaf %a2, %f2, %s1 : f32
      %s3 = scalar.fmaf %a3, %f3, %s2 : f32
      %na = scalar.fmaf %s3, %s3, %acc : f32
      scf.yield %na : f32
    }
    %tot = kernel.workgroup.reduce<addf> %sq : f32
    %mn = scalar.mulf %tot, %inv_d : f32
    %vr = scalar.addf %mn, %eps : f32
    %ir = scalar.rsqrtf %vr : f32
    %is0 = index.cmp eq, %m, %c0 : index
    %is1 = index.cmp eq, %m, %c1 : index
    %is2 = index.cmp eq, %m, %c2 : index
    %nv0 = buffer.view %n0[%zero] : buffer -> view<2304xf32>
    %nv1 = buffer.view %n1[%zero] : buffer -> view<2304xf32>
    %nv2 = buffer.view %n2[%zero] : buffer -> view<2304xf32>
    %nv3 = buffer.view %n3[%zero] : buffer -> view<2304xf32>
    %ov0 = buffer.view %mix0[%zero] : buffer -> view<[%tokens]x2304xf32>
    %ov1 = buffer.view %mix1[%zero] : buffer -> view<[%tokens]x2304xf32>
    %ov2 = buffer.view %mix2[%zero] : buffer -> view<[%tokens]x2304xf32>
    scf.if %is0 {
      scf.for %c = [%lane to %c2304 step %c256] {
        %cb = index.assume %c [range(%c, 0, 2303)] : index
        %f0 = view.load %fv[%tok, 0, %cb] : view<[%tokens]x4x2304xf32> -> f32
        %f1 = view.load %fv[%tok, 1, %cb] : view<[%tokens]x4x2304xf32> -> f32
        %f2 = view.load %fv[%tok, 2, %cb] : view<[%tokens]x4x2304xf32> -> f32
        %f3 = view.load %fv[%tok, 3, %cb] : view<[%tokens]x4x2304xf32> -> f32
        %s0 = scalar.mulf %a0, %f0 : f32
        %s1 = scalar.fmaf %a1, %f1, %s0 : f32
        %s2 = scalar.fmaf %a2, %f2, %s1 : f32
        %s3 = scalar.fmaf %a3, %f3, %s2 : f32
        %w = view.load %nv0[%cb] : view<2304xf32> -> f32
        %y = scalar.mulf %s3, %ir : f32
        %z = scalar.mulf %y, %w : f32
        view.store %z, %ov0[%tok, %cb] : f32, view<[%tokens]x2304xf32>
      }
    } else {
      scf.if %is1 {
        scf.for %c = [%lane to %c2304 step %c256] {
          %cb = index.assume %c [range(%c, 0, 2303)] : index
          %f0 = view.load %fv[%tok, 0, %cb] : view<[%tokens]x4x2304xf32> -> f32
          %f1 = view.load %fv[%tok, 1, %cb] : view<[%tokens]x4x2304xf32> -> f32
          %f2 = view.load %fv[%tok, 2, %cb] : view<[%tokens]x4x2304xf32> -> f32
          %f3 = view.load %fv[%tok, 3, %cb] : view<[%tokens]x4x2304xf32> -> f32
          %s0 = scalar.mulf %a0, %f0 : f32
          %s1 = scalar.fmaf %a1, %f1, %s0 : f32
          %s2 = scalar.fmaf %a2, %f2, %s1 : f32
          %s3 = scalar.fmaf %a3, %f3, %s2 : f32
          %w = view.load %nv1[%cb] : view<2304xf32> -> f32
          %y = scalar.mulf %s3, %ir : f32
          %z = scalar.mulf %y, %w : f32
          view.store %z, %ov1[%tok, %cb] : f32, view<[%tokens]x2304xf32>
        }
      } else {
        scf.if %is2 {
          scf.for %c = [%lane to %c2304 step %c256] {
            %cb = index.assume %c [range(%c, 0, 2303)] : index
            %f0 = view.load %fv[%tok, 0, %cb] : view<[%tokens]x4x2304xf32> -> f32
            %f1 = view.load %fv[%tok, 1, %cb] : view<[%tokens]x4x2304xf32> -> f32
            %f2 = view.load %fv[%tok, 2, %cb] : view<[%tokens]x4x2304xf32> -> f32
            %f3 = view.load %fv[%tok, 3, %cb] : view<[%tokens]x4x2304xf32> -> f32
            %s0 = scalar.mulf %a0, %f0 : f32
            %s1 = scalar.fmaf %a1, %f1, %s0 : f32
            %s2 = scalar.fmaf %a2, %f2, %s1 : f32
            %s3 = scalar.fmaf %a3, %f3, %s2 : f32
            %w = view.load %nv2[%cb] : view<2304xf32> -> f32
            %y = scalar.mulf %s3, %ir : f32
            %z = scalar.mulf %y, %w : f32
            view.store %z, %ov2[%tok, %cb] : f32, view<[%tokens]x2304xf32>
          }
        } else {
          // m == 3: Markov head context f_prev[row, 2304], written by the last committed token of each span.
          %span32 = view.load %iv[%tok, 3] : view<[%tokens]x4xi32> -> i32
          %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
          %span_v = index.cast %span_b : i32 to index
          %span = index.assume %span_v [lt(%span_v, %spans)] : index
          %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
          %end_tok = index.add %first, %length : index
          %last_tok = index.sub %end_tok, %c1 : index
          %is_last = index.cmp eq, %tok, %last_tok : index
          %write_f = scalar.andi %active, %is_last : i1
          scf.if %write_f {
            %kv_stride = index.constant 6144 : offset
            %fbase = index.scale %pool, %kv_stride : index, offset -> offset
            %row_b = index.assume %row [range(%row, 0, 1023)] : index
            %fv_out = buffer.view %dcache[%fbase] : buffer -> view<1024x2304xf32>
            scf.for %c = [%lane to %c2304 step %c256] {
              %cb = index.assume %c [range(%c, 0, 2303)] : index
              %f0 = view.load %fv[%tok, 0, %cb] : view<[%tokens]x4x2304xf32> -> f32
              %f1 = view.load %fv[%tok, 1, %cb] : view<[%tokens]x4x2304xf32> -> f32
              %f2 = view.load %fv[%tok, 2, %cb] : view<[%tokens]x4x2304xf32> -> f32
              %f3 = view.load %fv[%tok, 3, %cb] : view<[%tokens]x4x2304xf32> -> f32
              %s0 = scalar.mulf %a0, %f0 : f32
              %s1 = scalar.fmaf %a1, %f1, %s0 : f32
              %s2 = scalar.fmaf %a2, %f2, %s1 : f32
              %s3 = scalar.fmaf %a3, %f3, %s2 : f32
              %w = view.load %nv3[%cb] : view<2304xf32> -> f32
              %y = scalar.mulf %s3, %ir : f32
              %z = scalar.mulf %y, %w : f32
              view.store %z, %fv_out[%row_b, %cb] : f32, view<1024x2304xf32>
            }
          }
        }
      }
    }
  }
  kernel.return
}

// DSpark context KV store: head RMSNorm + NeoX RoPE (theta=1e6) + FP8 E4M3FN paged store.
kernel.def export("g_dspark_kv_store") @g_dspark_kv_store(%tokens: index) {
  %ckv = index.constant 8 : index
  %lanes = index.constant 64 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %ckv, %c1) workgroup_size(%lanes, %c1, %c1) : index
} launch(%tokens_arg: index, %layer_arg: index, %context_arg: index, %pool_arg: index, %k: buffer, %v: buffer, %w: buffer, %info: buffer, %origins: buffer, %cache: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 2)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 1048576)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %kh = index.assume %head_id [range(%head_id, 0, 7)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 63)] : index
  %half = index.constant 64 : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %w_view = buffer.view %w[%zero] : buffer -> view<128xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x8x128xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x8x128xf32>
  %a = view.load %k_view[%token, %kh, %lane] : view<[%tokens]x8x128xf32> -> f32
  %b = view.load %k_view[%token, %kh, %lane2] : view<[%tokens]x8x128xf32> -> f32
  %aa = scalar.mulf %a, %a : f32
  %bb = scalar.mulf %b, %b : f32
  %sq = scalar.addf %aa, %bb : f32
  %sum = kernel.workgroup.reduce<addf> %sq : f32
  %inv_d = scalar.constant 0.0078125 : f32
  %mean = scalar.mulf %sum, %inv_d : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %var = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %var : f32
  %wa = view.load %w_view[%lane] : view<128xf32> -> f32
  %wb = view.load %w_view[%lane2] : view<128xf32> -> f32
  %na = scalar.mulf %a, %inv : f32
  %nb = scalar.mulf %b, %inv : f32
  %sa = scalar.mulf %na, %wa : f32
  %sb = scalar.mulf %nb, %wb : f32
  %pair = index.cast %lane : index to i32
  %pair_f = scalar.uitofp %pair : i32 to f32
  %pos_f = scalar.sitofp %position32 : i32 to f32
  // -ln(1e6) / 64 = -0.21586735246819178
  %step = scalar.constant -0.21586735246819178 : f32
  %lf = scalar.mulf %pair_f, %step : f32
  %freq = scalar.expf<afn> %lf : f32
  %angle = scalar.mulf %pos_f, %freq : f32
  %sine = scalar.sinf<afn> %angle : f32
  %cosine = scalar.cosf<afn> %angle : f32
  %ac = scalar.mulf %sa, %cosine : f32
  %bs = scalar.mulf %sb, %sine : f32
  %bc = scalar.mulf %sb, %cosine : f32
  %as = scalar.mulf %sa, %sine : f32
  %ra = scalar.subf %ac, %bs : f32
  %rb = scalar.addf %bc, %as : f32
  %va = view.load %v_view[%token, %kh, %lane] : view<[%tokens]x8x128xf32> -> f32
  %vb = view.load %v_view[%token, %kh, %lane2] : view<[%tokens]x8x128xf32> -> f32
  scf.if %live {
    %c15 = index.constant 15 : index
    %c16 = index.constant 16 : index
    %c2 = index.constant 2 : index
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %row = index.cast %row_b : i32 to index
    %pos_b = scalar.assume %position32 [range(%position32, 0, 1048575)] : i32
    %position = index.cast %pos_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %k_off = func.call @g_kv_offset(%origins, %row, %position, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
    %pos_bytes = index.constant 1024 : offset
    %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
    %v_off = index.add %k_off, %plane_stride : offset
    %k_cache = buffer.view %cache[%k_off] : buffer -> view<8x128xi8>
    %v_cache = buffer.view %cache[%v_off] : buffer -> view<8x128xi8>
    %kv4 = vector.from_elements %ra, %rb, %va, %vb : vector<4xf32>
    %kv8 = func.call @g_f32_to_e4m3fn_v4(%kv4) : (vector<4xf32>) -> (vector<4xi8>)
    %ka8 = vector.extract %kv8[0] : vector<4xi8> -> i8
    %kb8 = vector.extract %kv8[1] : vector<4xi8> -> i8
    %va8 = vector.extract %kv8[2] : vector<4xi8> -> i8
    %vb8 = vector.extract %kv8[3] : vector<4xi8> -> i8
    view.store %ka8, %k_cache[%kh, %lane] : i8, view<8x128xi8>
    view.store %kb8, %k_cache[%kh, %lane2] : i8, view<8x128xi8>
    view.store %va8, %v_cache[%kh, %lane] : i8, view<8x128xi8>
    view.store %vb8, %v_cache[%kh, %lane2] : i8, view<8x128xi8>
  }
  kernel.return
}

// DFlash/DSpark block metadata & anchor/mask (-1) token prep.
kernel.def export("g_dfl_prepare") @g_dfl_prepare() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%spans_arg: index, %blocks_arg: index, %tokens_arg: index, %depth_arg: index, %metadata: buffer, %ids: buffer, %draft: buffer, %draft_ids: buffer, %slots: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %bs_full = index.constant 8 : index
  %width_b = index.add %depth, %c1i : index
  %bs = index.min %bs_full, %width_b : index
  %c0_i32 = scalar.constant 0 : i32
  %c2_i32 = scalar.constant 2 : i32
  %bs_i32 = index.cast %bs : index to i32
  %mask = scalar.constant -1 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %out = buffer.view %draft[%c16] : buffer -> view<[%blocks]x9xi32>
  %dt = index.mul %blocks, %bs : index
  %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%dt]xi32>
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%blocks]xi32>
  scf.for %i = [%c0 to %dt step %c1i] {
    %ib = index.assume %i [range(%i, 0, 8191), lt(%i, %dt)] : index
    view.store %mask, %dids[%ib] : i32, view<[%dt]xi32>
  }
  %total = scf.for %s = [%c0 to %count step %c1i](%k = %c0 : index) -> (index) {
    %sb = index.assume %s [range(%s, 0, 8191), lt(%s, %spans)] : index
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %prop_bits = scalar.andi %flags, %c2_i32 : i32
    %proposes = scalar.cmpi ne, %prop_bits, %c0_i32 : i32
    %fits = index.cmp ult, %k, %blocks : index
    %take = scalar.andi %proposes, %fits : i1
    %kn = scf.if %take -> (index) {
      %kb = index.assume %k [range(%k, 0, 1023), lt(%k, %blocks)] : index
      scf.for %w = [%c0 to %c9 step %c1i] {
        %wb = index.assume %w [range(%w, 0, 8)] : index
        %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
        view.store %v, %out[%kb, %wb] : i32, view<[%blocks]x9xi32>
      }
      %base = index.mul %kb, %bs : index
      %base32 = index.cast %base : index to i32
      view.store %bs_i32, %out[%kb, 0] : i32, view<[%blocks]x9xi32>
      view.store %base32, %out[%kb, 4] : i32, view<[%blocks]x9xi32>
      %cursor32 = view.load %records[%sb, 4] : view<[%spans]x9xi32> -> i32
      view.store %cursor32, %slot_v[%kb] : i32, view<[%blocks]xi32>
      %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
      %cursor = index.cast %cursor_b : i32 to index
      %ae = index.mul %cursor, %id_stride : index
      %ab = index.assume %ae [lt(%ae, %n_ids)] : index
      %anchor = view.load %ids_flat[%ab] : view<[%n_ids]xi32> -> i32
      %db = index.assume %base [range(%base, 0, 8191), lt(%base, %dt)] : index
      view.store %anchor, %dids[%db] : i32, view<[%dt]xi32>
      %k1 = index.add %k, %c1i : index
      scf.yield %k1 : index
    } else {
      scf.yield %k : index
    }
    scf.yield %kn : index
  }
  %dheader = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %tot32 = index.cast %total : index to i32
  view.store %tot32, %dheader[0] : i32, view<4xi32>
  view.store %eos, %dheader[1] : i32, view<4xi32>
  view.store %c0_i32, %dheader[2] : i32, view<4xi32>
  view.store %c0_i32, %dheader[3] : i32, view<4xi32>
  kernel.return
}

kernel.def export("g_dfl_publish") @g_dfl_publish() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%blocks_arg: index, %tokens_arg: index, %depth_arg: index, %metadata: buffer, %draft: buffer, %slots: buffer, %proposals: buffer, %ids: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c0_i32 = scalar.constant 0 : i32
  %mheader = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %mixed_word = view.load %mheader[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %dh = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %count32 = view.load %dh[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 1024)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %blocks : index
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%blocks]xi32>
  %row_cap = index.mul %blocks, %depth : index
  %prop = buffer.view %proposals[%zero] : buffer -> view<[%row_cap]xi32>
  scf.for %b = [%c0 to %count step %c1i] {
    %bb = index.assume %b [range(%b, 0, 1023), lt(%b, %blocks)] : index
    %slot32 = view.load %slot_v[%bb] : view<[%blocks]xi32> -> i32
    %slot_b = scalar.assume %slot32 [range(%slot32, 0, 8191)] : i32
    %slot = index.cast %slot_b : i32 to index
    scf.for %j = [%c0 to %depth step %c1i] {
      %r0 = index.mul %bb, %depth : index
      %r1 = index.add %r0, %j : index
      %rb = index.assume %r1 [range(%r1, 0, 8191), lt(%r1, %row_cap)] : index
      %token = view.load %prop[%rb] : view<[%row_cap]xi32> -> i32
      %t0 = index.add %slot, %j : index
      %t1 = index.add %t0, %c1i : index
      %e0 = index.mul %t1, %id_stride : index
      %e = index.assume %e0 [range(%e0, 0, 65535), lt(%e0, %n_ids)] : index
      view.store %token, %ids_flat[%e] : i32, view<[%n_ids]xi32>
    }
  }
  kernel.return
}

// Overwrite x[tok] with dspark.mask_embed whenever tokens[tok] < 0.
kernel.def export("g_dspark_mask_fill") @g_dspark_mask_fill(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %tokens: buffer, %mask_embed: buffer, %x: buffer) {
  %nt = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %nt)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %c256 = index.constant 256 : index
  %c2304 = index.constant 2304 : index
  %tv = buffer.view %tokens[%zero] : buffer -> view<[%nt]xi32>
  %tid = view.load %tv[%tok] : view<[%nt]xi32> -> i32
  %is_mask = scalar.cmpi slt, %tid, %c0_i32 : i32
  scf.if %is_mask {
    %mv = buffer.view %mask_embed[%zero] : buffer -> view<2304xf32>
    %xv = buffer.view %x[%zero] : buffer -> view<[%nt]x2304xf32>
    scf.for %c = [%lane to %c2304 step %c256] {
      %cb = index.assume %c [range(%c, 0, 2303)] : index
      %v = view.load %mv[%cb] : view<2304xf32> -> f32
      view.store %v, %xv[%tok, %cb] : f32, view<[%nt]x2304xf32>
    }
  }
  kernel.return
}

// DSpark block Q prep (24 Q heads, head_dim=128, NeoX RoPE theta=1e6, q_norm).
kernel.def export("g_dspark_q_prep") @g_dspark_q_prep(%tokens: index) {
  %cq = index.constant 24 : index
  %lanes = index.constant 64 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%lanes, %c1, %c1) : index
} launch(%tokens_arg: index, %q: buffer, %w: buffer, %info: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %qh = index.assume %head_id [range(%head_id, 0, 23)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 63)] : index
  %half = index.constant 64 : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %w_view = buffer.view %w[%zero] : buffer -> view<128xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x24x128xf32>
  %a = view.load %q_view[%token, %qh, %lane] : view<[%tokens]x24x128xf32> -> f32
  %b = view.load %q_view[%token, %qh, %lane2] : view<[%tokens]x24x128xf32> -> f32
  %aa = scalar.mulf %a, %a : f32
  %bb = scalar.mulf %b, %b : f32
  %sq = scalar.addf %aa, %bb : f32
  %sum = kernel.workgroup.reduce<addf> %sq : f32
  %inv_d = scalar.constant 0.0078125 : f32
  %mean = scalar.mulf %sum, %inv_d : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %var = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %var : f32
  %wa = view.load %w_view[%lane] : view<128xf32> -> f32
  %wb = view.load %w_view[%lane2] : view<128xf32> -> f32
  %na = scalar.mulf %a, %inv : f32
  %nb = scalar.mulf %b, %inv : f32
  %sa = scalar.mulf %na, %wa : f32
  %sb = scalar.mulf %nb, %wb : f32
  %pair = index.cast %lane : index to i32
  %pair_f = scalar.uitofp %pair : i32 to f32
  %pos_f = scalar.sitofp %position32 : i32 to f32
  %step = scalar.constant -0.21586735246819178 : f32
  %lf = scalar.mulf %pair_f, %step : f32
  %freq = scalar.expf<afn> %lf : f32
  %angle = scalar.mulf %pos_f, %freq : f32
  %sine = scalar.sinf<afn> %angle : f32
  %cosine = scalar.cosf<afn> %angle : f32
  %ac = scalar.mulf %sa, %cosine : f32
  %bs = scalar.mulf %sb, %sine : f32
  %bc = scalar.mulf %sb, %cosine : f32
  %as = scalar.mulf %sa, %sine : f32
  %ra = scalar.subf %ac, %bs : f32
  %rb = scalar.addf %bc, %as : f32
  view.store %ra, %q_view[%token, %qh, %lane] : f32, view<[%tokens]x24x128xf32>
  view.store %rb, %q_view[%token, %qh, %lane2] : f32, view<[%tokens]x24x128xf32>
  kernel.return
}

// DSpark block K prep (8 KV heads, head_dim=128, NeoX RoPE theta=1e6, k_norm).
kernel.def export("g_dspark_block_kv") @g_dspark_block_kv(%tokens: index) {
  %ckv = index.constant 8 : index
  %lanes = index.constant 64 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %ckv, %c1) workgroup_size(%lanes, %c1, %c1) : index
} launch(%tokens_arg: index, %k: buffer, %w: buffer, %info: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %kh = index.assume %head_id [range(%head_id, 0, 7)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 63)] : index
  %half = index.constant 64 : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %w_view = buffer.view %w[%zero] : buffer -> view<128xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x8x128xf32>
  %a = view.load %k_view[%token, %kh, %lane] : view<[%tokens]x8x128xf32> -> f32
  %b = view.load %k_view[%token, %kh, %lane2] : view<[%tokens]x8x128xf32> -> f32
  %aa = scalar.mulf %a, %a : f32
  %bb = scalar.mulf %b, %b : f32
  %sq = scalar.addf %aa, %bb : f32
  %sum = kernel.workgroup.reduce<addf> %sq : f32
  %inv_d = scalar.constant 0.0078125 : f32
  %mean = scalar.mulf %sum, %inv_d : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %var = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %var : f32
  %wa = view.load %w_view[%lane] : view<128xf32> -> f32
  %wb = view.load %w_view[%lane2] : view<128xf32> -> f32
  %na = scalar.mulf %a, %inv : f32
  %nb = scalar.mulf %b, %inv : f32
  %sa = scalar.mulf %na, %wa : f32
  %sb = scalar.mulf %nb, %wb : f32
  %pair = index.cast %lane : index to i32
  %pair_f = scalar.uitofp %pair : i32 to f32
  %pos_f = scalar.sitofp %position32 : i32 to f32
  %step = scalar.constant -0.21586735246819178 : f32
  %lf = scalar.mulf %pair_f, %step : f32
  %freq = scalar.expf<afn> %lf : f32
  %angle = scalar.mulf %pos_f, %freq : f32
  %sine = scalar.sinf<afn> %angle : f32
  %cosine = scalar.cosf<afn> %angle : f32
  %ac = scalar.mulf %sa, %cosine : f32
  %bs = scalar.mulf %sb, %sine : f32
  %bc = scalar.mulf %sb, %cosine : f32
  %as = scalar.mulf %sa, %sine : f32
  %ra = scalar.subf %ac, %bs : f32
  %rb = scalar.addf %bc, %as : f32
  view.store %ra, %k_view[%token, %kh, %lane] : f32, view<[%tokens]x8x128xf32>
  view.store %rb, %k_view[%token, %kh, %lane2] : f32, view<[%tokens]x8x128xf32>
  kernel.return
}

// DSpark attention: 24 Q / 8 KV heads (gq=3), FP8 E4M3FN context KV (window=1024 on layers 0,2; full on layer 1) + bidirectional F32 block KV.
kernel.def export("g_dspark_attention") @g_dspark_attention(%tokens: index) {
  %cq = index.constant 24 : index
  %c256 = index.constant 256 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %is_local_arg: i32, %context_arg: index, %pool_arg: index, %metadata: buffer, %info: buffer, %q: buffer, %bk: buffer, %bv: buffer, %origins: buffer, %cache: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 2)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 1048576)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %qh = index.assume %head_id [range(%head_id, 0, 23)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %neg = scalar.constant -3.0e38 : f32
  %scale = scalar.constant 0.08838834764831845 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %kh = index.div %qh, %c3 : index
  %khb = index.assume %kh [range(%kh, 0, 7)] : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %acc_bytes = index.constant 4160 : offset
  %merge = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
  %mv = buffer.view %merge[%zero] : buffer -> view<8x130xf32>
  scf.if %live {
    %records = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
    %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %span_v = index.cast %span_b : i32 to index
    %span = index.assume %span_v [lt(%span_v, %spans)] : index
    %length32 = view.load %records[%span, 0] : view<[%spans]x9xi32> -> i32
    %pos32 = view.load %records[%span, 1] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
    %len_b = scalar.assume %length32 [range(%length32, 1, 8)] : i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 1048575)] : i32
    %cur_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %block_len = index.cast %len_b : i32 to index
    %block_pos = index.cast %pos_b : i32 to index
    %cursor = index.cast %cur_b : i32 to index
    %is_local = scalar.cmpi ne, %is_local_arg, %c0_i32 : i32
    %win_i32 = scalar.constant 1024 : i32
    %st_raw = scalar.subi %pos_b, %win_i32 : i32
    %has_win = scalar.cmpi sgt, %pos_b, %win_i32 : i32
    %use_win = scalar.andi %is_local, %has_win : i1
    %cs_i32 = scf.select %use_win, %st_raw, %c0_i32 : i32
    %cs_b = scalar.assume %cs_i32 [range(%cs_i32, 0, 1048575)] : i32
    %ctx_start = index.cast %cs_b : i32 to index
    %ctx_count = index.sub %block_pos, %ctx_start : index
    %keys = index.add %ctx_count, %block_len : index
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %row = index.cast %row_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %pos_bytes = index.constant 1024 : offset
    %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
    %bk_view = buffer.view %bk[%zero] : buffer -> view<[%tokens]x8x32x4xf32>
    %bv_view = buffer.view %bv[%zero] : buffer -> view<[%tokens]x8x32x4xf32>
    %q4 = vector.load %q_view[%token, %qh, %lane, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>
    %scv = vector.splat %scale : vector<4xf32>
    %qs4 = vector.mulf %q4, %scv : vector<4xf32>
    %mf, %lf, %af = scf.for %key = [%wave to %keys step %c8](%m = %neg : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, f32, vector<4xf32>) {
      %in_ctx = index.cmp ult, %key, %ctx_count : index
      %k4, %v4 = scf.if %in_ctx -> (vector<4xf32>, vector<4xf32>) {
        %kpos = index.add %ctx_start, %key : index
        %koff = func.call @g_kv_offset(%origins, %row, %kpos, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %voff = index.add %koff, %plane_stride : offset
        %kc = buffer.view %cache[%koff] : buffer -> view<8x32x4xi8>
        %vc = buffer.view %cache[%voff] : buffer -> view<8x32x4xi8>
        %k8 = vector.load %kc[%khb, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>
        %v8 = vector.load %vc[%khb, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>
        %kh16 = func.call @g_e4m3fn_to_f16_v4(%k8) : (vector<4xi8>) -> (vector<4xf16>)
        %vh16 = func.call @g_e4m3fn_to_f16_v4(%v8) : (vector<4xi8>) -> (vector<4xf16>)
        %kf = vector.extf %kh16 : vector<4xf16> to vector<4xf32>
        %vf = vector.extf %vh16 : vector<4xf16> to vector<4xf32>
        scf.yield %kf, %vf : vector<4xf32>, vector<4xf32>
      } else {
        %bj = index.sub %key, %ctx_count : index
        %btok0 = index.add %cursor, %bj : index
        %btok = index.assume %btok0 [lt(%btok0, %tokens)] : index
        %kf = vector.load %bk_view[%btok, %khb, %lane, %c0] : view<[%tokens]x8x32x4xf32> -> vector<4xf32>
        %vf = vector.load %bv_view[%btok, %khb, %lane, %c0] : view<[%tokens]x8x32x4xf32> -> vector<4xf32>
        scf.yield %kf, %vf : vector<4xf32>, vector<4xf32>
      }
      %p4 = vector.dotf<reassoc|contract> %qs4, %k4, %zero_f : vector<4xf32>, vector<4xf32>, f32
      %s = kernel.subgroup.reduce<addf> %p4 : f32
      %nm = scalar.maxnumf %m, %s : f32
      %ms = scalar.subf %m, %nm : f32
      %al = scalar.expf<afn> %ms : f32
      %ls = scalar.mulf %l, %al : f32
      %alv = vector.splat %al : vector<4xf32>
      %as = vector.mulf %acc, %alv : vector<4xf32>
      %ds = scalar.subf %s, %nm : f32
      %pv = scalar.expf<afn> %ds : f32
      %nl = scalar.addf %ls, %pv : f32
      %pvv = vector.splat %pv : vector<4xf32>
      %na = vector.fmaf %pvv, %v4, %as : vector<4xf32>
      scf.yield %nm, %nl, %na : f32, f32, vector<4xf32>
    }
    %l4 = index.constant 4 : index
    %db = index.mul %lane, %l4 : index
    %c128 = index.constant 128 : index
    %c129 = index.constant 129 : index
    %e0 = vector.extract %af[0] : vector<4xf32> -> f32
    %e1 = vector.extract %af[1] : vector<4xf32> -> f32
    %e2 = vector.extract %af[2] : vector<4xf32> -> f32
    %e3 = vector.extract %af[3] : vector<4xf32> -> f32
    %d0 = index.assume %db [range(%db, 0, 124)] : index
    %d1 = index.add %d0, %c1i : index
    %d2 = index.add %d0, %c2 : index
    %d3 = index.add %d0, %c3 : index
    view.store %e0, %mv[%wave, %d0] : f32, view<8x130xf32>
    view.store %e1, %mv[%wave, %d1] : f32, view<8x130xf32>
    view.store %e2, %mv[%wave, %d2] : f32, view<8x130xf32>
    view.store %e3, %mv[%wave, %d3] : f32, view<8x130xf32>
    %is_l0 = index.cmp eq, %lane, %c0 : index
    scf.if %is_l0 {
      view.store %mf, %mv[%wave, %c128] : f32, view<8x130xf32>
      view.store %lf, %mv[%wave, %c129] : f32, view<8x130xf32>
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %is_w0 = index.cmp eq, %wave, %c0 : index
    scf.if %is_w0 {
      %gm = scf.for %w = [%c0 to %c8 step %c1i](%bm = %neg : f32) -> (f32) {
        %wb = index.assume %w [range(%w, 0, 7)] : index
        %mw = view.load %mv[%wb, %c128] : view<8x130xf32> -> f32
        %nm = scalar.maxnumf %bm, %mw : f32
        scf.yield %nm : f32
      }
      %gl, %g0, %g1, %g2, %g3 = scf.for %w = [%c0 to %c8 step %c1i](%sl = %zero_f : f32, %s0 = %zero_f : f32, %s1 = %zero_f : f32, %s2 = %zero_f : f32, %s3 = %zero_f : f32) -> (f32, f32, f32, f32, f32) {
        %wb = index.assume %w [range(%w, 0, 7)] : index
        %mw = view.load %mv[%wb, %c128] : view<8x130xf32> -> f32
        %lw = view.load %mv[%wb, %c129] : view<8x130xf32> -> f32
        %dm = scalar.subf %mw, %gm : f32
        %sc = scalar.expf<afn> %dm : f32
        %nl = scalar.fmaf %lw, %sc, %sl : f32
        %v0 = view.load %mv[%wb, %d0] : view<8x130xf32> -> f32
        %v1 = view.load %mv[%wb, %d1] : view<8x130xf32> -> f32
        %v2 = view.load %mv[%wb, %d2] : view<8x130xf32> -> f32
        %v3 = view.load %mv[%wb, %d3] : view<8x130xf32> -> f32
        %n0 = scalar.fmaf %v0, %sc, %s0 : f32
        %n1 = scalar.fmaf %v1, %sc, %s1 : f32
        %n2 = scalar.fmaf %v2, %sc, %s2 : f32
        %n3 = scalar.fmaf %v3, %sc, %s3 : f32
        scf.yield %nl, %n0, %n1, %n2, %n3 : f32, f32, f32, f32, f32
      }
      %inv_l = scalar.divf %one_f, %gl : f32
      %o0 = scalar.mulf %g0, %inv_l : f32
      %o1 = scalar.mulf %g1, %inv_l : f32
      %o2 = scalar.mulf %g2, %inv_l : f32
      %o3 = scalar.mulf %g3, %inv_l : f32
      %ov4 = vector.from_elements %o0, %o1, %o2, %o3 : vector<4xf32>
      %out_v = buffer.view %out[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
      vector.store %ov4, %out_v[%token, %qh, %lane, %c0] : vector<4xf32>, view<[%tokens]x24x32x4xf32>
    }
  }
  kernel.return
}

// SwiGLU activation: out = silu(gate) * up for width=9216.
kernel.def export("g_dspark_swiglu") @g_dspark_swiglu(%tokens: index) {
  %c36 = index.constant 36 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c36, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %gate: buffer, %up: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %blk = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %idx0 = index.mul %blk, %c256 : index
  %idx1 = index.add %idx0, %lane : index
  %idx = index.assume %idx1 [range(%idx1, 0, 9215)] : index
  %gv = buffer.view %gate[%zero] : buffer -> view<[%tokens]x9216xf32>
  %uv = buffer.view %up[%zero] : buffer -> view<[%tokens]x9216xf32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x9216xf32>
  %g = view.load %gv[%tok, %idx] : view<[%tokens]x9216xf32> -> f32
  %u = view.load %uv[%tok, %idx] : view<[%tokens]x9216xf32> -> f32
  %sig = scalar.logisticf<afn> %g : f32
  %silu = scalar.mulf %g, %sig : f32
  %res = scalar.mulf %silu, %u : f32
  view.store %res, %ov[%tok, %idx] : f32, view<[%tokens]x9216xf32>
  kernel.return
}

// Fast Q8_0 matvec for K=9216 (down_proj): 1 wave per output row, reuses each weight block across up to 8 tokens.
kernel.def export("g_dspark_mv_q8_0_k9216") @g_dspark_mv_q8_0_k9216(%n: index, %tokens: index) {
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c3 = index.constant 3 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c128 = index.constant 128 : index
  %np = index.add %n, %c3 : index
  %groups = index.div %np, %c4 : index
  %tp = index.add %tokens, %c7 : index
  %ttiles = index.div %tp, %c8 : index
  kernel.launch.config workgroups(%groups, %ttiles, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%n_arg: index, %tokens_arg: index, %xq: buffer, %w: buffer, %y: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 16384)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %wg = kernel.workgroup.id<x> : index
  %tt_id = kernel.workgroup.id<y> : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 3)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c288 = index.constant 288 : index
  %c9216 = index.constant 9216 : offset
  %row0 = index.mul %wg, %c4 : index
  %row = index.add %row0, %wave : index
  %in_n = index.cmp ult, %row, %n : index
  %t_base = index.mul %tt_id, %c8 : index
  %t_rem = index.sub %tokens, %t_base : index
  %t_cnt = index.min %t_rem, %c8 : index
  scf.if %in_n {
    %rb = index.assume %row [lt(%row, %n)] : index
    %q_view = buffer.view %xq[%zero] : buffer -> view<[%tokens]x288x8xi32>
    %m_off = index.scale %tokens, %c9216 : index, offset -> offset
    %m_view = buffer.view %xq[%m_off] : buffer -> view<[%tokens]x288x2xf32>
    %row_bytes = index.constant 9792 : offset
    %blk_bytes = index.constant 34 : offset
    %two_o = index.constant 2 : offset
    %row_org = index.scale %rb, %row_bytes : index, offset -> offset
    %zero_f = scalar.constant 0.0 : f32
    %zero_i = scalar.constant 0 : i32
    %zero8i = vector.splati 0 : vector<8xi32>
    %acc_bytes = index.constant 128 : offset
    %acc_lds = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
    %av = buffer.view %acc_lds[%zero] : buffer -> view<4x8xf32>
    %t_last = index.sub %tokens, %c1 : index
    %t0_r = index.min %t_base, %t_last : index
    %t0 = index.assume %t0_r [lt(%t0_r, %tokens)] : index
    %t1a = index.add %t_base, %c1 : index
    %t1_r = index.min %t1a, %t_last : index
    %t1 = index.assume %t1_r [lt(%t1_r, %tokens)] : index
    %c2 = index.constant 2 : index
    %c3 = index.constant 3 : index
    %c5 = index.constant 5 : index
    %c6 = index.constant 6 : index
    %c7 = index.constant 7 : index
    %t2a = index.add %t_base, %c2 : index
    %t2_r = index.min %t2a, %t_last : index
    %t2 = index.assume %t2_r [lt(%t2_r, %tokens)] : index
    %t3a = index.add %t_base, %c3 : index
    %t3_r = index.min %t3a, %t_last : index
    %t3 = index.assume %t3_r [lt(%t3_r, %tokens)] : index
    %t4a = index.add %t_base, %c4 : index
    %t4_r = index.min %t4a, %t_last : index
    %t4 = index.assume %t4_r [lt(%t4_r, %tokens)] : index
    %t5a = index.add %t_base, %c5 : index
    %t5_r = index.min %t5a, %t_last : index
    %t5 = index.assume %t5_r [lt(%t5_r, %tokens)] : index
    %t6a = index.add %t_base, %c6 : index
    %t6_r = index.min %t6a, %t_last : index
    %t6 = index.assume %t6_r [lt(%t6_r, %tokens)] : index
    %t7a = index.add %t_base, %c7 : index
    %t7_r = index.min %t7a, %t_last : index
    %t7 = index.assume %t7_r [lt(%t7_r, %tokens)] : index
    %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 = scf.for %g = [%lane to %c288 step %c32](%a0 = %zero_f : f32, %a1 = %zero_f : f32, %a2 = %zero_f : f32, %a3 = %zero_f : f32, %a4 = %zero_f : f32, %a5 = %zero_f : f32, %a6 = %zero_f : f32, %a7 = %zero_f : f32) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
      %gb = index.assume %g [range(%g, 0, 287)] : index
      %bo = index.scale %gb, %blk_bytes : index, offset -> offset
      %wo = index.add %row_org, %bo : offset
      %dv = buffer.view %w[%wo] : buffer -> view<1xf16>
      %dh = view.load %dv[0] : view<1xf16> -> f16
      %dw = scalar.extf %dh : f16 to f32
      %qo = index.add %wo, %two_o : offset
      %wv = buffer.view %w[%qo] : buffer -> view<32xi8>
      %wb = vector.load %wv[0] : view<32xi8> -> vector<32xi8>
      %x0w = vector.load %q_view[%t0, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x0b = vector.bitcast %x0w : vector<8xi32> to vector<32xi8>
      %d0 = view.load %m_view[%t0, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q0 = vector.dot4i<s8s8> %wb, %x0b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i0 = vector.reduce<addi> %q0, %zero_i : vector<8xi32>, i32
      %f0 = scalar.sitofp %i0 : i32 to f32
      %sc0 = scalar.mulf %dw, %d0 : f32
      %n0 = scalar.fmaf %f0, %sc0, %a0 : f32
      %x1w = vector.load %q_view[%t1, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x1b = vector.bitcast %x1w : vector<8xi32> to vector<32xi8>
      %d1 = view.load %m_view[%t1, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q1 = vector.dot4i<s8s8> %wb, %x1b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i1 = vector.reduce<addi> %q1, %zero_i : vector<8xi32>, i32
      %f1 = scalar.sitofp %i1 : i32 to f32
      %sc1 = scalar.mulf %dw, %d1 : f32
      %n1 = scalar.fmaf %f1, %sc1, %a1 : f32
      %x2w = vector.load %q_view[%t2, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x2b = vector.bitcast %x2w : vector<8xi32> to vector<32xi8>
      %d2 = view.load %m_view[%t2, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q2 = vector.dot4i<s8s8> %wb, %x2b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i2 = vector.reduce<addi> %q2, %zero_i : vector<8xi32>, i32
      %f2 = scalar.sitofp %i2 : i32 to f32
      %sc2 = scalar.mulf %dw, %d2 : f32
      %n2 = scalar.fmaf %f2, %sc2, %a2 : f32
      %x3w = vector.load %q_view[%t3, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x3b = vector.bitcast %x3w : vector<8xi32> to vector<32xi8>
      %d3 = view.load %m_view[%t3, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q3 = vector.dot4i<s8s8> %wb, %x3b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i3 = vector.reduce<addi> %q3, %zero_i : vector<8xi32>, i32
      %f3 = scalar.sitofp %i3 : i32 to f32
      %sc3 = scalar.mulf %dw, %d3 : f32
      %n3 = scalar.fmaf %f3, %sc3, %a3 : f32
      %x4w = vector.load %q_view[%t4, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x4b = vector.bitcast %x4w : vector<8xi32> to vector<32xi8>
      %d4 = view.load %m_view[%t4, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q4 = vector.dot4i<s8s8> %wb, %x4b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i4 = vector.reduce<addi> %q4, %zero_i : vector<8xi32>, i32
      %f4 = scalar.sitofp %i4 : i32 to f32
      %sc4 = scalar.mulf %dw, %d4 : f32
      %n4 = scalar.fmaf %f4, %sc4, %a4 : f32
      %x5w = vector.load %q_view[%t5, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x5b = vector.bitcast %x5w : vector<8xi32> to vector<32xi8>
      %d5 = view.load %m_view[%t5, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q5 = vector.dot4i<s8s8> %wb, %x5b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i5 = vector.reduce<addi> %q5, %zero_i : vector<8xi32>, i32
      %f5 = scalar.sitofp %i5 : i32 to f32
      %sc5 = scalar.mulf %dw, %d5 : f32
      %n5 = scalar.fmaf %f5, %sc5, %a5 : f32
      %x6w = vector.load %q_view[%t6, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x6b = vector.bitcast %x6w : vector<8xi32> to vector<32xi8>
      %d6 = view.load %m_view[%t6, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q6 = vector.dot4i<s8s8> %wb, %x6b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i6 = vector.reduce<addi> %q6, %zero_i : vector<8xi32>, i32
      %f6 = scalar.sitofp %i6 : i32 to f32
      %sc6 = scalar.mulf %dw, %d6 : f32
      %n6 = scalar.fmaf %f6, %sc6, %a6 : f32
      %x7w = vector.load %q_view[%t7, %gb, %c0] : view<[%tokens]x288x8xi32> -> vector<8xi32>
      %x7b = vector.bitcast %x7w : vector<8xi32> to vector<32xi8>
      %d7 = view.load %m_view[%t7, %gb, 0] : view<[%tokens]x288x2xf32> -> f32
      %q7 = vector.dot4i<s8s8> %wb, %x7b, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %i7 = vector.reduce<addi> %q7, %zero_i : vector<8xi32>, i32
      %f7 = scalar.sitofp %i7 : i32 to f32
      %sc7 = scalar.mulf %dw, %d7 : f32
      %n7 = scalar.fmaf %f7, %sc7, %a7 : f32
      scf.yield %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7 : f32, f32, f32, f32, f32, f32, f32, f32
    }
    %r0 = kernel.subgroup.reduce<addf> %s0 : f32
    %r1 = kernel.subgroup.reduce<addf> %s1 : f32
    %r2 = kernel.subgroup.reduce<addf> %s2 : f32
    %r3 = kernel.subgroup.reduce<addf> %s3 : f32
    %r4 = kernel.subgroup.reduce<addf> %s4 : f32
    %r5 = kernel.subgroup.reduce<addf> %s5 : f32
    %r6 = kernel.subgroup.reduce<addf> %s6 : f32
    %r7 = kernel.subgroup.reduce<addf> %s7 : f32
    %is_l0 = index.cmp eq, %lane, %c0 : index
    scf.if %is_l0 {
      view.store %r0, %av[%wave, 0] : f32, view<4x8xf32>
      view.store %r1, %av[%wave, 1] : f32, view<4x8xf32>
      view.store %r2, %av[%wave, 2] : f32, view<4x8xf32>
      view.store %r3, %av[%wave, 3] : f32, view<4x8xf32>
      view.store %r4, %av[%wave, 4] : f32, view<4x8xf32>
      view.store %r5, %av[%wave, 5] : f32, view<4x8xf32>
      view.store %r6, %av[%wave, 6] : f32, view<4x8xf32>
      view.store %r7, %av[%wave, 7] : f32, view<4x8xf32>
      %yv = buffer.view %y[%zero] : buffer -> view<[%tokens]x[%n]xf32>
      scf.for %j = [%c0 to %t_cnt step %c1] {
        %jb = index.assume %j [range(%j, 0, 7)] : index
        %tj = index.add %t_base, %jb : index
        %tjb = index.assume %tj [lt(%tj, %tokens)] : index
        %val = view.load %av[%wave, %jb] : view<4x8xf32> -> f32
        view.store %val, %yv[%tjb, %rb] : f32, view<[%tokens]x[%n]xf32>
      }
    }
  }
  kernel.return
}

// Gather f_prev[span] from draft_cache tail + init Markov state=0 and prev=anchor.
kernel.def export("g_dspark_markov_init") @g_dspark_markov_init(%blocks: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%blocks, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%blocks_arg: index, %bsz_arg: index, %pool_arg: index, %draft_meta: buffer, %draft_ids: buffer, %dcache: buffer, %f_out: buffer, %state: buffer, %prev: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %bsz = index.assume %bsz_arg [range(%bsz_arg, 2, 8)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %b_id = kernel.workgroup.id<x> : index
  %b = index.assume %b_id [lt(%b_id, %blocks)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c256 = index.constant 256 : index
  %c2304 = index.constant 2304 : index
  %zero_f = scalar.constant 0.0 : f32
  %dh = buffer.view %draft_meta[%zero] : buffer -> view<4xi32>
  %cnt32 = view.load %dh[0] : view<4xi32> -> i32
  %cnt_b = scalar.assume %cnt32 [range(%cnt32, 0, 1024)] : i32
  %cnt = index.cast %cnt_b : i32 to index
  %live = index.cmp ult, %b, %cnt : index
  %sv = buffer.view %state[%zero] : buffer -> view<[%blocks]x256xf32>
  view.store %zero_f, %sv[%b, %lane] : f32, view<[%blocks]x256xf32>
  scf.if %live {
    %recs = buffer.view %draft_meta[%c16o] : buffer -> view<[%blocks]x9xi32>
    %row32 = view.load %recs[%b, 3] : view<[%blocks]x9xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %row = index.cast %row_b : i32 to index
    %kv_stride = index.constant 6144 : offset
    %fbase = index.scale %pool, %kv_stride : index, offset -> offset
    %fsrc = buffer.view %dcache[%fbase] : buffer -> view<1024x2304xf32>
    %fdst = buffer.view %f_out[%zero] : buffer -> view<[%blocks]x2304xf32>
    scf.for %c = [%lane to %c2304 step %c256] {
      %cb = index.assume %c [range(%c, 0, 2303)] : index
      %v = view.load %fsrc[%row, %cb] : view<1024x2304xf32> -> f32
      view.store %v, %fdst[%b, %cb] : f32, view<[%blocks]x2304xf32>
    }
    %is_l0 = index.cmp eq, %lane, %c0 : index
    scf.if %is_l0 {
      %dt = index.mul %blocks, %bsz : index
      %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%dt]xi32>
      %at = index.mul %b, %bsz : index
      %atb = index.assume %at [lt(%at, %dt)] : index
      %anchor = view.load %dids[%atb] : view<[%dt]xi32> -> i32
      %pv = buffer.view %prev[%zero] : buffer -> view<[%blocks]xi32>
      view.store %anchor, %pv[%b] : i32, view<[%blocks]xi32>
    }
  }
  kernel.return
}

// Parallel coalesced Markov RNN step: 32 workgroups x 8 waves = 256 waves (1 wave per state/latent index).
kernel.def export("g_dspark_markov_rnn") @g_dspark_markov_rnn(%blocks: index) {
  %c32 = index.constant 32 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%blocks, %c32, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%blocks_arg: index, %bsz_arg: index, %step_arg: index, %draft_meta: buffer, %ctx_h: buffer, %ctx_f: buffer, %w1: buffer, %w_sp: buffer, %rnn_b: buffer, %state: buffer, %prev: buffer, %latent: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %bsz = index.assume %bsz_arg [range(%bsz_arg, 2, 8)] : index
  %step = index.assume %step_arg [range(%step_arg, 0, 6)] : index
  %b_id = kernel.workgroup.id<x> : index
  %b = index.assume %b_id [lt(%b_id, %blocks)] : index
  %wg_y = kernel.workgroup.id<y> : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %tid = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %c512 = index.constant 512 : index
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %dh = buffer.view %draft_meta[%zero] : buffer -> view<4xi32>
  %cnt32 = view.load %dh[0] : view<4xi32> -> i32
  %cnt_b = scalar.assume %cnt32 [range(%cnt32, 0, 1024)] : i32
  %cnt = index.cast %cnt_b : i32 to index
  %live = index.cmp ult, %b, %cnt : index
  %xs_bytes = index.constant 2048 : offset
  %lat_bytes = index.constant 1024 : offset
  %xs_lds = buffer.alloca<workgroup> align(16) %xs_bytes : buffer
  %xs = buffer.view %xs_lds[%zero] : buffer -> view<512xf32>
  scf.if %live {
    %sv = buffer.view %state[%zero] : buffer -> view<[%blocks]x256xf32>
    %s_in = view.load %sv[%b, %tid] : view<[%blocks]x256xf32> -> f32
    view.store %s_in, %xs[%tid] : f32, view<512xf32>
    %pv = buffer.view %prev[%zero] : buffer -> view<[%blocks]xi32>
    %raw_p = view.load %pv[%b] : view<[%blocks]xi32> -> i32
    %c0_i32 = scalar.constant 0 : i32
    %vmax = scalar.constant 200057 : i32
    %p0 = scalar.maxsi %raw_p, %c0_i32 : i32
    %p1 = scalar.minsi %p0, %vmax : i32
    %pb = scalar.assume %p1 [range(%p1, 0, 200057)] : i32
    %pid = index.cast %pb : i32 to index
    %in_g = index.cmp ult, %tid, %c8 : index
    scf.if %in_g {
      %rbytes = index.constant 272 : offset
      %bbytes = index.constant 34 : offset
      %ro = index.scale %pid, %rbytes : index, offset -> offset
      %bo = index.scale %tid, %bbytes : index, offset -> offset
      %org = index.add %ro, %bo : offset
      %v32 = func.call @g_q8_0_values(%w1, %org, %c0) : (buffer, offset, index) -> (vector<32xf32>)
      %xs_hi = buffer.view %xs_lds[%lat_bytes] : buffer -> view<8x32xf32>
      %lb = index.assume %tid [range(%tid, 0, 7)] : index
      vector.store %v32, %xs_hi[%lb, %c0] : vector<32xf32>, view<8x32xf32>
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %j0 = index.mul %wg_y, %c8 : index
    %j1 = index.add %j0, %wave : index
    %o0 = index.assume %j1 [range(%j1, 0, 255)] : index
    %o1 = index.add %o0, %c256 : index
    %o2 = index.add %o0, %c512 : index
    %xs4 = buffer.view %xs_lds[%zero] : buffer -> view<128x4xf32>
    %wsp4 = buffer.view %w_sp[%zero] : buffer -> view<768x128x4xf32>
    %a0, %a1, %a2 = scf.for %k = [%lane to %c128 step %c32](%s0 = %zero_f : f32, %s1 = %zero_f : f32, %s2 = %zero_f : f32) -> (f32, f32, f32) {
      %kb = index.assume %k [range(%k, 0, 127)] : index
      %xv = vector.load %xs4[%kb, %c0] : view<128x4xf32> -> vector<4xf32>
      %w0 = vector.load %wsp4[%o0, %kb, %c0] : view<768x128x4xf32> -> vector<4xf32>
      %w1v = vector.load %wsp4[%o1, %kb, %c0] : view<768x128x4xf32> -> vector<4xf32>
      %w2 = vector.load %wsp4[%o2, %kb, %c0] : view<768x128x4xf32> -> vector<4xf32>
      %n0 = vector.dotf<reassoc|contract> %w0, %xv, %s0 : vector<4xf32>, vector<4xf32>, f32
      %n1 = vector.dotf<reassoc|contract> %w1v, %xv, %s1 : vector<4xf32>, vector<4xf32>, f32
      %n2 = vector.dotf<reassoc|contract> %w2, %xv, %s2 : vector<4xf32>, vector<4xf32>, f32
      scf.yield %n0, %n1, %n2 : f32, f32, f32
    }
    %d0 = kernel.subgroup.reduce<addf> %a0 : f32
    %d1 = kernel.subgroup.reduce<addf> %a1 : f32
    %d2 = kernel.subgroup.reduce<addf> %a2 : f32
    %is_l0 = index.cmp eq, %lane, %c0 : index
    scf.if %is_l0 {
      %dt = index.mul %blocks, %bsz : index
      %t0 = index.mul %b, %bsz : index
      %t1 = index.add %t0, %step : index
      %tok = index.add %t1, %c1 : index
      %tok_b = index.assume %tok [lt(%tok, %dt)] : index
      %chv = buffer.view %ctx_h[%zero] : buffer -> view<[%dt]x768xf32>
      %cfv = buffer.view %ctx_f[%zero] : buffer -> view<[%blocks]x768xf32>
      %rbv = buffer.view %rnn_b[%zero] : buffer -> view<768xf32>
      %b0 = view.load %rbv[%o0] : view<768xf32> -> f32
      %b1 = view.load %rbv[%o1] : view<768xf32> -> f32
      %b2 = view.load %rbv[%o2] : view<768xf32> -> f32
      %h0 = view.load %chv[%tok_b, %o0] : view<[%dt]x768xf32> -> f32
      %h1 = view.load %chv[%tok_b, %o1] : view<[%dt]x768xf32> -> f32
      %h2 = view.load %chv[%tok_b, %o2] : view<[%dt]x768xf32> -> f32
      %f0 = view.load %cfv[%b, %o0] : view<[%blocks]x768xf32> -> f32
      %f1 = view.load %cfv[%b, %o1] : view<[%blocks]x768xf32> -> f32
      %f2 = view.load %cfv[%b, %o2] : view<[%blocks]x768xf32> -> f32
      %r0a = scalar.addf %d0, %b0 : f32
      %r0b = scalar.addf %r0a, %h0 : f32
      %r0 = scalar.addf %r0b, %f0 : f32
      %r1a = scalar.addf %d1, %b1 : f32
      %r1b = scalar.addf %r1a, %h1 : f32
      %r1 = scalar.addf %r1b, %f1 : f32
      %r2a = scalar.addf %d2, %b2 : f32
      %r2b = scalar.addf %r2a, %h2 : f32
      %r2 = scalar.addf %r2b, %f2 : f32
      %gate = scalar.logisticf<afn> %r0 : f32
      %cand = scalar.tanhf<afn> %r1 : f32
      %lat_v = scalar.tanhf<afn> %r2 : f32
      %s_cur = view.load %sv[%b, %o0] : view<[%blocks]x256xf32> -> f32
      %om = scalar.subf %one_f, %gate : f32
      %gs = scalar.mulf %gate, %s_cur : f32
      %s_out = scalar.fmaf %om, %cand, %gs : f32
      view.store %s_out, %sv[%b, %o0] : f32, view<[%blocks]x256xf32>
      %lv = buffer.view %latent[%zero] : buffer -> view<[%blocks]x256xf32>
      view.store %lat_v, %lv[%b, %o0] : f32, view<[%blocks]x256xf32>
    }
  }
  kernel.return
}

// Parallel coalesced Markov output projection: h_eff[b, c] = h[tok, c] + out_b[c] + dot(out_w[c], latent[b]).
kernel.def export("g_dspark_markov_out") @g_dspark_markov_out(%blocks: index) {
  %c36 = index.constant 36 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%blocks, %c36, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%blocks_arg: index, %bsz_arg: index, %step_arg: index, %draft_meta: buffer, %h: buffer, %latent: buffer, %out_w: buffer, %out_b: buffer, %h_eff: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %bsz = index.assume %bsz_arg [range(%bsz_arg, 2, 8)] : index
  %step = index.assume %step_arg [range(%step_arg, 0, 6)] : index
  %b_id = kernel.workgroup.id<x> : index
  %b = index.assume %b_id [lt(%b_id, %blocks)] : index
  %wg_y = kernel.workgroup.id<y> : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %tid = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %zero_f = scalar.constant 0.0 : f32
  %dh = buffer.view %draft_meta[%zero] : buffer -> view<4xi32>
  %cnt32 = view.load %dh[0] : view<4xi32> -> i32
  %cnt_b = scalar.assume %cnt32 [range(%cnt32, 0, 1024)] : i32
  %cnt = index.cast %cnt_b : i32 to index
  %live = index.cmp ult, %b, %cnt : index
  %lat_bytes = index.constant 1024 : offset
  %lat_lds = buffer.alloca<workgroup> align(16) %lat_bytes : buffer
  %lv_s = buffer.view %lat_lds[%zero] : buffer -> view<256xf32>
  scf.if %live {
    %lv_g = buffer.view %latent[%zero] : buffer -> view<[%blocks]x256xf32>
    %lval = view.load %lv_g[%b, %tid] : view<[%blocks]x256xf32> -> f32
    view.store %lval, %lv_s[%tid] : f32, view<256xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %lat4 = buffer.view %lat_lds[%zero] : buffer -> view<64x4xf32>
    %ow4 = buffer.view %out_w[%zero] : buffer -> view<2304x64x4xf32>
    %dt = index.mul %blocks, %bsz : index
    %t0 = index.mul %b, %bsz : index
    %t1 = index.add %t0, %step : index
    %tok = index.add %t1, %c1 : index
    %tok_b = index.assume %tok [lt(%tok, %dt)] : index
    %hv = buffer.view %h[%zero] : buffer -> view<[%dt]x2304xf32>
    %obv = buffer.view %out_b[%zero] : buffer -> view<2304xf32>
    %hev = buffer.view %h_eff[%zero] : buffer -> view<[%blocks]x2304xf32>
    %base_w = index.mul %wg_y, %c64 : index
    %base_c0 = index.mul %wave, %c8 : index
    %base_c = index.add %base_w, %base_c0 : index
    %is_l0 = index.cmp eq, %lane, %c0 : index
    %l4a = vector.load %lat4[%lane, %c0] : view<64x4xf32> -> vector<4xf32>
    %lane_hi = index.add %lane, %c32 : index
    %l4b = vector.load %lat4[%lane_hi, %c0] : view<64x4xf32> -> vector<4xf32>
    scf.for %r = [%c0 to %c8 step %c1] {
      %c_raw = index.add %base_c, %r : index
      %cb = index.assume %c_raw [range(%c_raw, 0, 2303)] : index
      %wa = vector.load %ow4[%cb, %lane, %c0] : view<2304x64x4xf32> -> vector<4xf32>
      %wb = vector.load %ow4[%cb, %lane_hi, %c0] : view<2304x64x4xf32> -> vector<4xf32>
      %p0 = vector.dotf<reassoc|contract> %wa, %l4a, %zero_f : vector<4xf32>, vector<4xf32>, f32
      %p1 = vector.dotf<reassoc|contract> %wb, %l4b, %p0 : vector<4xf32>, vector<4xf32>, f32
      %dot = kernel.subgroup.reduce<addf> %p1 : f32
      scf.if %is_l0 {
        %hc = view.load %hv[%tok_b, %cb] : view<[%dt]x2304xf32> -> f32
        %bc = view.load %obv[%cb] : view<2304xf32> -> f32
        %s0 = scalar.addf %hc, %bc : f32
        %s1 = scalar.addf %s0, %dot : f32
        view.store %s1, %hev[%b, %cb] : f32, view<[%blocks]x2304xf32>
      }
    }
  }
  kernel.return
}

// Fast Q4_0 proposal head (200058 x 2304 Q4_0 = 259 MB) -> chunk-local (max_val, max_idx) over 1563 chunks.
kernel.def export("g_dspark_head_q4_0") @g_dspark_head_q4_0(%blocks: index) {
  %c1563 = index.constant 1563 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1563, %blocks, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%blocks_arg: index, %draft_meta: buffer, %xq: buffer, %head_q4: buffer, %chunk_vals: buffer, %chunk_ids: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %chk_id = kernel.workgroup.id<x> : index
  %chk = index.assume %chk_id [range(%chk_id, 0, 1562)] : index
  %b_id = kernel.workgroup.id<y> : index
  %b = index.assume %b_id [lt(%b_id, %blocks)] : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c72 = index.constant 72 : index
  %c128 = index.constant 128 : index
  %c200058 = index.constant 200058 : index
  %dh = buffer.view %draft_meta[%zero] : buffer -> view<4xi32>
  %cnt32 = view.load %dh[0] : view<4xi32> -> i32
  %cnt_b = scalar.assume %cnt32 [range(%cnt32, 0, 1024)] : i32
  %cnt = index.cast %cnt_b : i32 to index
  %live = index.cmp ult, %b, %cnt : index
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero_f = scalar.constant 0.0 : f32
  %c0_i32 = scalar.constant 0 : i32
  %m8_f = scalar.constant -8.0 : f32
  %zero4i = vector.splati 0 : vector<4xi32>
  %mask15 = vector.splati 15 : vector<16xi8>
  %four8 = vector.splati 4 : vector<16xi8>
  %lds_v_bytes = index.constant 32 : offset
  %lds_v = buffer.alloca<workgroup> align(16) %lds_v_bytes : buffer
  %lds_i = buffer.alloca<workgroup> align(16) %lds_v_bytes : buffer
  %lvv = buffer.view %lds_v[%zero] : buffer -> view<8xf32>
  %liv = buffer.view %lds_i[%zero] : buffer -> view<8xi32>
  scf.if %live {
    %q_view = buffer.view %xq[%zero] : buffer -> view<[%blocks]x72x8xi32>
    %c2304_o = index.constant 2304 : offset
    %m_off = index.scale %blocks, %c2304_o : index, offset -> offset
    %m_view = buffer.view %xq[%m_off] : buffer -> view<[%blocks]x72x2xf32>
    %d_base = index.constant 231579648 : offset
    %wq_all = buffer.view %head_q4[%zero] : buffer -> view<201024x72x4xi32>
    %wd_all = buffer.view %head_q4[%d_base] : buffer -> view<201024x72xf16>
    // Hoist the 3 activation blocks per lane (g0=lane, g1=lane+32, g2=lane+64 for lane<8) into registers outside the 16-row loop.
    %g0 = index.assume %lane [range(%lane, 0, 31)] : index
    %g1_raw = index.add %lane, %c32 : index
    %g1 = index.assume %g1_raw [range(%g1_raw, 32, 63)] : index
    %has_g2 = index.cmp ult, %lane, %c8 : index
    %c64 = index.constant 64 : index
    %g2_raw = index.add %lane, %c64 : index
    %g2_cl = scf.select %has_g2, %g2_raw, %c64 : index
    %g2 = index.assume %g2_cl [range(%g2_cl, 64, 71)] : index
    %xw0 = vector.load %q_view[%b, %g0, %c0] : view<[%blocks]x72x8xi32> -> vector<8xi32>
    %xb0 = vector.bitcast %xw0 : vector<8xi32> to vector<32xi8>
    %xb0_lo = vector.slice %xb0[0] : vector<32xi8> -> vector<16xi8>
    %xb0_hi = vector.slice %xb0[16] : vector<32xi8> -> vector<16xi8>
    %dx0 = view.load %m_view[%b, %g0, 0] : view<[%blocks]x72x2xf32> -> f32
    %qs0 = view.load %m_view[%b, %g0, 1] : view<[%blocks]x72x2xf32> -> f32
    %m8_qs0 = scalar.mulf %m8_f, %qs0 : f32
    %xw1 = vector.load %q_view[%b, %g1, %c0] : view<[%blocks]x72x8xi32> -> vector<8xi32>
    %xb1 = vector.bitcast %xw1 : vector<8xi32> to vector<32xi8>
    %xb1_lo = vector.slice %xb1[0] : vector<32xi8> -> vector<16xi8>
    %xb1_hi = vector.slice %xb1[16] : vector<32xi8> -> vector<16xi8>
    %dx1 = view.load %m_view[%b, %g1, 0] : view<[%blocks]x72x2xf32> -> f32
    %qs1 = view.load %m_view[%b, %g1, 1] : view<[%blocks]x72x2xf32> -> f32
    %m8_qs1 = scalar.mulf %m8_f, %qs1 : f32
    %xw2 = vector.load %q_view[%b, %g2, %c0] : view<[%blocks]x72x8xi32> -> vector<8xi32>
    %xb2 = vector.bitcast %xw2 : vector<8xi32> to vector<32xi8>
    %xb2_lo = vector.slice %xb2[0] : vector<32xi8> -> vector<16xi8>
    %xb2_hi = vector.slice %xb2[16] : vector<32xi8> -> vector<16xi8>
    %dx2_raw = view.load %m_view[%b, %g2, 0] : view<[%blocks]x72x2xf32> -> f32
    %dx2 = scf.select %has_g2, %dx2_raw, %zero_f : f32
    %qs2 = view.load %m_view[%b, %g2, 1] : view<[%blocks]x72x2xf32> -> f32
    %m8_qs2 = scalar.mulf %m8_f, %qs2 : f32
    %vbase = index.mul %chk, %c128 : index
    // Each of 8 waves handles 16 vocab rows (wave + 8*r) with 16-byte aligned dwordx4 weight loads.
    %bv, %bi = scf.for %r = [%c0 to %c16 step %c1](%best_v = %neg_inf : f32, %best_i = %c0_i32 : i32) -> (f32, i32) {
      %off_r0 = index.mul %r, %c8 : index
      %off_r = index.add %off_r0, %wave : index
      %vrow = index.add %vbase, %off_r : index
      %in_v = index.cmp ult, %vrow, %c200058 : index
      %nv, %ni = scf.if %in_v -> (f32, i32) {
        %vb = index.assume %vrow [range(%vrow, 0, 200057)] : index
        // Block g0 = lane
        %pk0_4 = vector.load %wq_all[%vb, %g0, %c0] : view<201024x72x4xi32> -> vector<4xi32>
        %pk0 = vector.bitcast %pk0_4 : vector<4xi32> to vector<16xi8>
        %dh0 = view.load %wd_all[%vb, %g0] : view<201024x72xf16> -> f16
        %dw0 = scalar.extf %dh0 : f16 to f32
        %lo0 = vector.andi %pk0, %mask15 : vector<16xi8>
        %hi0 = vector.shrui %pk0, %four8 : vector<16xi8>
        %d0a = vector.dot4i<s8s8> %lo0, %xb0_lo, %zero4i : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %d0b = vector.dot4i<s8s8> %hi0, %xb0_hi, %d0a : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %di0 = vector.reduce<addi> %d0b, %c0_i32 : vector<4xi32>, i32
        %df0 = scalar.sitofp %di0 : i32 to f32
        %adj0 = scalar.addf %df0, %m8_qs0 : f32
        %sc0 = scalar.mulf %dw0, %dx0 : f32
        %s0 = scalar.mulf %adj0, %sc0 : f32
        // Block g1 = lane + 32
        %pk1_4 = vector.load %wq_all[%vb, %g1, %c0] : view<201024x72x4xi32> -> vector<4xi32>
        %pk1 = vector.bitcast %pk1_4 : vector<4xi32> to vector<16xi8>
        %dh1 = view.load %wd_all[%vb, %g1] : view<201024x72xf16> -> f16
        %dw1 = scalar.extf %dh1 : f16 to f32
        %lo1 = vector.andi %pk1, %mask15 : vector<16xi8>
        %hi1 = vector.shrui %pk1, %four8 : vector<16xi8>
        %d1a = vector.dot4i<s8s8> %lo1, %xb1_lo, %zero4i : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %d1b = vector.dot4i<s8s8> %hi1, %xb1_hi, %d1a : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %di1 = vector.reduce<addi> %d1b, %c0_i32 : vector<4xi32>, i32
        %df1 = scalar.sitofp %di1 : i32 to f32
        %adj1 = scalar.addf %df1, %m8_qs1 : f32
        %sc1 = scalar.mulf %dw1, %dx1 : f32
        %s1 = scalar.fmaf %adj1, %sc1, %s0 : f32
        // Block g2 = lane + 64 (dx2 == 0 for lane >= 8)
        %pk2_4 = vector.load %wq_all[%vb, %g2, %c0] : view<201024x72x4xi32> -> vector<4xi32>
        %pk2 = vector.bitcast %pk2_4 : vector<4xi32> to vector<16xi8>
        %dh2 = view.load %wd_all[%vb, %g2] : view<201024x72xf16> -> f16
        %dw2 = scalar.extf %dh2 : f16 to f32
        %lo2 = vector.andi %pk2, %mask15 : vector<16xi8>
        %hi2 = vector.shrui %pk2, %four8 : vector<16xi8>
        %d2a = vector.dot4i<s8s8> %lo2, %xb2_lo, %zero4i : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %d2b = vector.dot4i<s8s8> %hi2, %xb2_hi, %d2a : vector<16xi8>, vector<16xi8>, vector<4xi32>
        %di2 = vector.reduce<addi> %d2b, %c0_i32 : vector<4xi32>, i32
        %df2 = scalar.sitofp %di2 : i32 to f32
        %adj2 = scalar.addf %df2, %m8_qs2 : f32
        %sc2 = scalar.mulf %dw2, %dx2 : f32
        %acc = scalar.fmaf %adj2, %sc2, %s1 : f32
        %tot = kernel.subgroup.reduce<addf> %acc : f32
        %gt = scalar.cmpf ogt, %tot, %best_v : f32
        %v32 = index.cast %vb : index to i32
        %mv = scf.select %gt, %tot, %best_v : f32
        %mi = scf.select %gt, %v32, %best_i : i32
        scf.yield %mv, %mi : f32, i32
      } else {
        scf.yield %best_v, %best_i : f32, i32
      }
      scf.yield %nv, %ni : f32, i32
    }
    %is_l0 = index.cmp eq, %lane, %c0 : index
    scf.if %is_l0 {
      view.store %bv, %lvv[%wave] : f32, view<8xf32>
      view.store %bi, %liv[%wave] : i32, view<8xi32>
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %is_w0 = index.cmp eq, %wave, %c0 : index
    %is_t0 = scalar.andi %is_w0, %is_l0 : i1
    scf.if %is_t0 {
      %fv, %fi = scf.for %w = [%c0 to %c8 step %c1](%cv = %neg_inf : f32, %ci = %c0_i32 : i32) -> (f32, i32) {
        %wb = index.assume %w [range(%w, 0, 7)] : index
        %wv = view.load %lvv[%wb] : view<8xf32> -> f32
        %wi = view.load %liv[%wb] : view<8xi32> -> i32
        %gt = scalar.cmpf ogt, %wv, %cv : f32
        %nv = scf.select %gt, %wv, %cv : f32
        %ni = scf.select %gt, %wi, %ci : i32
        scf.yield %nv, %ni : f32, i32
      }
      %cv_out = buffer.view %chunk_vals[%zero] : buffer -> view<[%blocks]x1563xf32>
      %ci_out = buffer.view %chunk_ids[%zero] : buffer -> view<[%blocks]x1563xi32>
      view.store %fv, %cv_out[%b, %chk] : f32, view<[%blocks]x1563xf32>
      view.store %fi, %ci_out[%b, %chk] : i32, view<[%blocks]x1563xi32>
    }
  }
  kernel.return
}

// Reduce 1563 chunk argmaxes -> proposals[b * depth + step] and prev[b].
kernel.def export("g_dspark_argmax") @g_dspark_argmax(%blocks: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%blocks, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%blocks_arg: index, %depth_arg: index, %step_arg: index, %draft_meta: buffer, %chunk_vals: buffer, %chunk_ids: buffer, %proposals: buffer, %prev: buffer) {
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %step = index.assume %step_arg [range(%step_arg, 0, 6)] : index
  %b_id = kernel.workgroup.id<x> : index
  %b = index.assume %b_id [lt(%b_id, %blocks)] : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %tid = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c256 = index.constant 256 : index
  %c1563 = index.constant 1563 : index
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %c0_i32 = scalar.constant 0 : i32
  %dh = buffer.view %draft_meta[%zero] : buffer -> view<4xi32>
  %cnt32 = view.load %dh[0] : view<4xi32> -> i32
  %cnt_b = scalar.assume %cnt32 [range(%cnt32, 0, 1024)] : i32
  %cnt = index.cast %cnt_b : i32 to index
  %live = index.cmp ult, %b, %cnt : index
  %lds_bytes = index.constant 1024 : offset
  %lds_v = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  %lds_i = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  %lvv = buffer.view %lds_v[%zero] : buffer -> view<256xf32>
  %liv = buffer.view %lds_i[%zero] : buffer -> view<256xi32>
  scf.if %live {
    %cv = buffer.view %chunk_vals[%zero] : buffer -> view<[%blocks]x1563xf32>
    %ci = buffer.view %chunk_ids[%zero] : buffer -> view<[%blocks]x1563xi32>
    %tv, %ti = scf.for %k = [%tid to %c1563 step %c256](%bv = %neg_inf : f32, %bi = %c0_i32 : i32) -> (f32, i32) {
      %kb = index.assume %k [range(%k, 0, 1562)] : index
      %v = view.load %cv[%b, %kb] : view<[%blocks]x1563xf32> -> f32
      %id = view.load %ci[%b, %kb] : view<[%blocks]x1563xi32> -> i32
      %gt = scalar.cmpf ogt, %v, %bv : f32
      %nv = scf.select %gt, %v, %bv : f32
      %ni = scf.select %gt, %id, %bi : i32
      scf.yield %nv, %ni : f32, i32
    }
    view.store %tv, %lvv[%tid] : f32, view<256xf32>
    view.store %ti, %liv[%tid] : i32, view<256xi32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %is_t0 = index.cmp eq, %tid, %c0 : index
    scf.if %is_t0 {
      %fv, %fi = scf.for %j = [%c0 to %c256 step %c1](%bv = %neg_inf : f32, %bi = %c0_i32 : i32) -> (f32, i32) {
        %jb = index.assume %j [range(%j, 0, 255)] : index
        %v = view.load %lvv[%jb] : view<256xf32> -> f32
        %id = view.load %liv[%jb] : view<256xi32> -> i32
        %gt = scalar.cmpf ogt, %v, %bv : f32
        %nv = scf.select %gt, %v, %bv : f32
        %ni = scf.select %gt, %id, %bi : i32
        scf.yield %nv, %ni : f32, i32
      }
      %row_cap = index.mul %blocks, %depth : index
      %prop = buffer.view %proposals[%zero] : buffer -> view<[%row_cap]xi32>
      %p0 = index.mul %b, %depth : index
      %p1 = index.add %p0, %step : index
      %pb = index.assume %p1 [lt(%p1, %row_cap)] : index
      view.store %fi, %prop[%pb] : i32, view<[%row_cap]xi32>
      %pv = buffer.view %prev[%zero] : buffer -> view<[%blocks]xi32>
      view.store %fi, %pv[%b] : i32, view<[%blocks]xi32>
    }
  }
  kernel.return
}
"""


def _emit_q8_mv_call(k: int, n_var: str, t_var: str, xq_var: str, w_var: str, y_var: str, tag: str) -> str:
    """Emit a fast Q8_0 matvec dispatch using Inkling's existing k2304/k3072 kernels (or k9216)."""
    if k == 9216:
        return f"  kernel.launch @g_dspark_mv_q8_0_k9216[{n_var}, {t_var}]({n_var}, {t_var}, {xq_var}, {w_var}, {y_var}) : [index, index](index, index, buffer, view<22560768xi8>, buffer)\n"
    w_bytes = {"2304": {3072: 7520256, 1024: 2506752, 9216: 22560768, 768: 1880064}, "3072": {2304: 7520256}}
    assert k in (2304, 3072)
    sfx = f"{t_var.lstrip('%')}_{n_var.lstrip('%')}_{w_var.lstrip('%')}"
    return f"""  scf.if %single {{
    kernel.launch @g_mv_q8_0_k{k}_q8_r1[{n_var}, {t_var}]({n_var}, {t_var}, %zero_i32, {xq_var}, {w_var}, {y_var}) : [index, index](index, index, i32, buffer, {tag}, buffer)
  }} else {{
    %le4_{sfx} = index.cmp ule, {t_var}, %c4 : index
    %le8_{sfx} = index.cmp ule, {t_var}, %c8 : index
    scf.if %le4_{sfx} {{
      kernel.launch @g_mv_q8_0_k{k}_t4_q8_r1[{n_var}, {t_var}]({n_var}, {t_var}, %zero_i32, {xq_var}, {w_var}, {y_var}) : [index, index](index, index, i32, buffer, {tag}, buffer)
    }} else {{
      scf.if %le8_{sfx} {{
        kernel.launch @g_mv_q8_0_k{k}_t8_q8[{n_var}, {t_var}]({n_var}, {t_var}, %zero_i32, {xq_var}, {w_var}, {y_var}) : [index, index](index, index, i32, buffer, {tag}, buffer)
      }} else {{
        kernel.launch @g_mv_q8_0_k{k}_t16_q8[{n_var}, {t_var}]({n_var}, {t_var}, %zero_i32, {xq_var}, {w_var}, {y_var}) : [index, index](index, index, i32, buffer, {tag}, buffer)
      }}
    }}
  }}
"""


def _emit_dflash_warm() -> str:
    """Emit command.def @llm_dflash_warm for DSpark."""
    layers = []
    for l in range(3):
        layers.append(f"""  %dk_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_k_ctx.weight" : view<2506752xi8>
  %dv_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_v_ctx.weight" : view<2506752xi8>
  %dkn_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_k_ctx_norm.weight" : view<128xf32>
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %nH, %mix{l}, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)
    template.apply<@sel6_g_gemm_q8_0_store>(%tp, %nKV, %tp, %nH, %nKV, %h16, %s_h, %dk_{l}, %k) : (index, index, index, index, index, buffer, buffer, view<2506752xi8>, buffer)
    template.apply<@sel6_g_gemm_q8_0_store>(%tp, %nKV, %tp, %nH, %nKV, %h16, %s_h, %dv_{l}, %v) : (index, index, index, index, index, buffer, buffer, view<2506752xi8>, buffer)
  }} else {{
    kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %mix{l}, %h16q) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(2304, "%nKV", "%t", "%h16q", f"%dk_{l}", "%k", "view<2506752xi8>")}
{_emit_q8_mv_call(2304, "%nKV", "%t", "%h16q", f"%dv_{l}", "%v", "view<2506752xi8>")}
  }}
  %dl_{l} = index.constant {l} : index
  kernel.launch @g_dspark_kv_store[%t](%t, %dl_{l}, %context, %pool, %k, %v, %dkn_{l}, %info, %origins, %dcache) : [index](index, index, index, index, buffer, buffer, view<128xf32>, buffer, buffer, buffer)
""")
    return f"""
command.def public target(@device) @llm_dflash_warm() launch(%w_draft: buffer, %features: buffer, %metadata: buffer, %origins: buffer, %dcache: buffer, %tokens: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %zero_i32 = scalar.constant 0 : i32
  %tp_a = index.add %t, %c63 : index
  %tp_b = index.div %tp_a, %c64 : index
  %tp = index.mul %tp_b, %c64 : index
  %single = index.cmp eq, %t, %c1 : index
  %gemm = index.cmp ugt, %t, %c16 : index
  %nH = index.constant 2304 : index
  %nKV = index.constant 1024 : index
  %info_b = index.constant 16 : offset
  %info_sz = index.scale %tp, %info_b : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_sz : buffer
  %h_b = index.constant 9216 : offset
  %h_sz = index.scale %tp, %h_b : index, offset -> offset
  %mix0 = buffer.alloca<global> align(256) %h_sz : buffer
  %mix1 = buffer.alloca<global> align(256) %h_sz : buffer
  %mix2 = buffer.alloca<global> align(256) %h_sz : buffer
  %h16_b = index.constant 4608 : offset
  %h16_sz = index.scale %tp, %h16_b : index, offset -> offset
  %h16 = buffer.alloca<global> align(256) %h16_sz : buffer
  %h16q = buffer.alloca<global> align(256) %h16_sz : buffer
  %sc_b = index.constant 8 : offset
  %sc_sz = index.scale %tp, %sc_b : index, offset -> offset
  %s_h = buffer.alloca<global> align(256) %sc_sz : buffer
  %kv_b = index.constant 4096 : offset
  %kv_sz = index.scale %tp, %kv_b : index, offset -> offset
  %k = buffer.alloca<global> align(256) %kv_sz : buffer
  %v = buffer.alloca<global> align(256) %kv_sz : buffer
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  %alpha = command.parameter %w_draft, "dspark.fuse_alpha" : view<16xf32>
  %dn0 = command.parameter %w_draft, "dspark.blk.0.dflare_norm.weight" : view<2304xf32>
  %dn1 = command.parameter %w_draft, "dspark.blk.1.dflare_norm.weight" : view<2304xf32>
  %dn2 = command.parameter %w_draft, "dspark.blk.2.dflare_norm.weight" : view<2304xf32>
  %dn3 = command.parameter %w_draft, "dspark.markov.dflare_norm.weight" : view<2304xf32>
  kernel.launch @g_dspark_mix[%t](%t, %spans, %pool, %features, %alpha, %dn0, %dn1, %dn2, %dn3, %metadata, %info, %mix0, %mix1, %mix2, %dcache) : [index](index, index, index, buffer, view<16xf32>, view<2304xf32>, view<2304xf32>, view<2304xf32>, view<2304xf32>, buffer, buffer, buffer, buffer, buffer, buffer)
{"".join(layers)}  command.return
}}
"""


def _emit_dflash_propose() -> str:
    """Emit command.def @llm_dflash_propose for DSpark (3 dense layers + recurrent Markov head)."""
    layers = []
    for l in range(3):
        is_local = 1 if l in (0, 2) else 0
        layers.append(f"""  // ---- DSpark layer {l} (local={is_local}) ----
  %an_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_norm.weight" : view<2304xf32>
  %fn_{l} = command.parameter %w_draft, "dspark.blk.{l}.ffn_norm.weight" : view<2304xf32>
  %qn_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_q_norm.weight" : view<128xf32>
  %kn_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_k_norm.weight" : view<128xf32>
  %wq_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_q.weight" : view<7520256xi8>
  %wk_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_k.weight" : view<2506752xi8>
  %wv_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_v.weight" : view<2506752xi8>
  %wo_{l} = command.parameter %w_draft, "dspark.blk.{l}.attn_output.weight" : view<7520256xi8>
  %wg_{l} = command.parameter %w_draft, "dspark.blk.{l}.ffn_gate.weight" : view<22560768xi8>
  %wu_{l} = command.parameter %w_draft, "dspark.blk.{l}.ffn_up.weight" : view<22560768xi8>
  %wd_{l} = command.parameter %w_draft, "dspark.blk.{l}.ffn_down.weight" : view<22560768xi8>
  kernel.launch @g_ink_rms_norm_plane[%t](%t, %c1, %c0, %x, %an_{l}, %h) : [index](index, index, index, buffer, view<2304xf32>, buffer)
  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %h16q) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(2304, "%nQ", "%t", "%h16q", f"%wq_{l}", "%q", "view<7520256xi8>")}
{_emit_q8_mv_call(2304, "%nKV", "%t", "%h16q", f"%wk_{l}", "%k", "view<2506752xi8>")}
{_emit_q8_mv_call(2304, "%nKV", "%t", "%h16q", f"%wv_{l}", "%v", "view<2506752xi8>")}
  kernel.launch @g_dspark_q_prep[%t](%t, %q, %qn_{l}, %info) : [index](index, buffer, view<128xf32>, buffer)
  kernel.launch @g_dspark_block_kv[%t](%t, %k, %kn_{l}, %info) : [index](index, buffer, view<128xf32>, buffer)
  %dl_{l} = index.constant {l} : index
  %loc_{l} = scalar.constant {is_local} : i32
  kernel.launch @g_dspark_attention[%t](%t, %spans, %dl_{l}, %loc_{l}, %context, %pool, %metadata, %info, %q, %k, %v, %origins, %dcache, %o) : [index](index, index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_quant_q8[%t, %nQ](%t, %nQ, %o, %o16q) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(3072, "%nH", "%t", "%o16q", f"%wo_{l}", "%y", "view<7520256xi8>")}
  kernel.launch @g_ink_add[%t](%t, %x, %y, %x) : [index](index, buffer, buffer, buffer)
  kernel.launch @g_ink_rms_norm_plane[%t](%t, %c1, %c0, %x, %fn_{l}, %h) : [index](index, index, index, buffer, view<2304xf32>, buffer)
  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %h16q) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(2304, "%nFF", "%t", "%h16q", f"%wg_{l}", "%fg", "view<22560768xi8>")}
{_emit_q8_mv_call(2304, "%nFF", "%t", "%h16q", f"%wu_{l}", "%fu", "view<22560768xi8>")}
  kernel.launch @g_dspark_swiglu[%t](%t, %fg, %fu, %fa) : [index](index, buffer, buffer, buffer)
  kernel.launch @g_quant_q8[%t, %nFF](%t, %nFF, %fa, %faq) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(9216, "%nH", "%t", "%faq", f"%wd_{l}", "%y", "view<22560768xi8>")}
  kernel.launch @g_ink_add[%t](%t, %x, %y, %x) : [index](index, buffer, buffer, buffer)
""")
    markov_steps = []
    for s in range(7):
        markov_steps.append(f"""  %step_c_{s} = index.constant {s} : index
  %do_step_{s} = index.cmp ult, %step_c_{s}, %depth : index
  scf.if %do_step_{s} {{
    kernel.launch @g_dspark_markov_rnn[%spans](%spans, %bsz, %step_c_{s}, %metadata, %ctx_h, %ctx_f, %w1, %w_sp, %rnn_b, %state, %prev, %latent) : [index](index, index, index, buffer, buffer, buffer, view<54678528xi8>, view<393216xf32>, view<768xf32>, buffer, buffer, buffer)
    kernel.launch @g_dspark_markov_out[%spans](%spans, %bsz, %step_c_{s}, %metadata, %h, %latent, %out_w, %out_b, %h_eff) : [index](index, index, index, buffer, buffer, buffer, view<589824xf32>, view<2304xf32>, buffer)
    kernel.launch @g_quant_q8[%spans, %nH](%spans, %nH, %h_eff, %h_eff_q) : [index, index](index, index, buffer, buffer)
    kernel.launch @g_dspark_head_q4_0[%spans](%spans, %metadata, %h_eff_q, %head_q4, %chk_vals, %chk_ids) : [index](index, buffer, buffer, view<{HEAD_Q4_BYTES}xi8>, buffer, buffer)
    kernel.launch @g_dspark_argmax[%spans](%spans, %depth, %step_c_{s}, %metadata, %chk_vals, %chk_ids, %proposals, %prev) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer)
  }}
""")
    return f"""
command.def public target(@device) @llm_dflash_propose() launch(%w_embed: buffer, %w_draft: buffer, %w_phead: buffer, %epoch: buffer, %target_tokens: buffer, %origins: buffer, %dcache: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
  %epoch_spans = config.get @llm.epoch_spans : index
  %epoch_tokens = config.get @llm.epoch_tokens : index
  %width = config.get @llm.width : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %zero_i32 = scalar.constant 0 : i32
  %depth = index.sub %width, %c1 : index
  %bsz = index.min %c8, %width : index
  %single = index.cmp eq, %t, %c1 : index
  %single_sp = index.cmp eq, %spans, %c1 : index
  %nH = index.constant 2304 : index
  %nQ = index.constant 3072 : index
  %nKV = index.constant 1024 : index
  %nFF = index.constant 9216 : index
  %nRNN = index.constant 768 : index
  %meta_bytes = index.constant 40976 : offset
  %metadata = buffer.alloca<global> align(256) %meta_bytes : buffer
  %c4_o = index.constant 4 : offset
  %tok_sz = index.scale %t, %c4_o : index, offset -> offset
  %tokens = buffer.alloca<global> align(256) %tok_sz : buffer
  %slots_sz = index.scale %spans, %c4_o : index, offset -> offset
  %slots = buffer.alloca<global> align(256) %slots_sz : buffer
  %row_cap = index.mul %spans, %depth : index
  %prop_sz = index.scale %row_cap, %c4_o : index, offset -> offset
  %proposals = buffer.alloca<global> align(256) %prop_sz : buffer
  %info_b = index.constant 16 : offset
  %info_sz = index.scale %t, %info_b : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_sz : buffer
  %h_b = index.constant 9216 : offset
  %h_sz = index.scale %t, %h_b : index, offset -> offset
  %x = buffer.alloca<global> align(256) %h_sz : buffer
  %h = buffer.alloca<global> align(256) %h_sz : buffer
  %y = buffer.alloca<global> align(256) %h_sz : buffer
  %h16_b = index.constant 4608 : offset
  %h16_sz = index.scale %t, %h16_b : index, offset -> offset
  %h16q = buffer.alloca<global> align(256) %h16_sz : buffer
  %q_b = index.constant 12288 : offset
  %q_sz = index.scale %t, %q_b : index, offset -> offset
  %q = buffer.alloca<global> align(256) %q_sz : buffer
  %o = buffer.alloca<global> align(256) %q_sz : buffer
  %o16_b = index.constant 6144 : offset
  %o16_sz = index.scale %t, %o16_b : index, offset -> offset
  %o16q = buffer.alloca<global> align(256) %o16_sz : buffer
  %kv_b = index.constant 4096 : offset
  %kv_sz = index.scale %t, %kv_b : index, offset -> offset
  %k = buffer.alloca<global> align(256) %kv_sz : buffer
  %v = buffer.alloca<global> align(256) %kv_sz : buffer
  %ff_b = index.constant 36864 : offset
  %ff_sz = index.scale %t, %ff_b : index, offset -> offset
  %fg = buffer.alloca<global> align(256) %ff_sz : buffer
  %fu = buffer.alloca<global> align(256) %ff_sz : buffer
  %fa = buffer.alloca<global> align(256) %ff_sz : buffer
  %ffq_b = index.constant 18432 : offset
  %ffq_sz = index.scale %t, %ffq_b : index, offset -> offset
  %faq = buffer.alloca<global> align(256) %ffq_sz : buffer
  %embedding = command.parameter %w_embed, "token_embd.weight" : view<492106752xi8>
  %tok_norm = command.parameter %w_embed, "token_embd_norm.weight" : view<2304xf32>
  %mask_embed = command.parameter %w_draft, "dspark.mask_embed" : view<2304xf32>
  %head_q4 = command.parameter %w_phead, "output.weight" : view<{HEAD_Q4_BYTES}xi8>
  kernel.launch @g_dfl_prepare(%epoch_spans, %spans, %epoch_tokens, %depth, %epoch, %target_tokens, %metadata, %tokens, %slots) : (index, index, index, index, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  kernel.launch @g_embed[%t](%t, %tokens, %embedding, %tok_norm, %y, %x) : [index](index, buffer, view<492106752xi8>, view<2304xf32>, buffer, buffer)
  kernel.launch @g_dspark_mask_fill[%t](%t, %tokens, %mask_embed, %x) : [index](index, buffer, view<2304xf32>, buffer)
{"".join(layers)}
  // Final norm + recurrent Markov head.
  %dnorm = command.parameter %w_draft, "dspark.norm.weight" : view<2304xf32>
  kernel.launch @g_ink_rms_norm_plane[%t](%t, %c1, %c0, %x, %dnorm, %h) : [index](index, index, index, buffer, view<2304xf32>, buffer)
  %w1 = command.parameter %w_draft, "dspark.markov.w1.weight" : view<54678528xi8>
  %w_sp = command.parameter %w_draft, "dspark.markov.rnn_w_sp.weight" : view<393216xf32>
  %w_h = command.parameter %w_draft, "dspark.markov.rnn_w_h.weight" : view<1880064xi8>
  %w_f = command.parameter %w_draft, "dspark.markov.rnn_w_f.weight" : view<1880064xi8>
  %rnn_b = command.parameter %w_draft, "dspark.markov.rnn_b" : view<768xf32>
  %out_w = command.parameter %w_draft, "dspark.markov.out.weight" : view<589824xf32>
  %out_b = command.parameter %w_draft, "dspark.markov.out.bias" : view<2304xf32>
  %rnn_b_sz = index.constant 3072 : offset
  %ctx_h_sz = index.scale %t, %rnn_b_sz : index, offset -> offset
  %ctx_h = buffer.alloca<global> align(256) %ctx_h_sz : buffer
  %ctx_f_sz = index.scale %spans, %rnn_b_sz : index, offset -> offset
  %ctx_f = buffer.alloca<global> align(256) %ctx_f_sz : buffer
  %f_prev_sz = index.scale %spans, %h_b : index, offset -> offset
  %f_prev = buffer.alloca<global> align(256) %f_prev_sz : buffer
  %f_prev_q_sz = index.scale %spans, %h16_b : index, offset -> offset
  %f_prev_q = buffer.alloca<global> align(256) %f_prev_q_sz : buffer
  %st_b = index.constant 1024 : offset
  %st_sz = index.scale %spans, %st_b : index, offset -> offset
  %state = buffer.alloca<global> align(256) %st_sz : buffer
  %latent = buffer.alloca<global> align(256) %st_sz : buffer
  %prev = buffer.alloca<global> align(256) %slots_sz : buffer
  %h_eff = buffer.alloca<global> align(256) %f_prev_sz : buffer
  %h_eff_q = buffer.alloca<global> align(256) %f_prev_q_sz : buffer
  %chk_b = index.constant 6252 : offset
  %chk_sz = index.scale %spans, %chk_b : index, offset -> offset
  %chk_vals = buffer.alloca<global> align(256) %chk_sz : buffer
  %chk_ids = buffer.alloca<global> align(256) %chk_sz : buffer
  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %h16q) : [index, index](index, index, buffer, buffer)
{_emit_q8_mv_call(2304, "%nRNN", "%t", "%h16q", "%w_h", "%ctx_h", "view<1880064xi8>")}
  kernel.launch @g_dspark_markov_init[%spans](%spans, %bsz, %pool, %metadata, %tokens, %dcache, %f_prev, %state, %prev) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_quant_q8[%spans, %nH](%spans, %nH, %f_prev, %f_prev_q) : [index, index](index, index, buffer, buffer)
  scf.if %single_sp {{
    kernel.launch @g_mv_q8_0_k2304_q8_r1[%nRNN, %spans](%nRNN, %spans, %zero_i32, %f_prev_q, %w_f, %ctx_f) : [index, index](index, index, i32, buffer, view<1880064xi8>, buffer)
  }} else {{
    kernel.launch @g_mv_q8_0_k2304_t8_q8[%nRNN, %spans](%nRNN, %spans, %zero_i32, %f_prev_q, %w_f, %ctx_f) : [index, index](index, index, i32, buffer, view<1880064xi8>, buffer)
  }}
{"".join(markov_steps)}
  kernel.launch @g_dfl_publish(%spans, %epoch_tokens, %depth, %epoch, %metadata, %slots, %proposals, %target_tokens) : (index, index, index, buffer, buffer, buffer, buffer, buffer)
  command.return
}}
"""


def _inject_taps(body: str) -> str:
    """Inject the 4 DSpark target feature taps into llm_advance / llm_verify body."""
    tap_anchors = [
        ("  // ==== step 5 (phys blk.5,", 0, "%x"),
        ("  // ==== step 23 (phys blk.11,", 1, "%x"),
        ("  // ==== step 41 (phys blk.17,", 2, "%x"),
    ]
    for marker, slot, buf in tap_anchors:
        idx = body.find(marker)
        assert idx != -1, f"missing tap anchor {marker!r}"
        tap_code = (
            f"  %tap_s{slot} = index.constant {slot} : index\n"
            f"  kernel.launch @g_dspark_tap[%t](%t, %tap_s{slot}, {buf}, %residual) : [index](index, index, buffer, buffer)\n"
        )
        body = body[:idx] + tap_code + body[idx:]

    head_marker = '  %head = command.parameter %weights, "output.weight"'
    idx = body.find(head_marker)
    assert idx != -1, "missing output.weight anchor for slot 3 tap"
    tap3 = (
        "  %tap_s3 = index.constant 3 : index\n"
        "  kernel.launch @g_dspark_tap[%t](%t, %tap_s3, %h, %residual) : [index](index, index, buffer, buffer)\n"
    )
    body = body[:idx] + tap3 + body[idx:]
    return body


def _build_llm_verify(adv_body: str) -> str:
    """Derive @llm_verify from @llm_advance with deferred sconv/swa commit on %committed + g_spec_accept."""
    sig_old = "command.def public target(@device) @llm_advance() launch(%weights: buffer, %residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %outputs: buffer) {"
    sig_new = "command.def public target(@device) @llm_verify() launch(%weights: buffer, %residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %committed: buffer, %results: buffer) {"
    assert adv_body.startswith(sig_old)
    v = sig_new + adv_body[len(sig_old):]

    # Allocate staging buffers right after `%single = index.cmp eq, %t, %c1 : index`.
    alloc_anchor = "  %single = index.cmp eq, %t, %c1 : index\n"
    alloc_code = alloc_anchor + (
        "  %v_c64 = index.constant 64 : index\n"
        "  %v_c1_idx = index.constant 1 : index\n"
        "  %v_lt64 = index.cmp ult, %t, %v_c64 : index\n"
        "  %v_gt1 = index.cmp ugt, %t, %v_c1_idx : index\n"
        "  %v_defer = scalar.andi %v_gt1, %v_lt64 : i1\n"
        "  %v_st_t = scf.select %v_defer, %t, %v_c1_idx : index\n"
        "  %v_c1024_row = index.constant 393216 : offset\n"
        "  %v_c2304_row = index.constant 884736 : offset\n"
        "  %v_swa_row = index.constant 163840 : offset\n"
        "  %v_st1024_sz = index.scale %v_st_t, %v_c1024_row : index, offset -> offset\n"
        "  %v_st2304_sz = index.scale %v_st_t, %v_c2304_row : index, offset -> offset\n"
        "  %v_stswa_sz = index.scale %v_st_t, %v_swa_row : index, offset -> offset\n"
        "  %v_stage_c1024 = buffer.alloca<global> align(256) %v_st1024_sz : buffer\n"
        "  %v_stage_c2304 = buffer.alloca<global> align(256) %v_st2304_sz : buffer\n"
        "  %v_stage_swa = buffer.alloca<global> align(256) %v_stswa_sz : buffer\n"
        "  %v_w1024 = index.constant 1024 : index\n"
        "  %v_w2304 = index.constant 2304 : index\n"
    )
    assert v.count(alloc_anchor) == 1
    v = v.replace(alloc_anchor, alloc_code, 1)

    # Defer sconv c1024 commits (k/v), sconv c2304 commits (ap/fp), and swa commits when %v_defer.
    c1024_re = re.compile(
        r"^(?P<ind>[ \t]*)kernel\.launch @g_ink_sconv_commit_c1024\[%spans\]\(%t, %spans, %slot_(?P<vis>\d+), %off_(?P<kv>[kv]), %metadata, %origins, %arena, %(?P<src>\w+)\) : \[index\]\(index, index, index, index, buffer, buffer, buffer, buffer\)$",
        flags=re.M,
    )
    def _rep_c1024(m: re.Match) -> str:
        ind, vis, kv, src = m.group("ind"), int(m.group("vis")), m.group("kv"), m.group("src")
        slot = vis * 2 + (0 if kv == "k" else 1)
        return (
            f"{ind}scf.if %v_defer {{\n"
            f"{ind}  %vs_{vis}_{kv}_{m.start()} = index.constant {slot} : index\n"
            f"{ind}  kernel.launch @g_ink_stage_f32[%t](%t, %v_w1024, %vs_{vis}_{kv}_{m.start()}, %{src}, %v_stage_c1024) : [index](index, index, index, buffer, buffer)\n"
            f"{ind}}} else {{\n"
            f"{ind}  {m.group(0).strip()}\n"
            f"{ind}}}"
        )
    v, n1024 = c1024_re.subn(_rep_c1024, v)
    assert n1024 == 176, f"expected 176 c1024 commits, got {n1024}"

    c2304_re = re.compile(
        r"^(?P<ind>[ \t]*)kernel\.launch @g_ink_sconv_commit_c2304\[%spans\]\(%t, %spans, %slot_(?P<vis>\d+), %off_(?P<af>ap|fp), %metadata, %origins, %arena, %(?P<src>\w+)\) : \[index\]\(index, index, index, index, buffer, buffer, buffer, buffer\)$",
        flags=re.M,
    )
    def _rep_c2304(m: re.Match) -> str:
        ind, vis, af, src = m.group("ind"), int(m.group("vis")), m.group("af"), m.group("src")
        slot = vis * 2 + (0 if af == "ap" else 1)
        return (
            f"{ind}scf.if %v_defer {{\n"
            f"{ind}  %vs2_{vis}_{af}_{m.start()} = index.constant {slot} : index\n"
            f"{ind}  kernel.launch @g_ink_stage_f32[%t](%t, %v_w2304, %vs2_{vis}_{af}_{m.start()}, %{src}, %v_stage_c2304) : [index](index, index, index, buffer, buffer)\n"
            f"{ind}}} else {{\n"
            f"{ind}  {m.group(0).strip()}\n"
            f"{ind}}}"
        )
    v, n2304 = c2304_re.subn(_rep_c2304, v)
    assert n2304 == 96, f"expected 96 c2304 commits, got {n2304}"

    swa_re = re.compile(
        r"^(?P<ind>[ \t]*)kernel\.launch @g_ink_kv_prep_store_swa_ring\[%t\]\(%t, %spans, %swa_off_(?P<vis>\d+), %metadata, %info, %origins, %k_conv, %v_conv, %arena\) : \[index\]\(index, index, index, buffer, buffer, buffer, buffer, buffer, buffer\)$",
        flags=re.M,
    )
    swa_counter = [0]
    def _rep_swa(m: re.Match) -> str:
        ind, vis = m.group("ind"), m.group("vis")
        idx = swa_counter[0]
        swa_counter[0] += 1
        return (
            f"{ind}scf.if %v_defer {{\n"
            f"{ind}  %vswa_{idx} = index.constant {idx} : index\n"
            f"{ind}  kernel.launch @g_ink_swa_save_ring[%spans](%t, %spans, %vswa_{idx}, %swa_off_{vis}, %metadata, %origins, %arena, %v_stage_swa) : [index](index, index, index, index, buffer, buffer, buffer, buffer)\n"
            f"{ind}}}\n"
            f"{m.group(0)}"
        )
    v, nswa = swa_re.subn(_rep_swa, v)
    assert nswa == 40, f"expected 40 swa ring stores, got {nswa}"

    # Replace tail head + g_head_select with multi-row verify + g_spec_accept + deferred batch commits.
    head_anchor = '  %head = command.parameter %weights, "output.weight" : view<492106752xi8>\n'
    h_idx = v.find(head_anchor)
    assert h_idx != -1
    verify_tail = head_anchor + """  %v_width = config.get @llm.width : index
  %v_span_rows = index.mul %spans, %v_width : index
  %v_rows = index.min %v_span_rows, %t : index
  %v_c36_o = index.constant 36 : offset
  %v_c16_o = index.constant 16 : offset
  %v_c4_o = index.constant 4 : offset
  %v_rm_rec = index.scale %v_rows, %v_c36_o : index, offset -> offset
  %v_rm_sz = index.add %v_rm_rec, %v_c16_o : offset
  %v_rows_meta = buffer.alloca<global> align(256) %v_rm_sz : buffer
  %v_pred_sz = index.scale %v_rows, %v_c4_o : index, offset -> offset
  %v_preds = buffer.alloca<global> align(256) %v_pred_sz : buffer
  %v_part_sz = index.scale %v_rows, %part_bytes : index, offset -> offset
  %v_partials = buffer.alloca<global> align(256) %v_part_sz : buffer
  %v_log_sz = index.scale %v_rows, %logit_bytes : index, offset -> offset
  %v_logits = buffer.alloca<global> align(256) %v_log_sz : buffer
  kernel.launch @g_spec_rows(%spans, %v_rows, %metadata, %v_rows_meta) : (index, index, buffer, buffer)
  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %h16q) : [index, index](index, index, buffer, buffer)
  %v_one = index.cmp eq, %v_rows, %c1 : index
  %v_c4 = index.constant 4 : index
  %v_le4 = index.cmp ule, %v_rows, %v_c4 : index
  scf.if %v_one {
    kernel.launch @g_head_mv_q8[%v_rows](%v_rows, %t, %v_rows_meta, %h16q, %head, %v_partials, %v_logits) : [index](index, index, buffer, buffer, view<492106752xi8>, buffer, buffer)
  } else {
    scf.if %v_le4 {
      kernel.launch @g_head_mv_t4_q8[%v_rows](%v_rows, %t, %v_rows_meta, %h16q, %head, %v_partials, %v_logits) : [index](index, index, buffer, buffer, view<492106752xi8>, buffer, buffer)
    } else {
      %v_c8 = index.constant 8 : index
      %v_le8 = index.cmp ule, %v_rows, %v_c8 : index
      scf.if %v_le8 {
        kernel.launch @g_head_mv_t8_q8[%v_rows](%v_rows, %t, %v_rows_meta, %h16q, %head, %v_partials, %v_logits) : [index](index, index, buffer, buffer, view<492106752xi8>, buffer, buffer)
      } else {
        kernel.launch @g_head_mv_t16_q8[%v_rows](%v_rows, %t, %v_rows_meta, %h16q, %head, %v_partials, %v_logits) : [index](index, index, buffer, buffer, view<492106752xi8>, buffer, buffer)
      }
    }
  }
  kernel.launch @g_head_select[%v_rows](%v_rows, %v_rows_meta, %v_partials, %v_logits, %v_preds) : [index](index, buffer, buffer, buffer, buffer)
  kernel.launch @g_spec_accept(%spans, %v_rows, %t, %metadata, %tokens, %v_preds, %results, %committed) : (index, index, index, buffer, buffer, buffer, buffer, buffer)
  scf.if %v_defer {
    kernel.launch @g_ink_sconv_commit_batch_c1024[%spans](%t, %spans, %committed, %origins, %arena, %v_stage_c1024) : [index](index, index, buffer, buffer, buffer, buffer)
    kernel.launch @g_ink_sconv_commit_batch_c2304[%spans](%t, %spans, %committed, %origins, %arena, %v_stage_c2304) : [index](index, index, buffer, buffer, buffer, buffer)
    kernel.launch @g_ink_swa_restore_batch[%spans](%t, %spans, %metadata, %committed, %origins, %v_stage_swa, %arena) : [index](index, index, buffer, buffer, buffer, buffer, buffer)
  }
  command.return
}
"""
    return v[:h_idx] + verify_tail


def patch_llm_dspark(text: str) -> str:
    """Add DSpark speculative kernels, taps, @llm_verify, @llm_dflash_warm, and @llm_dflash_propose."""
    adv_marker = "command.def public target(@device) @llm_advance() "
    adv_idx = text.find(adv_marker)
    assert adv_idx != -1, "llm.loom: missing @llm_advance"
    adv_end = text.find("\n  command.return\n}\n", adv_idx)
    assert adv_end != -1, "llm.loom: missing @llm_advance closing"
    adv_end += len("\n  command.return\n}\n")

    prefix = text[:adv_idx]
    adv_orig = text[adv_idx:adv_end]
    suffix = text[adv_end:]

    adv_tapped = _inject_taps(adv_orig)
    verify_cmd = _build_llm_verify(adv_tapped)
    warm_cmd = _emit_dflash_warm()
    propose_cmd = _emit_dflash_propose()

    return (
        prefix
        + _DSPARK_CONFIG_DECLS
        + _DSPARK_KERNELS
        + "\n"
        + adv_tapped
        + "\n"
        + verify_cmd
        + "\n"
        + warm_cmd
        + "\n"
        + propose_cmd
        + suffix
    )
