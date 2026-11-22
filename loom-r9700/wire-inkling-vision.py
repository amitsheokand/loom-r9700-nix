#!/usr/bin/env python3
"""Wire Inkling 2 Mini's built-in 23-layer ViT vision encoder (mmproj-F16.gguf, inkling2v)
into hrx-loom 0.2.0 (dual-target RDNA3 gfx11 + RDNA4 gfx12).

Architecture (from mmproj-F16.gguf / Basalt inkling2v):
  - Patch size P = 20, spatial merge = 2x2 (cell = 40 px), n_embd E = 1024, FFN F = 4096,
    heads H = 16, head_dim D = 64, layers L = 23, eps = 1e-5, proj_dim = 2304.
  - Sequence length T = n_patches + 1 (token 0 = class token, tokens 1..n_patches = patches
    in 2x2 merge block order so cell m's 4 patches sit contiguously at 1 + 4*m .. 1 + 4*m + 3).
  - Token 0: v.class_embd + v.class_position_embd.weight.
  - Tokens 1..n_patches: Conv2d(3, 1024, k=20, s=20, no bias) on RGB normalized by (x/127.5 - 1.0)
    (top-left canvas padded with -1.0f when unscaled, or bicubic-resized when scaled down),
    plus 32x32 v.position_embd.weight bilinear hat interpolation (align_corners=false).
  - Pre-LayerNorm: v.pre_ln.weight / v.pre_ln.bias (eps=1e-5) across all T tokens (0..T-1).
  - 23 ViT blocks (l = 0..22):
      x = LayerNorm(h, v.blk.l.ln1)
      q = x @ attn_q.weight^T + attn_q.bias
      k = x @ attn_k.weight^T + attn_k.bias
      v = x @ attn_v.weight^T + attn_v.bias
      2D axial RoPE on q and k (interleaved pairs (2*j, 2*j+1), j=0..31:
        f = j % 16, inv = 10000^(-2*f/32), coord = (j < 16 ? px+1 : py+1) for t > 0 else 0)
      o = FullSelfAttention(q, k, v, scale=0.125)
      h += ls1.weight * ((o @ attn_out.weight^T) + attn_out.bias)
      x = LayerNorm(h, v.blk.l.ln2)
      f = gelu_erf((x @ ffn_up.weight^T) + ffn_up.bias)
      h += ls2.weight * ((f @ ffn_down.weight^T) + ffn_down.bias)
  - Spatial 2x2 merger + projector (class token 0 dropped; M = n_patches / 4 cells):
      m_in[m, 0..4095] = concat(h[1+4*m+0], h[1+4*m+1], h[1+4*m+2], h[1+4*m+3])
      m1 = gelu_erf(m_in @ mm.1.weight^T)   ([M, 4096], no bias)
      out = m1 @ mm.2.weight^T              ([M, 2304], no bias)
  - Injected into llm.loom @g_embed directly into both %x and %h without token_embd_norm.weight
    (matching Basalt's `x0 = emb`).
"""

from __future__ import annotations


def emit_vision_prepare_loom() -> str:
    return """// Image component preparation for Inkling 2 Mini (inkling2v, mmproj-F16.gguf).
func.decl import("prepare") @source_scope(%scope: buffer) -> (i32)
func.decl import("prepare") @parameter(%stage: i32, %binding: i32, %input: i32, %source: i32, %policy: buffer)
func.decl import("prepare") @memory_forecast(%root: buffer)

global.rodata.def @vision_forecast_root = bytes("766973696f6e5f666f7265636173745f6d656d6f7279")
global.rodata.def @vision_scope = bytes("766973696f6e")
global.rodata.def @vision_policy = bytes("776569676874732e6c6f6f6d")
global.rodata.def @vision_name = bytes("696e6b6c696e6732765f70726f6a6563746f72")
global.rodata.def @vision_empty = bytes("")

func.decl @vision_stage(%height: i64, %width: i64) -> (i32)

func.def public @text_image_prepare_v1(%directory: buffer, %options: buffer) -> (i64, i64, i64, i64, buffer, buffer, buffer) {
  %forecast = global.load @vision_forecast_root : buffer
  func.call @memory_forecast(%forecast) : (buffer)
  %h = scalar.constant 800 : i64
  %w = scalar.constant 800 : i64
  %stage = func.call @vision_stage(%h, %w) : (i64, i64) -> (i32)
  %scope = global.load @vision_scope : buffer
  %checkpoint = func.call @source_scope(%scope) : (buffer) -> (i32)
  %policy = global.load @vision_policy : buffer
  %c0 = scalar.constant 0 : i32
  func.call @parameter(%stage, %c0, %c0, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
  %units = scalar.constant 1 : i64
  %upload = scalar.constant 67174400 : i64
  %output = scalar.constant 37748736 : i64
  %state_bytes = index.constant 1 : offset
  %state = buffer.alloca<private> align(1) %state_bytes : buffer
  %empty = global.load @vision_empty : buffer
  %name = global.load @vision_name : buffer
  func.return %units, %upload, %output, %output, %empty, %name, %state : i64, i64, i64, i64, buffer, buffer, buffer
}
"""


def emit_vision_stages_loom() -> str:
    return """// Encoder stage per processed image size (shared by preparation and batching).
func.decl import("prepare") @stage(%root: buffer, %tag: i64) -> (i32)
func.decl import("prepare") @config_i64(%stage: i32, %key: buffer, %value: i64)
global.rodata.def @vision_root = bytes("766973696f6e5f656e636f6465")
global.rodata.def @vision_height_key = bytes("766973696f6e2e686569676874")
global.rodata.def @vision_width_key = bytes("766973696f6e2e7769647468")

func.def public @vision_stage(%height: i64, %width: i64) -> (i32) {
  %root = global.load @vision_root : buffer
  %one = scalar.constant 1 : i64
  %stage = func.call @stage(%root, %one) : (buffer, i64) -> (i32)
  %hk = global.load @vision_height_key : buffer
  %wk = global.load @vision_width_key : buffer
  func.call @config_i64(%stage, %hk, %height) : (i32, buffer, i64)
  func.call @config_i64(%stage, %wk, %width) : (i32, buffer, i64)
  func.return %stage : i32
}
"""


def emit_vision_control_loom() -> str:
    return """// Image component control for Inkling 2 Mini (cell = 40 px, row_bytes = 2304 * 4 = 9216).
type.decl import("finite", "batch") @FiniteBatch
func.decl import("finite", "batch_pack") @finite_batch_pack(%batch: ptr<@FiniteBatch>) -> (buffer, buffer)
func.decl import("finite", "batch_finish") @finite_batch_finish(%batch: ptr<@FiniteBatch>)
func.decl import("input") @require(%condition: i32, %diagnostic: buffer)
func.decl import("runner") @execute_2(%stage: i32, %input: hal.buffer, %output: hal.buffer) -> (i64)
func.decl @vision_stage(%height: i64, %width: i64) -> (i32)
global.rodata.def @vision_bad_rows = bytes("696d61676520726f7720636f756e7420646f6573206e6f74206d61746368207468652070726f6365737365642073697a65")

// Memory: private = largest packet + encoder scratch; retained = output rows (9216 B/row).
func.def public @vision_forecast_memory(%state: buffer, %ranges: buffer, %descriptors: buffer) -> (i64, i64, i64) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i64 = scalar.constant 0 : i64
  %c16 = scalar.constant 16 : i64
  %range_bytes = buffer.length %ranges
  %range_length = index.cast %range_bytes : offset to i64
  %count_value = scalar.divui %range_length, %c16 : i64
  %bounded = scalar.assume %count_value [range(%count_value, 1, 2147483647)] : i64
  %count = index.cast %bounded : i64 to index
  %records = buffer.view %ranges[%origin] : buffer -> view<[%count]x2xi64>
  %peak, %rows = scf.for %i = [%c0 to %count step %c1](%p = %c0_i64 : i64, %r = %c0_i64 : i64) -> (i64, i64) {
    %off = view.load %records[%i, 0] : view<[%count]x2xi64> -> i64
    %ob = scalar.assume %off [range(%off, 0, 9223372036854775807)] : i64
    %o = index.cast %ob : i64 to offset
    %header = buffer.view %descriptors[%o] : buffer -> view<8xi32>
    %len32 = view.load %header[7] : view<8xi32> -> i32
    %rows32 = view.load %header[2] : view<8xi32> -> i32
    %len = scalar.extui %len32 : i32 to i64
    %rr = scalar.extui %rows32 : i32 to i64
    %bigger = scalar.cmpi ugt, %len, %p : i64
    %np = scf.select %bigger, %len, %p : i64
    %nr = scalar.addi %r, %rr : i64
    scf.yield %np, %nr : i64, i64
  }
  %scratch = scalar.constant 536870912 : i64
  %private = scalar.addi %peak, %scratch : i64
  %row_bytes = scalar.constant 9216 : i64
  %retained = scalar.muli %rows, %row_bytes : i64
  func.return %private, %retained, %retained : i64, i64, i64
}

func.def public @text_image_prepare_request_v1(%state: buffer, %request: buffer) -> (buffer, i64, i64) {
  %origin = index.constant 0 : offset
  %header = buffer.view %request[%origin] : buffer -> view<8xi32>
  %meta32 = view.load %header[3] : view<8xi32> -> i32
  %meta64 = scalar.extui %meta32 : i32 to i64
  %mb = scalar.assume %meta64 [range(%meta64, 0, 4294967295)] : i64
  %mo = index.cast %mb : i64 to offset
  %meta = buffer.view %request[%mo] : buffer -> view<2xi32>
  %w = view.load %meta[0] : view<2xi32> -> i32
  %h = view.load %meta[1] : view<2xi32> -> i32
  %w64 = scalar.extui %w : i32 to i64
  %h64 = scalar.extui %h : i32 to i64
  %shift = scalar.constant 32 : i64
  %wbits = scalar.shli %w64, %shift : i64
  %tag = scalar.ori %wbits, %h64 : i64
  %units = scalar.constant 1 : i64
  func.return %request, %units, %tag : buffer, i64, i64
}

func.def public @text_image_prepare_batch_v1(%state: buffer, %tags: buffer, %batch: ptr<@FiniteBatch>) -> (i32, buffer, buffer) {
  %ranges, %packed = func.call @finite_batch_pack(%batch) : (ptr<@FiniteBatch>) -> (buffer, buffer)
  %origin = index.constant 0 : offset
  %range = buffer.view %ranges[%origin] : buffer -> view<1x2xi64>
  %source64 = view.load %range[0, 0] : view<1x2xi64> -> i64
  %sb = scalar.assume %source64 [range(%source64, 0, 4294967295)] : i64
  %source = index.cast %sb : i64 to offset
  %header = buffer.view %packed[%source] : buffer -> view<8xi32>
  %rows32 = view.load %header[2] : view<8xi32> -> i32
  %meta32 = view.load %header[3] : view<8xi32> -> i32
  %planes32 = view.load %header[5] : view<8xi32> -> i32
  %meta64 = scalar.extui %meta32 : i32 to i64
  %planes64 = scalar.extui %planes32 : i32 to i64
  %mb = scalar.assume %meta64 [range(%meta64, 0, 4294967295)] : i64
  %pb = scalar.assume %planes64 [range(%planes64, 0, 4294967295)] : i64
  %mo0 = index.cast %mb : i64 to offset
  %po0 = index.cast %pb : i64 to offset
  %mo = index.add %source, %mo0 : offset
  %po = index.add %source, %po0 : offset
  %meta = buffer.view %packed[%mo] : buffer -> view<2xi32>
  %dst_w = view.load %meta[0] : view<2xi32> -> i32
  %dst_h = view.load %meta[1] : view<2xi32> -> i32
  %plane = buffer.view %packed[%po] : buffer -> view<12xi32>
  %src_w = view.load %plane[1] : view<12xi32> -> i32
  %src_h = view.load %plane[2] : view<12xi32> -> i32
  %pix_rel = view.load %plane[6] : view<12xi32> -> i32
  %channels = view.load %plane[8] : view<12xi32> -> i32
  // rows check: cell = 40 px (patch 20 x merge 2)
  %pf = scalar.constant 40 : i32
  %gw = scalar.divui %dst_w, %pf : i32
  %gh = scalar.divui %dst_h, %pf : i32
  %expect = scalar.muli %gw, %gh : i32
  %ok = scalar.cmpi eq, %expect, %rows32 : i32
  %ok32 = scalar.extui %ok : i1 to i32
  %bad = global.load @vision_bad_rows : buffer
  func.call @require(%ok32, %bad) : (i32, buffer)
  // payload: 64-byte device header + the packed cohort
  %hdr_bytes = index.constant 64 : offset
  %packed_bytes = buffer.length %packed
  %payload_bytes = index.add %hdr_bytes, %packed_bytes : offset
  %payload = buffer.alloca<private> align(16) %payload_bytes : buffer
  buffer.copy %packed[%origin], %payload[%hdr_bytes], %packed_bytes
  %hdr = buffer.view %payload[%origin] : buffer -> view<16xi32>
  %source64b = index.cast %source : offset to i64
  %source32 = scalar.trunci %source64b : i64 to i32
  %sixty4 = scalar.constant 64 : i32
  %pix0 = scalar.addi %pix_rel, %source32 : i32
  %pix_abs = scalar.addi %pix0, %sixty4 : i32
  view.store %src_w, %hdr[0] : i32, view<16xi32>
  view.store %src_h, %hdr[1] : i32, view<16xi32>
  view.store %channels, %hdr[2] : i32, view<16xi32>
  view.store %pix_abs, %hdr[3] : i32, view<16xi32>
  view.store %dst_w, %hdr[4] : i32, view<16xi32>
  view.store %dst_h, %hdr[5] : i32, view<16xi32>
  // plan: [output bytes, result bytes, 1 request, rows] [first row, rows, 0] then [byte offset, width] per row
  %rows64 = scalar.extui %rows32 : i32 to i64
  %rb = scalar.assume %rows64 [range(%rows64, 1, 4096)] : i64
  %rows = index.cast %rb : i64 to index
  %c16 = index.constant 16 : offset
  %c56 = index.constant 56 : offset
  %row_bytes_total = index.scale %rows, %c16 : index, offset -> offset
  %plan_bytes = index.add %c56, %row_bytes_total : offset
  %plan = buffer.alloca<private> align(8) %plan_bytes : buffer
  %ph = buffer.view %plan[%origin] : buffer -> view<7xi64>
  %width = scalar.constant 2304 : i64
  %row_bytes = scalar.constant 9216 : i64
  %out_bytes = scalar.muli %rb, %row_bytes : i64
  %one = scalar.constant 1 : i64
  %zero = scalar.constant 0 : i64
  view.store %out_bytes, %ph[0] : i64, view<7xi64>
  view.store %out_bytes, %ph[1] : i64, view<7xi64>
  view.store %one, %ph[2] : i64, view<7xi64>
  view.store %rb, %ph[3] : i64, view<7xi64>
  view.store %zero, %ph[4] : i64, view<7xi64>
  view.store %rb, %ph[5] : i64, view<7xi64>
  view.store %zero, %ph[6] : i64, view<7xi64>
  %records = buffer.view %plan[%c56] : buffer -> view<[%rows]x2xi64>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %rows step %c1] {
    %i64 = index.cast %i : index to i64
    %offset = scalar.muli %i64, %row_bytes : i64
    view.store %offset, %records[%i, 0] : i64, view<[%rows]x2xi64>
    view.store %width, %records[%i, 1] : i64, view<[%rows]x2xi64>
  }
  %h64 = scalar.extui %dst_h : i32 to i64
  %w64 = scalar.extui %dst_w : i32 to i64
  %stage = func.call @vision_stage(%h64, %w64) : (i64, i64) -> (i32)
  func.return %stage, %payload, %plan : i32, buffer, buffer
}

func.def public @text_image_run_v1(%state: buffer, %stage: i32, %input: hal.buffer, %output: hal.buffer, %batch: ptr<@FiniteBatch>) {
  %work = func.call @execute_2(%stage, %input, %output) : (i32, hal.buffer, hal.buffer) -> (i64)
  func.return
}

func.def public @text_image_collect_batch_v1(%state: buffer, %batch: ptr<@FiniteBatch>) {
  func.call @finite_batch_finish(%batch) : (ptr<@FiniteBatch>)
  func.return
}
"""


def _emit_f16_gemm_pair(name: str, k_dim: int, n_dim: int, w_bytes: int, sel_idx: int) -> str:
    """Emit dual-target (@rdna3 + @rdna4) 64x64 WMMA F16 GEMM kernel + template selector."""
    return f"""kernel.def target(@rdna3) export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %n_const = index.constant {n_dim} : index
  %out_tiles = index.div %n_const, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, %w: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, 65536), multiple_of(%rows_arg, 64)] : index
  %k = index.constant {k_dim} : index
  %n = index.constant {n_dim} : index
  %origin = index.constant 0 : offset
  %block_bytes = index.constant 64 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %stage_bytes = index.constant 9216 : offset
  %row_tile_count = index.div %rows, %c64 : index
  %out_tile_count = index.div %n, %c64 : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_out_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row = index.mul %wave_row_group, %c16 : index
  %wave_out = index.mul %wave_out_group, %c32 : index
  %w_row0 = index.add %wave_out, %lane_row : index
  %w_row1 = index.add %w_row0, %c16 : index
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %c64 : index
  %blocks = index.div %k, %c32 : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x64xf32, %y_layout>
  %bounded_wave_row = index.assume %wave_row [range(%wave_row, 0, 48)] : index
  %bounded_wave_out = index.assume %wave_out [range(%wave_out, 0, 32)] : index
  %wave_out1 = index.add %bounded_wave_out, %c16 : index
  %x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<64x72xf16>
  %w_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %x_row = index.add %row_origin, %stage_row : index
  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index
  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %a0_raw, %a1_raw = scf.for %step = [%c0 to %k_steps step %c1](%a0_acc = %init : vector<8xf32>, %a1_acc = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {{
    %k_origin = index.mul %step, %c64 : index
    %x_col = index.add %k_origin, %stage_col : index
    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%rows]x[%k]xf16> -> vector<16xf16>
    vector.store %x_half, %x_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    %w_col = index.add %k_origin, %stage_col : index
    %deq0_two_b = index.constant 2 : offset
    %deq0_col_bytes = index.scale %w_col, %deq0_two_b : index, offset -> offset
    %deq0_origin = index.add %w_row_origin, %deq0_col_bytes : offset
    %deq0_view = buffer.view %w[%deq0_origin] : buffer -> view<16xf16>
    %deq0 = vector.load %deq0_view[0] : view<16xf16> -> vector<16xf16>
    vector.store %deq0, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %fc0 = index.constant 0 : index
    %lhs0 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc0] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>
    %rv000 = vector.load %w_stage_view[%w_row0, %fc0] : view<64x72xf16> -> vector<16xf16>
    %rhs000 = vector.fragment<rhs> %rv000 shape [%c16, %c16] : vector<16xf16>
    %na0_0 = vector.mma %lhs0, %rhs000, %a0_acc : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %rv010 = vector.load %w_stage_view[%w_row1, %fc0] : view<64x72xf16> -> vector<16xf16>
    %rhs010 = vector.fragment<rhs> %rv010 shape [%c16, %c16] : vector<16xf16>
    %na1_0 = vector.mma %lhs0, %rhs010, %a1_acc : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %fc1 = index.constant 16 : index
    %lhs1 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc1] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>
    %rv001 = vector.load %w_stage_view[%w_row0, %fc1] : view<64x72xf16> -> vector<16xf16>
    %rhs001 = vector.fragment<rhs> %rv001 shape [%c16, %c16] : vector<16xf16>
    %na0_1 = vector.mma %lhs1, %rhs001, %na0_0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %rv011 = vector.load %w_stage_view[%w_row1, %fc1] : view<64x72xf16> -> vector<16xf16>
    %rhs011 = vector.fragment<rhs> %rv011 shape [%c16, %c16] : vector<16xf16>
    %na1_1 = vector.mma %lhs1, %rhs011, %na1_0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %fc2 = index.constant 32 : index
    %lhs2 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc2] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>
    %rv002 = vector.load %w_stage_view[%w_row0, %fc2] : view<64x72xf16> -> vector<16xf16>
    %rhs002 = vector.fragment<rhs> %rv002 shape [%c16, %c16] : vector<16xf16>
    %na0_2 = vector.mma %lhs2, %rhs002, %na0_1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %rv012 = vector.load %w_stage_view[%w_row1, %fc2] : view<64x72xf16> -> vector<16xf16>
    %rhs012 = vector.fragment<rhs> %rv012 shape [%c16, %c16] : vector<16xf16>
    %na1_2 = vector.mma %lhs2, %rhs012, %na1_1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %fc3 = index.constant 48 : index
    %lhs3 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc3] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>
    %rv003 = vector.load %w_stage_view[%w_row0, %fc3] : view<64x72xf16> -> vector<16xf16>
    %rhs003 = vector.fragment<rhs> %rv003 shape [%c16, %c16] : vector<16xf16>
    %na0_3 = vector.mma %lhs3, %rhs003, %na0_2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %rv013 = vector.load %w_stage_view[%w_row1, %fc3] : view<64x72xf16> -> vector<16xf16>
    %rhs013 = vector.fragment<rhs> %rv013 shape [%c16, %c16] : vector<16xf16>
    %na1_3 = vector.mma %lhs3, %rhs013, %na1_2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %na0_3, %na1_3 : vector<8xf32>, vector<8xf32>
  }}
  %inv_bytes = index.constant 4096 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {{
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>
  %a0 = vector.mulf %a0_raw, %inv_frag : vector<8xf32>
  %a1 = vector.mulf %a1_raw, %inv_frag : vector<8xf32>
  vector.fragment.store<result> %a0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  kernel.return
}}
kernel.def target(@rdna4) export("{name}_rdna4") @{name}_rdna4(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %n_const = index.constant {n_dim} : index
  %out_tiles = index.div %n_const, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, %w: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, 65536), multiple_of(%rows_arg, 64)] : index
  %k = index.constant {k_dim} : index
  %n = index.constant {n_dim} : index
  %origin = index.constant 0 : offset
  %block_bytes = index.constant 64 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %stage_bytes = index.constant 9216 : offset
  %row_tile_count = index.div %rows, %c64 : index
  %out_tile_count = index.div %n, %c64 : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_out_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row = index.mul %wave_row_group, %c16 : index
  %wave_out = index.mul %wave_out_group, %c32 : index
  %w_row0 = index.add %wave_out, %lane_row : index
  %w_row1 = index.add %w_row0, %c16 : index
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %c64 : index
  %blocks = index.div %k, %c32 : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x64xf32, %y_layout>
  %bounded_wave_row = index.assume %wave_row [range(%wave_row, 0, 48)] : index
  %bounded_wave_out = index.assume %wave_out [range(%wave_out, 0, 32)] : index
  %wave_out1 = index.add %bounded_wave_out, %c16 : index
  %x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<64x72xf16>
  %w_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>
  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>
  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x64xf16, %w_stage_view_tl>
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %x_row = index.add %row_origin, %stage_row : index
  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index
  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %a0_raw, %a1_raw = scf.for %step = [%c0 to %k_steps step %c1](%a0_acc = %init : vector<8xf32>, %a1_acc = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {{
    %k_origin = index.mul %step, %c64 : index
    %x_col = index.add %k_origin, %stage_col : index
    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%rows]x[%k]xf16> -> vector<16xf16>
    vector.store %x_half, %x_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    %w_col = index.add %k_origin, %stage_col : index
    %deq0_two_b = index.constant 2 : offset
    %deq0_col_bytes = index.scale %w_col, %deq0_two_b : index, offset -> offset
    %deq0_origin = index.add %w_row_origin, %deq0_col_bytes : offset
    %deq0_view = buffer.view %w[%deq0_origin] : buffer -> view<16xf16>
    %deq0 = vector.load %deq0_view[0] : view<16xf16> -> vector<16xf16>
    vector.store %deq0, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %fc0 = index.constant 0 : index
    %lhs0 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc0] shape [%c16, %c16] : view<64x72xf16> -> vector<8xf16>
    %rv000_b = index.sub %w_row0, %lane_row : index
    %rhs000 = vector.fragment.load<rhs> %w_stage_view_t[%fc0, %rv000_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na0_0 = vector.mma %lhs0, %rhs000, %a0_acc : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %rv010_b = index.sub %w_row1, %lane_row : index
    %rhs010 = vector.fragment.load<rhs> %w_stage_view_t[%fc0, %rv010_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na1_0 = vector.mma %lhs0, %rhs010, %a1_acc : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %fc1 = index.constant 16 : index
    %lhs1 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc1] shape [%c16, %c16] : view<64x72xf16> -> vector<8xf16>
    %rv001_b = index.sub %w_row0, %lane_row : index
    %rhs001 = vector.fragment.load<rhs> %w_stage_view_t[%fc1, %rv001_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na0_1 = vector.mma %lhs1, %rhs001, %na0_0 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %rv011_b = index.sub %w_row1, %lane_row : index
    %rhs011 = vector.fragment.load<rhs> %w_stage_view_t[%fc1, %rv011_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na1_1 = vector.mma %lhs1, %rhs011, %na1_0 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %fc2 = index.constant 32 : index
    %lhs2 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc2] shape [%c16, %c16] : view<64x72xf16> -> vector<8xf16>
    %rv002_b = index.sub %w_row0, %lane_row : index
    %rhs002 = vector.fragment.load<rhs> %w_stage_view_t[%fc2, %rv002_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na0_2 = vector.mma %lhs2, %rhs002, %na0_1 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %rv012_b = index.sub %w_row1, %lane_row : index
    %rhs012 = vector.fragment.load<rhs> %w_stage_view_t[%fc2, %rv012_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na1_2 = vector.mma %lhs2, %rhs012, %na1_1 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %fc3 = index.constant 48 : index
    %lhs3 = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc3] shape [%c16, %c16] : view<64x72xf16> -> vector<8xf16>
    %rv003_b = index.sub %w_row0, %lane_row : index
    %rhs003 = vector.fragment.load<rhs> %w_stage_view_t[%fc3, %rv003_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na0_3 = vector.mma %lhs3, %rhs003, %na0_2 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %rv013_b = index.sub %w_row1, %lane_row : index
    %rhs013 = vector.fragment.load<rhs> %w_stage_view_t[%fc3, %rv013_b] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>
    %na1_3 = vector.mma %lhs3, %rhs013, %na1_2 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %na0_3, %na1_3 : vector<8xf32>, vector<8xf32>
  }}
  %inv_bytes = index.constant 4096 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {{
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>
  %a0 = vector.mulf %a0_raw, %inv_frag : vector<8xf32>
  %a1 = vector.mulf %a1_raw, %inv_frag : vector<8xf32>
  vector.fragment.store<result> %a0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  kernel.return
}}
template.decl @sel{sel_idx}_{name}(%g0: index, %g1: index, %a0: index, %a1: index, %a2: index, %a3: buffer, %a4: buffer, %a5: view<{w_bytes}xi8>, %a6: buffer)
template.def<@sel{sel_idx}_{name}> target(@rdna3) @sel{sel_idx}_{name}_rdna3(%g0: index, %g1: index, %a0: index, %a1: index, %a2: index, %a3: buffer, %a4: buffer, %a5: view<{w_bytes}xi8>, %a6: buffer) {{
  kernel.launch @{name}[%g0, %g1](%a0, %a1, %a2, %a3, %a4, %a5, %a6) : [index, index](index, index, index, buffer, buffer, view<{w_bytes}xi8>, buffer)
  template.return
}}
template.def<@sel{sel_idx}_{name}> target(@rdna4) @sel{sel_idx}_{name}_rdna4(%g0: index, %g1: index, %a0: index, %a1: index, %a2: index, %a3: buffer, %a4: buffer, %a5: view<{w_bytes}xi8>, %a6: buffer) {{
  kernel.launch @{name}_rdna4[%g0, %g1](%a0, %a1, %a2, %a3, %a4, %a5, %a6) : [index, index](index, index, index, buffer, buffer, view<{w_bytes}xi8>, buffer)
  template.return
}}
"""


def emit_vision_encoder_loom() -> str:
    gemm_1024x1024 = _emit_f16_gemm_pair("v_gemm_f16_store_1024x1024", 1024, 1024, 2097152, 1)
    gemm_1024x4096 = _emit_f16_gemm_pair("v_gemm_f16_store_1024x4096", 1024, 4096, 8388608, 2)
    gemm_4096x1024 = _emit_f16_gemm_pair("v_gemm_f16_store_4096x1024", 4096, 1024, 8388608, 3)
    gemm_4096x4096 = _emit_f16_gemm_pair("v_gemm_f16_store_4096x4096", 4096, 4096, 33554432, 4)
    gemm_4096x2304 = _emit_f16_gemm_pair("v_gemm_f16_store_4096x2304", 4096, 2304, 18874368, 5)

    param_lines = [
        '  %w_v_class_embd = command.parameter %weights, "v.class_embd" : view<4096xi8>',
        '  %w_v_class_pos = command.parameter %weights, "v.class_position_embd.weight" : view<4096xi8>',
        '  %w_v_patch_embd = command.parameter %weights, "v.patch_embd.weight" : view<2457600xi8>',
        '  %w_v_pos_embd = command.parameter %weights, "v.position_embd.weight" : view<4194304xi8>',
        '  %w_v_pre_ln_w = command.parameter %weights, "v.pre_ln.weight" : view<4096xi8>',
        '  %w_v_pre_ln_b = command.parameter %weights, "v.pre_ln.bias" : view<4096xi8>',
    ]
    block_calls = []
    for l in range(23):
        param_lines.extend([
            f'  %w_v_blk_{l}_ln1_w = command.parameter %weights, "v.blk.{l}.ln1.weight" : view<4096xi8>',
            f'  %w_v_blk_{l}_ln1_b = command.parameter %weights, "v.blk.{l}.ln1.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_attn_q_w = command.parameter %weights, "v.blk.{l}.attn_q.weight" : view<2097152xi8>',
            f'  %w_v_blk_{l}_attn_q_b = command.parameter %weights, "v.blk.{l}.attn_q.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_attn_k_w = command.parameter %weights, "v.blk.{l}.attn_k.weight" : view<2097152xi8>',
            f'  %w_v_blk_{l}_attn_k_b = command.parameter %weights, "v.blk.{l}.attn_k.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_attn_v_w = command.parameter %weights, "v.blk.{l}.attn_v.weight" : view<2097152xi8>',
            f'  %w_v_blk_{l}_attn_v_b = command.parameter %weights, "v.blk.{l}.attn_v.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_attn_out_w = command.parameter %weights, "v.blk.{l}.attn_out.weight" : view<2097152xi8>',
            f'  %w_v_blk_{l}_attn_out_b = command.parameter %weights, "v.blk.{l}.attn_out.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_ls1_w = command.parameter %weights, "v.blk.{l}.ls1.weight" : view<4096xi8>',
            f'  %w_v_blk_{l}_ln2_w = command.parameter %weights, "v.blk.{l}.ln2.weight" : view<4096xi8>',
            f'  %w_v_blk_{l}_ln2_b = command.parameter %weights, "v.blk.{l}.ln2.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_ffn_up_w = command.parameter %weights, "v.blk.{l}.ffn_up.weight" : view<8388608xi8>',
            f'  %w_v_blk_{l}_ffn_up_b = command.parameter %weights, "v.blk.{l}.ffn_up.bias" : view<16384xi8>',
            f'  %w_v_blk_{l}_ffn_down_w = command.parameter %weights, "v.blk.{l}.ffn_down.weight" : view<8388608xi8>',
            f'  %w_v_blk_{l}_ffn_down_b = command.parameter %weights, "v.blk.{l}.ffn_down.bias" : view<4096xi8>',
            f'  %w_v_blk_{l}_ls2_w = command.parameter %weights, "v.blk.{l}.ls2.weight" : view<4096xi8>',
        ])
        block_calls.extend([
            f'  kernel.launch @v_norm_ln_1024[%T](%T, %h, %w_v_blk_{l}_ln1_w, %w_v_blk_{l}_ln1_b, %x) : [index](index, buffer, view<4096xi8>, view<4096xi8>, buffer)',
            f'  kernel.launch @v_row_scale_f16[%Tp](%Tp, %k1024, %x, %sc, %x16) : [index](index, index, buffer, buffer, buffer)',
            f'  template.apply<@sel1_v_gemm_f16_store_1024x1024>(%Tp, %k1024, %Tp, %k1024, %k1024, %x16, %sc, %w_v_blk_{l}_attn_q_w, %q) : (index, index, index, index, index, buffer, buffer, view<2097152xi8>, buffer)',
            f'  kernel.launch @v_bias_none_1024[%T](%T, %q, %w_v_blk_{l}_attn_q_b, %q) : [index](index, buffer, view<4096xi8>, buffer)',
            f'  template.apply<@sel1_v_gemm_f16_store_1024x1024>(%Tp, %k1024, %Tp, %k1024, %k1024, %x16, %sc, %w_v_blk_{l}_attn_k_w, %k) : (index, index, index, index, index, buffer, buffer, view<2097152xi8>, buffer)',
            f'  kernel.launch @v_bias_none_1024[%T](%T, %k, %w_v_blk_{l}_attn_k_b, %k) : [index](index, buffer, view<4096xi8>, buffer)',
            f'  template.apply<@sel1_v_gemm_f16_store_1024x1024>(%Tp, %k1024, %Tp, %k1024, %k1024, %x16, %sc, %w_v_blk_{l}_attn_v_w, %v) : (index, index, index, index, index, buffer, buffer, view<2097152xi8>, buffer)',
            f'  kernel.launch @v_bias_none_1024[%T](%T, %v, %w_v_blk_{l}_attn_v_b, %v) : [index](index, buffer, view<4096xi8>, buffer)',
            f'  kernel.launch @v_rope_ink_axial_64x16[%T](%T, %gw, %q) : [index](index, index, buffer)',
            f'  kernel.launch @v_rope_ink_axial_64x16[%T](%T, %gw, %k) : [index](index, index, buffer)',
            f'  kernel.launch @v_attn_64x16[%T](%T, %gw, %q, %k, %v, %o) : [index](index, index, buffer, buffer, buffer, buffer)',
            f'  kernel.launch @v_row_scale_f16[%Tp](%Tp, %k1024, %o, %sc, %x16) : [index](index, index, buffer, buffer, buffer)',
            f'  template.apply<@sel1_v_gemm_f16_store_1024x1024>(%Tp, %k1024, %Tp, %k1024, %k1024, %x16, %sc, %w_v_blk_{l}_attn_out_w, %a) : (index, index, index, index, index, buffer, buffer, view<2097152xi8>, buffer)',
            f'  kernel.launch @v_bias_ls_residual_1024[%T](%T, %h, %a, %w_v_blk_{l}_attn_out_b, %w_v_blk_{l}_ls1_w) : [index](index, buffer, buffer, view<4096xi8>, view<4096xi8>)',
            f'  kernel.launch @v_norm_ln_1024[%T](%T, %h, %w_v_blk_{l}_ln2_w, %w_v_blk_{l}_ln2_b, %x) : [index](index, buffer, view<4096xi8>, view<4096xi8>, buffer)',
            f'  kernel.launch @v_row_scale_f16[%Tp](%Tp, %k1024, %x, %sc, %x16) : [index](index, index, buffer, buffer, buffer)',
            f'  template.apply<@sel2_v_gemm_f16_store_1024x4096>(%Tp, %k4096, %Tp, %k1024, %k4096, %x16, %sc, %w_v_blk_{l}_ffn_up_w, %f) : (index, index, index, index, index, buffer, buffer, view<8388608xi8>, buffer)',
            f'  kernel.launch @v_bias_gelu_erf_4096[%T](%T, %f, %w_v_blk_{l}_ffn_up_b, %f) : [index](index, buffer, view<16384xi8>, buffer)',
            f'  kernel.launch @v_row_scale_f16[%Tp](%Tp, %k4096, %f, %sc, %x16) : [index](index, index, buffer, buffer, buffer)',
            f'  template.apply<@sel3_v_gemm_f16_store_4096x1024>(%Tp, %k1024, %Tp, %k4096, %k1024, %x16, %sc, %w_v_blk_{l}_ffn_down_w, %a) : (index, index, index, index, index, buffer, buffer, view<8388608xi8>, buffer)',
            f'  kernel.launch @v_bias_ls_residual_1024[%T](%T, %h, %a, %w_v_blk_{l}_ffn_down_b, %w_v_blk_{l}_ls2_w) : [index](index, buffer, buffer, view<4096xi8>, view<4096xi8>)',
        ])
    param_lines.extend([
        '  %w_mm_1_w = command.parameter %weights, "mm.1.weight" : view<33554432xi8>',
        '  %w_mm_2_w = command.parameter %weights, "mm.2.weight" : view<18874368xi8>',
    ])

    params_str = "\n".join(param_lines)
    blocks_str = "\n".join(block_calls)

    return f"""target.decl @device
amdgpu.target<gfx11-generic> @rdna3
amdgpu.target<gfx12-generic> @rdna4

kernel.def export("v_row_scale_f16") @v_row_scale_f16(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer, %x16: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %k = index.assume %k_arg [range(%k_arg, 8, 65536), multiple_of(%k_arg, 8)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k]xf32>
  %h_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k]xf16>
  %local = scf.for %c = [%lane to %k step %c256](%m = %zero_f : f32) -> (f32) {{
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %a = scalar.absf %v : f32
    %n = scalar.maxnumf %m, %a : f32
    scf.yield %n : f32
  }}
  %maximum = kernel.workgroup.reduce<maxnumf> %local : f32
  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %c127 = scalar.constant 127 : i32
  %c13 = scalar.constant 13 : i32
  %c0_i32 = scalar.constant 0 : i32
  %c100 = scalar.constant 100 : i32
  %unbiased = scalar.subi %biased, %c127 : i32
  %raw_e = scalar.subi %unbiased, %c13 : i32
  %positive = scalar.maxsi %raw_e, %c0_i32 : i32
  %e = scalar.minsi %positive, %c100 : i32
  %scale_exp = scalar.subi %c127, %e : i32
  %inverse_exp = scalar.addi %c127, %e : i32
  %scale_bits = scalar.shli %scale_exp, %c23 : i32
  %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %inverse = scalar.bitcast %inverse_bits : i32 to f32
  scf.for %c = [%lane to %k step %c256] {{
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %sv = scalar.mulf %v, %scale : f32
    %hv = scalar.fptrunc %sv : f32 to f16
    view.store %hv, %h_view[%row, %bc] : f16, view<[%rows]x[%k]xf16>
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }}
  kernel.return
}}

// Prepare [img_h, img_w, 3] F32 normalized pixels in [-1, 1]:
// - If src_w <= img_w and src_h <= img_h (unscaled, canvas-padded to multiples of 40):
//   copy top-left (x < src_w && y < src_h) as rgb / 127.5 - 1.0, pad right/bottom with -1.0f.
// - Otherwise (scaled down from large image): read from bicubic-resized %pix buffer.
kernel.def export("v_ink_prep_pixels") @v_ink_prep_pixels(%n_pix: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n_pix, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%n_arg: index, %img_w_arg: index, %img_h_arg: index, %hdr_src: buffer, %resized_pix: buffer, %pixf: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %img_w = index.assume %img_w_arg [range(%img_w_arg, 40, 16384)] : index
  %img_h = index.assume %img_h_arg [range(%img_h_arg, 40, 16384)] : index
  %c256 = index.constant 256 : index
  %c3 = index.constant 3 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {{
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %hv = buffer.view %hdr_src[%zero] : buffer -> view<16xi32>
    %src_w32 = view.load %hv[0] : view<16xi32> -> i32
    %src_h32 = view.load %hv[1] : view<16xi32> -> i32
    %cin32 = view.load %hv[2] : view<16xi32> -> i32
    %pix32 = view.load %hv[3] : view<16xi32> -> i32
    %src_wb = scalar.assume %src_w32 [range(%src_w32, 1, 16384)] : i32
    %src_hb = scalar.assume %src_h32 [range(%src_h32, 1, 16384)] : i32
    %cin_b = scalar.assume %cin32 [range(%cin32, 3, 4)] : i32
    %pix_b = scalar.assume %pix32 [range(%pix32, 0, 2147483647)] : i32
    %src_w = index.cast %src_wb : i32 to index
    %src_h = index.cast %src_hb : i32 to index
    %cin = index.cast %cin_b : i32 to index
    %pix_off = index.cast %pix_b : i32 to index
    %unscaled_w = index.cmp ule, %src_w, %img_w : index
    %unscaled_h = index.cmp ule, %src_h, %img_h : index
    %unscaled = scalar.andi %unscaled_w, %unscaled_h : i1
    %scale = scalar.constant 0.00784313725490196 : f32
    %minus_one = scalar.constant -1.0 : f32
    %val = scf.if %unscaled -> (f32) {{
      %pixel = index.div %bi, %c3 : index
      %ch = index.rem %bi, %c3 : index
      %yy = index.div %pixel, %img_w : index
      %xx = index.rem %pixel, %img_w : index
      %in_x = index.cmp ult, %xx, %src_w : index
      %in_y = index.cmp ult, %yy, %src_h : index
      %inside = scalar.andi %in_x, %in_y : i1
      %v = scf.if %inside -> (f32) {{
        %src_px = index.madd %yy, %src_w, %xx : index
        %src_el0 = index.madd %src_px, %cin, %ch : index
        %src_el1 = index.add %src_el0, %pix_off : index
        %srcn = index.constant 268435456 : index
        %src_el = index.assume %src_el1 [lt(%src_el1, %srcn)] : index
        %sv = buffer.view %hdr_src[%zero] : buffer -> view<[%srcn]xi8>
        %raw = view.load %sv[%src_el] : view<[%srcn]xi8> -> i8
        %u = scalar.extui %raw : i8 to i32
        %f = scalar.uitofp %u : i32 to f32
        %r = scalar.fmaf %f, %scale, %minus_one : f32
        scf.yield %r : f32
      }} else {{
        scf.yield %minus_one : f32
      }}
      scf.yield %v : f32
    }} else {{
      %rv = buffer.view %resized_pix[%zero] : buffer -> view<[%n]xi8>
      %raw = view.load %rv[%bi] : view<[%n]xi8> -> i8
      %u = scalar.extui %raw : i8 to i32
      %f = scalar.uitofp %u : i32 to f32
      %r = scalar.fmaf %f, %scale, %minus_one : f32
      scf.yield %r : f32
    }}
    %yv = buffer.view %pixf[%zero] : buffer -> view<[%n]xf32>
    view.store %val, %yv[%bi] : f32, view<[%n]xf32>
  }}
  kernel.return
}}

// Compute row 0 (class token) and rows 1..n_patches (patch conv2d + 32x32 bilinear hat pos embed in 2x2 merge order).
kernel.def export("v_ink_patch_and_pos_1024") @v_ink_patch_and_pos_1024(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %gw_arg: index, %gh_arg: index, %img_w_arg: index, %pix: buffer, %w_patch: buffer, %w_cls: buffer, %w_cls_pos: buffer, %w_pos: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 65536)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 2, 1024)] : index
  %gh = index.assume %gh_arg [range(%gh_arg, 2, 1024)] : index
  %img_w = index.assume %img_w_arg [range(%img_w_arg, 40, 16384)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c20 = index.constant 20 : index
  %c400 = index.constant 400 : index
  %c1200 = index.constant 1200 : index
  %c1024 = index.constant 1024 : index
  %c256 = index.constant 256 : index
  %zf = scalar.constant 0.0 : f32
  %onef = scalar.constant 1.0 : f32
  %halff = scalar.constant 0.5 : f32
  %s32f = scalar.constant 32.0 : f32
  %c0i = scalar.constant 0 : i32
  %c1i = scalar.constant 1 : i32
  %c31i = scalar.constant 31 : i32
  %yv = buffer.view %y[%zero] : buffer -> view<[%tokens]x1024xf32>
  %is_cls = index.cmp eq, %t, %c0 : index
  %stage_bytes = index.constant 4800 : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %sv = buffer.view %stage[%zero] : buffer -> view<1200xf32>
  %tp = scf.select %is_cls, %c1, %t : index
  %t_patch0 = index.sub %tp, %c1 : index
  %t_patch = index.assume %t_patch0 [range(%t_patch0, 0, 65535)] : index
  %bw0 = index.div %gw, %c2 : index
  %bw1 = index.max %bw0, %c1 : index
  %bw = index.assume %bw1 [range(%bw1, 1, 1024)] : index
  %blk = index.div %t_patch, %c4 : index
  %sub = index.rem %t_patch, %c4 : index
  %ty = index.div %blk, %bw : index
  %tx = index.rem %blk, %bw : index
  %dy = index.div %sub, %c2 : index
  %dx = index.rem %sub, %c2 : index
  %py = index.madd %ty, %c2, %dy : index
  %px = index.madd %tx, %c2, %dx : index
  %y0 = index.mul %py, %c20 : index
  %x0 = index.mul %px, %c20 : index
  %pix_n = index.constant 268435456 : index
  %pv = buffer.view %pix[%zero] : buffer -> view<[%pix_n]xf32>
  scf.for %i = [%lane to %c1200 step %c256] {{
    %bi = index.assume %i [lt(%i, %c1200)] : index
    %v = scf.if %is_cls -> (f32) {{
      scf.yield %zf : f32
    }} else {{
      %ch = index.div %bi, %c400 : index
      %rem = index.rem %bi, %c400 : index
      %ky = index.div %rem, %c20 : index
      %kx = index.rem %rem, %c20 : index
      %yy = index.add %y0, %ky : index
      %xx = index.add %x0, %kx : index
      %rowpix = index.mul %yy, %img_w : index
      %pixel = index.add %rowpix, %xx : index
      %el = index.madd %pixel, %c3, %ch : index
      %bel = index.assume %el [ge(%el, %c0), lt(%el, %pix_n)] : index
      %pval = view.load %pv[%bel] : view<[%pix_n]xf32> -> f32
      scf.yield %pval : f32
    }}
    view.store %v, %sv[%bi] : f32, view<1200xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %is_cls {{
    %clsv = buffer.view %w_cls[%zero] : buffer -> view<1024xf32>
    %cposv = buffer.view %w_cls_pos[%zero] : buffer -> view<1024xf32>
    scf.for %e = [%lane to %c1024 step %c256] {{
      %be = index.assume %e [lt(%e, %c1024)] : index
      %c_val = view.load %clsv[%be] : view<1024xf32> -> f32
      %p_val = view.load %cposv[%be] : view<1024xf32> -> f32
      %s = scalar.addf %c_val, %p_val : f32
      view.store %s, %yv[%t, %be] : f32, view<[%tokens]x1024xf32>
    }}
  }} else {{
    // Bilinear hat taps on 32x32 table (align_corners=false): src = (i + 0.5) * 32 / size - 0.5
    %px_i = index.cast %px : index to i32
    %py_i = index.cast %py : index to i32
    %gw_i = index.cast %gw : index to i32
    %gh_i = index.cast %gh : index to i32
    %px_f = scalar.sitofp %px_i : i32 to f32
    %py_f = scalar.sitofp %py_i : i32 to f32
    %gw_f = scalar.sitofp %gw_i : i32 to f32
    %gh_f = scalar.sitofp %gh_i : i32 to f32
    %px_h = scalar.addf %px_f, %halff : f32
    %py_h = scalar.addf %py_f, %halff : f32
    %sx0 = scalar.mulf %px_h, %s32f : f32
    %sx1 = scalar.divf %sx0, %gw_f : f32
    %src_x = scalar.subf %sx1, %halff : f32
    %sy0 = scalar.mulf %py_h, %s32f : f32
    %sy1 = scalar.divf %sy0, %gh_f : f32
    %src_y = scalar.subf %sy1, %halff : f32
    %fl_x = scalar.floorf %src_x : f32
    %fl_y = scalar.floorf %src_y : f32
    %dx0 = scalar.subf %src_x, %fl_x : f32
    %adx0 = scalar.absf %dx0 : f32
    %wx0_raw = scalar.subf %onef, %adx0 : f32
    %wx0 = scalar.maxnumf %wx0_raw, %zf : f32
    %dx1 = scalar.subf %dx0, %onef : f32
    %adx1 = scalar.absf %dx1 : f32
    %wx1_raw = scalar.subf %onef, %adx1 : f32
    %wx1 = scalar.maxnumf %wx1_raw, %zf : f32
    %dy0 = scalar.subf %src_y, %fl_y : f32
    %ady0 = scalar.absf %dy0 : f32
    %wy0_raw = scalar.subf %onef, %ady0 : f32
    %wy0 = scalar.maxnumf %wy0_raw, %zf : f32
    %dy1 = scalar.subf %dy0, %onef : f32
    %ady1 = scalar.absf %dy1 : f32
    %wy1_raw = scalar.subf %onef, %ady1 : f32
    %wy1 = scalar.maxnumf %wy1_raw, %zf : f32
    %ax = scalar.fptosi %fl_x : f32 to i32
    %ax1 = scalar.addi %ax, %c1i : i32
    %ix0_a = scalar.maxsi %ax, %c0i : i32
    %ix0_b = scalar.minsi %ix0_a, %c31i : i32
    %ix0_c = scalar.assume %ix0_b [range(%ix0_b, 0, 31)] : i32
    %ix0 = index.cast %ix0_c : i32 to index
    %ix1_a = scalar.maxsi %ax1, %c0i : i32
    %ix1_b = scalar.minsi %ix1_a, %c31i : i32
    %ix1_c = scalar.assume %ix1_b [range(%ix1_b, 0, 31)] : i32
    %ix1 = index.cast %ix1_c : i32 to index
    %ay = scalar.fptosi %fl_y : f32 to i32
    %ay1 = scalar.addi %ay, %c1i : i32
    %iy0_a = scalar.maxsi %ay, %c0i : i32
    %iy0_b = scalar.minsi %iy0_a, %c31i : i32
    %iy0_c = scalar.assume %iy0_b [range(%iy0_b, 0, 31)] : i32
    %iy0 = index.cast %iy0_c : i32 to index
    %iy1_a = scalar.maxsi %ay1, %c0i : i32
    %iy1_b = scalar.minsi %iy1_a, %c31i : i32
    %iy1_c = scalar.assume %iy1_b [range(%iy1_b, 0, 31)] : i32
    %iy1 = index.cast %iy1_c : i32 to index
    %w00 = scalar.mulf %wy0, %wx0 : f32
    %w01 = scalar.mulf %wy0, %wx1 : f32
    %w10 = scalar.mulf %wy1, %wx0 : f32
    %w11 = scalar.mulf %wy1, %wx1 : f32
    %wn = index.constant 1228800 : index
    %wv = buffer.view %w_patch[%zero] : buffer -> view<[%wn]xf16>
    %tv = buffer.view %w_pos[%zero] : buffer -> view<32x32x1024xf32>
    scf.for %e = [%lane to %c1024 step %c256] {{
      %be = index.assume %e [lt(%e, %c1024)] : index
      %wrow = index.mul %be, %c1200 : index
      %dot = scf.for %k = [%c0 to %c1200 step %c1](%acc = %zf : f32) -> (f32) {{
        %bk = index.assume %k [lt(%k, %c1200)] : index
        %xk = view.load %sv[%bk] : view<1200xf32> -> f32
        %wi = index.add %wrow, %bk : index
        %bwi = index.assume %wi [lt(%wi, %wn)] : index
        %wkh = view.load %wv[%bwi] : view<[%wn]xf16> -> f16
        %wk = scalar.extf %wkh : f16 to f32
        %na = scalar.fmaf %wk, %xk, %acc : f32
        scf.yield %na : f32
      }}
      %p00 = view.load %tv[%iy0, %ix0, %be] : view<32x32x1024xf32> -> f32
      %p01 = view.load %tv[%iy0, %ix1, %be] : view<32x32x1024xf32> -> f32
      %p10 = view.load %tv[%iy1, %ix0, %be] : view<32x32x1024xf32> -> f32
      %p11 = view.load %tv[%iy1, %ix1, %be] : view<32x32x1024xf32> -> f32
      %pe0 = scalar.fmaf %p00, %w00, %dot : f32
      %pe1 = scalar.fmaf %p01, %w01, %pe0 : f32
      %pe2 = scalar.fmaf %p10, %w10, %pe1 : f32
      %pe3 = scalar.fmaf %p11, %w11, %pe2 : f32
      view.store %pe3, %yv[%t, %be] : f32, view<[%tokens]x1024xf32>
    }}
  }}
  kernel.return
}}

// Standard LayerNorm across 1024 columns (eps = 1e-5).
kernel.def export("v_norm_ln_1024") @v_norm_ln_1024(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %x: buffer, %w: buffer, %b: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant 32 : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %count = scalar.constant 1024.0 : f32
  %active = index.cmp ult, %lane, %groups : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x32x32xf32>
  %xv = scf.if %active -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, 31)] : index
    %a = vector.load %x_view[%row, %bl, %c0] : view<[%rows]x32x32xf32> -> vector<32xf32>
    scf.yield %a : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xlocal = vector.reduce<addf> %xv, %zero_f : vector<32xf32>, f32
  %xsum = kernel.workgroup.reduce<addf> %xlocal : f32
  %xmean = scalar.divf %xsum, %count : f32
  %xmeans = vector.splat %xmean : vector<32xf32>
  %xd = vector.subf %xv, %xmeans : vector<32xf32>
  %xc = scf.select %active, %xd, %zero32 : vector<32xf32>
  %sq = vector.mulf %xc, %xc : vector<32xf32>
  %local = vector.reduce<addf> %sq, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %mean = scalar.divf %sum, %count : f32
  %eps = scalar.constant 1.0e-5 : f32
  %stable = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %stable : f32
  %invs = vector.splat %inv : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, 31)] : index
    %xn = vector.mulf %xc, %invs : vector<32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<32x32xf32>
    %b_view = buffer.view %b[%zero] : buffer -> view<32x32xf32>
    %wv = vector.load %w_view[%bl, %c0] : view<32x32xf32> -> vector<32xf32>
    %bv = vector.load %b_view[%bl, %c0] : view<32x32xf32> -> vector<32xf32>
    %xw = vector.mulf %xn, %wv : vector<32xf32>
    %r = vector.addf %xw, %bv : vector<32xf32>
    %y_view = buffer.view %y[%zero] : buffer -> view<[%rows]x32x32xf32>
    vector.store %r, %y_view[%row, %bl, %c0] : vector<32xf32>, view<[%rows]x32x32xf32>
  }}
  kernel.return
}}

{gemm_1024x1024}
{gemm_1024x4096}
{gemm_4096x1024}
{gemm_4096x4096}
{gemm_4096x2304}

kernel.def export("v_bias_none_1024") @v_bias_none_1024(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cg = index.constant 4 : index
  kernel.launch.config workgroups(%rows, %cg, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %x: buffer, %bias: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %c256 = index.constant 256 : index
  %cols = index.constant 1024 : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %cg = kernel.workgroup.id<y> : index
  %lid = kernel.workitem.id<x> : index
  %c = index.madd %cg, %c256, %lid : index
  %live = index.cmp ult, %c, %cols : index
  scf.if %live {{
    %bc = index.assume %c [lt(%c, %cols)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%rows]x1024xf32>
    %bv = buffer.view %bias[%zero] : buffer -> view<1024xf32>
    %x0 = view.load %xv[%row, %bc] : view<[%rows]x1024xf32> -> f32
    %b0 = view.load %bv[%bc] : view<1024xf32> -> f32
    %a = scalar.addf %x0, %b0 : f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%rows]x1024xf32>
    view.store %a, %yv[%row, %bc] : f32, view<[%rows]x1024xf32>
  }}
  kernel.return
}}

// 2D axial RoPE for Inkling 2 Mini ViT (16 heads x 64 dim = 32 interleaved pairs per head).
// Token 0 (class token) is unrotated (coord = 0); tokens 1..n_patches use (px+1, py+1) in 2x2 merge order.
// Pair j in 0..31: f = j % 16, inv = exp(-f * ln(10000) / 16), coord = (j < 16 ? px+1 : py+1).
kernel.def export("v_rope_ink_axial_64x16") @v_rope_ink_axial_64x16(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %gw_arg: index, %x: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 65536)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 2, 4096)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %c512 = index.constant 512 : index
  %is_cls = index.cmp eq, %t, %c0 : index
  scf.if %is_cls {{
  }} else {{
    %t_patch0 = index.sub %t, %c1 : index
    %t_patch = index.assume %t_patch0 [range(%t_patch0, 0, 65535)] : index
    %bw0 = index.div %gw, %c2 : index
    %bw1 = index.max %bw0, %c1 : index
    %bw = index.assume %bw1 [range(%bw1, 1, 4096)] : index
    %blk = index.div %t_patch, %c4 : index
    %sub = index.rem %t_patch, %c4 : index
    %ty = index.div %blk, %bw : index
    %tx = index.rem %blk, %bw : index
    %dy = index.div %sub, %c2 : index
    %dx = index.rem %sub, %c2 : index
    %py0 = index.madd %ty, %c2, %dy : index
    %px0 = index.madd %tx, %c2, %dx : index
    %py1 = index.add %py0, %c1 : index
    %px1 = index.add %px0, %c1 : index
    %pxi = index.cast %px1 : index to i32
    %pyi = index.cast %py1 : index to i32
    %pxf = scalar.sitofp %pxi : i32 to f32
    %pyf = scalar.sitofp %pyi : i32 to f32
    %two_pi = scalar.constant 6.283185307179586 : f32
    %inv_two_pi = scalar.constant 0.15915494309189535 : f32
    %lnth = scalar.constant 9.210340371976184 : f32
    %c16f = scalar.constant 16.0 : f32
    %xv = buffer.view %x[%zero] : buffer -> view<[%tokens]x1024xf32>
    scf.for %w = [%lane to %c512 step %c256] {{
      %hd = index.div %w, %c32 : index
      %j0 = index.rem %w, %c32 : index
      %j = index.assume %j0 [range(%j0, 0, 31)] : index
      %is_x = index.cmp ult, %j, %c16 : index
      %f_idx = index.rem %j, %c16 : index
      %fi = index.cast %f_idx : index to i32
      %ff = scalar.sitofp %fi : i32 to f32
      %e0 = scalar.mulf %ff, %lnth : f32
      %e1 = scalar.divf %e0, %c16f : f32
      %e2 = scalar.negf %e1 : f32
      %inv = scalar.expf<afn> %e2 : f32
      %coord = scf.select %is_x, %pxf, %pyf : f32
      %ang0 = scalar.mulf %coord, %inv : f32
      %tr = scalar.mulf %ang0, %inv_two_pi : f32
      %trf = scalar.floorf %tr : f32
      %wrap = scalar.mulf %trf, %two_pi : f32
      %ang = scalar.subf %ang0, %wrap : f32
      %cs = scalar.cosf<afn> %ang : f32
      %sn = scalar.sinf<afn> %ang : f32
      %hcol = index.mul %hd, %c64 : index
      %j2 = index.mul %j, %c2 : index
      %clo0 = index.add %hcol, %j2 : index
      %chi0 = index.add %clo0, %c1 : index
      %clo = index.assume %clo0 [lt(%clo0, 1024)] : index
      %chi = index.assume %chi0 [lt(%chi0, 1024)] : index
      %a = view.load %xv[%t, %clo] : view<[%tokens]x1024xf32> -> f32
      %b = view.load %xv[%t, %chi] : view<[%tokens]x1024xf32> -> f32
      %acs = scalar.mulf %a, %cs : f32
      %bsn = scalar.mulf %b, %sn : f32
      %ra = scalar.subf %acs, %bsn : f32
      %asn = scalar.mulf %a, %sn : f32
      %rb = scalar.fmaf %b, %cs, %asn : f32
      view.store %ra, %xv[%t, %clo] : f32, view<[%tokens]x1024xf32>
      view.store %rb, %xv[%t, %chi] : f32, view<[%tokens]x1024xf32>
    }}
  }}
  kernel.return
}}

// Full bidirectional self-attention across all T tokens (16 heads, head_dim = 64, scale = 0.125).
kernel.def export("v_attn_64x16") @v_attn_64x16(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %heads = index.constant 16 : index
  kernel.launch.config workgroups(%tokens, %heads, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %gw_arg: index, %q: buffer, %k: buffer, %v: buffer, %o: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 65536)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, 4096)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %h_id = kernel.workgroup.id<y> : index
  %h = index.assume %h_id [range(%h_id, 0, 15)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cD = index.constant 64 : index
  %zf = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %scale = scalar.constant 0.125 : f32
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %qv = buffer.view %q[%zero] : buffer -> view<[%tokens]x1024xf32>
  %kv = buffer.view %k[%zero] : buffer -> view<[%tokens]x1024xf32>
  %vv = buffer.view %v[%zero] : buffer -> view<[%tokens]x1024xf32>
  %hcol = index.mul %h, %cD : index
  %c0_off = index.constant 0 : index
  %c32_off = index.constant 32 : index
  %qc0_r = index.add %hcol, %c0_off : index
  %qc0 = index.assume %qc0_r [range(%qc0_r, 0, 992)] : index
  %qr0 = vector.load %qv[%t, %qc0] : view<[%tokens]x1024xf32> -> vector<32xf32>
  %scv = vector.splat %scale : vector<32xf32>
  %q0 = vector.mulf %qr0, %scv : vector<32xf32>
  %qc1_r = index.add %hcol, %c32_off : index
  %qc1 = index.assume %qc1_r [range(%qc1_r, 0, 992)] : index
  %qr1 = vector.load %qv[%t, %qc1] : view<[%tokens]x1024xf32> -> vector<32xf32>
  %q1 = vector.mulf %qr1, %scv : vector<32xf32>
  %m_l, %l_l, %acc0_l, %acc1_l = scf.for %j = [%lane to %tokens step %c32](%m_c = %neg_big : f32, %l_c = %zf : f32, %acc0_c = %zero32 : vector<32xf32>, %acc1_c = %zero32 : vector<32xf32>) -> (f32, f32, vector<32xf32>, vector<32xf32>) {{
    %bj = index.assume %j [lt(%j, %tokens)] : index
    %kr0 = vector.load %kv[%bj, %qc0] : view<[%tokens]x1024xf32> -> vector<32xf32>
    %vr0 = vector.load %vv[%bj, %qc0] : view<[%tokens]x1024xf32> -> vector<32xf32>
    %pd0 = vector.mulf %q0, %kr0 : vector<32xf32>
    %ps0 = vector.reduce<addf> %pd0, %zf : vector<32xf32>, f32
    %kr1 = vector.load %kv[%bj, %qc1] : view<[%tokens]x1024xf32> -> vector<32xf32>
    %vr1 = vector.load %vv[%bj, %qc1] : view<[%tokens]x1024xf32> -> vector<32xf32>
    %pd1 = vector.mulf %q1, %kr1 : vector<32xf32>
    %ps1 = vector.reduce<addf> %pd1, %zf : vector<32xf32>, f32
    %s = scalar.addf %ps0, %ps1 : f32
    %mx = scalar.maxnumf %m_c, %s : f32
    %dm = scalar.subf %m_c, %mx : f32
    %corr = scalar.expf<afn> %dm : f32
    %ds = scalar.subf %s, %mx : f32
    %p = scalar.expf<afn> %ds : f32
    %lc = scalar.mulf %l_c, %corr : f32
    %ln = scalar.addf %lc, %p : f32
    %corrv = vector.splat %corr : vector<32xf32>
    %pv = vector.splat %p : vector<32xf32>
    %ac0 = vector.mulf %acc0_c, %corrv : vector<32xf32>
    %an0 = vector.fmaf %vr0, %pv, %ac0 : vector<32xf32>
    %ac1 = vector.mulf %acc1_c, %corrv : vector<32xf32>
    %an1 = vector.fmaf %vr1, %pv, %ac1 : vector<32xf32>
    scf.yield %mx, %ln, %an0, %an1 : f32, f32, vector<32xf32>, vector<32xf32>
  }}
  %mg = kernel.subgroup.reduce<maxnumf> %m_l : f32
  %dml = scalar.subf %m_l, %mg : f32
  %cl = scalar.expf<afn> %dml : f32
  %lw = scalar.mulf %l_l, %cl : f32
  %lg = kernel.subgroup.reduce<addf> %lw : f32
  %one = scalar.constant 1.0 : f32
  %invl = scalar.divf %one, %lg : f32
  %clv = vector.splat %cl : vector<32xf32>
  %stage_bytes = index.constant 8192 : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %st = buffer.view %stage[%zero] : buffer -> view<32x64xf32>
  %aw0 = vector.mulf %acc0_l, %clv : vector<32xf32>
  vector.store %aw0, %st[%lane, %c0_off] : vector<32xf32>, view<32x64xf32>
  %aw1 = vector.mulf %acc1_l, %clv : vector<32xf32>
  vector.store %aw1, %st[%lane, %c32_off] : vector<32xf32>, view<32x64xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %ov = buffer.view %o[%zero] : buffer -> view<[%tokens]x1024xf32>
  scf.for %col = [%lane to %cD step %c32] {{
    %bcol = index.assume %col [lt(%col, %cD)] : index
    %tot = scf.for %r = [%c0 to %c32 step %c1](%a = %zf : f32) -> (f32) {{
      %br = index.assume %r [range(%r, 0, 31)] : index
      %x = view.load %st[%br, %bcol] : view<32x64xf32> -> f32
      %na = scalar.addf %a, %x : f32
      scf.yield %na : f32
    }}
    %res = scalar.mulf %tot, %invl : f32
    %oc0 = index.add %hcol, %bcol : index
    %oc = index.assume %oc0 [lt(%oc0, 1024)] : index
    view.store %res, %ov[%t, %oc] : f32, view<[%tokens]x1024xf32>
  }}
  kernel.return
}}

// Fused bias + LayerScale + residual: h[r, c] += ls[c] * (a[r, c] + bias[c]).
kernel.def export("v_bias_ls_residual_1024") @v_bias_ls_residual_1024(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cg = index.constant 4 : index
  kernel.launch.config workgroups(%rows, %cg, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %h: buffer, %a: buffer, %bias: buffer, %ls: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %c256 = index.constant 256 : index
  %cols = index.constant 1024 : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %cg = kernel.workgroup.id<y> : index
  %lid = kernel.workitem.id<x> : index
  %c = index.madd %cg, %c256, %lid : index
  %live = index.cmp ult, %c, %cols : index
  scf.if %live {{
    %bc = index.assume %c [lt(%c, %cols)] : index
    %zero = index.constant 0 : offset
    %hv = buffer.view %h[%zero] : buffer -> view<[%rows]x1024xf32>
    %av = buffer.view %a[%zero] : buffer -> view<[%rows]x1024xf32>
    %bv = buffer.view %bias[%zero] : buffer -> view<1024xf32>
    %lsv = buffer.view %ls[%zero] : buffer -> view<1024xf32>
    %h0 = view.load %hv[%row, %bc] : view<[%rows]x1024xf32> -> f32
    %a0 = view.load %av[%row, %bc] : view<[%rows]x1024xf32> -> f32
    %b0 = view.load %bv[%bc] : view<1024xf32> -> f32
    %s0 = view.load %lsv[%bc] : view<1024xf32> -> f32
    %ab = scalar.addf %a0, %b0 : f32
    %res = scalar.fmaf %s0, %ab, %h0 : f32
    view.store %res, %hv[%row, %bc] : f32, view<[%rows]x1024xf32>
  }}
  kernel.return
}}

kernel.def export("v_bias_gelu_erf_4096") @v_bias_gelu_erf_4096(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cg = index.constant 16 : index
  kernel.launch.config workgroups(%rows, %cg, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %x: buffer, %bias: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %c256 = index.constant 256 : index
  %cols = index.constant 4096 : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %cg = kernel.workgroup.id<y> : index
  %lid = kernel.workitem.id<x> : index
  %c = index.madd %cg, %c256, %lid : index
  %live = index.cmp ult, %c, %cols : index
  scf.if %live {{
    %bc = index.assume %c [lt(%c, %cols)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%rows]x4096xf32>
    %bv = buffer.view %bias[%zero] : buffer -> view<4096xf32>
    %x0 = view.load %xv[%row, %bc] : view<[%rows]x4096xf32> -> f32
    %b0 = view.load %bv[%bc] : view<4096xf32> -> f32
    %a = scalar.addf %x0, %b0 : f32
    %one = scalar.constant 1.0 : f32
    %half = scalar.constant 0.5 : f32
    %rs2 = scalar.constant 0.7071067811865476 : f32
    %u = scalar.mulf %a, %rs2 : f32
    %au = scalar.absf %u : f32
    %hz = scalar.mulf %au, %half : f32
    %tden = scalar.addf %hz, %one : f32
    %t = scalar.divf %one, %tden : f32
    %p9 = scalar.constant 0.17087277 : f32
    %p8 = scalar.constant -0.82215223 : f32
    %p7 = scalar.constant 1.48851587 : f32
    %p6 = scalar.constant -1.13520398 : f32
    %p5 = scalar.constant 0.27886807 : f32
    %p4 = scalar.constant -0.18628806 : f32
    %p3 = scalar.constant 0.09678418 : f32
    %p2 = scalar.constant 0.37409196 : f32
    %p1 = scalar.constant 1.00002368 : f32
    %p0 = scalar.constant -1.26551223 : f32
    %q8 = scalar.fmaf %t, %p9, %p8 : f32
    %q7 = scalar.fmaf %t, %q8, %p7 : f32
    %q6 = scalar.fmaf %t, %q7, %p6 : f32
    %q5 = scalar.fmaf %t, %q6, %p5 : f32
    %q4 = scalar.fmaf %t, %q5, %p4 : f32
    %q3 = scalar.fmaf %t, %q4, %p3 : f32
    %q2 = scalar.fmaf %t, %q3, %p2 : f32
    %q1 = scalar.fmaf %t, %q2, %p1 : f32
    %q0 = scalar.fmaf %t, %q1, %p0 : f32
    %u2 = scalar.mulf %au, %au : f32
    %e0 = scalar.subf %q0, %u2 : f32
    %e1 = scalar.expf<afn> %e0 : f32
    %erfc = scalar.mulf %t, %e1 : f32
    %erfa = scalar.subf %one, %erfc : f32
    %zf_e = scalar.constant 0.0 : f32
    %neg = scalar.cmpf olt, %u, %zf_e : f32
    %erfn = scalar.negf %erfa : f32
    %erf = scf.select %neg, %erfn, %erfa : f32
    %onep = scalar.addf %erf, %one : f32
    %hx = scalar.mulf %a, %half : f32
    %r = scalar.mulf %hx, %onep : f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%rows]x4096xf32>
    view.store %r, %yv[%row, %bc] : f32, view<[%rows]x4096xf32>
  }}
  kernel.return
}}

kernel.def export("v_ew_gelu_erf") @v_ew_gelu_erf(%n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%n_arg: index, %x: buffer, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {{
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%n]xf32>
    %a = view.load %xv[%bi] : view<[%n]xf32> -> f32
    %one = scalar.constant 1.0 : f32
    %half = scalar.constant 0.5 : f32
    %rs2 = scalar.constant 0.7071067811865476 : f32
    %u = scalar.mulf %a, %rs2 : f32
    %au = scalar.absf %u : f32
    %hz = scalar.mulf %au, %half : f32
    %tden = scalar.addf %hz, %one : f32
    %t = scalar.divf %one, %tden : f32
    %p9 = scalar.constant 0.17087277 : f32
    %p8 = scalar.constant -0.82215223 : f32
    %p7 = scalar.constant 1.48851587 : f32
    %p6 = scalar.constant -1.13520398 : f32
    %p5 = scalar.constant 0.27886807 : f32
    %p4 = scalar.constant -0.18628806 : f32
    %p3 = scalar.constant 0.09678418 : f32
    %p2 = scalar.constant 0.37409196 : f32
    %p1 = scalar.constant 1.00002368 : f32
    %p0 = scalar.constant -1.26551223 : f32
    %q8 = scalar.fmaf %t, %p9, %p8 : f32
    %q7 = scalar.fmaf %t, %q8, %p7 : f32
    %q6 = scalar.fmaf %t, %q7, %p6 : f32
    %q5 = scalar.fmaf %t, %q6, %p5 : f32
    %q4 = scalar.fmaf %t, %q5, %p4 : f32
    %q3 = scalar.fmaf %t, %q4, %p3 : f32
    %q2 = scalar.fmaf %t, %q3, %p2 : f32
    %q1 = scalar.fmaf %t, %q2, %p1 : f32
    %q0 = scalar.fmaf %t, %q1, %p0 : f32
    %u2 = scalar.mulf %au, %au : f32
    %e0 = scalar.subf %q0, %u2 : f32
    %e1 = scalar.expf<afn> %e0 : f32
    %erfc = scalar.mulf %t, %e1 : f32
    %erfa = scalar.subf %one, %erfc : f32
    %zf_e = scalar.constant 0.0 : f32
    %neg = scalar.cmpf olt, %u, %zf_e : f32
    %erfn = scalar.negf %erfa : f32
    %erf = scf.select %neg, %erfn, %erfa : f32
    %onep = scalar.addf %erf, %one : f32
    %hx = scalar.mulf %a, %half : f32
    %r = scalar.mulf %hx, %onep : f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %r, %yv[%bi] : f32, view<[%n]xf32>
  }}
  kernel.return
}}

// Drop class token (row 0) and pack 4 contiguous 2x2-merged patch rows (1 + 4*m .. 1 + 4*m + 3) into [M, 4096].
kernel.def export("v_merge2x2_drop_cls_1024") @v_merge2x2_drop_cls_1024(%cells: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%cells, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%cells_arg: index, %tokens_arg: index, %h: buffer, %m_in: buffer) {{
  %cells = index.assume %cells_arg [range(%cells_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 5, 65536)] : index
  %m_id = kernel.workgroup.id<x> : index
  %m = index.assume %m_id [lt(%m_id, %cells)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c256 = index.constant 256 : index
  %c1024 = index.constant 1024 : index
  %c4096 = index.constant 4096 : index
  %hv = buffer.view %h[%zero] : buffer -> view<[%tokens]x1024xf32>
  %mv = buffer.view %m_in[%zero] : buffer -> view<[%cells]x4096xf32>
  %base0 = index.mul %m, %c4 : index
  %base = index.add %base0, %c1 : index
  scf.for %c = [%lane to %c4096 step %c256] {{
    %bc = index.assume %c [lt(%c, %c4096)] : index
    %sub = index.div %bc, %c1024 : index
    %col = index.rem %bc, %c1024 : index
    %bcol = index.assume %col [lt(%col, %c1024)] : index
    %r0 = index.add %base, %sub : index
    %r = index.assume %r0 [lt(%r0, %tokens)] : index
    %val = view.load %hv[%r, %bcol] : view<[%tokens]x1024xf32> -> f32
    view.store %val, %mv[%m, %bc] : f32, view<[%cells]x4096xf32>
  }}
  kernel.return
}}

kernel.def export("v_resize_bicubic_h_hdr") @v_resize_bicubic_h_hdr(%out_w: index, %max_rows: index) {{
  %count = index.mul %out_w, %max_rows : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %count, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%out_n_arg: index, %other_arg: index, %src: buffer, %dst: buffer) {{
  %zero_h = index.constant 0 : offset
  %hv = buffer.view %src[%zero_h] : buffer -> view<16xi32>
  %in_w32 = view.load %hv[0] : view<16xi32> -> i32
  %rows32 = view.load %hv[1] : view<16xi32> -> i32
  %cin32 = view.load %hv[2] : view<16xi32> -> i32
  %pix32 = view.load %hv[3] : view<16xi32> -> i32
  %in_b = scalar.assume %in_w32 [range(%in_w32, 1, 16384)] : i32
  %in_n = index.cast %in_b : i32 to index
  %rows_b = scalar.assume %rows32 [range(%rows32, 1, 16384)] : i32
  %rows_i = index.cast %rows_b : i32 to index
  %cin_b = scalar.assume %cin32 [range(%cin32, 3, 4)] : i32
  %cin_i = index.cast %cin_b : i32 to index
  %pix_b = scalar.assume %pix32 [range(%pix32, 0, 2147483647)] : i32
  %pix_i = index.cast %pix_b : i32 to index
  %out_n = index.assume %out_n_arg [range(%out_n_arg, 1, 65536)] : index
  %other = index.assume %other_arg [range(%other_arg, 1, 65536)] : index
  %count = index.mul %rows_i, %out_n : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %count : index
  scf.if %live {{
    %zero = index.constant 0 : offset
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %c3 = index.constant 3 : index
    %ccin = index.add %cin_i, %c0 : index
    %zf = scalar.constant 0.0 : f32
    %onef = scalar.constant 1.0 : f32
    %halff = scalar.constant 0.5 : f32
    %twof = scalar.constant 2.0 : f32
    %fourf = scalar.constant 4.0 : f32
    %fivef = scalar.constant 5.0 : f32
    %eightf = scalar.constant 8.0 : f32
    %c1p5 = scalar.constant 1.5 : f32
    %c2p5 = scalar.constant 2.5 : f32
    %mhalf = scalar.constant -0.5 : f32
    %fxp = scalar.constant 4194304.0 : f32
    %o = index.rem %i, %out_n : index
    %row = index.div %i, %out_n : index
    %in_i = index.cast %in_n : index to i32
    %out_i = index.cast %out_n : index to i32
    %in_f = scalar.sitofp %in_i : i32 to f32
    %out_f = scalar.sitofp %out_i : i32 to f32
    %scale = scalar.divf %in_f, %out_f : f32
    %fs = scalar.maxnumf %scale, %onef : f32
    %supc = scalar.constant 2.0 : f32
    %support = scalar.mulf %supc, %fs : f32
    %ss = scalar.divf %onef, %fs : f32
    %oi = index.cast %o : index to i32
    %of = scalar.sitofp %oi : i32 to f32
    %oh = scalar.addf %of, %halff : f32
    %center = scalar.mulf %oh, %scale : f32
    %lo0 = scalar.subf %center, %support : f32
    %lo1 = scalar.addf %lo0, %halff : f32
    %lo2 = scalar.fptosi %lo1 : f32 to i32
    %zi = scalar.constant 0 : i32
    %lo3 = scalar.maxsi %lo2, %zi : i32
    %hi0 = scalar.addf %center, %support : f32
    %hi1 = scalar.addf %hi0, %halff : f32
    %hi2 = scalar.fptosi %hi1 : f32 to i32
    %hi3 = scalar.minsi %hi2, %in_i : i32
    %lo4 = scalar.assume %lo3 [range(%lo3, 0, 65536)] : i32
    %hi4 = scalar.assume %hi3 [range(%hi3, 0, 65536)] : i32
    %xlo = index.cast %lo4 : i32 to index
    %xhi = index.cast %hi4 : i32 to index
    %ww = scf.for %xi = [%xlo to %xhi step %c1](%acc = %zf : f32) -> (f32) {{
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
      %wv_x = scalar.absf %d : f32
      %wv_lt1 = scalar.cmpf olt, %wv_x, %onef : f32
      %wv_lt2 = scalar.cmpf olt, %wv_x, %twof : f32
      %wv_p1a = scalar.mulf %wv_x, %c1p5 : f32
      %wv_p1b = scalar.subf %wv_p1a, %c2p5 : f32
      %wv_p1c = scalar.mulf %wv_p1b, %wv_x : f32
      %wv_p1d = scalar.mulf %wv_p1c, %wv_x : f32
      %wv_p1 = scalar.addf %wv_p1d, %onef : f32
      %wv_p2a = scalar.subf %wv_x, %fivef : f32
      %wv_p2b = scalar.mulf %wv_p2a, %wv_x : f32
      %wv_p2c = scalar.addf %wv_p2b, %eightf : f32
      %wv_p2d = scalar.mulf %wv_p2c, %wv_x : f32
      %wv_p2e = scalar.subf %wv_p2d, %fourf : f32
      %wv_p2 = scalar.mulf %wv_p2e, %mhalf : f32
      %wv_q = scf.select %wv_lt2, %wv_p2, %zf : f32
      %wv = scf.select %wv_lt1, %wv_p1, %wv_q : f32
      %na = scalar.addf %acc, %wv : f32
      scf.yield %na : f32
    }}
    %wnz = scalar.cmpf one, %ww, %zf : f32
    %wdiv = scf.select %wnz, %ww, %onef : f32
    %srcn = index.constant 268435456 : index
    %sv = buffer.view %src[%zero] : buffer -> view<[%srcn]xi8>
    %bias = scalar.constant 2097152 : i32
    %s0, %s1, %s2 = scf.for %xi = [%xlo to %xhi step %c1](%a0 = %bias : i32, %a1 = %bias : i32, %a2 = %bias : i32) -> (i32, i32, i32) {{
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
      %wv_x = scalar.absf %d : f32
      %wv_lt1 = scalar.cmpf olt, %wv_x, %onef : f32
      %wv_lt2 = scalar.cmpf olt, %wv_x, %twof : f32
      %wv_p1a = scalar.mulf %wv_x, %c1p5 : f32
      %wv_p1b = scalar.subf %wv_p1a, %c2p5 : f32
      %wv_p1c = scalar.mulf %wv_p1b, %wv_x : f32
      %wv_p1d = scalar.mulf %wv_p1c, %wv_x : f32
      %wv_p1 = scalar.addf %wv_p1d, %onef : f32
      %wv_p2a = scalar.subf %wv_x, %fivef : f32
      %wv_p2b = scalar.mulf %wv_p2a, %wv_x : f32
      %wv_p2c = scalar.addf %wv_p2b, %eightf : f32
      %wv_p2d = scalar.mulf %wv_p2c, %wv_x : f32
      %wv_p2e = scalar.subf %wv_p2d, %fourf : f32
      %wv_p2 = scalar.mulf %wv_p2e, %mhalf : f32
      %wv_q = scf.select %wv_lt2, %wv_p2, %zf : f32
      %wv = scf.select %wv_lt1, %wv_p1, %wv_q : f32
      %wn = scalar.divf %wv, %wdiv : f32
      %wfx = scalar.mulf %wn, %fxp : f32
      %neg = scalar.cmpf olt, %wn, %zf : f32
      %rnd = scf.select %neg, %mhalf, %halff : f32
      %wr = scalar.addf %wfx, %rnd : f32
      %wi = scalar.fptosi %wr : f32 to i32
      %src_idx = index.madd %row, %in_n, %xi : index
      %el_rel = index.mul %src_idx, %ccin : index
      %el0 = index.add %el_rel, %pix_i : index
      %el = index.assume %el0 [lt(%el0, %srcn)] : index
      %el1r = index.add %el0, %c1 : index
      %el1 = index.assume %el1r [lt(%el1r, %srcn)] : index
      %el2r = index.add %el1r, %c1 : index
      %el2 = index.assume %el2r [lt(%el2r, %srcn)] : index
      %p0r = view.load %sv[%el] : view<[%srcn]xi8> -> i8
      %p1r = view.load %sv[%el1] : view<[%srcn]xi8> -> i8
      %p2r = view.load %sv[%el2] : view<[%srcn]xi8> -> i8
      %p0 = scalar.extui %p0r : i8 to i32
      %p1 = scalar.extui %p1r : i8 to i32
      %p2 = scalar.extui %p2r : i8 to i32
      %m0 = scalar.muli %p0, %wi : i32
      %m1 = scalar.muli %p1, %wi : i32
      %m2 = scalar.muli %p2, %wi : i32
      %n0 = scalar.addi %a0, %m0 : i32
      %n1 = scalar.addi %a1, %m1 : i32
      %n2 = scalar.addi %a2, %m2 : i32
      scf.yield %n0, %n1, %n2 : i32, i32, i32
    }}
    %sh = scalar.constant 22 : i32
    %c255i = scalar.constant 255 : i32
    %dst_idx = index.madd %row, %out_n, %o : index
    %dstn = index.constant 268435456 : index
    %dv = buffer.view %dst[%zero] : buffer -> view<[%dstn]xi8>
    %de0 = index.mul %dst_idx, %c3 : index
    %c2x = index.constant 2 : index
    %q0a = scalar.shrsi %s0, %sh : i32
    %q0b = scalar.maxsi %q0a, %zi : i32
    %q0c = scalar.minsi %q0b, %c255i : i32
    %q0 = scalar.trunci %q0c : i32 to i8
    %dc0r = index.add %de0, %c0 : index
    %dc0 = index.assume %dc0r [lt(%dc0r, %dstn)] : index
    view.store %q0, %dv[%dc0] : i8, view<[%dstn]xi8>
    %q1a = scalar.shrsi %s1, %sh : i32
    %q1b = scalar.maxsi %q1a, %zi : i32
    %q1c = scalar.minsi %q1b, %c255i : i32
    %q1 = scalar.trunci %q1c : i32 to i8
    %dc1r = index.add %de0, %c1 : index
    %dc1 = index.assume %dc1r [lt(%dc1r, %dstn)] : index
    view.store %q1, %dv[%dc1] : i8, view<[%dstn]xi8>
    %q2a = scalar.shrsi %s2, %sh : i32
    %q2b = scalar.maxsi %q2a, %zi : i32
    %q2c = scalar.minsi %q2b, %c255i : i32
    %q2 = scalar.trunci %q2c : i32 to i8
    %dc2r = index.add %de0, %c2x : index
    %dc2 = index.assume %dc2r [lt(%dc2r, %dstn)] : index
    view.store %q2, %dv[%dc2] : i8, view<[%dstn]xi8>
  }}
  kernel.return
}}

kernel.def export("v_resize_bicubic_v_hdr") @v_resize_bicubic_v_hdr(%out_h: index, %cols: index) {{
  %count = index.mul %out_h, %cols : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %count, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%out_n_arg: index, %other_arg: index, %hdr: buffer, %src: buffer, %dst: buffer) {{
  %zero_h = index.constant 0 : offset
  %hv = buffer.view %hdr[%zero_h] : buffer -> view<16xi32>
  %in_w32 = view.load %hv[1] : view<16xi32> -> i32
  %in_b = scalar.assume %in_w32 [range(%in_w32, 1, 16384)] : i32
  %in_n = index.cast %in_b : i32 to index
  %out_n = index.assume %out_n_arg [range(%out_n_arg, 1, 65536)] : index
  %other = index.assume %other_arg [range(%other_arg, 1, 65536)] : index
  %count = index.mul %other, %out_n : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %count : index
  scf.if %live {{
    %zero = index.constant 0 : offset
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %c3 = index.constant 3 : index
    %ccin = index.constant 3 : index
    %zf = scalar.constant 0.0 : f32
    %onef = scalar.constant 1.0 : f32
    %halff = scalar.constant 0.5 : f32
    %twof = scalar.constant 2.0 : f32
    %fourf = scalar.constant 4.0 : f32
    %fivef = scalar.constant 5.0 : f32
    %eightf = scalar.constant 8.0 : f32
    %c1p5 = scalar.constant 1.5 : f32
    %c2p5 = scalar.constant 2.5 : f32
    %mhalf = scalar.constant -0.5 : f32
    %fxp = scalar.constant 4194304.0 : f32
    %o_v = index.div %i, %other : index
    %col = index.rem %i, %other : index
    %in_i = index.cast %in_n : index to i32
    %out_i = index.cast %out_n : index to i32
    %in_f = scalar.sitofp %in_i : i32 to f32
    %out_f = scalar.sitofp %out_i : i32 to f32
    %scale = scalar.divf %in_f, %out_f : f32
    %fs = scalar.maxnumf %scale, %onef : f32
    %supc = scalar.constant 2.0 : f32
    %support = scalar.mulf %supc, %fs : f32
    %ss = scalar.divf %onef, %fs : f32
    %oi = index.cast %o_v : index to i32
    %of = scalar.sitofp %oi : i32 to f32
    %oh = scalar.addf %of, %halff : f32
    %center = scalar.mulf %oh, %scale : f32
    %lo0 = scalar.subf %center, %support : f32
    %lo1 = scalar.addf %lo0, %halff : f32
    %lo2 = scalar.fptosi %lo1 : f32 to i32
    %zi = scalar.constant 0 : i32
    %lo3 = scalar.maxsi %lo2, %zi : i32
    %hi0 = scalar.addf %center, %support : f32
    %hi1 = scalar.addf %hi0, %halff : f32
    %hi2 = scalar.fptosi %hi1 : f32 to i32
    %hi3 = scalar.minsi %hi2, %in_i : i32
    %lo4 = scalar.assume %lo3 [range(%lo3, 0, 65536)] : i32
    %hi4 = scalar.assume %hi3 [range(%hi3, 0, 65536)] : i32
    %xlo = index.cast %lo4 : i32 to index
    %xhi = index.cast %hi4 : i32 to index
    %ww = scf.for %xi = [%xlo to %xhi step %c1](%acc = %zf : f32) -> (f32) {{
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
      %wv_x = scalar.absf %d : f32
      %wv_lt1 = scalar.cmpf olt, %wv_x, %onef : f32
      %wv_lt2 = scalar.cmpf olt, %wv_x, %twof : f32
      %wv_p1a = scalar.mulf %wv_x, %c1p5 : f32
      %wv_p1b = scalar.subf %wv_p1a, %c2p5 : f32
      %wv_p1c = scalar.mulf %wv_p1b, %wv_x : f32
      %wv_p1d = scalar.mulf %wv_p1c, %wv_x : f32
      %wv_p1 = scalar.addf %wv_p1d, %onef : f32
      %wv_p2a = scalar.subf %wv_x, %fivef : f32
      %wv_p2b = scalar.mulf %wv_p2a, %wv_x : f32
      %wv_p2c = scalar.addf %wv_p2b, %eightf : f32
      %wv_p2d = scalar.mulf %wv_p2c, %wv_x : f32
      %wv_p2e = scalar.subf %wv_p2d, %fourf : f32
      %wv_p2 = scalar.mulf %wv_p2e, %mhalf : f32
      %wv_q = scf.select %wv_lt2, %wv_p2, %zf : f32
      %wv = scf.select %wv_lt1, %wv_p1, %wv_q : f32
      %na = scalar.addf %acc, %wv : f32
      scf.yield %na : f32
    }}
    %wnz = scalar.cmpf one, %ww, %zf : f32
    %wdiv = scf.select %wnz, %ww, %onef : f32
    %srcn = index.constant 268435456 : index
    %sv = buffer.view %src[%zero] : buffer -> view<[%srcn]xi8>
    %bias = scalar.constant 2097152 : i32
    %s0, %s1, %s2 = scf.for %xi = [%xlo to %xhi step %c1](%a0 = %bias : i32, %a1 = %bias : i32, %a2 = %bias : i32) -> (i32, i32, i32) {{
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
      %wv_x = scalar.absf %d : f32
      %wv_lt1 = scalar.cmpf olt, %wv_x, %onef : f32
      %wv_lt2 = scalar.cmpf olt, %wv_x, %twof : f32
      %wv_p1a = scalar.mulf %wv_x, %c1p5 : f32
      %wv_p1b = scalar.subf %wv_p1a, %c2p5 : f32
      %wv_p1c = scalar.mulf %wv_p1b, %wv_x : f32
      %wv_p1d = scalar.mulf %wv_p1c, %wv_x : f32
      %wv_p1 = scalar.addf %wv_p1d, %onef : f32
      %wv_p2a = scalar.subf %wv_x, %fivef : f32
      %wv_p2b = scalar.mulf %wv_p2a, %wv_x : f32
      %wv_p2c = scalar.addf %wv_p2b, %eightf : f32
      %wv_p2d = scalar.mulf %wv_p2c, %wv_x : f32
      %wv_p2e = scalar.subf %wv_p2d, %fourf : f32
      %wv_p2 = scalar.mulf %wv_p2e, %mhalf : f32
      %wv_q = scf.select %wv_lt2, %wv_p2, %zf : f32
      %wv = scf.select %wv_lt1, %wv_p1, %wv_q : f32
      %wn = scalar.divf %wv, %wdiv : f32
      %wfx = scalar.mulf %wn, %fxp : f32
      %neg = scalar.cmpf olt, %wn, %zf : f32
      %rnd = scf.select %neg, %mhalf, %halff : f32
      %wr = scalar.addf %wfx, %rnd : f32
      %wi = scalar.fptosi %wr : f32 to i32
      %src_idx = index.madd %xi, %other, %col : index
      %el0 = index.mul %src_idx, %ccin : index
      %el = index.assume %el0 [lt(%el0, %srcn)] : index
      %el1r = index.add %el0, %c1 : index
      %el1 = index.assume %el1r [lt(%el1r, %srcn)] : index
      %el2r = index.add %el1r, %c1 : index
      %el2 = index.assume %el2r [lt(%el2r, %srcn)] : index
      %p0r = view.load %sv[%el] : view<[%srcn]xi8> -> i8
      %p1r = view.load %sv[%el1] : view<[%srcn]xi8> -> i8
      %p2r = view.load %sv[%el2] : view<[%srcn]xi8> -> i8
      %p0 = scalar.extui %p0r : i8 to i32
      %p1 = scalar.extui %p1r : i8 to i32
      %p2 = scalar.extui %p2r : i8 to i32
      %m0 = scalar.muli %p0, %wi : i32
      %m1 = scalar.muli %p1, %wi : i32
      %m2 = scalar.muli %p2, %wi : i32
      %n0 = scalar.addi %a0, %m0 : i32
      %n1 = scalar.addi %a1, %m1 : i32
      %n2 = scalar.addi %a2, %m2 : i32
      scf.yield %n0, %n1, %n2 : i32, i32, i32
    }}
    %sh = scalar.constant 22 : i32
    %c255i = scalar.constant 255 : i32
    %dst_idx = index.madd %o_v, %other, %col : index
    %dstn = index.constant 268435456 : index
    %dv = buffer.view %dst[%zero] : buffer -> view<[%dstn]xi8>
    %de0 = index.mul %dst_idx, %c3 : index
    %c2x = index.constant 2 : index
    %q0a = scalar.shrsi %s0, %sh : i32
    %q0b = scalar.maxsi %q0a, %zi : i32
    %q0c = scalar.minsi %q0b, %c255i : i32
    %q0 = scalar.trunci %q0c : i32 to i8
    %dc0r = index.add %de0, %c0 : index
    %dc0 = index.assume %dc0r [lt(%dc0r, %dstn)] : index
    view.store %q0, %dv[%dc0] : i8, view<[%dstn]xi8>
    %q1a = scalar.shrsi %s1, %sh : i32
    %q1b = scalar.maxsi %q1a, %zi : i32
    %q1c = scalar.minsi %q1b, %c255i : i32
    %q1 = scalar.trunci %q1c : i32 to i8
    %dc1r = index.add %de0, %c1 : index
    %dc1 = index.assume %dc1r [lt(%dc1r, %dstn)] : index
    view.store %q1, %dv[%dc1] : i8, view<[%dstn]xi8>
    %q2a = scalar.shrsi %s2, %sh : i32
    %q2b = scalar.maxsi %q2a, %zi : i32
    %q2c = scalar.minsi %q2b, %c255i : i32
    %q2 = scalar.trunci %q2c : i32 to i8
    %dc2r = index.add %de0, %c2x : index
    %dc2 = index.assume %dc2r [lt(%dc2r, %dstn)] : index
    view.store %q2, %dv[%dc2] : i8, view<[%dstn]xi8>
  }}
  kernel.return
}}

kernel.def export("v_copy") @v_copy(%n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%n_arg: index, %x: buffer, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {{
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%n]xf32>
    %a = view.load %xv[%bi] : view<[%n]xf32> -> f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %a, %yv[%bi] : f32, view<[%n]xf32>
  }}
  kernel.return
}}

config.decl @vision.height : %value: index where [range(%value, 1, 16384)]
config.decl @vision.width : %value: index where [range(%value, 1, 16384)]

command.def public target(@device) @vision_encode() launch(%weights: buffer, %input: buffer, %output: buffer) {{
  %img_h = config.get @vision.height : index
  %img_w = config.get @vision.width : index
  %cP = index.constant 20 : index
  %cm = index.constant 2 : index
  %c1 = index.constant 1 : index
  %c3 = index.constant 3 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %gh = index.div %img_h, %cP : index
  %gw = index.div %img_w, %cP : index
  %n_patches = index.mul %gh, %gw : index
  %T = index.add %n_patches, %c1 : index
  %mh = index.div %gh, %cm : index
  %mw = index.div %gw, %cm : index
  %M = index.mul %mh, %mw : index
  %T_up = index.add %T, %c255 : index
  %T_t = index.div %T_up, %c256 : index
  %Tp = index.mul %T_t, %c256 : index
  %M_up = index.add %M, %c63 : index
  %M_t = index.div %M_up, %c64 : index
  %Mp = index.mul %M_t, %c64 : index
  %img_px = index.mul %img_h, %img_w : index
  %n_pix = index.mul %img_px, %c3 : index
  %max_rows = index.constant 4096 : index
  %k1024 = index.constant 1024 : index
  %k2304 = index.constant 2304 : index
  %k4096 = index.constant 4096 : index
  %bpe4 = index.constant 4 : offset
  %bpe2 = index.constant 2 : offset
  %bpe1 = index.constant 1 : offset
  %h_n = index.mul %Tp, %k1024 : index
  %h_size = index.scale %h_n, %bpe4 : index, offset -> offset
  %h = buffer.alloca<global> align(256) %h_size : buffer
  %x = buffer.alloca<global> align(256) %h_size : buffer
  %q = buffer.alloca<global> align(256) %h_size : buffer
  %k = buffer.alloca<global> align(256) %h_size : buffer
  %v = buffer.alloca<global> align(256) %h_size : buffer
  %o = buffer.alloca<global> align(256) %h_size : buffer
  %a = buffer.alloca<global> align(256) %h_size : buffer
  %f_n = index.mul %Tp, %k4096 : index
  %f_size = index.scale %f_n, %bpe4 : index, offset -> offset
  %f = buffer.alloca<global> align(256) %f_size : buffer
  %x16_size = index.scale %f_n, %bpe2 : index, offset -> offset
  %x16 = buffer.alloca<global> align(256) %x16_size : buffer
  %sc_w = index.constant 2 : index
  %sc_n = index.mul %Tp, %sc_w : index
  %sc_size = index.scale %sc_n, %bpe4 : index, offset -> offset
  %sc = buffer.alloca<global> align(256) %sc_size : buffer
  %m_n = index.mul %Mp, %k4096 : index
  %m_size = index.scale %m_n, %bpe4 : index, offset -> offset
  %m_in = buffer.alloca<global> align(256) %m_size : buffer
  %m1 = buffer.alloca<global> align(256) %m_size : buffer
  %out_n = index.mul %Mp, %k2304 : index
  %out_size = index.scale %out_n, %bpe4 : index, offset -> offset
  %out = buffer.alloca<global> align(256) %out_size : buffer
  %wide_n = index.mul %max_rows, %img_w : index
  %wide_n3 = index.mul %wide_n, %c3 : index
  %wide_size = index.scale %wide_n3, %bpe1 : index, offset -> offset
  %wide = buffer.alloca<global> align(256) %wide_size : buffer
  %pix_size = index.scale %n_pix, %bpe1 : index, offset -> offset
  %pix = buffer.alloca<global> align(256) %pix_size : buffer
  %pixf_size = index.scale %n_pix, %bpe4 : index, offset -> offset
  %pixf = buffer.alloca<global> align(256) %pixf_size : buffer
  kernel.launch @v_resize_bicubic_h_hdr[%img_w, %max_rows](%img_w, %max_rows, %input, %wide) : [index, index](index, index, buffer, buffer)
  kernel.launch @v_resize_bicubic_v_hdr[%img_h, %img_w](%img_h, %img_w, %input, %wide, %pix) : [index, index](index, index, buffer, buffer, buffer)
  kernel.launch @v_ink_prep_pixels[%n_pix](%n_pix, %img_w, %img_h, %input, %pix, %pixf) : [index](index, index, index, buffer, buffer, buffer)
{params_str}
  kernel.launch @v_ink_patch_and_pos_1024[%T](%T, %gw, %gh, %img_w, %pixf, %w_v_patch_embd, %w_v_class_embd, %w_v_class_pos, %w_v_pos_embd, %h) : [index](index, index, index, index, buffer, view<2457600xi8>, view<4096xi8>, view<4096xi8>, view<4194304xi8>, buffer)
  kernel.launch @v_norm_ln_1024[%T](%T, %h, %w_v_pre_ln_w, %w_v_pre_ln_b, %h) : [index](index, buffer, view<4096xi8>, view<4096xi8>, buffer)
{blocks_str}
  kernel.launch @v_merge2x2_drop_cls_1024[%M](%M, %T, %h, %m_in) : [index](index, index, buffer, buffer)
  kernel.launch @v_row_scale_f16[%Mp](%Mp, %k4096, %m_in, %sc, %x16) : [index](index, index, buffer, buffer, buffer)
  template.apply<@sel4_v_gemm_f16_store_4096x4096>(%Mp, %k4096, %Mp, %k4096, %k4096, %x16, %sc, %w_mm_1_w, %m1) : (index, index, index, index, index, buffer, buffer, view<33554432xi8>, buffer)
  %p_M_4096 = index.mul %M, %k4096 : index
  kernel.launch @v_ew_gelu_erf[%p_M_4096](%p_M_4096, %m1, %m1) : [index](index, buffer, buffer)
  kernel.launch @v_row_scale_f16[%Mp](%Mp, %k4096, %m1, %sc, %x16) : [index](index, index, buffer, buffer, buffer)
  template.apply<@sel5_v_gemm_f16_store_4096x2304>(%Mp, %k2304, %Mp, %k4096, %k2304, %x16, %sc, %w_mm_2_w, %out) : (index, index, index, index, index, buffer, buffer, view<18874368xi8>, buffer)
  %final_n = index.mul %M, %k2304 : index
  kernel.launch @v_copy[%final_n](%final_n, %out, %output) : [index](index, buffer, buffer)
  command.return
}}
"""


_CONTROL_VISION_HELPERS = """
// ---- Inkling 2 Mini Vision (inkling2v, cell = 40 px) ----
global.rodata.def @ink_ct_image = bytes("3c7c636f6e74656e745f696d6167657c3e")
global.rodata.def @vision_sentinel = bytes("1a4c4f4f4d5f494d4147451a")
global.rodata.def @vision_image_key = bytes("22696d6167655f75726c22")
global.rodata.def @vision_lost = bytes("696d616765206d61726b6572206c6f7374207768696c652072656e646572696e6720746865206d657373616765")

func.def @vision_request(%request: buffer) -> (i1) {
  %origin = index.constant 0 : offset
  %end = buffer.length %request
  %key = global.load @vision_image_key : buffer
  %at = func.call @find(%request, %origin, %end, %key) : (buffer, offset, offset, buffer) -> (offset)
  %found = index.cmp ult, %at, %end : offset
  func.return %found : i1
}

// Compute processed (tw, th, rows) for Inkling 2 Mini ViT (cell = 40 px, max_cells = 1600):
// - If ceil(w/40) * ceil(h/40) <= 1600: pad right/bottom to multiples of 40 (no scaling).
// - Otherwise: scale down preserving aspect ratio so (tw/40)*(th/40) <= 1600.
func.def @vision_geometry(%w: i32, %h: i32) -> (i32, i32, i32) {
  %one32 = scalar.constant 1 : i32
  %c39 = scalar.constant 39 : i32
  %c40 = scalar.constant 40 : i32
  %max_cells = scalar.constant 1600 : i32
  %w_pos = scalar.maxsi %w, %one32 : i32
  %h_pos = scalar.maxsi %h, %one32 : i32
  %w_up = scalar.addi %w_pos, %c39 : i32
  %h_up = scalar.addi %h_pos, %c39 : i32
  %cw0 = scalar.divui %w_up, %c40 : i32
  %ch0 = scalar.divui %h_up, %c40 : i32
  %cw1 = scalar.maxsi %cw0, %one32 : i32
  %ch1 = scalar.maxsi %ch0, %one32 : i32
  %cells0 = scalar.muli %cw1, %ch1 : i32
  %over = scalar.cmpi ugt, %cells0, %max_cells : i32
  %cw, %ch = scf.if %over -> (i32, i32) {
    %wd = scalar.sitofp %w_pos : i32 to f64
    %hd = scalar.sitofp %h_pos : i32 to f64
    %max_px = scalar.constant 2560000.0 : f64
    %area = scalar.mulf %wd, %hd : f64
    %q = scalar.divf %max_px, %area : f64
    %q32 = scalar.fptrunc %q : f64 to f32
    %s32 = scalar.sqrtf %q32 : f32
    %s = scalar.extf %s32 : f32 to f64
    %c40d = scalar.constant 40.0 : f64
    %oned = scalar.constant 1.0 : f64
    %sw0 = scalar.mulf %wd, %s : f64
    %sh0 = scalar.mulf %hd, %s : f64
    %gcw0 = scalar.divf %sw0, %c40d : f64
    %gch0 = scalar.divf %sh0, %c40d : f64
    %gcw1 = scalar.floorf %gcw0 : f64
    %gch1 = scalar.floorf %gch0 : f64
    %gcw2 = scalar.maxnumf %gcw1, %oned : f64
    %gch2 = scalar.maxnumf %gch1, %oned : f64
    %icw = scalar.fptosi %gcw2 : f64 to i32
    %ich = scalar.fptosi %gch2 : f64 to i32
    scf.yield %icw, %ich : i32, i32
  } else {
    scf.yield %cw1, %ch1 : i32, i32
  }
  %tw = scalar.muli %cw, %c40 : i32
  %th = scalar.muli %ch, %c40 : i32
  %rows = scalar.muli %cw, %ch : i32
  func.return %tw, %th, %rows : i32, i32, i32
}

// Render content parts for Inkling 2 Mini:
// Each text part -> rtok + <|content_text|> + text + <|end_message|>
// Each image part -> rtok + <|content_image|> + SENTINEL + <|end_message|>
func.def @ink_vision_parts(%rtok: buffer, %content: buffer, %parts: buffer, %frames: buffer, %part_count: i64) -> (buffer, buffer, i64) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %zero32 = scalar.constant 0 : i32
  %text_kind = scalar.constant 0 : i32
  %bounded_count = scalar.assume %part_count [range(%part_count, 1, 4096)] : i64
  %count = index.cast %bounded_count : i64 to index
  %records = buffer.view %parts[%origin] : buffer -> view<[%count]x6xi32>
  %ct_text = global.load @ink_ct_text : buffer
  %ct_image = global.load @ink_ct_image : buffer
  %end_msg = global.load @ink_end_msg : buffer
  %sentinel = global.load @vision_sentinel : buffer
  %empty = global.load @empty : buffer
  %sentinel_length = buffer.length %sentinel
  %images = scf.for %i = [%c0 to %count step %c1](%n = %c0 : index) -> (index) {
    %kind = view.load %records[%i, 0] : view<[%count]x6xi32> -> i32
    %is_text = scalar.cmpi eq, %kind, %text_kind : i32
    %inc = scf.select %is_text, %c0, %c1 : index
    %nn = index.add %n, %inc : index
    scf.yield %nn : index
  }
  %c28 = index.constant 28 : offset
  %c8 = index.constant 8 : offset
  %record_bytes = index.scale %images, %c28 : index, offset -> offset
  %meta_bytes = index.scale %images, %c8 : index, offset -> offset
  %plan_bytes = index.add %record_bytes, %meta_bytes : offset
  %plan = buffer.alloca<private> align(4) %plan_bytes : buffer
  %frame_bytes = buffer.length %frames
  %frame_words0 = index.cast %frame_bytes : offset to index
  %frame_words = index.div %frame_words0, %c4 : index
  %frame_view = buffer.view %frames[%origin] : buffer -> view<[%frame_words]xi32>
  %text, %image_n = scf.for %i = [%c0 to %count step %c1](%acc = %empty : buffer, %image = %c0 : index) -> (buffer, index) {
    %kind = view.load %records[%i, 0] : view<[%count]x6xi32> -> i32
    %is_text = scalar.cmpi eq, %kind, %text_kind : i32
    %na, %ni = scf.if %is_text -> (buffer, index) {
      %source32 = view.load %records[%i, 1] : view<[%count]x6xi32> -> i32
      %length32 = view.load %records[%i, 2] : view<[%count]x6xi32> -> i32
      %source64 = scalar.extui %source32 : i32 to i64
      %length64 = scalar.extui %length32 : i32 to i64
      %sb = scalar.assume %source64 [range(%source64, 0, 4294967295)] : i64
      %lb = scalar.assume %length64 [range(%length64, 0, 4294967295)] : i64
      %source = index.cast %sb : i64 to offset
      %length = index.cast %lb : i64 to offset
      %piece = func.call @slice(%content, %source, %length) : (buffer, offset, offset) -> (buffer)
      %blk = func.call @join4(%rtok, %ct_text, %piece, %end_msg) : (buffer, buffer, buffer, buffer) -> (buffer)
      %joined = func.call @join(%acc, %blk) : (buffer, buffer) -> (buffer)
      scf.yield %joined, %image : buffer, index
    } else {
      %frame32 = view.load %records[%i, 3] : view<[%count]x6xi32> -> i32
      %frame_count32 = view.load %records[%i, 4] : view<[%count]x6xi32> -> i32
      %frame_bound = scalar.assume %frame32 [range(%frame32, 0, 4095)] : i32
      %frame = index.cast %frame_bound : i32 to index
      %wi0 = index.mul %frame, %c4 : index
      %hi0 = index.add %wi0, %c1 : index
      %wi = index.assume %wi0 [lt(%wi0, %frame_words)] : index
      %hi = index.assume %hi0 [lt(%hi0, %frame_words)] : index
      %width = view.load %frame_view[%wi] : view<[%frame_words]xi32> -> i32
      %height = view.load %frame_view[%hi] : view<[%frame_words]xi32> -> i32
      %tw, %th, %rows = func.call @vision_geometry(%width, %height) : (i32, i32) -> (i32, i32, i32)
      %blk = func.call @join4(%rtok, %ct_image, %sentinel, %end_msg) : (buffer, buffer, buffer, buffer) -> (buffer)
      %joined = func.call @join(%acc, %blk) : (buffer, buffer) -> (buffer)
      %bi = index.assume %image [range(%image, 0, 4095), lt(%image, %images)] : index
      %rec = buffer.view %plan[%origin] : buffer -> view<[%images]x7xi32>
      %sl64 = index.cast %sentinel_length : offset to i64
      %sl32 = scalar.trunci %sl64 : i64 to i32
      %mo0 = index.scale %bi, %c8 : index, offset -> offset
      %mo1 = index.add %record_bytes, %mo0 : offset
      %mo64 = index.cast %mo1 : offset to i64
      %mo32 = scalar.trunci %mo64 : i64 to i32
      %eight32 = scalar.constant 8 : i32
      view.store %frame32, %rec[%bi, 0] : i32, view<[%images]x7xi32>
      view.store %frame_count32, %rec[%bi, 1] : i32, view<[%images]x7xi32>
      view.store %zero32, %rec[%bi, 2] : i32, view<[%images]x7xi32>
      view.store %sl32, %rec[%bi, 3] : i32, view<[%images]x7xi32>
      view.store %rows, %rec[%bi, 4] : i32, view<[%images]x7xi32>
      view.store %mo32, %rec[%bi, 5] : i32, view<[%images]x7xi32>
      view.store %eight32, %rec[%bi, 6] : i32, view<[%images]x7xi32>
      %meta = buffer.view %plan[%mo1] : buffer -> view<2xi32>
      view.store %tw, %meta[0] : i32, view<2xi32>
      view.store %th, %meta[1] : i32, view<2xi32>
      %ni2 = index.add %image, %c1 : index
      scf.yield %joined, %ni2 : buffer, index
    }
    scf.yield %na, %ni : buffer, index
  }
  %images64 = index.cast %images : index to i64
  func.return %text, %plan, %images64 : buffer, buffer, i64
}

func.def @vision_locate(%fragment: buffer, %plan: buffer, %count64: i64) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %bounded = scalar.assume %count64 [range(%count64, 0, 4096)] : i64
  %count = index.cast %bounded : i64 to index
  %sentinel = global.load @vision_sentinel : buffer
  %sentinel_length = buffer.length %sentinel
  %end = buffer.length %fragment
  %rec = buffer.view %plan[%origin] : buffer -> view<[%count]x7xi32>
  %final = scf.for %i = [%c0 to %count step %c1](%from = %origin : offset) -> (offset) {
    %at = func.call @find(%fragment, %from, %end, %sentinel) : (buffer, offset, offset, buffer) -> (offset)
    %found = index.cmp ult, %at, %end : offset
    %f32 = scalar.extui %found : i1 to i32
    %error = global.load @vision_lost : buffer
    func.call @require(%f32, %error) : (i32, buffer)
    %at64 = index.cast %at : offset to i64
    %at32 = scalar.trunci %at64 : i64 to i32
    view.store %at32, %rec[%i, 2] : i32, view<[%count]x7xi32>
    %next = index.add %at, %sentinel_length : offset
    scf.yield %next : offset
  }
  func.return
}

func.def @vision_slice(%text: buffer, %begin: offset, %end: offset) -> (buffer) {
  %origin = index.constant 0 : offset
  %length = index.sub %end, %begin : offset
  %out = buffer.alloca<private> align(1) %length : buffer
  buffer.copy %text[%begin], %out[%origin], %length
  func.return %out : buffer
}

func.def @vision_prepare_input(%text: buffer, %markers: buffer, %marker_count: i64, %capacity: i64) -> (buffer, i64, buffer) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c0_i64 = scalar.constant 0 : i64
  %c1_i64 = scalar.constant 1 : i64
  %four = index.constant 4 : offset
  %bounded = scalar.assume %marker_count [range(%marker_count, 1, 4096)] : i64
  %count = index.cast %bounded : i64 to index
  %marks = buffer.view %markers[%origin] : buffer -> view<[%count]x6xi32>
  %text_end = buffer.length %text
  %probe = scalar.addi %capacity, %c1_i64 : i64
  %last, %total0 = scf.for %i = [%c0 to %count step %c1](%cursor = %origin : offset, %sum = %c0_i64 : i64) -> (offset, i64) {
    %off32 = view.load %marks[%i, 1] : view<[%count]x6xi32> -> i32
    %len32 = view.load %marks[%i, 2] : view<[%count]x6xi32> -> i32
    %rows32 = view.load %marks[%i, 3] : view<[%count]x6xi32> -> i32
    %off64 = scalar.extui %off32 : i32 to i64
    %len64 = scalar.extui %len32 : i32 to i64
    %ob = scalar.assume %off64 [range(%off64, 0, 4294967295)] : i64
    %lb = scalar.assume %len64 [range(%len64, 0, 4294967295)] : i64
    %off = index.cast %ob : i64 to offset
    %len = index.cast %lb : i64 to offset
    %segment = func.call @vision_slice(%text, %cursor, %off) : (buffer, offset, offset) -> (buffer)
    %ids, %n = func.call @encode(%segment, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
    %rows64 = scalar.extui %rows32 : i32 to i64
    %s1 = scalar.addi %sum, %n : i64
    %s2 = scalar.addi %s1, %rows64 : i64
    %next = index.add %off, %len : offset
    scf.yield %next, %s2 : offset, i64
  }
  %tail = func.call @vision_slice(%text, %last, %text_end) : (buffer, offset, offset) -> (buffer)
  %tail_ids, %tail_n = func.call @encode(%tail, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
  %total = scalar.addi %total0, %tail_n : i64
  %fits = scalar.cmpi ule, %total, %capacity : i64
  %fits32 = scalar.extui %fits : i1 to i32
  %error = global.load @overflow : buffer
  func.call @require(%fits32, %error) : (i32, buffer)
  %tb = scalar.assume %total [range(%total, 0, 1073741823)] : i64
  %total_index = index.cast %tb : i64 to index
  %token_bytes = index.scale %total_index, %four : index, offset -> offset
  %tokens = buffer.alloca<private> align(4) %token_bytes : buffer
  buffer.fill %c0_i32, %tokens[%origin], %token_bytes : i32
  %c12 = index.constant 12 : offset
  %span_bytes = index.scale %count, %c12 : index, offset -> offset
  %spans = buffer.alloca<private> align(4) %span_bytes : buffer
  %span_view = buffer.view %spans[%origin] : buffer -> view<[%count]x3xi32>
  %last2, %out_end = scf.for %i = [%c0 to %count step %c1](%cursor = %origin : offset, %out = %origin : offset) -> (offset, offset) {
    %proj = view.load %marks[%i, 0] : view<[%count]x6xi32> -> i32
    %off32 = view.load %marks[%i, 1] : view<[%count]x6xi32> -> i32
    %len32 = view.load %marks[%i, 2] : view<[%count]x6xi32> -> i32
    %rows32 = view.load %marks[%i, 3] : view<[%count]x6xi32> -> i32
    %off64 = scalar.extui %off32 : i32 to i64
    %len64 = scalar.extui %len32 : i32 to i64
    %ob = scalar.assume %off64 [range(%off64, 0, 4294967295)] : i64
    %lb = scalar.assume %len64 [range(%len64, 0, 4294967295)] : i64
    %off = index.cast %ob : i64 to offset
    %len = index.cast %lb : i64 to offset
    %segment = func.call @vision_slice(%text, %cursor, %off) : (buffer, offset, offset) -> (buffer)
    %ids, %n = func.call @encode(%segment, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
    %nb = scalar.assume %n [range(%n, 0, 1073741823)] : i64
    %ni = index.cast %nb : i64 to index
    %id_bytes = index.scale %ni, %four : index, offset -> offset
    buffer.copy %ids[%origin], %tokens[%out], %id_bytes
    %image_at = index.add %out, %id_bytes : offset
    %image_at64 = index.cast %image_at : offset to i64
    %four64 = scalar.constant 4 : i64
    %input_offset64 = scalar.divui %image_at64, %four64 : i64
    %input_offset = scalar.trunci %input_offset64 : i64 to i32
    view.store %proj, %span_view[%i, 0] : i32, view<[%count]x3xi32>
    view.store %input_offset, %span_view[%i, 1] : i32, view<[%count]x3xi32>
    view.store %rows32, %span_view[%i, 2] : i32, view<[%count]x3xi32>
    %rows64 = scalar.extui %rows32 : i32 to i64
    %rb = scalar.assume %rows64 [range(%rows64, 0, 1073741823)] : i64
    %ri = index.cast %rb : i64 to index
    %row_bytes = index.scale %ri, %four : index, offset -> offset
    %after = index.add %image_at, %row_bytes : offset
    %next = index.add %off, %len : offset
    scf.yield %next, %after : offset, offset
  }
  %tail_b = scalar.assume %tail_n [range(%tail_n, 0, 1073741823)] : i64
  %tail_i = index.cast %tail_b : i64 to index
  %tail_bytes = index.scale %tail_i, %four : index, offset -> offset
  buffer.copy %tail_ids[%origin], %tokens[%out_end], %tail_bytes
  func.return %tokens, %total, %spans : buffer, i64, buffer
}

func.def public @prepare_input(%text: buffer, %markers: buffer, %marker_count: i64, %format: i32, %boundary: i32, %capacity: i64) -> (buffer, i64, buffer) {
  %zero = scalar.constant 0 : i64
  %plain = scalar.cmpi eq, %marker_count, %zero : i64
  %tokens, %count, %spans = scf.if %plain -> (buffer, i64, buffer) {
    %t, %n, %s = func.call @plain_prepare_input(%text, %markers, %marker_count, %format, %boundary, %capacity) : (buffer, buffer, i64, i32, i32, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  } else {
    %t, %n, %s = func.call @vision_prepare_input(%text, %markers, %marker_count, %capacity) : (buffer, buffer, i64, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  }
  func.return %tokens, %count, %spans : buffer, i64, buffer
}

// Per-message Inkling 2 Mini renderer used when the request contains image_url parts.
func.def @ink_vision_chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %zero64 = scalar.constant 0 : i64
  %array_type = scalar.constant 3 : i64
  %false = scalar.constant false : i1
  %empty = global.load @empty : buffer
  %request_bytes = buffer.length %request
  %request_length = index.cast %request_bytes : offset to i64
  %messages_key = global.load @messages_key : buffer
  %fm, %bm, %lm, %tm = func.call @json_lookup(%request, %zero64, %request_length, %messages_key) : (buffer, i64, i64, buffer) -> (i1, i64, i64, i64)
  %is_array = scalar.cmpi eq, %tm, %array_type : i64
  %has_messages = scalar.andi %fm, %is_array : i1
  func.call @check(%has_messages) : (i1)
  %elements, %etype, %ecount = func.call @members_range(%request, %bm, %lm) : (buffer, i64, i64) -> (buffer, i32, i64)
  %bounded = scalar.assume %ecount [range(%ecount, 0, 2147483647)] : i64
  %extent = index.cast %bounded : i64 to index
  %records = buffer.view %elements[%origin] : buffer -> view<[%extent]x5xi64>
  %role_key = global.load @role_key : buffer
  %system_role = global.load @system : buffer
  %is_sys_first = scf.for %i = [%c0 to %extent step %c1](%acc = %false : i1) -> (i1) {
    %is_first = index.cmp eq, %i, %c0 : index
    %val = scf.if %is_first -> (i1) {
      %mb = view.load %records[%i, 2] : view<[%extent]x5xi64> -> i64
      %ml = view.load %records[%i, 3] : view<[%extent]x5xi64> -> i64
      %role = func.call @ink_string(%request, %mb, %ml, %role_key) : (buffer, i64, i64, buffer) -> (buffer)
      %is_sys = func.call @equal(%role, %system_role) : (buffer, buffer) -> (i1)
      scf.yield %is_sys : i1
    } else {
      scf.yield %acc : i1
    }
    scf.yield %val : i1
  }
  %first_is_sys = scalar.andi %has_messages, %is_sys_first : i1
  %tools_block = func.call @ink_tools_block(%tools) : (buffer) -> (buffer)
  %effort_block = func.call @ink_effort_block(%request) : (buffer) -> (buffer)
  %pre = scf.select %first_is_sys, %empty, %effort_block : buffer
  %out = func.call @join(%tools_block, %pre) : (buffer, buffer) -> (buffer)
  %bit_vision = scalar.constant 2305843009213693952 : i64
  %bit_effort = scalar.constant 4611686018427387904 : i64
  %e_state = scf.select %first_is_sys, %zero64, %bit_effort : i64
  %state = scalar.ori %bit_vision, %e_state : i64
  func.return %out, %state : buffer, i64
}

func.def @ink_vision_chat_message(%raw: buffer, %role: buffer, %content: buffer, %parts: buffer, %frames: buffer, %reasoning: buffer, %calls: buffer, %state: i64, %ordinal: i64, %fields: i64, %part_count: i64) -> (buffer, i64, buffer, i64) {
  %origin = index.constant 0 : offset
  %zero64 = scalar.constant 0 : i64
  %false = scalar.constant false : i1
  %true = scalar.constant true : i1
  %empty = global.load @empty : buffer
  %system_role = global.load @system : buffer
  %user_role = global.load @user : buffer
  %assistant_role = global.load @assistant : buffer
  %tool_role = global.load @tool_role : buffer
  %is_system = func.call @equal(%role, %system_role) : (buffer, buffer) -> (i1)
  %is_user = func.call @equal(%role, %user_role) : (buffer, buffer) -> (i1)
  %is_assistant = func.call @equal(%role, %assistant_role) : (buffer, buffer) -> (i1)
  %is_tool = func.call @equal(%role, %tool_role) : (buffer, buffer) -> (i1)
  %known1 = scalar.ori %is_system, %is_user : i1
  %known2 = scalar.ori %is_assistant, %is_tool : i1
  %known = scalar.ori %known1, %known2 : i1
  func.call @check(%known) : (i1)
  %bit_effort = scalar.constant 4611686018427387904 : i64
  %masked_e = scalar.andi %state, %bit_effort : i64
  %effort_done = scalar.cmpi ne, %masked_e, %zero64 : i64
  %not_done = scalar.cmpi eq, %effort_done, %false : i1
  %not_sys = scalar.cmpi eq, %is_system, %false : i1
  %emit_effort = scalar.andi %not_done, %not_sys : i1
  %default_effort = global.load @ink_default_effort_block : buffer
  %pre = scf.select %emit_effort, %default_effort, %empty : buffer
  %set_e = scf.select %emit_effort, %bit_effort, %zero64 : i64
  %next_state = scalar.ori %state, %set_e : i64
  %msg_sys = global.load @ink_msg_sys : buffer
  %msg_user = global.load @ink_msg_user : buffer
  %msg_model = global.load @ink_msg_model : buffer
  %msg_tool = global.load @ink_msg_tool : buffer
  %ct_text = global.load @ink_ct_text : buffer
  %ct_think = global.load @ink_ct_think : buffer
  %end_msg = global.load @ink_end_msg : buffer
  %end_sampling = global.load @ink_end_sampling : buffer
  %name_key = global.load @ink_k_name : buffer
  %raw_len_off = buffer.length %raw
  %raw_len = index.cast %raw_len_off : offset to i64
  %rtok = scf.if %is_system -> (buffer) {
    scf.yield %msg_sys : buffer
  } else {
    %r1 = scf.if %is_user -> (buffer) {
      scf.yield %msg_user : buffer
    } else {
      %r2 = scf.if %is_tool -> (buffer) {
        %tname = func.call @ink_string(%raw, %zero64, %raw_len, %name_key) : (buffer, i64, i64, buffer) -> (buffer)
        %h = func.call @join(%msg_tool, %tname) : (buffer, buffer) -> (buffer)
        scf.yield %h : buffer
      } else {
        %aname = func.call @ink_string(%raw, %zero64, %raw_len, %name_key) : (buffer, i64, i64, buffer) -> (buffer)
        %h = func.call @join(%msg_model, %aname) : (buffer, buffer) -> (buffer)
        scf.yield %h : buffer
      }
      scf.yield %r2 : buffer
    }
    scf.yield %r1 : buffer
  }
  %has_parts = scalar.cmpi ne, %part_count, %zero64 : i64
  %content_block, %vision_plan, %vision_count = scf.if %has_parts -> (buffer, buffer, i64) {
    %cb, %vp, %vc = func.call @ink_vision_parts(%rtok, %content, %parts, %frames, %part_count) : (buffer, buffer, buffer, buffer, i64) -> (buffer, buffer, i64)
    scf.yield %cb, %vp, %vc : buffer, buffer, i64
  } else {
    %clen = buffer.length %content
    %nonempty = index.cmp ne, %clen, %origin : offset
    %rlen = buffer.length %reasoning
    %has_r = index.cmp ne, %rlen, %origin : offset
    %klen = buffer.length %calls
    %has_k = index.cmp ne, %klen, %origin : offset
    %has_other = scalar.ori %has_r, %has_k : i1
    %not_asst = scalar.cmpi eq, %is_assistant, %false : i1
    %no_other = scalar.cmpi eq, %has_other, %false : i1
    %allow_empty = scalar.ori %not_asst, %no_other : i1
    %emit_c = scalar.ori %nonempty, %allow_empty : i1
    %cb = scf.if %emit_c -> (buffer) {
      %b = func.call @join4(%rtok, %ct_text, %content, %end_msg) : (buffer, buffer, buffer, buffer) -> (buffer)
      scf.yield %b : buffer
    } else {
      scf.yield %empty : buffer
    }
    scf.yield %cb, %empty, %zero64 : buffer, buffer, i64
  }
  %body = scf.if %is_assistant -> (buffer) {
    %rlen = buffer.length %reasoning
    %has_r = index.cmp ne, %rlen, %origin : offset
    %rblock = scf.if %has_r -> (buffer) {
      %v = func.call @join4(%rtok, %ct_think, %reasoning, %end_msg) : (buffer, buffer, buffer, buffer) -> (buffer)
      scf.yield %v : buffer
    } else {
      scf.yield %empty : buffer
    }
    %all = func.call @join4(%rblock, %content_block, %calls, %end_sampling) : (buffer, buffer, buffer, buffer) -> (buffer)
    scf.yield %all : buffer
  } else {
    scf.yield %content_block : buffer
  }
  %fragment = func.call @join(%pre, %body) : (buffer, buffer) -> (buffer)
  func.call @vision_locate(%fragment, %vision_plan, %vision_count) : (buffer, buffer, i64)
  func.return %fragment, %next_state, %vision_plan, %vision_count : buffer, i64, buffer, i64
}
global.rodata.def @ink_default_effort_block = bytes("3c7c6d6573736167655f73797374656d7c3e3c7c636f6e74656e745f746578747c3e5468696e6b696e67206566666f7274206c6576656c3a20302e393c7c656e645f6d6573736167657c3e")

func.def public @chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64) {
  %images = func.call @vision_request(%request) : (buffer) -> (i1)
  %text, %state = scf.if %images -> (buffer, i64) {
    %t, %s = func.call @ink_vision_chat_begin(%request, %tools) : (buffer, buffer) -> (buffer, i64)
    scf.yield %t, %s : buffer, i64
  } else {
    %t, %s = func.call @format_chat_begin(%request, %tools) : (buffer, buffer) -> (buffer, i64)
    scf.yield %t, %s : buffer, i64
  }
  func.return %text, %state : buffer, i64
}

func.def public @chat_message(%raw: buffer, %role: buffer, %content: buffer, %parts: buffer, %frames: buffer, %reasoning: buffer, %calls: buffer, %state: i64, %ordinal: i64, %fields: i64, %part_count: i64) -> (buffer, i64, buffer, i64) {
  %bit_vision = scalar.constant 2305843009213693952 : i64
  %zero = scalar.constant 0 : i64
  %masked = scalar.andi %state, %bit_vision : i64
  %images = scalar.cmpi ne, %masked, %zero : i64
  %f, %n, %p, %c = scf.if %images -> (buffer, i64, buffer, i64) {
    %f1, %n1, %p1, %c1 = func.call @ink_vision_chat_message(%raw, %role, %content, %parts, %frames, %reasoning, %calls, %state, %ordinal, %fields, %part_count) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, i64, i64, i64, i64) -> (buffer, i64, buffer, i64)
    scf.yield %f1, %n1, %p1, %c1 : buffer, i64, buffer, i64
  } else {
    %f2, %n2, %p2, %c2 = func.call @format_chat_message(%raw, %role, %content, %parts, %frames, %reasoning, %calls, %state, %ordinal, %fields, %part_count) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, i64, i64, i64, i64) -> (buffer, i64, buffer, i64)
    scf.yield %f2, %n2, %p2, %c2 : buffer, i64, buffer, i64
  }
  func.return %f, %n, %p, %c : buffer, i64, buffer, i64
}
"""


def patch_control_vision(text: str) -> str:
    """Wire multimodal prepare_input, chat_begin/chat_message, and header[2] mixed flag into control.loom."""
    # 0. Raise @chat_image_limits compressed_bytes from 1 MiB to 64 MiB for large screenshots
    old_lim = "  %compressed_bytes = scalar.constant 1048576 : i64\n"
    new_lim = "  %compressed_bytes = scalar.constant 67108864 : i64\n"
    assert text.count(old_lim) == 1, "control.loom: expected 1 @chat_image_limits compressed_bytes"
    text = text.replace(old_lim, new_lim, 1)

    # 1. Rename @prepare_input -> @plain_prepare_input
    old_prep = "func.def public @prepare_input("
    new_prep = "func.def @plain_prepare_input("
    assert text.count(old_prep) == 1, "control.loom: expected 1 public @prepare_input"
    text = text.replace(old_prep, new_prep, 1)

    # 2. Rename Inkling's @chat_begin / @chat_message -> @format_chat_begin / @format_chat_message
    old_cb = "// Render the full conversation in chat_begin.\nfunc.def public @chat_begin("
    new_cb = "// Render the full conversation in chat_begin.\nfunc.def @format_chat_begin("
    assert text.count(old_cb) == 1, "control.loom: expected 1 Inkling @chat_begin"
    text = text.replace(old_cb, new_cb, 1)

    old_cm = "// Every message was already rendered by chat_begin.\nfunc.def public @chat_message("
    new_cm = "// Every message was already rendered by chat_begin.\nfunc.def @format_chat_message("
    assert text.count(old_cm) == 1, "control.loom: expected 1 Inkling @chat_message"
    text = text.replace(old_cm, new_cm, 1)

    # 3. Store %vision_flag in header[2] inside @plain_encode_epoch (or @encode_epoch if DSpark is off)
    old_hdr = (
        "  view.store %first_count, %header[0] : i32, view<4xi32>\n"
        "  view.store %eos, %header[1] : i32, view<4xi32>\n"
        "  view.store %c0, %header[2] : i32, view<4xi32>\n"
        "  view.store %c0, %header[3] : i32, view<4xi32>"
    )
    new_hdr = (
        "  view.store %first_count, %header[0] : i32, view<4xi32>\n"
        "  view.store %eos, %header[1] : i32, view<4xi32>\n"
        "  %vision_media = scalar.addi %first_media_count, %next_media_count : i32\n"
        "  %vision_has = scalar.cmpi ne, %vision_media, %c0 : i32\n"
        "  %vision_flag = scalar.extui %vision_has : i1 to i32\n"
        "  view.store %vision_flag, %header[2] : i32, view<4xi32>\n"
        "  view.store %c0, %header[3] : i32, view<4xi32>"
    )
    assert text.count(old_hdr) == 1, "control.loom: expected 1 header[2] store in encode_epoch"
    text = text.replace(old_hdr, new_hdr, 1)

    return text + "\n" + _CONTROL_VISION_HELPERS


_NEW_G_EMBED = """kernel.def export("g_embed") @g_embed(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %ids: buffer, %meta: buffer, %tinfo: buffer, %table: buffer, %w: buffer, %x: buffer, %h: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant 1 : index
  %groups = index.constant 72 : index
  %zero_f = scalar.constant 0.0 : f32
  %meta_view = buffer.view %meta[%zero] : buffer -> view<4xi32>
  %mixed_word = view.load %meta_view[2] : view<4xi32> -> i32
  %mixed_zero = scalar.constant 0 : i32
  %mixed = scalar.cmpi ne, %mixed_word, %mixed_zero : i32
  %stride_one = index.constant 1 : index
  %stride_eight = index.constant 8 : index
  %id_stride = scf.select %mixed, %stride_eight, %stride_one : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %raw_e0 = index.mul %token, %id_stride : index
  %raw_e = index.assume %raw_e0 [lt(%raw_e0, %n_ids)] : index
  %raw = view.load %ids_flat[%raw_e] : view<[%n_ids]xi32> -> i32
  %feat_word = scf.if %mixed -> (i32) {
    %fw_e0 = index.mul %token, %id_stride : index
    %fw_e1 = index.add %fw_e0, %stride_one : index
    %fw_e = index.assume %fw_e1 [lt(%fw_e1, %n_ids)] : index
    %fw = view.load %ids_flat[%fw_e] : view<[%n_ids]xi32> -> i32
    scf.yield %fw : i32
  } else {
    scf.yield %mixed_zero : i32
  }
  %tinfo_view = buffer.view %tinfo[%zero] : buffer -> view<[%tokens]x4xi32>
  %live_word = view.load %tinfo_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %is_live = scalar.cmpi ne, %live_word, %mixed_zero : i32
  %has_feat = scalar.cmpi ne, %feat_word, %mixed_zero : i32
  %is_feat = scalar.andi %has_feat, %is_live : i1
  %feat_one_byte = index.constant 1 : offset
  %c0_i32 = scalar.constant 0 : i32
  %vocab_last = scalar.constant 200057 : i32
  %low = scalar.maxsi %raw, %c0_i32 : i32
  %clamped = scalar.minsi %low, %vocab_last : i32
  %bounded = scalar.assume %clamped [range(%clamped, 0, 200057)] : i32
  %id = index.cast %bounded : i32 to index
  %row_bytes = index.constant 2448 : offset
  %block_bytes = index.constant 34 : offset
  %row_origin = index.scale %id, %row_bytes : index, offset -> offset
  %active = index.cmp ult, %lane, %groups : index
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %values = scf.if %active -> (vector<32xf32>) {
    %block = index.div %lane, %c8 : index
    %within = index.rem %lane, %c8 : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %origin = index.add %row_origin, %block_add : offset
    %v = func.call @g_q8_0_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    %vv = scf.if %is_feat -> (vector<32xf32>) {
      %fb_lane = index.assume %lane [range(%lane, 0, 71)] : index
      %fo_u = scalar.maxsi %feat_word, %mixed_zero : i32
      %fo_i = index.cast %fo_u : i32 to index
      %fo = index.scale %fo_i, %feat_one_byte : index, offset -> offset
      %fview = buffer.view %ids[%fo] : buffer -> view<72x32xf32>
      %fv = vector.load %fview[%fb_lane, %c0] : view<72x32xf32> -> vector<32xf32>
      scf.yield %fv : vector<32xf32>
    } else {
      scf.yield %v : vector<32xf32>
    }
    scf.yield %vv : vector<32xf32>
  } else {
    scf.yield %zero32 : vector<32xf32>
  }
  %normalized = vector.addf %values, %zero32 : vector<32xf32>
  %count = scalar.constant 2304.0 : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %squares2 = vector.mulf %normalized, %normalized : vector<32xf32>
  %local2 = vector.reduce<addf> %squares2, %zero_f : vector<32xf32>, f32
  %sum2 = kernel.workgroup.reduce<addf> %local2 : f32
  %mean2 = scalar.divf %sum2, %count : f32
  %stable2 = scalar.addf %mean2, %eps : f32
  %inverse2 = scalar.rsqrtf %stable2 : f32
  %inverses2 = vector.splat %inverse2 : vector<32xf32>
  scf.if %active {
    %bl = index.assume %lane [range(%lane, 0, 71)] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x72x32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x72x32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<72x32xf32>
    vector.store %normalized, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x72x32xf32>
    %weights = vector.load %w_view[%bl, %c0] : view<72x32xf32> -> vector<32xf32>
    %n2 = vector.mulf %normalized, %inverses2 : vector<32xf32>
    %hv_norm = vector.mulf %n2, %weights : vector<32xf32>
    // Vision rows replace embeddings as-is without token_embd_norm.weight (Basalt: x0 = emb).
    %hv = scf.select %is_feat, %normalized, %hv_norm : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x72x32xf32>
  }
  kernel.return
}"""


def patch_llm_vision(text: str) -> str:
    """Update @g_embed and its launch sites in llm.loom (and @g_dfl_prepare if present) for vision features."""
    start_marker = 'kernel.def export("g_embed") @g_embed(%tokens: index) {'
    end_marker = '\n// x[t] += y[t]; h[t] = rmsnorm(x[t]) * w_next.\nkernel.def export("g_post_add_norm")'
    s_idx = text.find(start_marker)
    e_idx = text.find(end_marker, s_idx)
    assert s_idx != -1 and e_idx != -1, "llm.loom: could not locate @g_embed block"
    text = text[:s_idx] + _NEW_G_EMBED + "\n" + text[e_idx:]

    # Update @g_embed launch sites in @llm_advance, @llm_verify, and @llm_dflash_propose
    old_launch_xh = "  kernel.launch @g_embed[%t](%t, %tokens, %embedding, %tok_norm, %x, %h) : [index](index, buffer, view<492106752xi8>, view<2304xf32>, buffer, buffer)"
    new_launch_xh = "  kernel.launch @g_embed[%t](%t, %tokens, %metadata, %info, %embedding, %tok_norm, %x, %h) : [index](index, buffer, buffer, buffer, view<492106752xi8>, view<2304xf32>, buffer, buffer)"
    assert text.count(old_launch_xh) in (1, 2), f"unexpected g_embed launch count: {text.count(old_launch_xh)}"
    text = text.replace(old_launch_xh, new_launch_xh)

    old_launch_yx = "  kernel.launch @g_embed[%t](%t, %tokens, %embedding, %tok_norm, %y, %x) : [index](index, buffer, view<492106752xi8>, view<2304xf32>, buffer, buffer)"
    new_launch_yx = "  kernel.launch @g_embed[%t](%t, %tokens, %metadata, %info, %embedding, %tok_norm, %y, %x) : [index](index, buffer, buffer, buffer, view<492106752xi8>, view<2304xf32>, buffer, buffer)"
    if old_launch_yx in text:
        text = text.replace(old_launch_yx, new_launch_yx)

    # Also make @g_dfl_prepare stride-aware when header[2] != 0 (%mixed)
    old_dfl = (
        "  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>\n"
        "  %count32 = view.load %header[0] : view<4xi32> -> i32\n"
        "  %eos = view.load %header[1] : view<4xi32> -> i32\n"
        "  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>"
    )
    new_dfl = (
        "  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>\n"
        "  %count32 = view.load %header[0] : view<4xi32> -> i32\n"
        "  %eos = view.load %header[1] : view<4xi32> -> i32\n"
        "  %mixed_word = view.load %header[2] : view<4xi32> -> i32\n"
        "  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32\n"
        "  %id_stride = scf.select %mixed, %c8, %c1i : index\n"
        "  %n_ids = index.mul %tokens, %id_stride : index\n"
        "  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>"
    )
    if old_dfl in text:
        text = text.replace(old_dfl, new_dfl, 1)
        old_anchor = (
            "      %cursor = index.cast %cursor_b : i32 to index\n"
            "      %ae = index.assume %cursor [range(%cursor, 0, 8191), lt(%cursor, %tokens)] : index\n"
            "      %anchor = view.load %ids_flat[%ae] : view<[%tokens]xi32> -> i32"
        )
        new_anchor = (
            "      %cursor = index.cast %cursor_b : i32 to index\n"
            "      %ae0 = index.mul %cursor, %id_stride : index\n"
            "      %ae = index.assume %ae0 [range(%ae0, 0, 65535), lt(%ae0, %n_ids)] : index\n"
            "      %anchor = view.load %ids_flat[%ae] : view<[%n_ids]xi32> -> i32"
        )
        assert old_anchor in text, "llm.loom: expected anchor load in g_dfl_prepare"
        text = text.replace(old_anchor, new_anchor, 1)

    return text
