#!/usr/bin/env python3
"""Audit a completed V4.1 GGUF and sample its payloads against the source.

Checks the complete tensor layout, then re-encodes selected expert tensors and
samples native Engram rows. This is artifact validation, not inference QA.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import zlib

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import (SourceDB, NativeQuantizer, Imatrix, build_plan,
                                  validate_scales, scale_name, QUANTIZATION)
from glm53_quantize import (GGUF_ARRAY, GGUF_UINT32, QTYPE_F16, QTYPE_F32,
                            QTYPE_IQ2_XXS, QTYPE_NAMES, QTYPE_Q2_K, QTYPE_Q4_K,
                            QTYPE_Q8_0, TensorPlan, align, qtype_nbytes,
                            read_exact, read_u32, read_u64, read_gguf_string,
                            skip_gguf_value)
from glm53_validate_gguf import read_selected_metadata

# The DSpark support file is written by gguf-tools/deepseek4-quantize.c, not by
# deepseek41_quantize.py: it is a separate artifact beside the main GGUF. These
# tables re-derive its plan from the checkpoint independently of the C tool.
DSPARK_ALIGNMENT = 32
DSPARK_ARCHITECTURE = "deepseek41-dspark"
DSPARK_NAME = "DeepSeek V4.1 Flash DSpark support"
DSPARK_QUANTIZATION = {
    "q4": "Q4_K gate/up/down; Q8_0 dense; F32 router and confidence head",
    "q2": "IQ2_XXS gate/up; Q2_K down; Q8_0 dense; F32 router and confidence head",
}

DSPARK_STAGE_NAMES = {
    "hc_attn_base": "hc_attn_base.weight",
    "hc_attn_fn": "hc_attn_fn.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_base": "hc_ffn_base.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight",
    "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn.attn_sink": "attn_sinks.weight",
    "attn.wq_a.weight": "attn_q_a.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wq_b.weight": "attn_q_b.weight",
    "attn.wkv.weight": "attn_kv.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "main_proj.weight": "main_proj.weight",
    "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight",
    "markov_head.embed.weight": "markov_head.markov_w1.weight",
    "markov_head.head.weight": "markov_head.markov_w2.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}
# The drafts are text-only, so the visual router bias is dropped, and every
# FP8/FP4 scale is consumed by the weight it belongs to.
DSPARK_DROPPED = ("ffn.gate.bias_vl",)
DSPARK_EXPERT_PARTS = {"w1": "ffn_gate_exps.weight",
                       "w3": "ffn_up_exps.weight",
                       "w2": "ffn_down_exps.weight"}
DSPARK_F32_MARKERS = ("_base.weight", "_scale.weight", "_norm.weight",
                      ".norm.weight", "attn_sinks.weight", "exp_probs_b.bias",
                      "ffn_gate_inp.weight", "confidence_head.proj.weight")
DSPARK_F16_MARKERS = ("hc_attn_fn.weight", "hc_ffn_fn.weight")


def dspark_expert_source(name):
    """mtp.{S}.ffn.experts.{X}.{w}.weight -> (stage, expert, part) or None."""
    parts = name.split(".")
    if len(parts) != 7 or parts[0] != "mtp" or parts[2:4] != ["ffn", "experts"]:
        return None
    if parts[5] not in DSPARK_EXPERT_PARTS or parts[6] not in ("weight", "scale"):
        return None
    if not parts[1].isdigit() or not parts[4].isdigit():
        return None
    return int(parts[1]), int(parts[4]), parts[5], parts[6]


def dspark_qtype(name, shape, quant, part=None):
    """The converter's per-role type policy, re-derived from the GGUF name."""
    if part is not None:
        if quant == "q4":
            return QTYPE_Q4_K
        return QTYPE_Q2_K if part == "w2" else QTYPE_IQ2_XXS
    if any(marker in name for marker in DSPARK_F16_MARKERS):
        return QTYPE_F16
    if any(marker in name for marker in DSPARK_F32_MARKERS):
        return QTYPE_F32
    if len(shape) > 1 and shape[0] % 32 == 0:
        return QTYPE_Q8_0
    return QTYPE_F16


def dspark_expert_sample(item, all_experts=False):
    """Which expert slabs of one DSpark tensor get a full byte comparison.

    check_payload's own sample is keyed on expert_layer, which on the main
    model is the layer index and therefore spans 40 different values.  A DSpark
    tensor's expert_layer is the draft *stage* (0/1/2), so that formula would
    give all three matrices of a stage the same three expert ids and the whole
    file only three distinct ids out of 128.  Stride over the expert range
    instead, with the start offset taken from the tensor name: every tensor
    then covers its whole range (no gap wider than the stride, both ends
    included) and different tensors land on different ids.
    """
    count = item.expert_count
    if all_experts:
        return set(range(count))
    stride = max(1, count // 8)
    start = zlib.crc32(item.name.encode()) % stride
    return {0, count - 1} | set(range(start, count, stride))


def build_dspark_plan(db, config, quant="q4"):
    """Mirror the support file the C converter writes, from the source headers."""
    if quant not in DSPARK_QUANTIZATION:
        raise ValueError(f"unknown DSpark quantization recipe: {quant}")
    c = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    experts = c["dspark_n_routed_experts"]
    stages = c["num_nextn_predict_layers"]
    plan, consumed, seen = {}, set(), {}

    for name in sorted(db.tensors):
        if not name.startswith("mtp."):
            continue
        stage, rest = name[4:].split(".", 1)
        if not stage.isdigit():
            raise ValueError(f"{name}: unparsable DSpark stage")
        stage = int(stage)
        if not 0 <= stage < stages:
            raise ValueError(f"{name}: stage outside 0..{stages - 1}")
        expert = dspark_expert_source(name)
        if expert is not None:
            _, index, part, field = expert
            if not 0 <= index < experts:
                raise ValueError(f"{name}: expert outside 0..{experts - 1}")
            consumed.add(name)
            if field == "scale":
                continue
            info = db.info(name)
            if info["dtype"] != "I8" or len(info["shape"]) != 2:
                raise ValueError(f"{name}: expected a packed FP4 matrix")
            target = f"mtp.{stage}.{DSPARK_EXPERT_PARTS[part]}"
            shape = (info["shape"][1] * 2, info["shape"][0], experts)
            previous = seen.setdefault(target, shape)
            if previous != shape:
                raise ValueError(f"{target}: inconsistent expert shape {shape}")
            if target not in plan:
                plan[target] = TensorPlan(
                    target, shape, dspark_qtype(target, shape, quant, part),
                    "experts", source=f"mtp.{stage}.ffn.experts.{{expert}}.{part}.weight",
                    expert_layer=stage, expert_part=part, expert_count=experts)
            continue
        if rest in DSPARK_DROPPED:
            consumed.add(name)
            continue
        if rest.endswith(".scale") and rest[:-len(".scale")] + ".weight" in DSPARK_STAGE_NAMES:
            consumed.add(name)
            continue
        if rest not in DSPARK_STAGE_NAMES:
            raise ValueError(f"{name}: no DSpark name rule")
        consumed.add(name)
        info = db.info(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            consumed.add(scale_name(name))
        shape = tuple(reversed(info["shape"]))
        target = f"mtp.{stage}.{DSPARK_STAGE_NAMES[rest]}"
        if target in plan:
            raise ValueError(f"{target}: emitted twice")
        plan[target] = TensorPlan(target, shape, dspark_qtype(target, shape, quant),
                                  "dspark", source=name)

    missing = sorted(name for name in db.tensors
                     if name.startswith("mtp.") and name not in consumed)
    if missing:
        raise ValueError(f"unclaimed DSpark source tensors: {missing[:10]}")
    # The C converter sorts its tensor table by GGUF name before assigning
    # offsets; reproduce that order so the layout comparison is meaningful.
    items = [plan[key] for key in sorted(plan)]
    if not items:
        raise ValueError("no DSpark tensors in the source checkpoint")
    offset = 0
    for item in items:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, DSPARK_ALIGNMENT)
    return items


def read_dspark_metadata(fp, count):
    """Read the whole KV block, keeping the keys the support file must carry."""
    wanted = {"general.architecture", "general.name", "general.alignment",
              "dspark.block_size", "dspark.markov_rank", "dspark.noise_token_id",
              "dspark.stage_count", "dspark.n_layers", "dspark.n_expert",
              "dspark.n_expert_used", "dspark.target_layer_ids"}
    metadata = {}
    for _ in range(count):
        key = read_gguf_string(fp, "metadata key")
        kind = read_u32(fp, "metadata type")
        if key not in wanted:
            skip_gguf_value(fp, kind)
            continue
        if key in metadata:
            raise ValueError(f"duplicate metadata: {key}")
        if key == "dspark.target_layer_ids":
            if kind != GGUF_ARRAY:
                raise ValueError("dspark.target_layer_ids is not an array")
            element = read_u32(fp, "array element type")
            length = read_u64(fp, "array length")
            if element != GGUF_UINT32:
                raise ValueError("dspark.target_layer_ids is not a uint32 array")
            metadata[key] = tuple(read_u32(fp, "target layer") for _ in range(length))
        else:
            metadata[key] = read_selected_metadata(fp, kind)
    return metadata


def dequantize_q8_0(payload, ncols, np):
    """Independent Q8_0 reader: 32 int8 codes behind one f16 block scale."""
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    codes = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (codes * scales).reshape(-1, ncols)


def dequantize_q4_k(payload, ncols, np):
    """Independent Q4_K reader: d/dmin f16, 6-bit scale+min pairs, 4-bit codes."""
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 144)
    n = blocks.shape[0]
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).reshape(n)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32).reshape(n)
    packed = blocks[:, 4:16].astype(np.uint32)
    qs = blocks[:, 16:144]
    out = np.empty((n, 256), dtype=np.float32)
    for j in range(8):
        if j < 4:
            sc = packed[:, j] & 63
            mn = packed[:, j + 4] & 63
        else:
            sc = (packed[:, j + 4] & 0xF) | ((packed[:, j - 4] >> 6) << 4)
            mn = (packed[:, j + 4] >> 4) | ((packed[:, j] >> 6) << 4)
        half, low = divmod(j, 2)
        nibbles = qs[:, half * 32:(half + 1) * 32]
        codes = (nibbles & 0xF) if low == 0 else (nibbles >> 4)
        out[:, j * 32:(j + 1) * 32] = (codes.astype(np.float32) * (d * sc)[:, None]
                                       - (dmin * mn)[:, None])
    return out.reshape(-1, ncols)


def dspark_numeric_spot_check(fp, start, plan, db, quantizer):
    """Decode two payloads back to floats and compare with the source weights.

    The byte comparison above proves the file matches the recipe; this proves
    the recipe preserved the weights, using a reader written from the block
    layout rather than the encoder that produced them.
    """
    np = quantizer.np
    readers = {QTYPE_Q8_0: dequantize_q8_0, QTYPE_Q4_K: dequantize_q4_k}
    samples = []
    dense = next((t for t in plan if t.qtype == QTYPE_Q8_0 and not t.is_expert), None)
    expert = next((t for t in plan if t.is_expert and t.qtype in readers), None)
    if dense is not None:
        samples.append((dense, dense.source, dense.offset, dense.nbytes))
    if expert is not None:
        stride = expert.nbytes // expert.expert_count
        samples.append((expert, expert.source.format(expert=0), expert.offset, stride))
    for item, source, offset, nbytes in samples:
        fp.seek(start + offset)
        decoded = readers[item.qtype](read_exact(fp, nbytes, item.name), item.shape[0], np)
        values = quantizer.to_f32(db, source)
        if decoded.shape != values.shape:
            raise ValueError(f"{item.name}: decoded {decoded.shape}, source {values.shape}")
        scale = float(np.abs(values).max()) or 1.0
        error = float(np.abs(decoded - values).max()) / scale
        rms = float(np.sqrt(np.mean(np.square(decoded - values)))) / scale
        limit = 0.02 if item.qtype == QTYPE_Q8_0 else 0.25
        print(f"PASS: {item.name} {QTYPE_NAMES[item.qtype]} round-trip "
              f"max_rel={error:.5f} rms_rel={rms:.5f} (limit {limit})", flush=True)
        if not error < limit:
            raise ValueError(f"{item.name}: dequantized payload is not the source weight")


def validate_dspark(args):
    config = json.loads((Path(args.hf) / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("--dspark expects a DeepSeek V4.1 checkpoint")
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales)
    try:
        plan = build_dspark_plan(db, config, args.quant)
        c = config["text_config"]
        stages = c["num_nextn_predict_layers"]
        with open(args.gguf, "rb") as fp:
            if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
                raise ValueError("expected GGUF v3")
            if read_u64(fp, "tensor count") != len(plan):
                raise ValueError(f"tensor count differs from source plan ({len(plan)})")
            metadata = read_dspark_metadata(fp, read_u64(fp, "metadata count"))
            expected = {"general.architecture": DSPARK_ARCHITECTURE,
                        "general.name": DSPARK_NAME,
                        "general.alignment": DSPARK_ALIGNMENT,
                        "dspark.block_size": c["dspark_block_size"],
                        "dspark.markov_rank": c["dspark_markov_rank"],
                        "dspark.noise_token_id": c["dspark_noise_token_id"],
                        "dspark.target_layer_ids": tuple(c["dspark_target_layer_ids"]),
                        "dspark.stage_count": stages,
                        "dspark.n_layers": stages,
                        "dspark.n_expert": c["dspark_n_routed_experts"],
                        "dspark.n_expert_used": c["dspark_num_experts_per_tok"]}
            if metadata != expected:
                raise ValueError(f"metadata mismatch: {metadata}")
            for item in plan:
                name = read_gguf_string(fp, "tensor name")
                rank = read_u32(fp, "tensor rank")
                if rank != len(item.shape):
                    raise ValueError(f"{name}: unexpected rank {rank}")
                shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
                kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
                if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f"{item.name}: tensor layout mismatch")
            start = align(fp.tell(), DSPARK_ALIGNMENT)
            size = start + plan[-1].offset + align(plan[-1].nbytes, DSPARK_ALIGNMENT)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f"expected file size {size}")
            print(f"PASS: {len(plan)} DSpark tensor layouts; complete {size}-byte "
                  f"support GGUF ({DSPARK_QUANTIZATION[args.quant]})", flush=True)
            if args.payload:
                q = NativeQuantizer(args.quants_library)
                imatrix = Imatrix(args.imatrix, q.np)
                for item in plan:
                    if item.qtype == QTYPE_IQ2_XXS and not args.imatrix:
                        raise ValueError(f"{item.name}: re-encoding IQ2_XXS without an "
                                         "imatrix depends on the converter's synthetic "
                                         "importance accumulation order; pass --imatrix")
                slabs, ids, total = 0, set(), 0
                for index, item in enumerate(plan):
                    experts = (dspark_expert_sample(item, args.all_experts)
                               if item.is_expert else None)
                    if experts is not None:
                        slabs, total = slabs + len(experts), total + item.expert_count
                        ids.update(experts)
                    check_payload(fp, start + item.offset, item, db, q, imatrix, experts)
                    if (index + 1) % 10 == 0 or index + 1 == len(plan):
                        print(f"PASS: source payload checks {index + 1}/{len(plan)}", flush=True)
                print(f"PASS: expert payload coverage {slabs}/{total} slabs over "
                      f"{len(ids)} distinct expert ids", flush=True)
                dspark_numeric_spot_check(fp, start, plan, db, q)
    finally:
        db.close()


def check_payload(fp, offset, item, db, quantizer, imatrix, expert_ids=None):
    if item.role == "engram_disk":
        rows = item.shape[1]
        rng = random.Random(41 + rows)
        selected = {0, 1, rows - 1, 16383, 16384, (1 << 32) // 264}
        selected.update(rng.randrange(rows) for _ in range(128))
        for row in sorted(selected):
            if row >= rows:
                continue
            expected = b"".join(db.iter_read(item.source, row * 256, 256))
            expected += b"".join(db.iter_read(scale_name(item.source), row * 8, 8))
            fp.seek(offset + row * 264)
            if read_exact(fp, 264, item.name) != expected:
                raise ValueError(f"{item.name}: Engram row {row} differs from source")
    elif item.is_expert:
        selected = set(expert_ids) if expert_ids is not None else {
            0, (item.expert_layer * 17 + 41) % item.expert_count, item.expert_count - 1}
        stride = item.nbytes // item.expert_count
        for expert in sorted(selected):
            values = quantizer.to_f32(db, item.source.format(expert=expert))
            importance = imatrix.expert(item.name, expert, item.shape[0], item.expert_count)
            expected = quantizer.encode(values, item.qtype, importance)
            fp.seek(offset + expert * stride)
            if len(expected) != stride or read_exact(fp, stride, item.name) != expected:
                raise ValueError(f"{item.name}: encoded expert {expert} differs from source recipe")
    else:
        values = quantizer.to_f32(db, item.source)
        expected = quantizer.encode(values, item.qtype)
        fp.seek(offset)
        if len(expected) != item.nbytes or read_exact(fp, item.nbytes, item.name) != expected:
            raise ValueError(f"{item.name}: payload differs from source recipe")


def validate(args):
    config = json.loads((Path(args.hf) / "config.json").read_text())
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales)
    try:
        plan = build_plan(db, config, args.quant)
        with open(args.gguf, "rb") as fp:
            if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
                raise ValueError("expected GGUF v3")
            if read_u64(fp, "tensor count") != len(plan):
                raise ValueError("tensor count differs from source plan")
            metadata = {}
            for _ in range(read_u64(fp, "metadata count")):
                key = read_gguf_string(fp, "metadata key")
                kind = read_u32(fp, "metadata type")
                if key in {"general.architecture", "general.alignment", "general.source.revision",
                            "deepseek41.calibration", "deepseek41.quantization"}:
                    if key in metadata:
                        raise ValueError(f"duplicate metadata: {key}")
                    metadata[key] = read_selected_metadata(fp, kind)
                else:
                    skip_gguf_value(fp, kind)
            expected = {"general.architecture": "deepseek41", "general.alignment": GGUF_ALIGNMENT,
                        "general.source.revision": args.source_revision,
                        "deepseek41.quantization": QUANTIZATION[args.quant],
                        "deepseek41.calibration": "imatrix" if args.imatrix else "weight-energy bootstrap"}
            if metadata != expected:
                raise ValueError(f"metadata mismatch: {metadata}")
            for item in plan:
                name = read_gguf_string(fp, "tensor name")
                rank = read_u32(fp, "tensor rank")
                if rank != len(item.shape):
                    raise ValueError(f"{name}: unexpected rank {rank}")
                shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
                kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
                if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f"{item.name}: tensor layout mismatch")
            start = align(fp.tell(), GGUF_ALIGNMENT)
            size = start + plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f"expected file size {size}")
            print(f"PASS: {len(plan)} tensor layouts; complete {size}-byte GGUF", flush=True)
            if args.payload:
                q = NativeQuantizer(args.quants_library)
                imatrix = Imatrix(args.imatrix, q.np)
                if args.imatrix and any(t.is_expert and t.name not in imatrix.entries for t in plan):
                    raise ValueError("imatrix is missing expert tensors")
                for index, item in enumerate(plan):
                    check_payload(fp, start + item.offset, item, db, q, imatrix)
                    if (index + 1) % 25 == 0 or index + 1 == len(plan):
                        print(f"PASS: source payload checks {index + 1}/{len(plan)}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--quant")
    parser.add_argument("--imatrix")
    parser.add_argument("--payload", action="store_true")
    parser.add_argument("--dspark", action="store_true",
                        help="audit a V4.1 DSpark support GGUF instead of the main model")
    parser.add_argument("--all-experts", action="store_true",
                        help="with --dspark: byte-check every expert slab, not a stride sample")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args()
    choices = DSPARK_QUANTIZATION if args.dspark else QUANTIZATION
    if args.quant is None:
        args.quant = "q4" if args.dspark else "q2"
    if args.quant not in choices:
        parser.error(f"--quant must be one of {sorted(choices)}")
    if not args.dspark and not args.source_revision:
        parser.error("--source-revision is required for the main model")
    if args.all_experts and not args.dspark:
        parser.error("--all-experts applies to --dspark audits")
    try:
        validate_dspark(args) if args.dspark else validate(args)
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-validate: {error}")
