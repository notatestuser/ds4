#!/usr/bin/env python3
"""End-to-end check of the V4.1 DSpark support-GGUF converter.

Builds a tiny synthetic V4.1 checkpoint (real safetensors shards, real FP8/FP4
block layouts, three draft stages), runs gguf-tools/deepseek4-quantize
--dspark-support --v41-dspark over it, and audits the result with
deepseek41_validate_gguf.py --dspark --payload. No model weights and no GPU.
"""

import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_quantize import NativeQuantizer, SourceDB, validate_scales
from deepseek41_validate_gguf import (DSPARK_ALIGNMENT, build_dspark_plan,
                                      dequantize_q4_k, dequantize_q8_0,
                                      dspark_expert_sample)
from glm53_quantize import (QTYPE_F16, QTYPE_F32, QTYPE_IQ2_XXS, QTYPE_Q2_K,
                            QTYPE_Q4_K, QTYPE_Q8_0, TensorPlan, align,
                            read_gguf_string, read_u32, skip_gguf_value)

CONVERTER = ROOT / "gguf-tools" / "deepseek4-quantize"
VALIDATOR = ROOT / "gguf-tools" / "deepseek41_validate_gguf.py"

# Small but structurally faithful: FP8 dense weights blocked 32x32, FP4 experts
# with one E8M0 scale per 32 columns, Q4_K rows a multiple of 256.
DIM = 256
INTER = 256
HC = 4
HC_MIX = 2 * HC + HC * HC
Q_LORA = 64
HEADS_X_HEADDIM = 128
OUT_LOW = 64
OUT_A_ROWS = 64
KV = 64
VOCAB = 512
RANK = 32
EXPERTS = 4
# dspark_expert_sample strides by max(1, count // 8), so the stride sample is
# only a strict subset of the expert range once the count reaches 16.  One
# fixture at that width keeps --all-experts falsifiable end to end.
WIDE_EXPERTS = 16
STAGES = 3
TARGET_LAYERS = [5, 6, 7]


def fp8_codes(rng, rows, cols):
    codes = rng.integers(0, 256, (rows, cols), dtype=np.uint16).astype(np.uint8)
    codes[(codes & 127) == 127] = 0          # no NaN/Inf codes in the source
    return codes


def e8m0(rng, rows, cols):
    # A realistic per-block exponent spread: a 2^8 range inside one row keeps
    # the Q4_K/Q8_0 round-trip bounds below meaningful rather than vacuous.
    return rng.integers(124, 133, (rows, cols), dtype=np.uint16).astype(np.uint8)


def bf16(rng, *shape):
    values = rng.standard_normal(shape).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype("<u2")


class Shard:
    """Minimal safetensors writer."""

    def __init__(self):
        self.entries = {}
        self.blobs = []
        self.offset = 0

    def add(self, name, dtype, shape, payload):
        data = payload.tobytes()
        self.entries[name] = {"dtype": dtype, "shape": list(shape),
                              "data_offsets": [self.offset, self.offset + len(data)]}
        self.blobs.append(data)
        self.offset += len(data)
        return name

    def write(self, path):
        header = json.dumps(self.entries).encode()
        header += b" " * (-len(header) % 8)
        with open(path, "wb") as fp:
            fp.write(struct.pack("<Q", len(header)))
            fp.write(header)
            for blob in self.blobs:
                fp.write(blob)
        return self.entries


def build_checkpoint(directory, seed=41, experts=EXPERTS):
    rng = np.random.default_rng(seed)
    shard = Shard()
    names = []

    def fp8(name, rows, cols):
        names.append(shard.add(name, "F8_E4M3", (rows, cols), fp8_codes(rng, rows, cols)))
        names.append(shard.add(name.removesuffix(".weight") + ".scale", "F8_E8M0",
                               (rows // 32, cols // 32), e8m0(rng, rows // 32, cols // 32)))

    def raw(name, dtype, shape, payload):
        names.append(shard.add(name, dtype, shape, payload))

    for stage in range(STAGES):
        p = f"mtp.{stage}"
        for site in ("attn", "ffn"):
            raw(f"{p}.hc_{site}_fn", "F32", (HC_MIX, HC * DIM),
                rng.standard_normal((HC_MIX, HC * DIM)).astype("<f4"))
            raw(f"{p}.hc_{site}_base", "F32", (HC_MIX,),
                rng.standard_normal(HC_MIX).astype("<f4"))
            raw(f"{p}.hc_{site}_scale", "F32", (3,), rng.standard_normal(3).astype("<f4"))
            raw(f"{p}.{site}_norm.weight", "BF16", (DIM,), bf16(rng, DIM))
        raw(f"{p}.attn.attn_sink", "F32", (8,), rng.standard_normal(8).astype("<f4"))
        fp8(f"{p}.attn.wq_a.weight", Q_LORA, DIM)
        raw(f"{p}.attn.q_norm.weight", "BF16", (Q_LORA,), bf16(rng, Q_LORA))
        fp8(f"{p}.attn.wq_b.weight", HEADS_X_HEADDIM, Q_LORA)
        fp8(f"{p}.attn.wkv.weight", KV, DIM)
        raw(f"{p}.attn.kv_norm.weight", "BF16", (KV,), bf16(rng, KV))
        fp8(f"{p}.attn.wo_a.weight", OUT_LOW, OUT_A_ROWS)
        fp8(f"{p}.attn.wo_b.weight", DIM, OUT_LOW)
        raw(f"{p}.ffn.gate.weight", "BF16", (experts, DIM), bf16(rng, experts, DIM))
        raw(f"{p}.ffn.gate.bias", "F32", (experts,), rng.standard_normal(experts).astype("<f4"))
        raw(f"{p}.ffn.gate.bias_vl", "F32", (experts,),
            rng.standard_normal(experts).astype("<f4"))
        fp8(f"{p}.ffn.shared_experts.w1.weight", INTER, DIM)
        fp8(f"{p}.ffn.shared_experts.w3.weight", INTER, DIM)
        fp8(f"{p}.ffn.shared_experts.w2.weight", DIM, INTER)
        for expert in range(experts):
            for part, rows, cols in (("w1", INTER, DIM), ("w3", INTER, DIM), ("w2", DIM, INTER)):
                base = f"{p}.ffn.experts.{expert}.{part}"
                raw(f"{base}.weight", "I8", (rows, cols // 2),
                    rng.integers(0, 256, (rows, cols // 2), dtype=np.uint16).astype(np.uint8))
                raw(f"{base}.scale", "F8_E8M0", (rows, cols // 32),
                    e8m0(rng, rows, cols // 32))
        if stage == 0:
            fp8(f"{p}.main_proj.weight", DIM, len(TARGET_LAYERS) * DIM)
            raw(f"{p}.main_norm.weight", "BF16", (DIM,), bf16(rng, DIM))
        if stage == STAGES - 1:
            raw(f"{p}.norm.weight", "BF16", (DIM,), bf16(rng, DIM))
            raw(f"{p}.markov_head.embed.weight", "BF16", (VOCAB, RANK), bf16(rng, VOCAB, RANK))
            raw(f"{p}.markov_head.head.weight", "BF16", (VOCAB, RANK), bf16(rng, VOCAB, RANK))
            raw(f"{p}.confidence_head.proj.weight", "BF16", (1, DIM + RANK),
                bf16(rng, 1, DIM + RANK))

    shard_name = "model-00001-of-00001.safetensors"
    shard.write(Path(directory) / shard_name)
    (Path(directory) / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": shard.offset},
         "weight_map": {name: shard_name for name in names}}))
    (Path(directory) / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32],
                                "expert_dtype": "fp4"},
        "text_config": {
            "hidden_size": DIM, "moe_intermediate_size": INTER, "vocab_size": VOCAB,
            "rms_norm_eps": 1e-20, "compress_ratios": [0] * (8 + STAGES),
            "num_nextn_predict_layers": STAGES, "dspark_block_size": 5,
            "dspark_noise_token_id": 499, "dspark_target_layer_ids": TARGET_LAYERS,
            "dspark_markov_rank": RANK, "dspark_n_routed_experts": experts,
            "dspark_num_experts_per_tok": 3,
        },
    }))
    return Path(directory)


def build_v4_checkpoint(directory, seed=11):
    """A V4-shaped DSpark source: 128x128 FP8 blocks, hc_head_*, markov_w1/w2.

    The V4 recipe must keep working untouched beside the new V4.1 one, so this
    fixture pins its inventory, its types and its metadata.
    """
    rng = np.random.default_rng(seed)
    shard = Shard()
    names = []

    def fp8(name, rows, cols):
        names.append(shard.add(name, "F8_E4M3", (rows, cols), fp8_codes(rng, rows, cols)))
        names.append(shard.add(name.removesuffix(".weight") + ".scale", "F8_E8M0",
                               (rows // 128, cols // 128),
                               e8m0(rng, rows // 128, cols // 128)))

    def raw(name, dtype, shape, payload):
        names.append(shard.add(name, dtype, shape, payload))

    v4_dim, v4_lora = 256, 128
    for stage in range(STAGES):
        p = f"mtp.{stage}"
        for site in ("attn", "ffn"):
            raw(f"{p}.hc_{site}_fn", "F32", (HC_MIX, HC * v4_dim),
                rng.standard_normal((HC_MIX, HC * v4_dim)).astype("<f4"))
            raw(f"{p}.hc_{site}_base", "F32", (HC_MIX,),
                rng.standard_normal(HC_MIX).astype("<f4"))
            raw(f"{p}.hc_{site}_scale", "F32", (3,), rng.standard_normal(3).astype("<f4"))
            raw(f"{p}.{site}_norm.weight", "BF16", (v4_dim,), bf16(rng, v4_dim))
        raw(f"{p}.attn.attn_sink", "F32", (8,), rng.standard_normal(8).astype("<f4"))
        fp8(f"{p}.attn.wq_a.weight", v4_lora, v4_dim)
        raw(f"{p}.attn.q_norm.weight", "BF16", (v4_lora,), bf16(rng, v4_lora))
        fp8(f"{p}.attn.wq_b.weight", v4_dim, v4_lora)
        fp8(f"{p}.attn.wkv.weight", v4_lora, v4_dim)
        raw(f"{p}.attn.kv_norm.weight", "BF16", (v4_lora,), bf16(rng, v4_lora))
        fp8(f"{p}.attn.wo_a.weight", v4_lora, v4_dim)
        fp8(f"{p}.attn.wo_b.weight", v4_dim, v4_lora)
        raw(f"{p}.ffn.gate.weight", "BF16", (EXPERTS, v4_dim), bf16(rng, EXPERTS, v4_dim))
        raw(f"{p}.ffn.gate.bias", "F32", (EXPERTS,), rng.standard_normal(EXPERTS).astype("<f4"))
        raw(f"{p}.ffn.gate.bias_vl", "F32", (EXPERTS,),
            rng.standard_normal(EXPERTS).astype("<f4"))
        fp8(f"{p}.ffn.shared_experts.w1.weight", INTER, v4_dim)
        fp8(f"{p}.ffn.shared_experts.w3.weight", INTER, v4_dim)
        fp8(f"{p}.ffn.shared_experts.w2.weight", v4_dim, INTER)
        for expert in range(EXPERTS):
            for part, rows, cols in (("w1", INTER, v4_dim), ("w3", INTER, v4_dim),
                                     ("w2", v4_dim, INTER)):
                base = f"{p}.ffn.experts.{expert}.{part}"
                raw(f"{base}.weight", "I8", (rows, cols // 2),
                    rng.integers(0, 256, (rows, cols // 2), dtype=np.uint16).astype(np.uint8))
                raw(f"{base}.scale", "F8_E8M0", (rows, cols // 32), e8m0(rng, rows, cols // 32))
        if stage == 0:
            fp8(f"{p}.main_proj.weight", v4_dim, 3 * v4_dim)
            raw(f"{p}.main_norm.weight", "BF16", (v4_dim,), bf16(rng, v4_dim))
        if stage == STAGES - 1:
            raw(f"{p}.norm.weight", "BF16", (v4_dim,), bf16(rng, v4_dim))
            raw(f"{p}.hc_head_base", "F32", (HC,), rng.standard_normal(HC).astype("<f4"))
            raw(f"{p}.hc_head_fn", "F32", (HC, HC * v4_dim),
                rng.standard_normal((HC, HC * v4_dim)).astype("<f4"))
            raw(f"{p}.hc_head_scale", "F32", (1,), rng.standard_normal(1).astype("<f4"))
            raw(f"{p}.markov_head.markov_w1.weight", "BF16", (VOCAB, RANK), bf16(rng, VOCAB, RANK))
            raw(f"{p}.markov_head.markov_w2.weight", "BF16", (VOCAB, RANK), bf16(rng, VOCAB, RANK))
            raw(f"{p}.confidence_head.proj.weight", "BF16", (1, v4_dim + RANK),
                bf16(rng, 1, v4_dim + RANK))

    shard_name = "model-00001-of-00001.safetensors"
    shard.write(Path(directory) / shard_name)
    (Path(directory) / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": shard.offset},
         "weight_map": {name: shard_name for name in names}}))
    (Path(directory) / "config.json").write_text(json.dumps(
        {"model_type": "deepseek_v4", "compress_ratios": [0] * (8 + STAGES),
         "rms_norm_eps": 1e-6}))
    return Path(directory)


def read_gguf_header(path):
    """(kv dict, [(name, type, shape)]) of a GGUF file."""
    with open(path, "rb") as fp:
        assert fp.read(4) == b"GGUF" and read_u32(fp, "version") == 3
        n_tensors = struct.unpack("<Q", fp.read(8))[0]
        n_kv = struct.unpack("<Q", fp.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            key = read_gguf_string(fp, "key")
            kind = read_u32(fp, "type")
            if kind == 8:
                kv[key] = read_gguf_string(fp, "value")
            elif kind == 4:
                kv[key] = read_u32(fp, "value")
            elif kind == 9:
                element = read_u32(fp, "element type")
                length = struct.unpack("<Q", fp.read(8))[0]
                kv[key] = tuple(read_u32(fp, "element") for _ in range(length))
            else:
                skip_gguf_value(fp, kind)
        tensors = []
        for _ in range(n_tensors):
            name = read_gguf_string(fp, "name")
            rank = read_u32(fp, "rank")
            shape = struct.unpack(f"<{rank}Q", fp.read(8 * rank))
            kind = read_u32(fp, "tensor type")
            fp.read(8)
            tensors.append((name, kind, shape))
        return kv, tensors


class DsparkConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONVERTER.exists():
            raise unittest.SkipTest(f"missing {CONVERTER}; run make -C gguf-tools")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.hf = build_checkpoint(cls.tmp.name)
        cls.out = Path(cls.tmp.name) / "dspark-support-q4k.gguf"
        cls.convert(["--out", str(cls.out), "--overwrite"])
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.library = ROOT / "gguf-tools" / f"libds4quants.{suffix}"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def convert(cls, extra, check=True):
        command = [str(CONVERTER), "--hf", str(cls.hf), "--dspark-support",
                   "--v41-dspark", "--threads", "2"] + extra
        result = subprocess.run(command, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def plan(self, quant="q4"):
        db = SourceDB(str(self.hf), index_validator=lambda _: None,
                      scale_validator=validate_scales)
        try:
            return build_dspark_plan(db, json.loads((self.hf / "config.json").read_text()), quant)
        finally:
            db.close()

    def coverage(self, stdout):
        """(checked slabs, total slabs, distinct ids) of a --payload audit."""
        match = re.search(r"expert payload coverage (\d+)/(\d+) slabs over "
                          r"(\d+) distinct expert ids", stdout)
        self.assertIsNotNone(match, stdout)
        return tuple(int(value) for value in match.groups())

    def test_name_mapping_and_inventory(self):
        plan = {item.name: item for item in self.plan()}
        # 21 per-stage dense + 3 expert tensors per stage, 6 stage-specific.
        self.assertEqual(len(plan), STAGES * 24 + 6)
        self.assertIn("mtp.2.markov_head.markov_w1.weight", plan)
        self.assertIn("mtp.2.markov_head.markov_w2.weight", plan)
        self.assertNotIn("mtp.0.visual_router_bias", plan)
        self.assertEqual(plan["mtp.2.markov_head.markov_w1.weight"].source,
                         "mtp.2.markov_head.embed.weight")
        self.assertEqual(plan["mtp.0.main_proj.weight"].shape,
                         (len(TARGET_LAYERS) * DIM, DIM))
        self.assertEqual(plan["mtp.0.ffn_down_exps.weight"].shape, (INTER, DIM, EXPERTS))

    def test_quant_policy(self):
        plan = {item.name: item for item in self.plan()}
        self.assertEqual(plan["mtp.0.ffn_gate_inp.weight"].qtype, QTYPE_F32)
        self.assertEqual(plan["mtp.2.confidence_head.proj.weight"].qtype, QTYPE_F32)
        self.assertEqual(plan["mtp.0.hc_attn_fn.weight"].qtype, QTYPE_F16)
        self.assertEqual(plan["mtp.0.attn_norm.weight"].qtype, QTYPE_F32)
        self.assertEqual(plan["mtp.0.attn_q_a.weight"].qtype, QTYPE_Q8_0)
        self.assertEqual(plan["mtp.2.markov_head.markov_w1.weight"].qtype, QTYPE_Q8_0)
        for part in ("gate", "up", "down"):
            self.assertEqual(plan[f"mtp.1.ffn_{part}_exps.weight"].qtype, QTYPE_Q4_K)
        legacy = {item.name: item for item in self.plan("q2")}
        self.assertEqual(legacy["mtp.1.ffn_gate_exps.weight"].qtype, QTYPE_IQ2_XXS)
        self.assertEqual(legacy["mtp.1.ffn_up_exps.weight"].qtype, QTYPE_IQ2_XXS)
        self.assertEqual(legacy["mtp.1.ffn_down_exps.weight"].qtype, QTYPE_Q2_K)

    def test_expert_sampling_spreads_over_the_whole_range(self):
        """Every expert tensor's byte check must span its own expert range.

        check_payload's default set is keyed on expert_layer, which for a
        DSpark tensor is the draft stage, so the three matrices of a stage
        would sample identical ids and a 128-expert file only three of them.
        """
        items = [item for item in self.plan() if item.is_expert]
        self.assertEqual(len(items), STAGES * 3)
        for item in items:                       # four experts in the fixture
            self.assertEqual(dspark_expert_sample(item), set(range(EXPERTS)))
        union, legacy = set(), set()
        for item in items:                       # the same names at 128 experts
            wide = TensorPlan(item.name, item.shape, item.qtype, item.role,
                              expert_layer=item.expert_layer, expert_count=128)
            sample = sorted(dspark_expert_sample(wide))
            union.update(sample)
            legacy.update({0, (wide.expert_layer * 17 + 41) % 128, 127})
            self.assertGreaterEqual(len(sample), 9)
            self.assertEqual((sample[0], sample[-1]), (0, 127))
            self.assertLessEqual(max(b - a for a, b in zip(sample, sample[1:])), 16)
            self.assertEqual(dspark_expert_sample(wide, all_experts=True), set(range(128)))
        self.assertEqual(len(legacy), 5)          # what the shared formula gives
        self.assertGreaterEqual(len(union), 24)

    def test_manifest_maps_every_source_tensor(self):
        result = subprocess.run([str(CONVERTER), "--hf", str(self.hf), "--dspark-manifest"],
                                capture_output=True, text=True, check=True)
        self.assertIn("# unknown_dspark_tensors=0", result.stdout)
        self.assertIn("# dspark_stages=3", result.stdout)
        self.assertIn("markov_head.embed.weight\tmtp.2.markov_head.markov_w1.weight",
                      result.stdout)
        self.assertIn("consume_visual_sidecar", result.stdout)

    def test_converter_reports_the_family_and_geometry(self):
        result = self.convert(["--dry-run"])
        self.assertIn("dspark_family: deepseek41", result.stdout)
        self.assertIn(f"dspark_experts: count={EXPERTS} used=3", result.stdout)
        self.assertIn(f"target_layers={','.join(map(str, TARGET_LAYERS))}", result.stdout)

    def test_expert_count_mismatch_is_refused(self):
        result = self.convert(["--dry-run", "--n-experts", str(EXPERTS + 1)], check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match DSpark tensor", result.stderr)

    def test_unknown_mtp_tensor_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            hf = build_checkpoint(scratch)
            index_path = hf / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["mtp.0.mystery.weight"] = next(iter(index["weight_map"].values()))
            index_path.write_text(json.dumps(index))
            result = subprocess.run(
                [str(CONVERTER), "--hf", str(hf), "--dspark-support", "--v41-dspark",
                 "--dry-run"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown tensors", result.stderr)

    def test_validator_accepts_the_converted_file(self):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--hf", str(self.hf), "--gguf", str(self.out),
             "--dspark", "--payload", "--quants-library", str(self.library)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DSpark tensor layouts", result.stdout)
        slabs = STAGES * 3 * EXPERTS
        self.assertIn(f"expert payload coverage {slabs}/{slabs} slabs over "
                      f"{EXPERTS} distinct expert ids", result.stdout)
        self.assertIn("round-trip", result.stdout)

    def test_all_experts_flag(self):
        """The flag is accepted for a DSpark audit and refused for a main one."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--hf", str(self.hf), "--gguf", str(self.out),
             "--dspark", "--payload", "--all-experts", "--quants-library", str(self.library)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        slabs = STAGES * 3 * EXPERTS
        self.assertIn(f"expert payload coverage {slabs}/{slabs} slabs", result.stdout)
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--hf", str(self.hf), "--gguf", str(self.out),
             "--all-experts", "--source-revision", "x"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--all-experts applies to --dspark audits", result.stderr)

    def test_all_experts_widens_the_sample(self):
        """--all-experts must reach dspark_expert_sample, not merely parse.

        At the shared fixture's four experts the stride sample is already the
        whole range (stride = max(1, 4 // 8) = 1, start = crc32 % 1 = 0), so
        both audits print the same coverage and dropping the flag at the
        validator's call site would go unnoticed.  WIDE_EXPERTS crosses the
        stride threshold, where the default audit must byte-check strictly
        fewer slabs than --all-experts.
        """
        with tempfile.TemporaryDirectory() as scratch:
            hf = build_checkpoint(scratch, seed=17, experts=WIDE_EXPERTS)
            out = Path(scratch) / "wide-dspark.gguf"
            result = subprocess.run(
                [str(CONVERTER), "--hf", str(hf), "--dspark-support", "--v41-dspark",
                 "--threads", "2", "--out", str(out), "--overwrite"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = [sys.executable, str(VALIDATOR), "--hf", str(hf), "--gguf", str(out),
                     "--dspark", "--payload", "--quants-library", str(self.library)]
            sampled = subprocess.run(audit, capture_output=True, text=True)
            self.assertEqual(sampled.returncode, 0, sampled.stdout + sampled.stderr)
            every = subprocess.run(audit + ["--all-experts"], capture_output=True, text=True)
            self.assertEqual(every.returncode, 0, every.stdout + every.stderr)
        total = STAGES * 3 * WIDE_EXPERTS
        checked, reported, ids = self.coverage(sampled.stdout)
        self.assertEqual(reported, total)
        self.assertLess(checked, total)          # a sample, not the whole file
        self.assertLessEqual(ids, WIDE_EXPERTS)
        self.assertEqual(self.coverage(every.stdout), (total, total, WIDE_EXPERTS))

    def test_validator_rejects_a_flipped_payload_byte(self):
        corrupt = Path(self.tmp.name) / "corrupt.gguf"
        data = bytearray(self.out.read_bytes())
        data[-1] ^= 0xFF
        corrupt.write_bytes(bytes(data))
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--hf", str(self.hf), "--gguf", str(corrupt),
             "--dspark", "--payload", "--quants-library", str(self.library)],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from source recipe", result.stdout + result.stderr)

    def test_fp8_32x32_blocks_round_trip(self):
        """The converter must use the checkpoint's 32x32 FP8 grid, not V4's 128."""
        plan = {item.name: item for item in self.plan()}
        item = plan["mtp.0.attn_q_a.weight"]
        start = self.data_start(plan)
        with open(self.out, "rb") as fp:
            fp.seek(start + item.offset)
            payload = fp.read(item.nbytes)
        quantizer = NativeQuantizer(str(self.library))
        db = SourceDB(str(self.hf), index_validator=lambda _: None,
                      scale_validator=validate_scales)
        try:
            values = quantizer.to_f32(db, item.source)
        finally:
            db.close()
        decoded = dequantize_q8_0(payload, item.shape[0], np)
        self.assertEqual(decoded.shape, values.shape)
        scale = float(np.abs(values).max())
        self.assertLess(float(np.abs(decoded - values).max()) / scale, 0.02)

    def test_expert_payload_round_trips_within_q4k_error(self):
        plan = {item.name: item for item in self.plan()}
        item = plan["mtp.1.ffn_gate_exps.weight"]
        stride = item.nbytes // item.expert_count
        start = self.data_start(plan)
        quantizer = NativeQuantizer(str(self.library))
        db = SourceDB(str(self.hf), index_validator=lambda _: None,
                      scale_validator=validate_scales)
        try:
            for expert in (0, EXPERTS - 1):
                with open(self.out, "rb") as fp:
                    fp.seek(start + item.offset + expert * stride)
                    payload = fp.read(stride)
                values = quantizer.to_f32(db, item.source.format(expert=expert))
                decoded = dequantize_q4_k(payload, item.shape[0], np)
                self.assertEqual(decoded.shape, values.shape)
                scale = float(np.abs(values).max())
                self.assertLess(float(np.abs(decoded - values).max()) / scale, 0.25)
        finally:
            db.close()

    def test_v4_recipe_is_unchanged(self):
        """Without a V4.1 config the tool must still emit V4's exact recipe."""
        with tempfile.TemporaryDirectory() as scratch:
            hf = build_v4_checkpoint(scratch)
            out = Path(scratch) / "v4-dspark.gguf"
            result = subprocess.run(
                [str(CONVERTER), "--hf", str(hf), "--dspark-support", "--threads", "2",
                 "--out", str(out), "--overwrite"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("dspark_family: deepseek4\n", result.stdout)
            self.assertIn("target_layers=40,41,42", result.stdout)
            self.assertIn("tensor_types: f32=34 f16=7 q8_0=31 q2_K=3 iq2_xxs=6",
                          result.stdout)
            kv, tensors = read_gguf_header(out)
            self.assertEqual(kv, {
                "general.architecture": "deepseek4-dspark",
                "general.name": "DeepSeek V4 Flash DSpark support",
                "general.alignment": DSPARK_ALIGNMENT,
                "dspark.block_size": 5, "dspark.markov_rank": 256,
                "dspark.noise_token_id": 128799,
                "dspark.target_layer_ids": (40, 41, 42),
                "dspark.stage_count": STAGES, "dspark.n_layers": STAGES})
            names = {name: kind for name, kind, _ in tensors}
            self.assertEqual(len(tensors), STAGES * 24 + 9)
            self.assertIn("mtp.2.hc_head_fn.weight", names)
            self.assertIn("mtp.2.markov_head.markov_w1.weight", names)
            # V4 keeps the router and the confidence head in the dense role.
            self.assertEqual(names["mtp.0.ffn_gate_inp.weight"], QTYPE_Q8_0)
            self.assertEqual(names["mtp.2.confidence_head.proj.weight"], QTYPE_Q8_0)
            self.assertEqual(names["mtp.0.ffn_gate_exps.weight"], QTYPE_IQ2_XXS)
            self.assertEqual(names["mtp.0.ffn_down_exps.weight"], QTYPE_Q2_K)

    def data_start(self, plan):
        with open(self.out, "rb") as fp:
            fp.seek(8)
            tensors, records = struct.unpack("<QQ", fp.read(16))
            for _ in range(records):
                read_gguf_string(fp, "key")
                skip_gguf_value(fp, read_u32(fp, "type"))
            for _ in range(tensors):
                read_gguf_string(fp, "name")
                rank = read_u32(fp, "rank")
                fp.read(8 * rank + 12)
            return align(fp.tell(), DSPARK_ALIGNMENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
