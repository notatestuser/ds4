# V4.1 Decode Gainers on Apple Silicon

[README](../README.md) | [Metal](METAL.md) | [Performance](PERFORMANCE.md) | [Serving](SERVER.md)

This branch (`v41-gainers`) makes DeepSeek V4.1 Flash decode faster on pre-M5
Apple GPUs without changing what the model emits. Every landed change is a
scheduling, batching or fusion change that keeps the same floating-point work
in the same order; none of them changes a reduction, and none of them is on by
default on any other device.

Read [Metal](METAL.md) first for build and model selection, [serving](SERVER.md)
for `--batched-session`, and [testing](TESTING.md) for the test targets named
here. For DSpark, read the V4.1 section of [speculative
decoding](SPECULATIVE_DECODING.md#deepseek-v41-flash-dspark) and not the V4
Flash one above it: the V4 download table, its five-token drafts and its 5.6 GiB
support file do not describe V4.1, whose support file is built rather than
downloaded and whose default draft width is 3. The command is under 3.7 stage 3
below.

## What this branch is

It sits on `v41-pre-m5-decode-queue` (`dceba87`), the branch of upstream
`8db1d1d` that carries phases 1.1–1.5 and is open as PR #1067 against
antirez/ds4. On top of that base it adds 12 feature and fix commits, plus four
reconciliation commits and their four review records.

Everything below was measured on one machine and one checkpoint:

| | |
| --- | --- |
| Machine | Mac Studio M3 Ultra, 512 GB |
| Model | DeepSeek V4.1 Flash Q4 GGUF (Q4_K routed experts, Q8 dense) |
| Residency | 294 GiB of main weights resident; Engram tables read from the file, never resident |
| Upstream control | untouched `8db1d1d`, same box, same GGUF, same prompts |
| Measured memory roofline | 715 GB/s over a 5 GiB working set, 87 % of the 819 GB/s nominal; ≈13.5 GB streamed per decode step |

See the [model guide](MODELS.md#deepseek-v41-flash) for the download and the
memory figures.

### The two rules every change obeys

**A single-token decode step is bit-identical to the legacy path.** The gate is
an in-process strict A/B: `speed-bench/metal_decode_schedule_bench` alternates
control and candidate splits on one live engine, unsetting the rollback
environment variable for the control and setting it for the candidate, over a
2048-token prefix, 16 warm-up and 512 measured tokens with
`--include-selection`, and compares every logit row and every argmax selection
between the two. All 529 rows and 528 selections must match bit for bit
(1,041 and 1,040 at the 1024-token setting 3.4 used). This rule is not
cosmetic: a reduction-order change flips one BF16 ulp on roughly 1 element in
256 at every rounding boundary, and V4.1 rounds nearly every matmul output, so
the divergence reaches a greedy token within a few hundred steps. That is
exactly how 2.1 was rejected.

**Every row of a batched decode step is bit-identical to the same row decoded
alone.** The gates are `session_concurrency_bench --verify` at 1, 2, 4 and 8
sessions, the session-batch oracle `tests/test_metal_session_batch` at
`DS4_TEST_LOGIT_TOLERANCE=2e-4`, and the six-prompt greedy id set (63 / 4,564 /
512 × 4 prompt tokens, 128 greedy ids each), which must reproduce the upstream
control exactly.

Every decode and batching gate below tests
`ds4_gpu_device_is_pre_m5_apple_silicon()`, so none of them is on by default on
an M5-class GPU or off Apple silicon. The 3.7 DSpark stages are the exception:
they are gated on `--dspark`, an `--mtp-model` file and a Metal backend
(`ds4_v41_dspark_support_requested()`, ds4.c:73710), not on pre-M5, so they will
run on an M5 Mac — unmeasured.

The batched stages additionally require `tp_world == 1` and keep the per-row
path for prefill rows and for any row that is a streaming, quality, imatrix or
image session, or whose weights are not Q8_0/F16/F32. The single-token fusions
are scoped by graph flags to `ds41_graph_step`'s layer loop, so the prefill
sweep and the batched step keep the dispatches they always had.

## Results

`v41-gainers` against the untouched upstream control and against the PR branch
it sits on. Single-stream figures are `ds4-bench` steady tok/s; batched figures
are `session_concurrency_bench` at ctx 4096, 128 generated tokens.

| Measurement | Upstream `8db1d1d` | PR branch (1.1–1.5) | **`v41-gainers`** |
| --- | ---: | ---: | ---: |
| Prefill, 2048-token chunks | 329–336 t/s | 326–333 t/s | **338–347 t/s** |
| Single stream at 2K / 4K / 6K / 8K | 18.06 / 17.97 / 17.92 / 17.95 | 25.8 / 26.0 / 25.8 / 25.7 | **28.34 / 28.27 / 28.15 / 28.02** |
| Step p50 at 1 / 2 / 4 / 8 sessions | 53.6 / 73.3 / 127.6 / 238.9 ms | — / — / — / 186.7 ms | **35.5 / 51.3 / 81.3 / 146.6 ms** |
| Aggregate at 1 / 2 / 4 / 8 sessions | 18.6 / 27.3 / 31.4 / 33.4 t/s | — / — / — / 42.9 t/s | **28.2 / 38.9 / 49.2 / 54.6 t/s** |
| HTTP, 1 stream / 8 streams aggregate | 17.2–18.1 / 32.1 t/s | — | **28.6 / 54.8 t/s** |

Prefill is unchanged within noise; nothing here touches the prefill sweep.
`--verify` is clean at 1/2/4/8 and the six greedy id sets are identical to the
control.

DSpark is opt-in and off in the rows above. With `--dspark --mtp-model`, on the
same branch:

| Prompt | Serial | `--dspark` k=3 | `--dspark` k=5 | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Code (a diff to continue) | 28.4 t/s | **30.8 t/s** | **34.4 t/s** | 90–92 % |
| Code review (3,028-token prompt, thinking mode) | 27.20 t/s | 26.53 t/s (−2.5 %) | 27.73 t/s (+2.0 %) | 71–73 % |
| Prose (*I Promessi Sposi*) | ≈28 t/s | 21.2 t/s (≈−25 %) | — | 29 % |

The code and prose acceptance rates are from the equivalent run on the
development branch, which this branch reproduces within run-to-run noise; the
review row was measured here, on the reconciled tip. The prose serial figure is
the single-stream walk above, not a separate run.

### What each stage is worth at 8 sessions

Measured on this branch's reconciled tip by exporting one rollback at a time,
8 sessions, ctx 4096, 128 generated tokens:

| Configuration | Step p50 | Aggregate |
| --- | ---: | ---: |
| Default, run 1 | 148.7 ms | 53.8 t/s |
| Default, run 2 | **146.7 ms** | **54.3 t/s** |
| `…_LATE_ENGRAM_JOIN` off | 148.7 ms | 53.3 t/s |
| `…_BATCH_ATTENTION_FLASH` off | 154.6 ms | 51.7 t/s |
| `…_BATCH_ATTENTION_ROWS_A` off | 156.1 ms | 51.2 t/s |
| `…_BATCH_ATTENTION_OUTPUT_ROWS` off | 160.3 ms | 49.9 t/s |
| `…_BATCH_ATTENTION_OUTPUT` off | 170.9 ms | 46.7 t/s |

Against run 1's default: the attention-output pair is worth ≈22 ms a step, the
rows-templated kernels ≈12 ms, the batched row table ≈7 ms, the batched flash
≈6 ms, and the late Engram join sits inside run-to-run noise. Two default runs
are shown because the spread between them (2 ms) is the size of the smallest
effect.

The four rows are nested, not additive. `…_BATCH_ATTENTION_OUTPUT` off also
disables the rows kernels above it — `ds41_batch_attention_output_rows_gate()`
is reached only after the output gate has passed (ds4.c:44379, :44410) — so its
≈22 ms *contains* their ≈12 ms, and 3.6d on its own is 170.9 − 160.3 ≈ 10 ms.
Summing the column over-counts by that 12 ms.

None of this buys a single session anything. At concurrency 1 each of the four
stages measured 35.9–36.3 ms a step with the stage on and with it rolled back,
flat within noise; the single-stream gain over the PR branch is 2.3, 3.5 and
3.4. 3.6 is what you buy by serving 2–8 sessions.

## The changes, in commit order

### 1.1–1.5 — the base branch (PR #1067)

Not part of this branch's own commits; it inherits them. `ds41_graph_step`
commits command buffers without waiting and drains once per token instead of
after every layer (1.1); both Engram tables are read on reader threads straight
into their upload buffers and joined before the first commit (1.2); decode
pipeline lookups are allocation-free (1.3); word-aligned copies of 1 MiB or
less run as a copy kernel on the open encoder instead of a blit (1.4); and the
batched step gets the same treatment (1.5).

Strict A/B on the base branch's own binaries: 26.07 vs 17.94 tok/s single
stream, 529 rows and 528 selections bit-identical, 42.87 tok/s aggregate at 8
sessions. Per stage, in the single-token strict A/B: 1.1 +28.5 %, 1.2 +11.9 %,
1.3 0.0 %, 1.4 −0.1 %; 1.5 is a batched change and is worth +24 % at 8 sessions.
1.3 and 1.4 are kept because they are bit-identical, gated and free, not because
they are fast. 1.2 is the large one after 1.1: Engram rows cost 5.5 ms a token
synchronously and 0.30 ms with the readers.

Rollbacks: `DS4_METAL_DISABLE_PRE_M5_V41_DECODE_QUEUE`,
`…_ENGRAM_PREFETCH`, `…_DECODE_PIPELINE_FAST_LOOKUP`,
`DS4_METAL_DISABLE_PRE_M5_SMALL_COMPUTE_COPY`,
`DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ENGRAM_READERS`.

### 2.3 — BF16 epilogues folded into the producing kernels

**Commit** `dfd14df`.

V4.1 rounds nearly every matmul and elementwise output to BF16, and upstream
does it with a separate `kernel_dsv41_bf16_linear` dispatch after the producer.
This applies the rounding at the producer's own store instead: the Q8_0
single-token matvec, the weighted RMS norm, the attention low projection,
`out_b`, the shared-expert SwiGLU, the routed+shared add, both hyper-connection
weighted sums and HC expand4 each gain a `_bf16` twin with the same K walk and
the same reduction tree. About 480 of the ~660 rounding dispatches per token
disappear.

**Measured** +4.1 % in the strict A/B (27.17 vs 26.10 tok/s), 529 logit rows
bit-identical. **Exactness:** each twin writes the value its producer wrote,
rounded at the same point, so the store is the only thing that moves.
**Rollback** `DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE`. **Test**
`tests/test_deepseek41_metal --fused-bf16` (23 checks; also `make
test-deepseek41-fused-bf16`). Five of those checks run a third pass with the
rollback exported and require the rolled-back result to match the fused one bit
for bit — the only model-free exercise of a documented escape hatch.

### 3.5 — the three remaining single-token elementwise fusions

**Commit** `77c0f97`.

Three fusions inside `ds41_graph_step`'s layer loop, each with its own switch:
the per-layer `ffn_split → pre` copy is dropped for layers 0–38 (layers 1 and
up read the mixer weights straight from `ffn_split`; the last layer still lands
them in `pre` for the head); `kernel_dsv41_rope2` ropes q and kv in one dispatch
through the device function the single RoPE kernel now uses; and
`kernel_dsv41_quantize_store2` is the quantize kernel with a second store of the
same rounded value into the window slot the copy used to fill. All three are
scoped by graph flags to the single-token layer loop, so the prefill sweep and
the batched step keep the legacy pairs.

**Measured**, each as its own strict A/B with the rollback as candidate: pre
copy +1.0 %, RoPE pair +0.4 %, quantize+store +1.1 %; ≈+2.4 % together. Every
run bit-identical over 529 rows and 528 selections. Dispatches per token fell to
2,011, about 120 fewer. **Exactness:** same values, same per-element order; the
RoPE and quantize kernel bodies are pinned by digest against the pre-patch
kernel file. **Rollbacks** `DS4_METAL_DISABLE_PRE_M5_V41_PRE_COPY`,
`…_ROPE_PAIR`, `…_QUANTIZE_STORE` (fusion-gate bits 1, 2, 4). **Tests**
`--rope-pair`, `--quantize-store`, `--rope-digest`, `--rope-freqs`,
`--fusion-gates`.

### 3.4 — concurrent-dispatch section for the MoE layer

**Commit** `1624e53`.

The shared expert and the routed experts share one concurrent compute encoder in
three levels — {shared gate, shared up, routed pair-SwiGLU}, {shared SwiGLU +
BF16}, {shared down, routed sum6} — separated by the two global execution
barriers a concurrent encoder's `memoryBarrierWithResources` gives, instead of
six dispatches in six serial encoders. The host factors the single-token Q8_0
matvec encode and the flat SwiGLU encode into helpers that both the serial path
and the section call, so the kernels, arguments, buffers and per-element order
are identical and only the schedule changes. Single-token decode only; the
prefill per-row fallback is excluded explicitly.

**Measured** +1.06 % over 1024 tokens (28.06 vs 27.77 tok/s), 1,041 rows and
1,040 selections bit-identical. The design pass expected 2–3 ms a token and got
≈0.4: the six dispatches already shared one serial encoder, and overlapping the
shared expert with the routed pair gains little when both stream from the same
memory system.

The encoder timeline cannot see this section (see
[Reproducing the measurements](#reproducing-the-measurements)), so it counts its
own arms: `DS4_METAL_V41_PARALLEL_FFN_REPORT=1` prints
`planned/armed_id/armed_slots/declined/refused/unjoined`, and a real decode
reports `planned=320 armed_id=320 unjoined=0` over 8 tokens × 40 layers.
**Rollback** `DS4_METAL_DISABLE_PRE_M5_V41_PARALLEL_FFN`, or implicitly whenever
2.3's `DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE` is set: the section arms only
on the fused-epilogue path (ds4.c:42246), so bisecting with the 2.3 rollback
switches off both stages and charges 3.4's ≈1 % to 2.3.
`DS4_METAL_V41_PARALLEL_FFN_SCOPE_BARRIER` swaps the barrier form for bisecting.
**Tests** `--v41-parallel-ffn` and `--v41-parallel-ffn-slots` (both run by `make
test-deepseek41-metal`); `DS4_METAL_TEST_Q4_SELECTED_MIN_TENSOR_MB` lets the
test reach the selected-slots arm point, which the resident decode cannot.

### 3.6d — one attention-output projection pair for all N rows

**Commit** `0c597c4`.

All N rows of a batched decode step take one `out_a` dispatch and one `out_b`
dispatch a layer instead of two a row: 640 dispatches a token become 80 at
N = 8. `out_a` goes through the `out_a` half of the prefill entry point, one
(group, row) threadgroup per row with the same k-split and reduction tree;
`out_b` goes through `ds41_matmul_batch`'s row-exact Q8 kernel, the single-row
geometry on a grid of `(out_dim/nr0, n_rows, 1)`.

**Measured** +9.6 % at 8 sessions (185.9 → 168.0 ms, 43.0 → 47.6 tok/s aggregate;
ms per added row 21.4 → 18.9). **Exactness:** every row's floats are the per-row
loop's at every width; the two BF16 roundings move to the rounding kernels that
already followed them. Projecting `out_b` with `mul_mv_ext` would read each
weight chunk once for `r1ptg` rows, but it reorders the reduction and picks its
lane map by the row count — on the real model that flipped tokens within 33
decode steps and took the 2e-4 oracle from 2.9e-6 to 2.94, so it stays opt-in
behind `DS4_METAL_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_MV_EXT` for pricing only.
**Rollback** `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT` (bit 8).
**Test** `--attention-output-decode`, which is bit-identical at widths 1–8 and
fails under the `mv_ext` environment.

### 3.6d2 — rows-templated attention-output kernels, and the late Engram join

**Commit** `ac67cd1`.

3.6d collapsed the dispatches but kept a threadgroup per row, so each layer
still streamed `out_a`'s 35.6 MB and `out_b`'s 44.6 MB N times. These kernels —
`kernel_dsv4_attn_out_low_q8_0_f32{,_bf16}_rows<R>` and
`kernel_mul_mv_q8_0_f32{,_bf16}_nrows<R>`, R in 2/4/8 — hold all N activation
rows in one threadgroup and dot each Q8 block against every row, taking the
traffic without `mul_mv_ext`'s reduction change. Independently, the Engram
reader join moves off the layer-1 encode point to the join the first flush
already performs, which is what its committed-command invariant actually
requires.

**Measured** +4.5 % at 8 sessions (167.6 → 160.1 ms, 47.7 → 50.0 tok/s; ms per
added row 18.8 → 17.7); the late join is +0.4 %, inside noise.
**Exactness:** per (activation row, output row) the K walk, the k-split, the
eight-term dot, the block order and the `simd_sum` reduction tree are the
single-row kernel's, and the BF16 rounding stays fused at the store. The join
move changes when the host waits, not what is computed. **Rollbacks**
`DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_ROWS` (bit 16) and
`DS4_METAL_DISABLE_PRE_M5_V41_LATE_ENGRAM_JOIN` (bit 32). **Test**
`--attn-out-rows` (28 memcmp cases against R single-row dispatches, both
epilogues, order-sensitive data, also under `MTL_SHADER_VALIDATION=1`).

When attributing a number to this stage, check the run log for the presence of
`ds4: V4.1 batched attention output enabled` **and** the absence of
`ds4: V4.1 batched attention output rows kernels unavailable`. The announce is
printed from the intent flag before anything is encoded, so presence alone
admits a run in which every layer fell back to 3.6d's bit-identical pair.

### 3.6c — one FlashAttention dispatch with ne03 = N

**Commit** `662f596`.

The decode attention of a batched step runs as one
`kernel_flash_attn_ext_vec_f16_dk512_dv512` dispatch with `ne03 = N` plus one
reduce over N×64 rows, and each row's KV gather, two f32→f16 staging copies and
pad kernel collapse into one `kernel_dsv41_flash_stage_rows` dispatch: 48
dispatches a layer at N = 8 become 10, and the gathered rows are converted
straight out of `compressed[owner]` instead of through a 1 MB f32 round trip.

**Measured** +3.7 % at 8 sessions (160.5 → 154.5 ms, 49.8 → 51.8 tok/s; ms per
added row 17.8 → 16.9). **Exactness:** the vector kernel already indexes q/K/V
and the mask by `iq3` and the reduce by `rid`, and its block-to-workgroup map
depends only on `iwg`/`NSG`/`NWG`, never on `ne11`, so padding the batch to a
common key count with zero K/V and a `-MAXHALF` mask only adds blocks the kernel
skips. The encoder refuses the batch, with nothing encoded, unless
`nsg(n_keys_max) == 1` and `nwg == 32`. **Rollback**
`DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_FLASH` (bit 64). **Test**
`--flash-rows` (12 memcmp cases at N = 1/2/4/8 over gathered, raw-only and mixed
batches, including key counts that are not multiples of 32).

Same caution as 3.6d2: require the presence of
`ds4: V4.1 batched attention flash enabled` and the absence of
`ds4: V4.1 batched attention flash fell back at layer`. A whole run can print
`enabled` and batch nothing — the refusals behind the per-layer fallback
(quality mode, a `DS4_METAL_FLASH_NWG` other than 32, an `nsg` above 1, a
pipeline that failed to compile, a scratch buffer that could not grow) are
run-level, not properties of one layer.

### 3.6a — the row table, and one RoPE / quantize / window / publish for all N rows

**Commit** `1856632`.

A batched decode step runs the front of attention — q and kv RoPE, the FP8
quantize, the sliding-window store and the compressor publish — once for all N
rows instead of once per row, and the per-row front loop starts at the indexer.
Session-private caches reach the kernels as GPU addresses in a
64-byte-per-row table uploaded with `setBytes`, the bindless mechanism
`kernel_touch_u8_stride_table` already uses, with `useResources` per dispatch
and the destination bounds re-checked on the host. About 245 dispatches per row
per token become about 16 per step.

**Measured** +4.6 % at 8 sessions (154.5 → 147.4 ms, 51.8 → 54.3 tok/s; ms per
added row 16.9 → 15.9). **Exactness:** the rows kernels are the per-row ones with
the position and destination taken from the table, the norms are one threadgroup
a row of the same pipeline, and `ds4_gpu_dsv41_projection_rows` is the single-row
matvec branch. With this stage in, every part of the batched step is
width-invariant, so the width probe and the tolerance-0 oracle become gates
rather than recordings. **Rollback**
`DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_ROWS_A` (bit 128; the default mask
on a pre-M5 device is therefore 255). **Tests** `--rows-a` at N = 1/2/4/8,
`--quantize-digest` (the hoisted quantizer body against the pre-patch kernel
file), `--index-projection` at decode widths 1–8.

### 3.7 stage 2 — the DSpark support GGUF, converter and validator

**Commit** `ded470e`.

`--v41-dspark` teaches the DSpark support converter V4.1: the `text_config`
layout, 32×32 FP8 blocks derived from the scale grid, `markov_head` embed/head
names, an F32 router and confidence head, a q4/q2 preset, and
`dspark.n_expert` 128 / `n_expert_used` 3. V4 output stays byte-identical.

```sh
./gguf-tools/deepseek4-quantize --hf /path/to/DeepSeek-V4.1-Flash \
  --dspark-support --v41-dspark --dspark-quant q4 --n-experts 128 \
  --threads 12 --out DeepSeek-V4.1-Flash-DSpark-support-q4k.gguf --overwrite

python3 gguf-tools/deepseek41_validate_gguf.py \
  --hf /path/to/DeepSeek-V4.1-Flash \
  --gguf DeepSeek-V4.1-Flash-DSpark-support-q4k.gguf \
  --dspark --quant q4 --payload --all-experts
```

The support file is 8,328,374,496 bytes (7.756 GiB), 78 tensors (36 F32, 6 F16,
27 Q8_0, 9 Q4_K), converted in 57 s. `--all-experts` byte-checks all 1,152
expert slabs instead of a stride sample; it takes about 10 minutes and is the
right trade for a file written once. **Test** `make
test-deepseek41-dspark-conversion` (13 tests). Note that at that battery's
4-expert fixture the default stride sample already spans every expert, so the
tests cannot distinguish `--all-experts` from the default at the validator's
call site; the validator prints the coverage it achieved
(`expert payload coverage X/Y slabs over Z distinct expert ids`), and on a
128-expert file `--all-experts` must print Y/Y.

### 3.7 stage 3 — load, validate and report the support file

**Commit** `88b9c64`.

`--dspark --mtp-model FILE` binds and validates the V4.1 draft block. Its
routing width comes from the support file's own `dspark.n_expert*` (128 top-3),
not the target's 384/6 globals, and `mtp.*.hc_head_*` is V4-only (V4.1 mixes the
head with the last block's `ffn_pre`). A V4.1 file that fails binding or
validation is refused rather than counted. At this stage the draft graph does
not exist yet, so V4.1 binds the file, prints the summary and decodes serially.
From this commit on, that is also how a V4.1 run is started — there is no
download for the V4.1 support file, only the converter above:

```sh
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --dspark --mtp-model DeepSeek-V4.1-Flash-DSpark-support-q4k.gguf
```

`ds4-agent` and non-batched `ds4-server` take the same two flags.

**Measured:** the strict bench with `--dspark` off is bit-identical (27.42 vs
27.44 tok/s, 529 rows). **Rollback** `DS4_DISABLE_V41_DSPARK_LOAD=1`. **Test**
`make test-deepseek41-gguf` (five synthetic GGUF-header fixtures covering the
78-tensor geometry, the no-metadata fallback, the wrong family, and V4 with and
without `hc_head_*`).

### 3.7 stage 4 — the draft graph, in shadow mode

**Commit** `8d084bf`.

`ds41_graph_step` captures the hyper-connection mean of the residual at the
attention input of the target layers (one small dispatch each, only with
`--dspark`), and `ds41_graph_dspark_step` runs the three draft blocks on their
own scratch: private decode-seeded 128-row windows, the support file's own 128/3
MoE, `ds4_gpu_dsv41_rope` directly, then `mtp.2.norm`, the main head, the Markov
chain and the confidence head. Single-rank only; network TP keeps stage 3's
behaviour.

**Measured:** the draft graph costs 12.934 ms a call (median over 32 calls).
**Exactness** by being opt-in twice: without `--dspark` there is no draft state
at all, and with it the draft runs only under `DS4_V41_DSPARK_SHADOW=1`, which
throws the proposal away — the emitted ids and the strict bench are unchanged.
The draft itself is gated against a NumPy reference (`dspark_ref.py`) over the
same dequantised weights, per sub-block rather than end to end: both sides differ
only in f32 reduction order, and ten BF16 boundaries a block turn a 1e-6
difference into 2–4e-3 on the block hidden, so checking each sub-block from
ds4's own dumped input puts the gate back at 1e-5…1e-3.
**Switches** `DS4_V41_DSPARK_SHADOW`, `…_DUMP_DIR`, `…_DUMP_CALLS` (default 4),
`…_TIMING`; **rollback** `DS4_DISABLE_V41_DSPARK_DRAFT`. **Test**
`--dspark-draft` — the capture self-test, the HC split/pre/post decomposition,
the dump layout, the five stage-4 switch names, and, because it chains
`check_dspark_verify()`, stage 5's rewind bookkeeping as well.

The capture rides on `--dspark` alone, not on shadow mode, and
`DS4_V41_DSPARK_TIMING` times `ds41_graph_dspark_step` only. The control for the
shadow cost is a run with no `--dspark`, or `DS4_DISABLE_V41_DSPARK_DRAFT=1`.

### 3.7 stage 5 — the verify loop, prefill seeding and the rewind

**Commit** `79d4370`; the non-Apple build fix that follows it is `19e4b82`.

Up to k proposals run as rows of one `ds41_graph_step_batch` call at consecutive
positions of the same session; the target's argmax at row i−1 accepts proposal
i, and everything past the first rejection is rewound. The argmax is taken on
logits from a **per-row** head, because the row-exact batched matmul refuses the
vocabulary head and a batched one would reorder the reduction. The rewind covers
the positions, the raw-KV window ring, the ratio-2 compressor pair carry and the
Engram tail. The three draft windows are seeded from the target's last prompt
positions and from every committed speculative position. With a temperature,
V4.1 now takes V4's opportunistic semantics: the first token of a cycle is
sampled, the verified continuation is greedy.

**Exactness:** every emitted token is either the caller's own or one the target's
argmax reproduced at the position it occupies, so the stream is serial greedy
decode however wrong the draft is. Gates: six-prompt ids with `--dspark` off
identical to the control; 128-token ids with `--dspark` versus serial identical
on four prompts and 512-token ids identical on two; with the batched head
disabled, batched-versus-sequential logits bit-identical at 1/2/4/8.
`--dspark` off is byte-identical.

**Economics.** A cycle costs about 36 ms (the target step) + 13 ms (the draft
call) + 16 ms per verify row, because each verify row streams its own 6 experts
across 40 layers, roughly 4.8 GB. It yields about 1 + 0.9 + 0.8 + … tokens. So
even 90 % acceptance buys only +22 % at k = 5. When stage 5 was designed the
cycle model put break-even near 60 % acceptance; measured, the break-even region
is 71–73 %, where the review prompt came out −2.5 % at k = 3 and +2.0 % at
k = 5. Measured acceptance: 90–92 % on code output, 71–73 % on a code-review
request whose output is reasoning prose with quoted identifiers, 29 % on prose.
On prose the confidence rule declined to draft in 177 of 234 cycles, yet every
cycle still paid the 12.9 ms draft call — that is the −25 %.

This is below the project's 1.3× gate, so DSpark stays opt-in; the command is
the one under stage 3 above, with `DS4_V41_DSPARK_MAX_DRAFTS` for k.
[Speculative decoding](SPECULATIVE_DECODING.md#deepseek-v41-flash-dspark)
covers the flags and the sampling semantics; its V4 Flash section above that
anchor — the download table, the five-token drafts, the 5.6 GiB support file —
does not describe V4.1.

**Memory:** the support file adds 7.8 GiB of accelerator views, and each
`--dspark` session adds about 7.3 MiB of device tensors (+0.2 MiB while dumping)
and about 3.1 MiB of host arrays. `ds41_dspark_configure` runs after the memory
admission call, so none of this is counted in `ds41_session_bytes`; a run without
`--dspark` allocates none of it.

**Switches** `DS4_V41_DSPARK_MAX_DRAFTS` (default 3),
`DS4_V41_DSPARK_VERIFY_BATCH_HEAD` (diagnostic; forfeits the identity gate);
**rollbacks** `DS4_DISABLE_V41_DSPARK_VERIFY` (back to stage 4's shadow mode),
`DS4_DISABLE_V41_DSPARK_VERIFY_BATCH`, `DS4_DISABLE_V41_DSPARK_PREFILL_SEED`.
**Tests** `--dspark-verify` (the rewind bookkeeping and the window-ring cycle)
and `--dspark-verify-env`, which makes one assertion against whatever
environment the caller left: that `ds4_v41_dspark_verify_gates()` reports the
five switches exactly as they were exported. That is how a harness proves an
exported name reaches `ds4.c`. Sweeping it over a set of environments is the
stage runner's job, and that runner is not in this tree.

`DS4_DSPARK_STATS=1` prints acceptance, tokens per cycle and the accepted-length
histogram at the end of a run.

**Known coverage gap.** Nothing model-free reaches `ds41_spec_window_copy`'s
`[first_row, rows)` loop or the two literals its call sites pass: those
functions are `static` in `ds4.c` and unreachable from `tests/`. A save or
restore one row too short or too long passes the entire model-free battery. Both
call sites are pinned by their own guards today (verify rejects fewer than two
rows, so its save range is always the full `[0, rows)`; the commit's guard is
`committed < verify_rows`), but any future change to those ranges must be proved
by a real `--dspark` run against serial decode over more than 128 positions,
where the ring wraps.

### The non-Apple build fix

**Commit** `19e4b82`. Stage 5 left `ds41_dspark_max_drafts()` inside the
`DS4_HAS_DEEPSEEK41_GPU` block while calling it from the unguarded `--dspark`
banner, so `cc -fsyntax-only -DDS4_NO_GPU -std=c99 ds4.c` failed on an undeclared
function. It and the uint parser it calls move out of the block unchanged — both
are pure `getenv()` — so the GPU build keeps the same bodies, and no default,
switch or banner text changes.

## Measured and not landed

These were implemented and measured on the development branch and are **not** on
this branch. They are recorded so nobody retries them blind.

| Item | Result | Why not |
| --- | --- | --- |
| 3.1 flush interval (`DS4_METAL_V41_DECODE_FLUSH_LAYERS`) | k=0 −1.9 %, k=2 0.0 %, k=8 −0.2 % | No interval beats the default of 4 by ≥1 %. k=0 loses because the GPU idles while the whole token is encoded. |
| 3.3 r4 kernels (four rows per simdgroup) | sum6 +0.5 %, pair +0.0 %, Q8 matvec −0.8 % | Bit-identical but not faster: doubling rows per lane adds no memory-level parallelism these kernels can use. Left opt-in on the development branch only. |
| 3.6b batched indexer | +0.82 % at 8 sessions (147.7 → 146.5 ms) | Below the 1 % landing rule, and half the bucket cannot be batched at all — see below. |
| 2.1 Q8 rows matvec | Rejected | No faster in situ (the microbenchmark gain was a system-level-cache artefact), and it flipped a greedy token at step 264. This is the measurement that set the bit-exactness rule. |
| 2.2 Q4_K threadgroup variants (nsg 1/4/8) | ±0 % | Threadgroup size is not what limits `pair_swiglu` (528 GB/s) or `sum6` (397 GB/s). |
| 2.6 Engram table in RAM | +0.5 % for 94 GiB | With the reads already off the critical path by 1.2, residency recovers nothing. Excluded from this branch on memory grounds. |

3.6b is the interesting one. The indexer projection, RoPE, quantize and score do
batch exactly, but the top-k has to stay per row: the bitonic network's width is
derived from the row's own `n_comp` and is also the threadgroup size of the
dispatch, so one dispatch cannot give rows at different positions their own
networks without changing tie order at the k-th boundary. Half the bucket was
already priced at 0.26 ms a row, so +0.82 % is the expected size, not a
disappointment.

## Environment switches

Every switch this branch adds, plus the base-branch switches it relies on. The
Effect column says what happens when the variable **is** set. The `DISABLE`
names are presence-tested, so any value at all turns the feature off — as is
every other switch below except `DS4_DSPARK_STATS`, which is value-tested and
therefore the one place where `=0` means off. Most are read at step time or
snapshotted per command batch, which is what lets an A/B harness flip one on a
live engine; none of them needs to be set for normal use.

### Base branch (1.1–1.5)

| Name | Default | Effect |
| --- | --- | --- |
| `DS4_METAL_DISABLE_PRE_M5_V41_DECODE_QUEUE` | unset (queue on) | Back to the legacy drain-after-every-layer schedule |
| `DS4_METAL_V41_DECODE_QUEUE_ALL_DEVICES` | unset | Force the queued schedule on M5-class devices, for measurement |
| `DS4_METAL_V41_DECODE_FLUSH_LAYERS` | 4 | Layers per non-blocking commit; 0 keeps the token in one buffer |
| `DS4_METAL_DISABLE_PRE_M5_V41_ENGRAM_PREFETCH` | unset (readers on) | Synchronous Engram reads in single-token decode |
| `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ENGRAM_READERS` | unset (readers on) | Synchronous Engram reads in the batched step |
| `DS4_METAL_DISABLE_PRE_M5_V41_DECODE_PIPELINE_FAST_LOOKUP` | unset (fast on) | Allocating pipeline lookups |
| `DS4_METAL_DISABLE_PRE_M5_SMALL_COMPUTE_COPY` | unset (compute copy on) | Small aligned copies go back to a blit pass |

### 2.3 BF16 epilogues

| Name | Default | Effect |
| --- | --- | --- |
| `DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE` | unset (fused) | Separate `kernel_dsv41_bf16_linear` dispatch after every producer. Also disables 3.4's concurrent MoE section, which arms only on the fused-epilogue path (ds4.c:42246) |

### 3.4 concurrent MoE section

| Name | Default | Effect |
| --- | --- | --- |
| `DS4_METAL_DISABLE_PRE_M5_V41_PARALLEL_FFN` | unset (section on) | Six dispatches in six serial encoders |
| `DS4_METAL_V41_PARALLEL_FFN_REPORT` | unset | Print planned/armed/declined/refused/unjoined counts |
| `DS4_METAL_V41_PARALLEL_FFN_SCOPE_BARRIER` | unset | Swap the barrier form, for bisecting |
| `DS4_METAL_TEST_Q4_SELECTED_MIN_TENSOR_MB` | 2048 | Test-only: lowering it lets the model-free test reach the selected-slots route |

### 3.5 / 3.6 — the eight fusion-gate bits

`tests/test_deepseek41_metal --fusion-gates` (built by `make
tests/test_deepseek41_metal`) prints the live mask and asserts that each name
clears exactly its own bit of it and no other. (That is a statement about the
mask; bit 8 additionally decides whether bit 16's kernels ever arm, as the table
below says.) On a pre-M5 Apple GPU with nothing exported:

```
V4.1 decode fusion gates: pre_m5=1 live=0xff, each of the 8 rollback envs disables exactly its own fusion PASS
fusion_gates=255
```

255 is all eight on. Off pre-M5 Apple silicon the mask is 0. This is the only
model-free proof that the name an A/B harness exports is the name `ds4.c` reads:
a misspelt name would leave the schedule bench comparing the fused path with
itself and reporting it exact.

| Bit | Name | Feature |
| ---: | --- | --- |
| 1 | `DS4_METAL_DISABLE_PRE_M5_V41_PRE_COPY` | 3.5 `ffn_split → pre` copy dropped |
| 2 | `DS4_METAL_DISABLE_PRE_M5_V41_ROPE_PAIR` | 3.5 q+kv RoPE in one dispatch |
| 4 | `DS4_METAL_DISABLE_PRE_M5_V41_QUANTIZE_STORE` | 3.5 quantize + window store in one dispatch |
| 8 | `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT` | 3.6d one out_a/out_b pair per step. Setting it also disables bit 16, whose kernels arm only inside this gate |
| 16 | `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_ROWS` | 3.6d2 rows-templated kernels |
| 32 | `DS4_METAL_DISABLE_PRE_M5_V41_LATE_ENGRAM_JOIN` | 3.6d2 join at the first commit |
| 64 | `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_FLASH` | 3.6c one flash dispatch with ne03 = N |
| 128 | `DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_ROWS_A` | 3.6a row table, batched RoPE/quantize/window/publish |

Two more switches belong to this group without being gate bits: one is an
opt-in rather than a rollback, the other a base-branch knob 3.6c reads.

| Name | Default | Effect |
| --- | --- | --- |
| `DS4_METAL_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_MV_EXT` | unset | Project out_b through `mul_mv_ext`. **Not bit-identical** — informational pricing only; `--attention-output-decode` fails under it by design |
| `DS4_METAL_FLASH_NWG` | unset (adaptive, which resolves to 32) | Force the flash split-K work-group count. Any value but 32 makes 3.6c refuse the batch, for the whole run, and every layer takes the per-row path |

### 3.7 DSpark

| Name | Default | Effect |
| --- | --- | --- |
| `DS4_DISABLE_V41_DSPARK_LOAD` | unset | Refuse to bind the support file (stage 3 rollback) |
| `DS4_DISABLE_V41_DSPARK_DRAFT` | unset | No draft graph (stage 4 rollback) |
| `DS4_DISABLE_V41_DSPARK_VERIFY` | unset | Back to stage 4's shadow mode (stage 5 rollback) |
| `DS4_DISABLE_V41_DSPARK_VERIFY_BATCH` | unset | Verify the proposals one row at a time |
| `DS4_DISABLE_V41_DSPARK_PREFILL_SEED` | unset | Do not seed the draft windows from the prompt |
| `DS4_V41_DSPARK_MAX_DRAFTS` | 3 | Proposals verified per cycle (the k of the sweep above) |
| `DS4_V41_DSPARK_SHADOW` | unset | Run the draft and discard the proposal |
| `DS4_V41_DSPARK_DUMP_DIR` | unset | Write per-block sub-block dumps for the NumPy reference |
| `DS4_V41_DSPARK_DUMP_CALLS` | 4 | How many draft calls to dump |
| `DS4_V41_DSPARK_TIMING` | unset | Time `ds41_graph_dspark_step` (the capture is not covered) |
| `DS4_V41_DSPARK_VERIFY_BATCH_HEAD` | unset | Project the verify head batched. **Forfeits the identity gate**; diagnostic only |
| `DS4_DSPARK_STATS` | unset | Print acceptance, tokens per cycle and the length histogram. Set it to any value other than `0`; unlike every other switch here this one is value-tested (ds4.c:75998), so `DS4_DSPARK_STATS=0` prints nothing |

## Reproducing the measurements

Plain `make` builds none of the binaries below; build them first:

```sh
make ds4-bench metal-decode-schedule-bench session-concurrency-bench \
  tests/test_deepseek41_metal
```

Strict single-token A/B, one fusion at a time. `--candidate-env NAME` unsets
`NAME` for the control run and sets `NAME=1` for the candidate, in one process,
so the control is the fused default and the candidate is the rollback:

```sh
./speed-bench/metal_decode_schedule_bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 2048 --warmup 16 --tokens 512 --include-selection \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE
```

Batched sweep, timed and then verified (`--verify` zeroes the timing columns, so
it takes two runs):

```sh
./speed-bench/session_concurrency_bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --concurrency 1,2,4,8 --ctx 4096 --gen 128 --csv batched.csv
./speed-bench/session_concurrency_bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --concurrency 1,2,4,8 --ctx 4096 --gen 128 --verify --csv verify.csv
```

Single-stream walk:

```sh
./ds4-bench --metal -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-incr 2048 --ctx-alloc 9000 \
  --gen-tokens 256 --warm-weights --csv single.csv
```

Diagnostics: `DS4_METAL_CB_TIMES=1` reports command buffers per token and
encode/commit/GPU spans; `DS4_METAL_ENCODER_TIMELINE=timeline.txt` gives every
dispatch its own pass and appends the per-kernel durations to that file. Its
value is a **path, not a flag** — `DS4_METAL_ENCODER_TIMELINE=1` writes a file
named `1` in the working directory, and nothing at all appears on the terminal
but the `ds4: encoder timeline -> …` line. Each dispatch is one record,
`E <seq> <idx> <start_ns> <end_ns> <dur_us> <gap_us> <caller_unslid> <n_dispatch> <tg> <tpt> <kernel>`,
under a `B` line per batch (ds4_metal.m:207, :235, :273). Both add
synchronisation — their numbers are anatomy, not throughput. See
[performance](PERFORMANCE.md) for the general comparison rules.

### Caveats learned the hard way

- **Run with the desktop idle.** WindowServer and GPU compositing share the GPU
  and cost about 5 % at 8 sessions. One re-measurement here ran with the
  compositor at 52 % CPU and a 3D application rendering; it read ≈5 % slow and
  was discarded. Check `uptime` and `top` before trusting a run.
- **The encoder timeline runs out of counter buffers after about 18k records.**
  A 2048-token prefill exhausts them (one batch alone held 18,432), so the
  decode that follows records nothing and an empty trace looks like a feature
  that never ran. That is what made 3.4 look inert for a day. Gate on in-code
  counters — `DS4_METAL_V41_PARALLEL_FFN_REPORT`, the announce lines — not on
  the timeline.
- **The tolerance-0 session-batch oracle fails on upstream too.** A binary built
  from the untouched upstream commit reports 101,095 of 129,280 logits differing
  by up to 2.86e-6 at its prefill frontier: that is upstream's own
  batched-versus-serial prefill numerics, not a regression. Run the oracle at its
  documented tolerance, `DS4_TEST_LOGIT_TOLERANCE=2e-4`, and treat tolerance 0
  only as a comparison against the base tree's own level.
- **Rebuild test binaries after any change to `metal/*.metal`.** ds4 compiles
  those sources at runtime against a preamble baked into the binary, so a binary
  older than a preamble change fails shader compilation from the new tree.
- **The clock is not the problem.** `sudo powermetrics --samplers gpu_power
  -i 1000` during a 4096-token single-stream decode: 1380 MHz (the top P-state),
  100 % active residency, 60.4 W median (63.3 W max) over 145 samples in the
  decode window. That is what a bandwidth-bound workload with lightly loaded
  ALUs looks like; the 715 GB/s roofline stands.

## Limitations

- **Tensor parallel and multi-node paths are untouched.** The queued schedule,
  the concurrent MoE section and every batched stage require `tp_world == 1`,
  and the BF16 fold of `out_b` is disabled whenever a TP partial sum follows.
  Two-Mac TP and pipeline routes take the code they always took; see
  [distributed inference](DISTRIBUTED.md).
- **M5-class GPUs are unmeasured.** They take a different code path, and every
  decode and batching gate tests `ds4_gpu_device_is_pre_m5_apple_silicon()`.
  `DS4_METAL_V41_DECODE_QUEUE_ALL_DEVICES=1` force-enables the queued schedule on
  them for measurement; nothing in it is device-specific, but nobody has run it.
  The 3.7 DSpark stages are the exception: they ask for `--dspark` and a Metal
  backend, not for a pre-M5 device, so they do run on an M5 Mac — also
  unmeasured.
- **DSpark is below the 1.3× gate** and stays opt-in. It pays on output the
  drafter predicts well — code, at 90 %+ acceptance, is +22 % at k = 5. It is
  roughly neutral in the low 70s (−2.5 % at k = 3, +2.0 % at k = 5 on the review
  prompt) and loses about 25 % on prose. It is also not combined with native
  session batching: with `--batched-session`, the server uses ordinary target
  decoding and logs that it has done so. See
  [serving](SERVER.md#multiple-sessions).
- **Non-Apple builds compile but are untested with these changes.**
  `cc -fsyntax-only -DDS4_NO_GPU` is clean after the stage-5 fix, and the only
  symbol compiled outside `DS4_HAS_DEEPSEEK41_GPU` is a `getenv()` wrapper — but
  no CUDA or ROCm run has been made against this branch.
- **Single-session decode is still far from the roofline.** 28 tok/s × 13.5 GB
  is 378 GB/s, about 53 % of the measured 715 GB/s. The remaining gap is the
  below-roofline bandwidth of `sum6` (397 GB/s), the shared gate/up pair and the
  small attention projections, and closing it needs reduction-order changes that
  the bit-exactness rule excludes.
