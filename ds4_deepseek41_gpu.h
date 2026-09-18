#ifndef DS4_DEEPSEEK41_GPU_H
#define DS4_DEEPSEEK41_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* V4.1 activation/cache formats. Buffers are float-addressable but the
 * rounded values follow the released BF16/FP8/FP4 inference graph. */
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                          ds4_v41_activation_format format);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* CUDA scalar Q8 shared expert. 1: queued; 0: unsupported, no work queued;
 * -1: failure. After 1, keep the input/output tensors alive and unchanged
 * until join, which orders the result before subsequent main-stream work.
 * Other work may run between start and join using separate tensors. */
int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp);
int ds4_gpu_dsv41_shared_join(void);
#endif
/* Full-head prefill, with BF16 rounding between the two Q8 projections. */
int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);
/* The out_a half of the call above for N rows at once, with the same BF16 rounding of `low`
 * and bit-identical per row to the single-row projection; the caller supplies out_b.
 * Metal-only (ds4_metal.m): ds4.c calls it from ds41_batch_attention_output_project(), which is
 * #ifdef __APPLE__, as ds4_gpu_dsv41_quantize_store()/ds4_gpu_dsv41_rope_pair() are. */
int ds4_gpu_dsv41_attention_output_low_batch(
        ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);
/* Packed local 32-head input and BF16 low projection; output is an unrounded
 * rank partial. The graph sums ranks before rounding the attention block. */
int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank);
/* Adjacent-pair, unit-magnitude RoPE with the released V4.1 frequencies. */
int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                      uint32_t rows, uint32_t start, bool compressed, bool inverse);
/* Compressed pairs advance two absolute token positions per stored row. */
int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                             uint32_t rows, uint32_t start, uint32_t stride,
                             bool compressed, bool inverse);
#ifdef __APPLE__
/* PRE_M5 V4.1 decode fusions; each is bit-identical to the pair it replaces.
 * rope_pair: the single-row q and kv RoPE in one dispatch (same frequencies, same position).
 * quantize_store: quantize a row and also store the rounded values at dst + dst_offset,
 * replacing the sliding-window copy that followed the quantize. */
int ds4_gpu_dsv41_rope_pair(ds4_gpu_tensor *x0, uint32_t heads0,
                            ds4_gpu_tensor *x1, uint32_t heads1,
                            uint32_t width, uint32_t start,
                            bool compressed, bool inverse);
int ds4_gpu_dsv41_quantize_store(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                 ds4_v41_activation_format format,
                                 ds4_gpu_tensor *dst, uint64_t dst_offset);
/* The 32 released RoPE frequencies for the plain (false) and compressed (true) layers, built
 * once under precise float control. Exported so the test can prove the host-side table is
 * unchanged; every RoPE dispatch's theta is built from it. */
const float *ds4_gpu_dsv41_rope_frequencies(bool compressed);
/* Test oracle for the rollback switches, which are read by file-static helpers in ds4.c:
 * bit 0 pre copy, bit 1 q+kv RoPE, bit 2 quantize + window store, bit 3 batched attention output
 * (3.6d), bit 4 rows-templated attention output and bit 5 late Engram join (3.6d2), bit 6 batched
 * attention flash (3.6c), bit 7 batched attention rows stage (a) (3.6a); each bit is set when that
 * fusion is live for this process.
 * tests/test_deepseek41_metal --fusion-gates indexes its name table by bit position, so a bit
 * added here must be appended there. */
int ds4_v41_decode_fusion_gates(void);

/* PRE_M5 3.6a (2026-09-17): the V4.1 batched-decode row table and the four rows kernels that read
 * it. A batched decode step runs N <= DS4_GPU_V41_MAX_ROWS rows of DIFFERENT sessions; their
 * activations are row-contiguous workspace views, but their caches (window[il],
 * compressed[owner], index_cache[owner], previous_kv/previous_score[owner]) are session-private
 * buffers that no stride can reach. The table carries their GPU addresses to the kernels, the way
 * kernel_touch_u8_stride_table's address table already does, and every buffer behind an address is
 * passed to useResource on the encoder that dereferences it. */
enum { DS4_GPU_V41_MAX_ROWS = 8 };   /* mirrors DS4_TP_BATCH_MAX_ROWS (ds4_tp.h) */
typedef struct {
    ds4_gpu_tensor *window;          /* window[il],           f32[slots][512]      */
    ds4_gpu_tensor *compressed;      /* compressed[owner],    f32[cap][512]        */
    ds4_gpu_tensor *index_cache;     /* index_cache[owner],   f32[cap][128]        */
    ds4_gpu_tensor *previous_kv;     /* previous_kv[owner],   f32[512]             */
    ds4_gpu_tensor *previous_score;  /* previous_score[owner],f32[512]             */
    uint32_t pos;                    /* absolute position of this row              */
    uint32_t n_comp;                 /* ratio ? (pos + 1) / ratio : 0              */
    uint32_t publish;                /* ratio && (pos + 1) % ratio == 0            */
} ds4_gpu_v41_row;
/* 1 when GPU buffer addresses are usable on this device; 0 means the caller must keep the per-row
 * loop for the whole step (design-3_6.md 4.4: here a 0 address is a fault, not a prefetch). */
int ds4_gpu_v41_rows_available(void);
/* Start a table of `rows` entries; every entry must then be set before a rows kernel is called. */
int ds4_gpu_v41_rows_begin(uint32_t rows);
/* Resolve one row's tensors to GPU addresses and record their buffers and lengths. A NULL tensor
 * is "unused at this layer" and resolves to 0; a non-NULL tensor whose address is unavailable, and
 * a publishing row with n_comp == 0 (slot n_comp - 1 would wrap), fail the call. Each dispatch
 * below then refuses a row whose slot or width would fall outside the destination it was given --
 * the bound ds4_gpu_tensor_copy checked, which a raw GPU address cannot carry. */
int ds4_gpu_v41_rows_set(uint32_t index, const ds4_gpu_v41_row *row);
/* RoPE N rows at N absolute positions in one dispatch; per row identical to ds4_gpu_dsv41_rope. */
int ds4_gpu_dsv41_rope_rows(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                            uint32_t rows, const uint32_t *positions,
                            bool compressed, bool inverse);
/* Quantize N rows and store each rounded row into its own session's window slot (pos % slots). */
int ds4_gpu_dsv41_quantize_window_rows(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                       ds4_v41_activation_format format, uint32_t slots);
/* Pool N single-position rows against their own previous_kv/previous_score carry (ratio 2), or
 * copy kv into out (ratio 1). */
int ds4_gpu_dsv41_pool2_rows(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                             const ds4_gpu_tensor *scores, uint32_t width,
                             uint32_t rows, uint32_t ratio);
/* Store each publishing row's index_k and latent at slot n_comp - 1 of its own caches. */
int ds4_gpu_dsv41_publish_scatter_rows(const ds4_gpu_tensor *index_k, const ds4_gpu_tensor *latent,
                                       uint32_t key_width, uint32_t value_width, uint32_t rows);
int ds4_v41_decode_batch_out_b_row_exact(uint32_t rows, uint32_t outputs);
/* Which attention-output projection a batched decode step of `rows` eligible rows selects: 0 the
 * per-row loop, 1 3.6d's out_a+out_b pair, 2 the opt-in mv_ext out_b, 3 3.6d2's rows<R> kernels.
 * The fusion-gates oracle above proves the rollback names are read; this proves they route, which
 * for a stage whose two sides are bit-identical is the only model-free way to show that its
 * rollback does anything at all. */
int ds4_v41_batch_attention_output_path(unsigned rows);
/* Test oracle (tests/test_deepseek41_metal --flash-rows-desc): where each pointer of the
 * descriptor ds41_batch_attention_flash() builds came from, and the key counts beside it.
 * `*_kind` is 0 for a NULL pointer, 1 for a session's window[], 2 for a session's compressed[],
 * 3 for the workspace's rows_view[].selected_comp, and -1 for anything else; `*_row` is the row
 * whose structure the pointer came from and `*_index` the array index inside it. */
typedef struct {
    int raw_kv_kind, raw_kv_row, raw_kv_index;
    int comp_kv_kind, comp_kv_row, comp_kv_index;
    int comp_ids_kind, comp_ids_row, comp_ids_index;
    unsigned n_raw, raw_cap, raw_start, source_rows, attended;
} ds4_v41_flash_desc_probe;
/* Fill `out` with the descriptor a batched decode step of `rows` rows at layer `il` (compress
 * ratio `ratio`), whose rows sit at `positions`, hands the N-row dispatch for row `row`.  1 when
 * it answered, 0 when the arguments are outside what a batched step can present.  --flash-rows
 * proves that dispatch is bit-identical to N single-row dispatches over descriptors the test
 * builds itself; this proves ds4.c builds them out of the right row's caches. */
int ds4_v41_batch_attention_flash_desc(unsigned rows, const unsigned *positions, unsigned il,
                                       unsigned ratio, unsigned row,
                                       ds4_v41_flash_desc_probe *out);
#endif
/* The three oracles below are defined in ds4.c, so they exist only where the
 * test binary links ds4.o.  tests/test_deepseek41_metal does (Makefile:235,
 * $(CORE_OBJS)); tests/test_deepseek41_cuda compiles this same header and the
 * same test source but links ds4_cuda.o, ds4_image.o and $(MMQ_OBJS) only
 * (Makefile:474).  Declaring them there would invite exactly the link failure
 * ds4_v41_decode_fusion_gates avoids by living inside this guard.
 */
#ifdef __APPLE__
/* 3.7 stage 4 test oracle: the DS4_V41_DSPARK_* switch state ds4.c actually
 * read.  Bit 0 draft allocated (DS4_DISABLE_V41_DSPARK_DRAFT clears it),
 * 1 DS4_V41_DSPARK_SHADOW, 2 a non-empty DS4_V41_DSPARK_DUMP_DIR,
 * 3 DS4_V41_DSPARK_TIMING; bits 4+ hold DS4_V41_DSPARK_DUMP_CALLS.  Each name
 * is spelled once in ds4.c, in the helper both this and the real consumer
 * call, so a typo cannot pass here and fail there. */
int ds4_v41_dspark_draft_gates(void);

/* 3.7 stage 4 test oracle: drives ds41_dspark_capture() over a synthetic
 * residual and checks the HC mean, the slot ordering and the target-layer
 * mask without a model.  Non-zero on success. */
int ds4_v41_dspark_capture_selftest(void);

/* 3.7 stage 4 test oracle AND the writer's own table: the per-stage dump
 * layout scratchpad/dspark_ref.py parses.  Fills `name` with the file suffix
 * for slot `index` of `stage` and returns its element count for `rows` draft
 * rows, or 0 past the last slot.  `name` may be NULL to ask only the count. */
uint64_t ds4_v41_dspark_dump_slot(unsigned index, uint32_t stage, uint32_t rows,
                                  char *name, uint64_t cap);
#endif /* __APPLE__ */
int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual,
                           const ds4_gpu_tensor *kv,
                           const ds4_gpu_tensor *q_weight,
                           const ds4_gpu_tensor *k_weight,
                           const ds4_gpu_tensor *mask,
                           uint32_t width, uint32_t rows, float eps);
/* Pool complete pairs and retain the last unpaired projection in previous_*.
 * start is the absolute token position, including earlier chunks. */
int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out,
                      const ds4_gpu_tensor *kv, const ds4_gpu_tensor *scores,
                      ds4_gpu_tensor *previous_kv, ds4_gpu_tensor *previous_scores,
                       uint32_t width, uint32_t rows, uint32_t start);
/* Candidate blocks contain eight compressed positions. Produce causal block
 * maxima, pinning the newest block; filter consumes a 0/-inf block mask. */
int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks,
                                  const ds4_gpu_tensor *scores,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores,
                                  const ds4_gpu_tensor *block_mask,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
/* Causal index scores over ratio-1/2 compressed keys, without an extra cast
 * of the already quantized FP4 queries/keys. Scores have source_rows stride. */
int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                     const ds4_gpu_tensor *q,
                                     const ds4_gpu_tensor *weights,
                                     const ds4_gpu_tensor *keys,
                                     uint32_t source_rows, uint32_t rows,
                                     uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_tensor_ops_available(void);
/* Reuse exact BF16 views of FP4 queries/keys across score tiles. Invalid
 * casts retain the F32 arithmetic for the affected tile. */
uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed,
                              const ds4_gpu_tensor *q, const ds4_gpu_tensor *keys,
                              uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores,
                                      const ds4_gpu_tensor *q,
                                      const ds4_gpu_tensor *weights,
                                      const ds4_gpu_tensor *keys,
                                      const ds4_gpu_tensor *packed,
                                      uint32_t source_rows, uint32_t rows,
                                      uint32_t start, uint32_t ratio,
                                      uint32_t packed_rows, uint32_t offset);
/* Exact row-sort ordering with independent causal widths; at least 1024
 * visible keys per row. Output stride is 512 indices. */
int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *scores,
                                    uint32_t width, uint32_t rows,
                                    uint32_t start, uint32_t ratio);
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };
/* Lossless storage for already-BF16 activations or 0/-inf candidate masks.
 * Packed rows are padded to whole uint32_t words. Plain rows remain F32. */
int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                            ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                            uint32_t format, bool pack);
/* Batched F16 projections with the same arithmetic as individual matvecs. */
int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out,
                                 const void *model_map, uint64_t model_size,
                                 uint64_t weight_offset, uint32_t width,
                                 uint32_t outputs, uint32_t rows,
                                 const ds4_gpu_tensor *in);
/* Gather 512-wide F32 KV rows; IDs must come from top-k over source_rows. */
int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                           const ds4_gpu_tensor *ids, uint32_t source_rows,
                           uint32_t selected_rows);

#ifdef __cplusplus
}
#endif
#endif
