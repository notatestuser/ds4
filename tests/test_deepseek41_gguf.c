/* Header-only sparse fixtures: no real model or GPU allocation. */
#include "../ds4.c"
#include "../ds4_engram.h"
#include <assert.h>
#include <sys/wait.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

static void check_unmapped(const void *ptr, size_t page) {
#ifdef __APPLE__
    (void)page;
    mach_vm_address_t address = (uintptr_t)ptr;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    kern_return_t result = mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    assert(result == KERN_INVALID_ADDRESS ||
           (result == KERN_SUCCESS && address > (uintptr_t)ptr));
#else
    unsigned char resident;
    errno = 0;
    assert(mincore((void *)ptr, page, &resident) == -1 && errno == ENOMEM);
#endif
}

static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }
static void putstr(FILE *fp, const char *s) {
    put64(fp, strlen(s));
    assert(fwrite(s, 1, strlen(s), fp) == strlen(s));
}

static void string_kv(FILE *fp, const char *key, const char *value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_STRING); putstr(fp, value);
}

static void tensor(FILE *fp, const char *name, uint32_t type,
                   uint64_t width, uint64_t rows, uint64_t offset) {
    putstr(fp, name); put32(fp, 2); put64(fp, width); put64(fp, rows);
    put32(fp, type); put64(fp, offset);
}

static int run_fixture(int bad_layout) {
    enum { ALIGN = 16384 };
    const uint32_t rows = (1u << 24) + 3;
    const uint64_t table_bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    const uint64_t first = 2 * ALIGN + (bad_layout ? 32 : 0);
    const uint64_t second = align_up(first + table_bytes, ALIGN);
    const uint64_t file_size = second + table_bytes;
    char path[] = "/tmp/ds41-gguf.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 3); put64(fp, 3);
    string_kv(fp, "general.architecture", "deepseek41");
    string_kv(fp, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    putstr(fp, "general.alignment"); put32(fp, GGUF_VALUE_UINT32); put32(fp, ALIGN);
    tensor(fp, "test.weight", DS4_TENSOR_F32, 16, 1, 0);
    tensor(fp, "blk.1.engram_embd.weight", 24, 264, rows, first - ALIGN);
    tensor(fp, "blk.14.engram_embd.weight", 24, 264, rows, second - ALIGN);
    assert(ftell(fp) < ALIGN);
    assert(fflush(fp) == 0 && ftruncate(fd, (off_t)file_size) == 0);
    uint8_t row[DS4_ENGRAM_ROW_BYTES];
    memset(row, 56, DS4_ENGRAM_DIM);
    memset(row + DS4_ENGRAM_DIM, 127, DS4_ENGRAM_ROW_BYTES - DS4_ENGRAM_DIM);
    assert(pwrite(fd, row, sizeof(row), (off_t)(second + table_bytes - sizeof(row))) == sizeof(row));
    if (bad_layout) {
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            ds4_model m;
            model_open(&m, path, false, false);
            model_close(&m);
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    } else for (int shared = 0; shared < 2; shared++) {
        ds4_model m;
        model_open(&m, path, shared != 0, false);
        assert(m.file_size == file_size && m.size == first);
        assert(m.max_tensor_bytes == 64);
        char resident;
        assert(mincore((void *)m.map, ALIGN, &resident) == 0);
        check_unmapped(m.map + first, ALIGN);
        const ds4_tensor *t = model_find_tensor(&m, "blk.14.engram_embd.weight");
        assert(t && t->abs_offset == second && t->bytes == table_bytes);
        assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "test.weight")) == 0);
        model_warm_weights(&m);
        ds4_engram_table table;
        assert(ds4_engram_table_open(&table, path, t->abs_offset, rows));
        const uint32_t id = rows - 1;
        float values[DS4_ENGRAM_DIM];
        assert(ds4_engram_read(&table, &id, 1, values));
        for (int i = 0; i < DS4_ENGRAM_DIM; i++) assert(values[i] == 1);
        ds4_engram_table_close(&table);
        model_close(&m);
    }
    assert(fclose(fp) == 0 && unlink(path) == 0);
    return 0;
}

static void check_model_layout(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    assert(weights_layers_bound(&w, 0, UINT32_MAX));
    assert(weights_have_output_head(&w) && !w.output_hc_fn);
    check_unmapped(m.map + m.size, 16384);
    ds4_model_map_span_vec all = {0}, dense = {0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        assert((w.layer[il].attn_compressor_kv != NULL) == ds41_kv_source(il));
        assert((w.layer[il].indexer_attn_q_b != NULL) == ds41_index_source(il));
        assert((w.layer[il].engram_kv != NULL) == ds41_engram_layer(il));
        model_map_span_vec_include_layer(&all, &w.layer[il]);
        model_map_span_vec_include_layer_decode_static(&dense, &w.layer[il]);
    }
    for (uint32_t i = 0; i < all.len; i++) assert(all.v[i].end <= m.size);
    for (uint32_t i = 0; i < dense.len; i++) assert(dense.v[i].end <= m.size);
    free(all.v);
    free(dense.v);
    model_summary(&m);
    model_close(&m);
    puts("V4.1 complete model layout: PASS");
}

/* =========================================================================
 * DSpark support-file binding (3.7 stage 3), model-free.
 *
 * The fixture is a GGUF *header*: the names, shapes, types and offsets of the
 * tensors the DSpark support converter emits, over a sparse file whose tensor
 * payload is never read.  That is all support_model_detect,
 * dspark_weights_bind_optional and dspark_weights_validate_layout look at, so
 * the real V4.1 geometry (spec-3_7.md 2) can be checked without the 7.76 GiB
 * support model or a GPU.
 * ========================================================================= */

typedef struct {
    FILE *fp;
    uint64_t offset;
    uint32_t count;
} dspark_fixture;

static void u32_kv(FILE *fp, const char *key, uint32_t value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_UINT32); put32(fp, value);
}

static void u32_array_kv(FILE *fp, const char *key,
                         const uint32_t *values, uint32_t n) {
    putstr(fp, key); put32(fp, GGUF_VALUE_ARRAY);
    put32(fp, GGUF_VALUE_UINT32); put64(fp, n);
    for (uint32_t i = 0; i < n; i++) put32(fp, values[i]);
}

static void dspark_tensor(dspark_fixture *f, uint32_t stage, const char *suffix,
                          uint32_t type, uint32_t ndim,
                          uint64_t d0, uint64_t d1, uint64_t d2) {
    const uint64_t dims[3] = { d0, d1, d2 };
    char name[96];
    snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix);
    putstr(f->fp, name);
    put32(f->fp, ndim);
    uint64_t elements = 1;
    for (uint32_t i = 0; i < ndim; i++) { put64(f->fp, dims[i]); elements *= dims[i]; }
    put32(f->fp, type);
    put64(f->fp, f->offset);
    uint64_t bytes = 0;
    assert(tensor_nbytes(type, elements, &bytes));
    f->offset = align_up(f->offset + bytes, 32);
    f->count++;
}

static void dspark_block_tensors(dspark_fixture *f, uint32_t stage,
                                 uint64_t n_expert) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    dspark_tensor(f, stage, "hc_attn_fn.weight", DS4_TENSOR_F16, 2, hc_dim, hc_mix_dim, 0);
    dspark_tensor(f, stage, "hc_attn_base.weight", DS4_TENSOR_F32, 1, hc_mix_dim, 0, 0);
    dspark_tensor(f, stage, "hc_attn_scale.weight", DS4_TENSOR_F32, 1, 3, 0, 0);
    dspark_tensor(f, stage, "attn_norm.weight", DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    dspark_tensor(f, stage, "attn_sinks.weight", DS4_TENSOR_F32, 1, DS4_N_HEAD, 0, 0);
    dspark_tensor(f, stage, "attn_q_a.weight", DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
    dspark_tensor(f, stage, "attn_q_a_norm.weight", DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0, 0);
    dspark_tensor(f, stage, "attn_q_b.weight", DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
    dspark_tensor(f, stage, "attn_kv.weight", DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    dspark_tensor(f, stage, "attn_kv_a_norm.weight", DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
    dspark_tensor(f, stage, "attn_output_a.weight", DS4_TENSOR_Q8_0, 2,
                  (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
    dspark_tensor(f, stage, "attn_output_b.weight", DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);
    dspark_tensor(f, stage, "hc_ffn_fn.weight", DS4_TENSOR_F16, 2, hc_dim, hc_mix_dim, 0);
    dspark_tensor(f, stage, "hc_ffn_base.weight", DS4_TENSOR_F32, 1, hc_mix_dim, 0, 0);
    dspark_tensor(f, stage, "hc_ffn_scale.weight", DS4_TENSOR_F32, 1, 3, 0, 0);
    dspark_tensor(f, stage, "ffn_norm.weight", DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    dspark_tensor(f, stage, "ffn_gate_inp.weight", DS4_TENSOR_F32, 2, DS4_N_EMBD, n_expert, 0);
    dspark_tensor(f, stage, "exp_probs_b.bias", DS4_TENSOR_F32, 1, n_expert, 0, 0);
    dspark_tensor(f, stage, "ffn_gate_exps.weight", DS4_TENSOR_Q4_K, 3, DS4_N_EMBD, DS4_N_FF_EXP, n_expert);
    dspark_tensor(f, stage, "ffn_up_exps.weight", DS4_TENSOR_Q4_K, 3, DS4_N_EMBD, DS4_N_FF_EXP, n_expert);
    dspark_tensor(f, stage, "ffn_down_exps.weight", DS4_TENSOR_Q4_K, 3, DS4_N_FF_EXP, DS4_N_EMBD, n_expert);
    dspark_tensor(f, stage, "ffn_gate_shexp.weight", DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_tensor(f, stage, "ffn_up_shexp.weight", DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_tensor(f, stage, "ffn_down_shexp.weight", DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

typedef struct {
    const char *architecture;   /* general.architecture */
    uint64_t block_n_expert;    /* routing width of the block's tensors */
    uint32_t kv_n_expert;       /* 0: emit no dspark.n_expert* metadata */
    uint32_t kv_n_expert_used;
    bool hc_head;               /* emit the three V4-only head tensors */
} dspark_fixture_spec;

/* Writes the fixture to `path_out` (caller unlinks) and returns its tensors. */
static uint32_t write_dspark_fixture(const dspark_fixture_spec *spec, char *path) {
    enum { STAGES = 3, ALIGN = 32 };
    const uint64_t n_expert = spec->block_n_expert;
    const uint32_t target_layers[3] = { DS4_N_LAYER - 3u, DS4_N_LAYER - 2u, DS4_N_LAYER - 1u };
    const uint32_t markov_rank = 256;
    const uint32_t n_kv = 9 + (spec->kv_n_expert ? 2u : 0u);
    assert(n_expert != 0);

    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w+b");
    assert(fp);

    /* The tensor count is written before the entries, so count them first. */
    const uint32_t n_tensors = STAGES * 24u + 2u + 4u + (spec->hc_head ? 3u : 0u);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3);
    put64(fp, n_tensors); put64(fp, n_kv);
    string_kv(fp, "general.architecture", spec->architecture);
    string_kv(fp, "general.name", "DSpark support fixture");
    u32_kv(fp, "general.alignment", ALIGN);
    u32_kv(fp, "dspark.block_size", 5);
    u32_kv(fp, "dspark.markov_rank", markov_rank);
    u32_kv(fp, "dspark.noise_token_id", 128799);
    u32_array_kv(fp, "dspark.target_layer_ids", target_layers, 3);
    u32_kv(fp, "dspark.stage_count", STAGES);
    u32_kv(fp, "dspark.n_layers", STAGES);
    if (spec->kv_n_expert) {
        u32_kv(fp, "dspark.n_expert", spec->kv_n_expert);
        u32_kv(fp, "dspark.n_expert_used", spec->kv_n_expert_used);
    }

    dspark_fixture f = { .fp = fp, .offset = 0, .count = 0 };
    for (uint32_t stage = 0; stage < STAGES; stage++) {
        dspark_block_tensors(&f, stage, n_expert);
        if (stage == 0) {
            dspark_tensor(&f, stage, "main_proj.weight", DS4_TENSOR_Q8_0, 2,
                          (uint64_t)3 * DS4_N_EMBD, DS4_N_EMBD, 0);
            dspark_tensor(&f, stage, "main_norm.weight", DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        }
    }
    const uint32_t final_stage = STAGES - 1u;
    dspark_tensor(&f, final_stage, "norm.weight", DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    if (spec->hc_head) {
        dspark_tensor(&f, final_stage, "hc_head_base.weight", DS4_TENSOR_F32, 1, DS4_N_HC, 0, 0);
        dspark_tensor(&f, final_stage, "hc_head_fn.weight", DS4_TENSOR_F16, 2,
                      (uint64_t)DS4_N_EMBD * DS4_N_HC, DS4_N_HC, 0);
        dspark_tensor(&f, final_stage, "hc_head_scale.weight", DS4_TENSOR_F32, 1, 1, 0, 0);
    }
    dspark_tensor(&f, final_stage, "markov_head.markov_w1.weight", DS4_TENSOR_Q8_0, 2,
                  markov_rank, DS4_N_VOCAB, 0);
    dspark_tensor(&f, final_stage, "markov_head.markov_w2.weight", DS4_TENSOR_Q8_0, 2,
                  markov_rank, DS4_N_VOCAB, 0);
    dspark_tensor(&f, final_stage, "confidence_head.proj.weight", DS4_TENSOR_F32, 2,
                  (uint64_t)DS4_N_EMBD + markov_rank, 1, 0);
    assert(f.count == n_tensors);

    const long header_end = ftell(fp);
    assert(header_end > 0);
    /* Sparse: the payload is addressed by the directory but never touched. */
    const uint64_t total = align_up((uint64_t)header_end, ALIGN) + f.offset;
    assert(fflush(fp) == 0 && ftruncate(fd, (off_t)total) == 0);
    assert(fclose(fp) == 0);
    return n_tensors;
}

static void bind_dspark_fixture(const dspark_fixture_spec *spec,
                                ds4_dspark_weights *dw,
                                uint32_t *n_tensors_out) {
    char path[] = "/tmp/ds41-dspark-support.XXXXXX";
    const uint32_t n_tensors = write_dspark_fixture(spec, path);
    ds4_model m;
    model_open(&m, path, false, false);
    assert(m.n_tensors == n_tensors);
    ds4_dspark_summary summary = {0};
    uint32_t stages = 0;
    assert(support_model_detect(&m, &stages, &summary) == DS4_SUPPORT_DSPARK);
    assert(stages == 3);
    assert(summary.has_expert_counts == (spec->kv_n_expert != 0));
    dspark_weights_bind_optional(dw, &m, &summary);
    model_close(&m);
    assert(unlink(path) == 0);
    if (n_tensors_out) *n_tensors_out = n_tensors;
}

static void check_dspark_support_binding(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    ds4_dspark_weights dw;
    uint32_t n_tensors = 0;

    /* 1. The real V4.1 support file: 78 tensors, no hc_head_*, 128 top-3. */
    g_ds4_shape = DS4_SHAPE_FLASH41;
    const dspark_fixture_spec v41 = {
        .architecture = "deepseek41-dspark", .block_n_expert = 128,
        .kv_n_expert = 128, .kv_n_expert_used = 3, .hc_head = false,
    };
    bind_dspark_fixture(&v41, &dw, &n_tensors);
    assert(n_tensors == 78);
    assert(dw.present_tensors == 78);
    assert(dw.missing_tensors == 0);
    assert(dw.invalid_tensors == 0);
    assert(dw.metadata_errors == 0);
    assert(dw.n_expert == 128 && dw.n_expert_used == 3 && dw.has_expert_counts);
    assert(dw.head_mix_from_last_ffn);
    assert(dspark_weights_usable(&dw));   /* the engine starts on this file */
    assert(!dw.stage[2].hc_head_base && !dw.stage[2].hc_head_fn &&
           !dw.stage[2].hc_head_scale);
    assert(dw.stage[0].main_proj && dw.stage[0].main_norm);
    assert(dw.stage[2].norm && dw.stage[2].markov_w1 && dw.stage[2].markov_w2 &&
           dw.stage[2].confidence_proj);
    assert(dw.stage[0].block.ffn_gate_exps && dw.stage[2].block.ffn_down_exps);

    /* 2. Without the metadata the target model's 384 is assumed, and the five
     *    routing tensors of each of the three blocks then mismatch. */
    const dspark_fixture_spec v41_no_meta = {
        .architecture = "deepseek41-dspark", .block_n_expert = 128,
        .kv_n_expert = 0, .kv_n_expert_used = 0, .hc_head = false,
    };
    bind_dspark_fixture(&v41_no_meta, &dw, NULL);
    assert(dw.n_expert == DS4_N_EXPERT && dw.n_expert_used == DS4_N_EXPERT_USED);
    assert(dw.missing_tensors == 0);
    assert(dw.invalid_tensors == 15);
    assert(dw.metadata_errors == 0);
    assert(dw.head_mix_from_last_ffn);    /* still a V4.1-dialect file */
    assert(!dspark_weights_usable(&dw));  /* ... and the engine refuses it */

    /* 3. A V4 support file against a V4.1 target is named, not drowned. */
    const dspark_fixture_spec wrong_family = {
        .architecture = "deepseek4-dspark", .block_n_expert = 128,
        .kv_n_expert = 128, .kv_n_expert_used = 3, .hc_head = true,
    };
    bind_dspark_fixture(&wrong_family, &dw, NULL);
    assert(dw.metadata_errors == 1);
    /* A V4-dialect file keeps the V4 head rule even on a V4.1 target: the
     * three hc_head_* tensors are bound, so head_mix_from_last_ffn is false
     * rather than contradicting them.  The engine refuses it either way. */
    assert(!dw.head_mix_from_last_ffn);
    assert(dw.stage[2].hc_head_base && dw.stage[2].hc_head_fn &&
           dw.stage[2].hc_head_scale);
    assert(!dspark_weights_usable(&dw));

    /* 4. V4 is unchanged: hc_head_* required, expert width from the globals. */
    g_ds4_shape = DS4_SHAPE_FLASH;
    const dspark_fixture_spec v4 = {
        .architecture = "deepseek4-dspark", .block_n_expert = DS4_N_EXPERT,
        .kv_n_expert = 0, .kv_n_expert_used = 0, .hc_head = true,
    };
    bind_dspark_fixture(&v4, &dw, &n_tensors);
    assert(n_tensors == 81);
    assert(dw.present_tensors == 81);
    assert(dw.missing_tensors == 0);
    assert(dw.invalid_tensors == 0);
    assert(dw.metadata_errors == 0);
    assert(dw.n_expert == DS4_N_EXPERT && dw.n_expert_used == DS4_N_EXPERT_USED);
    assert(!dw.head_mix_from_last_ffn);
    assert(dw.stage[2].hc_head_base && dw.stage[2].hc_head_fn &&
           dw.stage[2].hc_head_scale);
    assert(dspark_weights_usable(&dw));

    /* 5. ... and a V4 file that lost them is still three tensors short. */
    const dspark_fixture_spec v4_no_head = {
        .architecture = "deepseek4-dspark", .block_n_expert = DS4_N_EXPERT,
        .kv_n_expert = 0, .kv_n_expert_used = 0, .hc_head = false,
    };
    bind_dspark_fixture(&v4_no_head, &dw, NULL);
    assert(dw.missing_tensors == 3);
    assert(dw.invalid_tensors == 0);
    assert(!dspark_weights_usable(&dw));

    g_ds4_shape = saved_shape;
    puts("DSpark support binding (V4.1 128/3, V4 unchanged): PASS");
}

/* The engine-level rule the V4.1 `supported` predicate applies to the flags. */
static void check_v41_dspark_predicate(void) {
    ds4_engine_options opt = {0};
    opt.backend = DS4_BACKEND_METAL;
    assert(unsetenv("DS4_DISABLE_V41_DSPARK_LOAD") == 0);

    assert(!ds4_v41_dspark_support_requested(&opt));      /* neither flag */
    opt.mtp_path = "support.gguf";
    assert(!ds4_v41_dspark_support_requested(&opt));      /* --mtp-model alone */
    opt.dspark = true;
    assert(ds4_v41_dspark_support_requested(&opt));       /* both, on Metal */
    opt.backend = DS4_BACKEND_CPU;
    assert(!ds4_v41_dspark_support_requested(&opt));      /* Metal only */
    opt.backend = DS4_BACKEND_METAL;
    assert(setenv("DS4_DISABLE_V41_DSPARK_LOAD", "1", 1) == 0);
    assert(!ds4_v41_dspark_support_requested(&opt));      /* rollback */
    assert(unsetenv("DS4_DISABLE_V41_DSPARK_LOAD") == 0);
    assert(ds4_v41_dspark_support_requested(&opt));
    opt.mtp_path = "";
    assert(!ds4_v41_dspark_support_requested(&opt));      /* --dspark alone */
    puts("V4.1 DSpark support predicate + DS4_DISABLE_V41_DSPARK_LOAD: PASS");
}

/*
 * What a front end sees on a stage-3 V4.1 engine: a bound DSpark support model
 * and no draft width.  ds4-bench must read that as "serial decode", not as a
 * failed support model, and must keep reading a V4 engine the old way.
 */
static void check_dspark_draft_pending(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->backend = DS4_BACKEND_METAL;
    e->support_kind = DS4_SUPPORT_DSPARK;
    e->dspark = true;
    e->dspark_weights.block_size = 5;

    g_ds4_shape = DS4_SHAPE_FLASH41;
    assert(!ds4_engine_dspark_draft_available(e));
    assert(ds4_engine_dspark_draft_pending(e));
    /* Stage 4 is what makes this non-zero.  (The DSpark arm of
     * ds4_engine_mtp_draft_tokens is inside #ifndef DS4_NO_GPU, so in this
     * CPU-only translation it reads zero for both families; the two predicates
     * above and below are the part that differs.) */
    assert(ds4_engine_mtp_draft_tokens(e) == 0);

    g_ds4_shape = DS4_SHAPE_FLASH;
    assert(ds4_engine_dspark_draft_available(e));
    assert(!ds4_engine_dspark_draft_pending(e));   /* V4: bench still refuses */

    g_ds4_shape = DS4_SHAPE_FLASH41;
    e->dspark = false;
    assert(!ds4_engine_dspark_draft_pending(e));   /* no --dspark */
    e->dspark = true;
    e->support_kind = DS4_SUPPORT_NONE;
    assert(!ds4_engine_dspark_draft_pending(e));   /* no support model */

    free(e);
    g_ds4_shape = saved_shape;
    puts("V4.1 DSpark draft-pending predicate (serial-decode front ends): PASS");
}

int main(int argc, char **argv) {
    if (argc == 2) {
        check_model_layout(argv[1]);
        return 0;
    }
    assert(argc == 1);
    run_fixture(0);
    run_fixture(1);
    puts("V4.1 disk-only GGUF extent: PASS");
    check_dspark_support_binding();
    check_v41_dspark_predicate();
    check_dspark_draft_pending();
    return 0;
}
