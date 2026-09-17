#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;
static float random_value(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, 4);
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] : i < 8 ? ldexpf(i, -9) :
                        ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error || (error == best_error && !(i & 1) && (best & 1))) {
            best = i; best_value = v; best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int check_router(void) {
    enum { ROWS = 129, MAX_EXPERTS = 384, USED = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(bias_bytes >= MAX_EXPERTS * sizeof(float));
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    float *logits = malloc(ROWS * MAX_EXPERTS * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(bias && logits);
    for (unsigned e = 0; e < MAX_EXPERTS; e++) bias[e] = (e % 7) * 0.125f;
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *p = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *ids = upload(NULL, ROWS * USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, ROWS * USED * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    CHECK(x && p && ids && weights && tok);
    const unsigned counts[] = {1, 3, 17, ROWS};
    for (unsigned n = 256; n <= MAX_EXPERTS; n += 128) {
        for (unsigned mode = 0; mode < 3; mode++) {
            for (unsigned i = 0; i < ROWS * n; i++) logits[i] = mode == 1 ? 0 :
                mode == 2 ? (i % 2 ? -80 : 40) : random_value() * 8;
            CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * n * sizeof(float)));
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                CHECK(ds4_gpu_router_select_batch_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, 0, true, false,
                    x, tok, n, USED, 1.5f, rows));
                CHECK(ds4_gpu_synchronize());
                const int32_t *selected = ds4_gpu_tensor_contents(ids);
                const float *actual = ds4_gpu_tensor_contents(weights);
                const float *probs = ds4_gpu_tensor_contents(p);
                CHECK(selected && actual && probs);
                for (unsigned t = 0; t < rows; t++) {
                    double ref[MAX_EXPERTS];
                    int best[USED];
                    for (unsigned k = 0; k < USED; k++) best[k] = -1;
                    for (unsigned e = 0; e < n; e++) {
                        const double v = logits[t * n + e];
#ifdef __APPLE__
                        /* Match Metal's established FP32 addition before log;
                         * log1p has different rounding near zero. */
                        ref[e] = sqrtf(v > 20 ? (float)v : logf(1.0f + expf((float)v)));
#else
                        ref[e] = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v)));
#endif
                        if (!(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e])))
                            fprintf(stderr, "router n=%u mode=%u rows=%u row=%u expert=%u logit=%.9g actual=%.9g ref=%.9g\n",
                                    n, mode, rows, t, e, v, probs[t * n + e], ref[e]);
                        CHECK(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e]));
                        const double score = ref[e] + bias[e];
                        for (unsigned k = 0; k < USED; k++) {
                            if (best[k] < 0 || score > ref[best[k]] + bias[best[k]]) {
                                for (unsigned j = USED - 1; j > k; j--) best[j] = best[j - 1];
                                best[k] = (int)e;
                                break;
                            }
                        }
                    }
                    double sum = 0;
                    for (unsigned k = 0; k < USED; k++) {
                        const int id = selected[t * USED + k];
                        CHECK(id >= 0 && id < (int)n);
                        for (unsigned j = 0; j < k; j++) CHECK(id != selected[t * USED + j]);
#ifdef __APPLE__
                        /* Bitonic selection does not promise CUDA's lower-ID
                         * tie break. Still require the correct top-k scores. */
                        CHECK(fabs((ref[id] + bias[id]) - (ref[best[k]] + bias[best[k]])) < 3e-6);
#else
                        CHECK(id == best[k]);
#endif
                        sum += ref[id];
                    }
                    for (unsigned k = 0; k < USED; k++) {
                        CHECK(fabs(actual[t * USED + k] - 1.5 * ref[selected[t * USED + k]] /
                              fmax(sum, 0x1p-14)) < 3e-6);
                    }
                }
                int32_t first_ids[USED]; float first_weights[USED];
                memcpy(first_ids, selected, sizeof(first_ids));
                memcpy(first_weights, actual, sizeof(first_weights));
                CHECK(ds4_gpu_router_select_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, n, USED, 1.5f,
                    0, 0, true, false, x));
                CHECK(ds4_gpu_synchronize());
                const int32_t *scalar_ids = ds4_gpu_tensor_contents(ids);
                const float *scalar_weights = ds4_gpu_tensor_contents(weights);
#ifdef __APPLE__
                for (unsigned k = 0; k < USED; k++) {
                    CHECK(scalar_ids[k] >= 0 && scalar_ids[k] < (int32_t)n);
                    CHECK(probs[first_ids[k]] + bias[first_ids[k]] ==
                          probs[scalar_ids[k]] + bias[scalar_ids[k]]);
                    CHECK(fabsf(first_weights[k] - scalar_weights[k]) < 3e-6f);
                }
#else
                CHECK(!memcmp(first_ids, scalar_ids, sizeof(first_ids)));
                CHECK(!memcmp(first_weights, scalar_weights, sizeof(first_weights)));
#endif
            }
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(tok);
    ds4_gpu_cleanup();
    free(bias); free(logits);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "256/384-expert routing, ties/extremes and scalar/batch oracle: PASS\n");
    return 1;
}

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f))));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            if (mode == DS4_V41_FP4_E4M3) scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode) expected = bf16(nearest(expected / scale, mode != 1) * scale);
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "quantization mode=%d index=%d: %.9g != %.9g\n",
                            mode, start + i, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

static double monotonic_seconds(void);

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t bf16_input(size_t i) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    if (i < 65536u * 5u) return ((uint32_t)(i / 5u) << 16u) | low[i % 5u];
    return (uint32_t)i * 2654435761u;
}

static int check_bf16_linear(void) {
    const struct { uint32_t width, rows, offset; } shapes[] = {
        {1, 1, 0}, {3, 1, 0}, {5, 1, 0}, {24, 1, 0}, {31, 3, 0},
        {32, 7, 1}, {129, 3, 4}, {1023, 1, 0}, {1024, 1, 0}, {1025, 1, 0},
        {65536u * 5u, 1, 0}, {5120, 1, 0}, {20480, 512, 0},
        {32768, 4096, 0}, {32768, 8192, 0}
    };
    const unsigned modes[] = {0, 1, 1, 0};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const size_t count = (size_t)shapes[s].width * shapes[s].rows;
        const size_t offset = shapes[s].offset;
        ds4_gpu_tensor *storage = upload(NULL, (count + 8) * sizeof(uint32_t));
        CHECK(storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage,
            offset * sizeof(uint32_t), count * sizeof(uint32_t));
        CHECK(view);
        uint32_t *bits = ds4_gpu_tensor_contents(storage);
        CHECK(bits);
        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned mode = modes[pass];
            if (mode) unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
            else setenv("DS4_METAL_DISABLE_V41_LINEAR_BF16", "1", 1);
            for (size_t i = 0; i < count + 8; i++) bits[i] = 0x12345678;
            for (size_t i = 0; i < count; i++) bits[offset + i] = bf16_input(i);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 5; repeat++) {
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_dsv41_quantize(view, shapes[s].width, shapes[s].rows, DS4_V41_BF16));
                CHECK(ds4_gpu_synchronize());
                if (repeat) elapsed += (monotonic_seconds() - begin) * 250;
            }
            for (size_t i = 0; i < count; i++) CHECK(bits[offset + i] == bf16_bits(bf16_input(i)));
            for (size_t i = 0; i < offset; i++) CHECK(bits[i] == 0x12345678);
            for (size_t i = offset + count; i < count + 8; i++) CHECK(bits[i] == 0x12345678);
            fprintf(stderr, "BF16 width=%u rows=%u offset=%zu mode=%u pass=%u %.3f ms: exact\n",
                shapes[s].width, shapes[s].rows, offset, mode, pass, elapsed);
        }
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(storage);
    }
    unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
    return 1;
}

#ifdef __APPLE__
static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24, ROWS = 8192 };
    const size_t weight_bytes = WIDTH * OUT * sizeof(_Float16);
    const size_t page = (size_t)getpagesize();
    const size_t mapped_bytes = (weight_bytes + page - 1) / page * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, mapped_bytes));
    _Float16 *weights = model;
    for (size_t i = 0; i < WIDTH * OUT; i++) weights[i] = (_Float16)(random_value() / 64);
    CHECK(ds4_gpu_set_model_map(model, mapped_bytes));
    ds4_gpu_tensor *x = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *norm = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *scales = upload(NULL, (ROWS + 1) * 4);
    ds4_gpu_tensor *out = upload(NULL, (OUT * ROWS + 1) * 4);
    ds4_gpu_tensor *ref = upload(NULL, (OUT * ROWS + 1) * 4);
    CHECK(x && norm && scales && out && ref);
    float *input = ds4_gpu_tensor_contents(x), *y = ds4_gpu_tensor_contents(out);
    float *expected = ds4_gpu_tensor_contents(ref), *s = ds4_gpu_tensor_contents(scales);
    for (size_t i = 0; i < (size_t)WIDTH * ROWS; i++)
        input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    const unsigned rows[] = {9, 31, 32, 33, 127, 512, 2048, 4096, 8192};
    for (size_t shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        const uint32_t n = rows[shape];
        double elapsed[2] = {0};
        for (unsigned repeat = 0; repeat < 7; repeat++) {
            y[n * OUT] = expected[n * OUT] = s[n] = 12345;
            for (unsigned j = 0; j < 2; j++) {
                const unsigned mode = j ^ (repeat & 1);
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode) {
                    CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(out, n >= 512u ? scales : norm,
                        model, mapped_bytes, 0, WIDTH, OUT, x, n, 1e-6f));
                } else {
                    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(norm, x, WIDTH, n, 1e-6f));
                    CHECK(ds4_gpu_matmul_f16_tensor(ref, model, mapped_bytes, 0,
                        WIDTH, OUT, norm, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / 6);
            }
            CHECK(!memcmp(y, expected, n * OUT * 4));
            CHECK(y[n * OUT] == 12345 && expected[n * OUT] == 12345 && s[n] == 12345);
        }
        fprintf(stderr, "HC scaled rows=%u: exact %.3f -> %.3f ms\n", n, elapsed[0], elapsed[1]);
    }
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

#endif

static int check_engram(void) {
    enum { D = 5120, ROWS = 5, N = ROWS * 4 * D };
    float *x = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    float *kv = malloc(ROWS * 5 * D * sizeof(float));
    float *qw = malloc(4 * D * sizeof(float)), *kw = malloc(4 * D * sizeof(float));
    uint8_t mask[] = {1, 1, 0, 1, 0};
    CHECK(x && actual && kv && qw && kw);
    for (int i = 0; i < N; i++) x[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) { qw[i] = random_value(); kw[i] = random_value(); }
    memset(x, 0, D * sizeof(float));
    memset(kv + D, 0, D * sizeof(float));
    ds4_gpu_tensor *xt = upload(x, N * sizeof(float));
    ds4_gpu_tensor *kt = upload(kv, ROWS * 5 * D * sizeof(float));
    ds4_gpu_tensor *qwt = upload(qw, 4 * D * sizeof(float));
    ds4_gpu_tensor *kwt = upload(kw, 4 * D * sizeof(float));
    ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
    CHECK(xt && kt && qwt && kwt && mt);
    size_t rounded_differently = 0;
    double error2 = 0, norm2 = 0;
    for (int masked = 0; masked < 2; masked++) {
        CHECK(ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
        CHECK(ds4_gpu_tensor_read(xt, 0, actual, N * sizeof(float)));
        for (int row = 0; row < ROWS; row++) for (int h = 0; h < 4; h++) {
            double dot = 0, h2 = 0, k2 = 0;
            for (int i = 0; i < D; i++) {
                const double a = x[(row * 4 + h) * D + i], b = kv[(row * 5 + h) * D + i];
                h2 += a * a; k2 += b * b;
                dot += a * (float)(qw[h * D + i] * kw[h * D + i]) * b;
            }
            dot /= sqrt(h2 / D + 1e-20) * sqrt(k2 / D + 1e-20) * sqrt(D);
            double gate = 1 / (1 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            if (masked && !mask[row]) gate = 0;
            for (int i = 0; i < D; i++) {
                const int off = (row * 4 + h) * D + i;
                const float expected = bf16(x[off] + (float)gate * kv[(row * 5 + 4) * D + i]);
                const double error = actual[off] - expected;
                CHECK(isfinite(actual[off]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected) / 128));
                if (masked && !mask[row]) CHECK(actual[off] == x[off]);
                rounded_differently += actual[off] != expected;
                error2 += error * error; norm2 += (double)expected * expected;
            }
        }
    }
    CHECK(rounded_differently < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    fprintf(stderr, "V4.1 Engram gate: %zu BF16 boundary differences, relative RMS %.8g\n",
            rounded_differently, sqrt(error2 / norm2));
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

static int check_rope_stride(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, (N + 1u) * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, (N + 1u) * sizeof(float));
    CHECK(a && b);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N; j++) x[j] = y[j] = bf16(random_value());
            x[N] = y[N] = 12345;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], 2, kind, inverse));
            for (uint32_t row = 0; row < ROWS; row++) {
                ds4_gpu_tensor *v = ds4_gpu_tensor_view(b, (uint64_t)row * WIDTH * HEADS * 4u,
                                                       WIDTH * HEADS * 4u);
                CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, HEADS, 1, starts[i] + row * 2u, kind, inverse));
                ds4_gpu_tensor_free(v);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x, y, (N + 1u) * sizeof(float)) && x[N] == 12345);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS + 1, 0, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, UINT32_MAX, true, false));
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 strided RoPE: exact row parity through 1M positions, guards PASS\n");
    return 1;
}

#ifdef __APPLE__
/* 3.5(b): q and kv roped in one dispatch must equal the two separate dispatches, byte for byte,
 * and must touch nothing else. */
static int check_rope_pair(void) {
    enum { WIDTH = 512, HEADS = 32, N0 = WIDTH * HEADS, N1 = WIDTH };
    ds4_gpu_tensor *a0 = upload(NULL, (N0 + 1u) * sizeof(float));
    ds4_gpu_tensor *b0 = upload(NULL, (N0 + 1u) * sizeof(float));
    ds4_gpu_tensor *a1 = upload(NULL, (N1 + 1u) * sizeof(float));
    ds4_gpu_tensor *b1 = upload(NULL, (N1 + 1u) * sizeof(float));
    CHECK(a0 && b0 && a1 && b1);
    float *x0 = ds4_gpu_tensor_contents(a0), *y0 = ds4_gpu_tensor_contents(b0);
    float *x1 = ds4_gpu_tensor_contents(a1), *y1 = ds4_gpu_tensor_contents(b1);
    const uint32_t starts[] = {0, 1, 127, 2048, 32766, 1048575};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N0; j++) x0[j] = y0[j] = bf16(random_value());
            for (size_t j = 0; j < N1; j++) x1[j] = y1[j] = bf16(random_value());
            x0[N0] = y0[N0] = 12345;
            x1[N1] = y1[N1] = 54321;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_pair(a0, HEADS, a1, 1, WIDTH, starts[i], kind, inverse));
            CHECK(ds4_gpu_dsv41_rope(b0, WIDTH, HEADS, 1, starts[i], kind, inverse));
            CHECK(ds4_gpu_dsv41_rope(b1, WIDTH, 1, 1, starts[i], kind, inverse));
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x0, y0, (N0 + 1u) * sizeof(float)));
            CHECK(!memcmp(x1, y1, (N1 + 1u) * sizeof(float)));
            CHECK(x0[N0] == 12345 && x1[N1] == 54321);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_pair(NULL, HEADS, a1, 1, WIDTH, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS, NULL, 1, WIDTH, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, 0, a1, 1, WIDTH, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS, a1, 0, WIDTH, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS, a1, 1, 32, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS, a1, 1, WIDTH, 1048576, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS + 1u, a1, 1, WIDTH, 0, false, false));
    CHECK(!ds4_gpu_dsv41_rope_pair(a0, HEADS, a1, 2, WIDTH, 0, false, false));
    ds4_gpu_tensor_free(b1); ds4_gpu_tensor_free(a1);
    ds4_gpu_tensor_free(b0); ds4_gpu_tensor_free(a0);
    fprintf(stderr, "V4.1 paired q+kv RoPE: bit-identical to two dispatches, guards PASS\n");
    return 1;
}

/* 3.5(c): the fused quantize + window store must equal quantize followed by the copy, and must
 * leave the rest of the window untouched. */
static int check_quantize_store(void) {
    enum { WIDTH = 512, ROWS = 3, SLOT = 5, SLOTS = 128, N = WIDTH * ROWS, WN = SLOTS * WIDTH };
    float *source = malloc(N * sizeof(float));
    CHECK(source);
    ds4_gpu_tensor *xa = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *xb = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *wa = upload(NULL, WN * sizeof(float));
    ds4_gpu_tensor *wb = upload(NULL, WN * sizeof(float));
    CHECK(xa && xb && wa && wb);
    float *pa = ds4_gpu_tensor_contents(wa), *pb = ds4_gpu_tensor_contents(wb);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WN; i++) pa[i] = pb[i] = (float)(i % 251) - 125.0f;
        CHECK(ds4_gpu_tensor_write(xa, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(xb, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_quantize(xa, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_copy(wa, (uint64_t)SLOT * WIDTH * 4u, xa, 0, (uint64_t)N * 4u));
        CHECK(ds4_gpu_dsv41_quantize_store(xb, WIDTH, ROWS, (ds4_v41_activation_format)mode,
                                           wb, (uint64_t)SLOT * WIDTH * 4u));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(xa), ds4_gpu_tensor_contents(xb), N * sizeof(float)));
        CHECK(!memcmp(pa, pb, WN * sizeof(float)));
    }
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, WIDTH, ROWS, DS4_V41_FP8_E8M0, NULL, 0));
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, WIDTH, ROWS, (ds4_v41_activation_format)4, wb, 0));
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, 24, 1, DS4_V41_FP8_E8M0, wb, 0));
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, WIDTH, ROWS, DS4_V41_FP8_E8M0, wb,
                                        (uint64_t)(WN - N + 1) * 4u));
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, WIDTH, ROWS, DS4_V41_FP8_E8M0, wb, 2));
    CHECK(!ds4_gpu_dsv41_quantize_store(xa, UINT32_MAX, UINT32_MAX, DS4_V41_BF16, wb, 0));
    ds4_gpu_tensor_free(wb); ds4_gpu_tensor_free(wa);
    ds4_gpu_tensor_free(xb); ds4_gpu_tensor_free(xa);
    free(source);
    fprintf(stderr, "V4.1 fused quantize + window store: bit-identical to quantize + copy, guards PASS\n");
    return 1;
}

/* Exactness witness for the shared RoPE device function: an FNV-1a digest of kernel_dsv41_rope's
 * output over a fixed matrix of shapes. It must not move when metal/dsv41.metal is edited - run it
 * once with DS4_METAL_DSV41_SOURCE pointing at the previous file and once without. */
static int check_rope_digest(void) {
    enum { WIDTH = 512, HEADS = 3, ROWS = 9, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, N * sizeof(float));
    CHECK(a);
    float *x = ds4_gpu_tensor_contents(a);
    const uint32_t starts[] = {0, 1, 127, 2048, 32766, 1048318};
    uint64_t digest = 14695981039346656037ull;
    seed = 7919;
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (uint32_t stride = 1; stride <= 2; stride++) {
                for (size_t j = 0; j < N; j++) x[j] = bf16(random_value());
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], stride,
                                                kind, inverse));
                CHECK(ds4_gpu_end_commands());
                const unsigned char *bytes = (const unsigned char *)x;
                for (size_t j = 0; j < N * sizeof(float); j++) {
                    digest ^= bytes[j];
                    digest *= 1099511628211ull;
                }
            }
        }
    }
    ds4_gpu_tensor_free(a);
    printf("rope_digest=%016llx\n", (unsigned long long)digest);
    fprintf(stderr, "V4.1 RoPE digest over 48 fixed dispatches: %016llx\n",
            (unsigned long long)digest);
    return 1;
}

/* Host-side witness for the other half of the RoPE change. The digest above runs one binary
 * against two kernel files, so it cannot see that the frequency table moved out of
 * ds4_gpu_dsv41_rope_stride into ds4_gpu_dsv41_rope_frequencies - and that table is what every
 * theta on the untouched default path is built from. Recompute it here from the pre-patch source
 * text, verbatim, under the same precise float control (ds4_metal.m is compiled with -ffast-math,
 * so leaving the pragma region is exactly the way the move could have changed a value), and
 * require the 64 floats to be byte-identical. */
#pragma float_control(precise, on, push)
static int check_rope_freqs(void) {
    float expected[2][32];
    for (int kind = 0; kind < 2; kind++) {
        const float base = kind ? 160000.0f : 10000.0f;
        const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
        const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (kind) {
                const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                const float interpolate = (f / 16.0f) * (1.0f - smooth);
                const float extrapolate = f * smooth;
                f = interpolate + extrapolate;
            }
            expected[kind][i] = f;
        }
    }
    uint64_t digest = 14695981039346656037ull;
    for (int kind = 0; kind < 2; kind++) {
        const float *actual = ds4_gpu_dsv41_rope_frequencies(kind != 0);
        CHECK(actual);
        CHECK(!memcmp(actual, expected[kind], sizeof(expected[kind])));
        const unsigned char *bytes = (const unsigned char *)actual;
        for (size_t j = 0; j < sizeof(expected[kind]); j++) {
            digest ^= bytes[j];
            digest *= 1099511628211ull;
        }
    }
    printf("rope_freqs=%016llx\n", (unsigned long long)digest);
    fprintf(stderr, "V4.1 RoPE frequencies: host table matches the pre-patch expressions, "
                    "digest %016llx\n", (unsigned long long)digest);
    return 1;
}
#pragma float_control(pop)

/* The three rollback switches are read by file-static helpers in ds4.c, which no test can reach
 * by name; the fused entry points above are called directly and are deliberately unaware of them.
 * ds4_v41_decode_fusion_gates() is the oracle: assert that each env name, spelled as ds4.c spells
 * it, disables exactly its own fusion and nothing else. A misspelt name would otherwise survive
 * every check - including the strict bench, which would then A/B the fused path against itself. */
static int check_fusion_gates(void) {
    static const char *const names[3] = {
        "DS4_METAL_DISABLE_PRE_M5_V41_PRE_COPY",
        "DS4_METAL_DISABLE_PRE_M5_V41_ROPE_PAIR",
        "DS4_METAL_DISABLE_PRE_M5_V41_QUANTIZE_STORE",
    };
    /* ds4_gpu_init() must have run: the pre-M5 test reads the Metal device name, and without it
     * every gate reads false and this whole check would pass vacuously. */
    const int pre_m5 = ds4_gpu_device_is_pre_m5_apple_silicon();
    const int base = ds4_v41_decode_fusion_gates();
    printf("fusion_gates=%d\n", base);
    CHECK(pre_m5 || base == 0);          /* nothing fuses off pre-M5 Apple silicon */
    for (int i = 0; i < 3; i++) {
        const int bit = 1 << i;
        if (getenv(names[i])) {
            /* Inherited from the caller (the runner sets one per process): must read as off. */
            CHECK(!(base & bit));
            continue;
        }
        CHECK(!pre_m5 || (base & bit));  /* default on */
        CHECK(setenv(names[i], "1", 1) == 0);
        const int masked = ds4_v41_decode_fusion_gates();
        CHECK(!(masked & bit));
        CHECK((masked & ~bit) == (base & ~bit));
        CHECK(unsetenv(names[i]) == 0);
        CHECK(ds4_v41_decode_fusion_gates() == base);
    }
    fprintf(stderr, "V4.1 decode fusion gates: pre_m5=%d live=0x%x, each rollback env disables "
                    "exactly its own fusion PASS\n", pre_m5, base);
    return 1;
}
#endif

static int check_pool(void) {
    enum { D = 512, ROWS = 257, PAIRS = ROWS / 2 };
    float *kv = malloc(ROWS * D * sizeof(float)), *scores = malloc(ROWS * D * sizeof(float));
    float *got = malloc(PAIRS * D * sizeof(float)), *reference = malloc(PAIRS * D * sizeof(float));
    CHECK(kv && scores && got && reference);
    for (int i = 0; i < ROWS * D; i++) { kv[i] = random_value(); scores[i] = random_value() * 25; }
    ds4_gpu_tensor *kt = upload(kv, ROWS * D * sizeof(float));
    ds4_gpu_tensor *st = upload(scores, ROWS * D * sizeof(float));
    ds4_gpu_tensor *pk = upload(NULL, D * sizeof(float)), *ps = upload(NULL, D * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, PAIRS * D * sizeof(float));
    CHECK(kt && st && pk && ps && out);
    const uint32_t chunks[] = {257, 1, 2, 3, 17, 127, 128, 129};
    for (size_t c = 0; c < sizeof(chunks) / sizeof(*chunks); c++) {
        CHECK(ds4_gpu_tensor_fill_f32(pk, NAN, D));
        CHECK(ds4_gpu_tensor_fill_f32(ps, NAN, D));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t start = 0; start < ROWS;) {
            uint32_t n = chunks[c] < ROWS - start ? chunks[c] : ROWS - start;
            uint32_t pairs = (n + (start & 1u)) / 2;
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(kt, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(st, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *o = pairs ? ds4_gpu_tensor_view(out, (uint64_t)(start / 2) * D * 4,
                                                         (uint64_t)pairs * D * 4) : NULL;
            CHECK(k && s && (!pairs || o));
            CHECK(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        if (!c) {
            memcpy(reference, got, PAIRS * D * sizeof(float));
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        } else CHECK(!memcmp(reference, got, PAIRS * D * sizeof(float)));
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, D, ROWS - 1, 0));
    CHECK(ds4_gpu_end_commands());
    CHECK(!memcmp(ds4_gpu_tensor_contents(pk), kv + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(ps), scores + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference, PAIRS * D * sizeof(float)));
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: exact across eight chunk sizes\n");
    return 1;
}

typedef struct { float score; uint32_t index; } candidate;
static int candidate_desc(const void *a, const void *b) {
    const candidate *x = a, *y = b;
    return x->score > y->score ? -1 : x->score < y->score ? 1 :
           x->index < y->index ? -1 : x->index > y->index;
}

static int check_candidates(void) {
    const uint32_t widths[] = {1, 7, 8, 9, 127, 16385, 17017};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, rows = 17;
        const uint32_t top = blocks < 2048 ? blocks : 2048;
        float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
        float *maxima = malloc((size_t)blocks * rows * 4);
        candidate *sorted = malloc(blocks * sizeof(candidate));
        uint8_t *kept = malloc(blocks);
        CHECK(scores && got && maxima && sorted && kept);
        /* Unique finite values avoid unspecified top-k tie ordering. */
        for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
            scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
        ds4_gpu_tensor *s = upload(NULL, (size_t)n * rows * 4);
        ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
        ds4_gpu_tensor *t = upload(NULL, (size_t)top * rows * 4);
        ds4_gpu_tensor *m = upload(NULL, (size_t)blocks * rows * 4);
        CHECK(s && b && t && m);
        for (uint32_t ratio = 1; ratio <= 2; ratio++) for (int late = 0; late < 2; late++) {
            const uint32_t start = late ? n * ratio - 1 : 0;
            CHECK(ds4_gpu_tensor_write(s, 0, scores, (size_t)n * rows * 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            CHECK(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(b, 0, maxima, (size_t)blocks * rows * 4));
            CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t visible = (start + r + 1) / ratio;
                if (visible > n) visible = n;
                for (uint32_t j = 0; j < blocks; j++) {
                    float best = -INFINITY;
                    for (uint32_t i = j * 8; i < (j + 1) * 8 && i < visible; i++)
                        best = fmaxf(best, scores[(size_t)r * n + i]);
                    if (visible && j == (visible - 1) / 8) best = INFINITY;
                    CHECK(maxima[(size_t)r * blocks + j] == best);
                    sorted[j] = (candidate){best, j};
                }
                qsort(sorted, blocks, sizeof(*sorted), candidate_desc);
                memset(kept, 0, blocks);
                for (uint32_t j = 0; j < top; j++)
                    if (sorted[j].score > -INFINITY) kept[sorted[j].index] = 1;
                for (uint32_t i = 0; i < n; i++) {
                    float expected = i < visible && kept[i / 8] ? scores[(size_t)r * n + i] : -INFINITY;
                    CHECK(got[(size_t)r * n + i] == expected);
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, UINT32_MAX, 1));
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, 0, 0));
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const uint32_t rows = sizes[i], selected = rows < 512 ? rows : 512;
        ds4_gpu_tensor *source = upload(NULL, (uint64_t)rows * 512 * 4);
        ds4_gpu_tensor *ids = upload(NULL, selected * 4);
        ds4_gpu_tensor *out = upload(NULL, (uint64_t)selected * 512 * 4);
        CHECK(source && ids && out);
        float *input = ds4_gpu_tensor_contents(source);
        int32_t *indices = ds4_gpu_tensor_contents(ids);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
            input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
        for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
        CHECK(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
        const float *result = ds4_gpu_tensor_contents(out);
        for (uint32_t r = 0; r < selected; r++)
            CHECK(!memcmp(result + r * 512, input + (size_t)indices[r] * 512, 512 * 4));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
        ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    }
    fprintf(stderr, "V4.1 bounded sparse KV gather: exact\n");
    return 1;
}

static int check_attention_output(bool large) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120 };
    const uint32_t ROWS = large ? 8192u : 513u;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000; /* Exact 1/128 scale and binary-fraction inputs. */
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int)(random_value() * 8192) % 8;
    }
    float *x = malloc(xb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = (int)(random_value() * 8192) % 51 / 256.0f;
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(xr && lr && yr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(yr, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    const uint32_t sizes[] = {31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        const uint32_t rows = sizes[n];
        if (rows > ROWS) break;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
            0, a_bytes, xt, rows));
        const float *got = ds4_gpu_tensor_contents(out);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(!memcmp(got_low, reference_low, (uint64_t)rows * GROUPS * RANK * 4));
        for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
            if (!isfinite(got[i]) || fabsf(got[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i])))
                fprintf(stderr, "attention output rows=%u index=%llu actual=%.9g reference=%.9g\n",
                    rows, (unsigned long long)i, got[i], reference[i]);
            CHECK(isfinite(got[i]) && fabsf(got[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        }
        if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * OUT]));
        fprintf(stderr, "V4.1 batched Q8 output, BF16 boundary, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, xt, ROWS));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, xt, ROWS + 1));
    ds4_gpu_tensor *packed = upload(NULL, xb / 2);
    ds4_gpu_tensor *partial[2] = {upload(NULL, ob), upload(NULL, ob)};
    CHECK(packed && partial[0] && partial[1]);
    const uint32_t tp_sizes[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 512, 513};
    const uint32_t q_half = GROUPS / 2 * GROUP, low_half = GROUPS / 2 * RANK;
    for (size_t n = 0; n < sizeof(tp_sizes) / sizeof(*tp_sizes); n++) {
        const uint32_t rows = tp_sizes[n];
        for (uint32_t rank = 0; rank < 2; rank++) {
            float *px = ds4_gpu_tensor_contents(packed);
            for (uint32_t t = 0; t < rows; t++)
                memcpy(px + (size_t)t * q_half, x + (size_t)t * 2 * q_half + rank * q_half,
                       q_half * sizeof(float));
            CHECK(ds4_gpu_tensor_fill_f32(partial[rank], NAN, ob / 4));
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_tp_batch(partial[rank], low,
                model, a_bytes + b_bytes, 0, a_bytes, packed, rows, rank));
            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(partial[rank]);
            for (uint32_t t = 0; t < rows; t++) {
                CHECK(!memcmp(got_low + (size_t)t * low_half,
                    reference_low + (size_t)t * 2 * low_half + rank * low_half,
                    low_half * sizeof(float)));
                /* Independent double sums sample both ends and interior output
                 * rows, including rank one's strided input slice. */
                const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
                for (size_t c = 0; c < sizeof(columns) / sizeof(*columns); c++) {
                    const uint32_t col = columns[c];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * 2 * low_half / 32 + rank * low_half / 32;
                    double sum = 0;
                    double rounding_bound = 0;
#ifndef __APPLE__
                    /* CUDA's established Q8 dot first rounds each activation
                     * block to signed bytes. TP must keep that same boundary. */
                    for (uint32_t k = 0; k < low_half; k += 32) {
                        const float *v = reference_low + (size_t)t * 2 * low_half + rank * low_half + k;
                        float amax = 0;
                        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
                        const float scale = amax / 127.0f;
                        const float inv = scale ? 1.0f / scale : 0;
                        int dot = 0;
                        for (uint32_t j = 0; j < 32; j++) {
                            dot += bw[k / 32].qs[j] * (int)lrintf(v[j] * inv);
                            /* The fast GPU reciprocal can approach an exact
                             * half-integer from the other side. Bound only
                             * those ambiguous bins, not arbitrary dot error. */
                            const double q = amax ? fabs((double)v[j] * 127.0 / amax) : 0;
                            if (q - floor(q) == 0.5)
                                rounding_bound += abs(bw[k / 32].qs[j]) * (double)scale / 128.0;
                        }
                        sum += dot * (double)scale / 128.0;
                    }
#else
                    for (uint32_t k = 0; k < low_half; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            reference_low[(size_t)t * 2 * low_half + rank * low_half + k];
#endif
                    if (!isfinite(got[(size_t)t * OUT + col]) ||
                        fabs(got[(size_t)t * OUT + col] - sum) > rounding_bound + 2e-5 * (1 + fabs(sum)))
                        fprintf(stderr, "TP output rank=%u rows=%u row=%u col=%u actual=%.9g oracle=%.9g tie_bound=%.9g\n",
                            rank, rows, t, col, got[(size_t)t * OUT + col], sum, rounding_bound);
                    CHECK(isfinite(got[(size_t)t * OUT + col]) &&
                        fabs(got[(size_t)t * OUT + col] - sum) <= rounding_bound + 2e-5 * (1 + fabs(sum)));
                }
            }
            CHECK(isnan(got_low[(size_t)rows * low_half]));
            if (rows < ROWS) CHECK(isnan(got[(size_t)rows * OUT]));
        }
        const float *a = ds4_gpu_tensor_contents(partial[0]);
        const float *b = ds4_gpu_tensor_contents(partial[1]);
        for (size_t i = 0; i < (size_t)rows * OUT; i++)
            CHECK(isfinite(a[i]) && isfinite(b[i]) &&
                fabsf(a[i] + b[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        fprintf(stderr, "V4.1 TP Q8 output, both ranks, BF16 and double oracle, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, 1, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, packed, 1, 1));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, ROWS + 1, 0));
    ds4_gpu_tensor_free(partial[0]); ds4_gpu_tensor_free(partial[1]); ds4_gpu_tensor_free(packed);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}

static int check_indexer_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * sizeof(float));
    ds4_gpu_tensor *st = upload(NULL, (size_t)(ROWS * KEYS + 1) * sizeof(float));
    const uint64_t packed_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *pt = upload(NULL, (size_t)packed_bytes + 4);
    CHECK(qt && kt && wt && st && pt);
#ifndef __APPLE__
    ds4_gpu_tensor *reference = upload(NULL, (size_t)KEYS * sizeof(float));
    CHECK(reference);
#endif
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt);
    float *w = ds4_gpu_tensor_contents(wt), *s = ds4_gpu_tensor_contents(st);
    CHECK(q && k && w && s);
    const bool packed_available = ds4_gpu_dsv41_tensor_ops_available();
    for (uint32_t range = 0; range < 3; range++) {
        for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)KEYS * DIM; i++)
            k[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
        if (range == 2) { q[0] = 1.0001f; k[67u * DIM] = 1.0003f; }
        uint32_t *packed_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(pt) + packed_bytes);
        *packed_guard = 0xabcdef01;
        if (packed_available) CHECK(ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS));
        CHECK(*packed_guard == 0xabcdef01);
        const uint32_t counts[] = {1, 7, 8, 9, 31, 32, 33};
        for (uint32_t ratio = 1; ratio <= 2; ratio++) {
            for (uint32_t early = 0; early <= 1; early++) {
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    const uint32_t start = early ? 0 : KEYS * ratio - rows;
                    for (uint32_t mode = 0; mode < (packed_available ? 3u : 1u); mode++) {
                        const uint32_t offset = mode == 2 && rows < ROWS ? 1u : 0u;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, (uint64_t)offset * HEADS * DIM * 4u,
                            (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, (uint64_t)offset * HEADS * 4u,
                            (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        s[(size_t)rows * KEYS] = 12345;
                        CHECK(mode ? ds4_gpu_dsv41_indexer_scores_packed(st, qv, wv, kt, pt,
                            KEYS, rows, start, ratio, ROWS, offset) :
                            ds4_gpu_dsv41_indexer_scores_batch(st, qv, wv, kt, KEYS, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        CHECK(s[(size_t)rows * KEYS] == 12345);
                        CHECK(*packed_guard == 0xabcdef01);
#ifndef __APPLE__
                        /* Compare the CUDA warp scorer with the original
                         * shared-memory reduction, not only a loose oracle. */
                        ds4_gpu_set_quality(true);
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            if (!visible) continue;
                            ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qt,
                                (uint64_t)t * HEADS * DIM * 4u, HEADS * DIM * 4u);
                            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt,
                                (uint64_t)t * HEADS * 4u, HEADS * 4u);
                            CHECK(qr && wr && ds4_gpu_glm_indexer_score_one_tensor(
                                reference, qr, wr, kt, visible, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                            CHECK(!memcmp(s + (size_t)t * KEYS,
                                ds4_gpu_tensor_contents(reference), visible * sizeof(float)));
                            ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(wr);
                        }
                        ds4_gpu_set_quality(false);
#endif
                        double worst = 0;
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            for (uint32_t j = 0; j < KEYS; j++) {
                                const float actual = s[(size_t)t * KEYS + j];
                                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                                double expected = 0, magnitude = 0;
                                for (uint32_t h = 0; h < HEADS; h++) {
                                    double dot = 0;
                                    for (uint32_t d = 0; d < DIM; d++)
                                        dot += (double)q[((size_t)(t + offset) * HEADS + h) * DIM + d] *
                                               k[(size_t)j * DIM + d];
                                    const double term = fmax(dot / 64.0, 0) * w[(t + offset) * HEADS + h];
                                    expected += term;
                                    magnitude += fabs(term);
                                }
                                CHECK(isfinite(actual));
                                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                                worst = fmax(worst, error);
                                CHECK(error < 0.00001);
                            }
                        }
                        fprintf(stderr, "V4.1 index scores mode=%u range=%u ratio=%u start=%u rows=%u error=%.9g\n",
                                mode, range, ratio, start, rows, worst);
                    }
                }
            }
        }
    }
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, UINT32_MAX, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS + 1, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, KEYS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_pack(st, qt, kt, KEYS, ROWS));
    CHECK(!ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS + 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 0, ROWS, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, ROWS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, UINT32_MAX, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, st,
        KEYS, ROWS, 0, 1, ROWS, 0));
    ds4_gpu_tensor_free(pt);
#ifndef __APPLE__
    ds4_gpu_tensor_free(reference);
#endif
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 causal batched index scores: double-precision oracle PASS\n");
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int check_embedding(void) {
    enum { VOCAB = 19, WIDTH = 5120, HC = 4, ROWS = 257 };
    const size_t row_bytes = WIDTH * HC * sizeof(float);
    const size_t weight_bytes = VOCAB * WIDTH * sizeof(uint16_t);
    uint16_t *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), weight_bytes) == 0);
    for (unsigned i = 0; i < VOCAB * WIDTH; i++)
        model[i] = (uint16_t)(0x2000u | (i % 1024u) | ((i & 1u) ? 0x8000u : 0));
    int32_t ids[ROWS];
    for (unsigned i = 0; i < ROWS; i++) ids[i] = (int32_t)((i * 7u) % VOCAB);
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *tokens = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out = upload(NULL, ROWS * row_bytes + sizeof(float));
    ds4_gpu_tensor *ref = upload(NULL, VOCAB * row_bytes);
    CHECK(tokens && out && ref);
    for (unsigned token = 0; token < VOCAB; token++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(ref, token * row_bytes, row_bytes);
        CHECK(row && ds4_gpu_embed_token_hc_tensor(row, model, weight_bytes, 0,
                                                   VOCAB, token, WIDTH, HC));
        ds4_gpu_tensor_free(row);
    }
    const unsigned counts[] = {1, 7, 127, 128, ROWS};
    const float marker = 12345;
    float *actual = malloc(ROWS * row_bytes + sizeof(float));
    float *expected = malloc(VOCAB * row_bytes);
    CHECK(actual && expected && ds4_gpu_tensor_read(ref, 0, expected, VOCAB * row_bytes));
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        const unsigned rows = counts[i];
        CHECK(ds4_gpu_tensor_write(out, rows * row_bytes, &marker, sizeof(marker)));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, weight_bytes, 0,
                                             VOCAB, rows, WIDTH, HC));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, rows * row_bytes + sizeof(float)));
        CHECK(actual[rows * WIDTH * HC] == marker);
        for (unsigned row = 0; row < rows; row++) {
            CHECK(!memcmp(actual + row * WIDTH * HC, expected + ids[row] * WIDTH * HC, row_bytes));
            for (unsigned hc = 0; hc < HC; hc++) for (unsigned d = 0; d < WIDTH; d++) {
                const unsigned w = (unsigned)ids[row] * WIDTH + d;
                const float value = (1.0f + (w % 1024u) / 1024.0f) / 128.0f * ((w & 1u) ? -1 : 1);
                CHECK(actual[(row * HC + hc) * WIDTH + d] == value);
            }
        }
        fprintf(stderr, "V4.1 batched embedding rows=%u: exact scalar and independent oracle PASS\n", rows);
    }
    free(expected); free(actual);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup(); free(model);
    CHECK(ds4_gpu_init());
    return 1;
}

static int check_index_projection(void) {
    enum { MAX_ROWS = 4096 };
    const uint32_t widths[] = {1280, 5120, 5120, 512}, outputs[] = {4096, 32, 512, 128};
    const uint32_t counts[] = {1, 31, 64, 65, 513, 2048, MAX_ROWS};
    for (uint32_t shape = 0; shape < 4; shape++) {
        const uint32_t width = widths[shape], output = outputs[shape];
        const size_t weight_bytes = (size_t)width * output * sizeof(uint16_t);
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), weight_bytes) == 0);
        uint16_t *weights = model;
        for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
            const int value = (int)(random_value() * 8192);
            weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                    (value < 0 ? 0x8000u : 0));
        }
        CHECK(ds4_gpu_set_model_map(model, weight_bytes));
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci];
            y[(size_t)rows * output] = 12345;
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < rows; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr && ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands());
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, in));
            CHECK(ds4_gpu_synchronize());
            const double seconds = monotonic_seconds() - start;
            CHECK(y[(size_t)rows * output] == 12345);
            double error = 0, norm = 0;
            for (size_t i = 0; i < (size_t)rows * output; i++) {
                CHECK(isfinite(y[i]) && isfinite(expected[i]));
                const double d = (double)y[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 exact F16 projection width=%u output=%u rows=%u RMS=%.9g %.3f ms\n",
                width, output, rows, relative, seconds * 1000);
            CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t o = j * (output / 8), t = rows - 1;
                double sum = 0, magnitude = 0;
                for (uint32_t k = 0; k < width; k++) {
                    const uint16_t w = weights[(size_t)o * width + k];
                    const float value = (1.0f + (w & 1023) / 1024.0f) / 128.0f *
                                        (w & 0x8000 ? -1 : 1);
                    const double term = (double)value * x[(size_t)t * width + k];
                    sum += term; magnitude += fabs(term);
                }
                CHECK(fabs(y[(size_t)t * output + o] - sum) < 0.000001 * fmax(magnitude, 1));
            }
        }
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, 1, in));
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, MAX_ROWS + 1, in));
        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

typedef struct { float value; uint32_t id; } topk_entry;

static int compare_topk_entry(const void *a, const void *b) {
    const topk_entry *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int check_general_topk(void) {
    const uint32_t widths[] = {1, 2, 31, 127, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 3;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *selected = upload(NULL, ((size_t)width * rows + 1) * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && selected && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        uint32_t *ids = ds4_gpu_tensor_contents(selected);
        const uint32_t counts[] = {1, 2, 31, 128, 511, 512, 513, 2048, width};
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    const float value = pattern == 0 ? random_value() :
                        pattern == 1 ? (float)(i % 7) :
                        pattern == 2 ? -INFINITY : (i & 1 ? -0.0f : 0.0f);
                    const float v = pattern == 1 && i % 17 == 0 ? INFINITY : value;
                    s[t * width + i] = v;
                    reference[t * width + i] = (topk_entry){v, i};
                }
                qsort(reference + t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                if (k > width) continue;
                ids[rows * k] = 0xabcdef01;
                CHECK(ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, k));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * k] == 0xabcdef01);
                for (uint32_t t = 0; t < rows; t++) {
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const uint32_t id = ids[t * k + i];
                        CHECK(id < width && !seen[id]);
                        seen[id] = true;
#ifdef __APPLE__
                        /* Metal's bitonic order may permute exactly tied keys. */
                        CHECK(s[t * width + id] == reference[t * width + i].value);
#else
                        if (ids[t * k + i] != reference[t * width + i].id) {
                            fprintf(stderr, "top-k width=%u k=%u pattern=%u row=%u rank=%u: %u vs %u\n",
                                width, k, pattern, t, i, ids[t * k + i], reference[t * width + i].id);
                            CHECK(0);
                        }
#endif
                    }
                }
            }
        }
        fprintf(stderr, "V4.1 variable-k sort width=%u: independent sorted-value/unique-ID oracle PASS\n", width);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores); free(reference); free(seen);
    }
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {1024, 1025, 1535, 2047, 2048, 4095, 16383, 32767, 65535};
    const uint32_t counts[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
    for (size_t fi = 0; fi < sizeof(frontiers) / sizeof(*frontiers); fi++) {
        const uint32_t start = frontiers[fi] * ratio - 1u;
        const uint32_t width = (start + 129u) / ratio + 1024u;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * 129u * sizeof(float));
        ds4_gpu_tensor *selected = upload(NULL, (129u * 512u + 1u) * sizeof(int32_t));
        ds4_gpu_tensor *reference = upload(NULL, 129u * 512u * sizeof(int32_t));
        CHECK(scores && selected && reference);
        float *s = ds4_gpu_tensor_contents(scores);
        int32_t *ids = ds4_gpu_tensor_contents(selected);
        const int32_t *ref = ds4_gpu_tensor_contents(reference);
        for (uint32_t pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < 129u; t++) {
                const uint32_t visible = (start + t + 1u) / ratio;
                for (uint32_t j = 0; j < width; j++) {
                    /* Future entries are deliberately attractive. Causality
                     * must come from each row's bounds, not score contents. */
                    float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7u) : -INFINITY;
                    s[(size_t)t * width + j] = j < visible ? v : 12345.0f;
                }
            }
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci];
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t t = 0; t < rows; t++) {
                    const uint32_t visible = (start + t + 1u) / ratio;
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4u, visible * 4u);
                    ds4_gpu_tensor *out = ds4_gpu_tensor_view(reference, (size_t)t * 512u * 4u, 512u * 4u);
                    CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, 512));
                    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
                }
                CHECK(ds4_gpu_end_commands());
                ids[rows * 512u] = 123456;
                CHECK(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * 512u] == 123456);
                for (uint32_t t = 0; t < rows; t++) {
                    for (uint32_t k = 0; k < 512u; k++) {
                        if (ids[t * 512u + k] != ref[t * 512u + k]) {
                            fprintf(stderr, "top-k mismatch ratio=%u visible=%u rows=%u pattern=%u row=%u rank=%u: %d vs %d\n",
                                ratio, frontiers[fi], rows, pattern, t, k, ids[t * 512u + k], ref[t * 512u + k]);
                            CHECK(0);
                        }
                        CHECK(ids[t * 512u + k] >= 0 && (uint32_t)ids[t * 512u + k] < (start + t + 1u) / ratio);
                    }
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 130, start, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 0, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
        fprintf(stderr, "V4.1 causal top-k ratio=%u visible=%u: exact IDs including ties PASS\n", ratio, frontiers[fi]);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores);
    }
    }
    return 1;
}


static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t rows = 129, offset = 2;
    for (uint32_t format = 0; format <= DS4_V41_CARRY_MASK; format++) {
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
            const uint32_t width = widths[wi];
            const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
            const size_t count = (size_t)width * rows;
            const size_t packed_bytes = (size_t)(rows + 4) * words * sizeof(uint32_t);
            ds4_gpu_tensor *plain = upload(NULL, (count + 1) * sizeof(float));
            ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
            uint32_t *expected = malloc(count * sizeof(uint32_t));
            CHECK(plain && packed && expected);
            uint32_t *bits = ds4_gpu_tensor_contents(plain);
            unsigned char *storage = ds4_gpu_tensor_contents(packed);
            for (size_t i = 0; i < count; i++) {
                const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
                expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
            }
            memcpy(bits, expected, count * sizeof(uint32_t));
            bits[count] = 0x12345678u;
            memset(storage, 0xa5, packed_bytes);
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)));
            memset(bits, 0, count * sizeof(uint32_t));
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)) && bits[count] == 0x12345678u);
            for (size_t i = 0; i < (size_t)offset * words * 4u; i++) CHECK(storage[i] == 0xa5);
            for (size_t i = (size_t)(offset + rows) * words * 4u; i < packed_bytes; i++)
                CHECK(storage[i] == 0xa5);
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, 0, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
            fprintf(stderr, "V4.1 compact carry format=%u width=%u: exact round-trip and bounds PASS\n", format, width);
            free(expected);
            ds4_gpu_tensor_free(plain);
            ds4_gpu_tensor_free(packed);
        }
    }
    return 1;
}

/* Exercise the full-head and compact TP layouts with the same causal keys.
 * Selected rows are shuffled; include masked future rows at odd frontiers. */
static int check_tp_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    const uint32_t sizes[] = {1, 31, 32, 33, 129, 257, 2048};
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(*sizes); s++) {
        const uint32_t n = sizes[s], nr = n + 127, start = 3073;
        const uint32_t ratio = 1u + s % 2u;
        const size_t nq = (size_t)n * H * D;
        float *q = malloc(nq * 4), *actual = malloc(nq * 4);
        float *compact = malloc(nq * 2), *part = malloc(nq * 2);
        float *raw = malloc((size_t)nr * D * 4), *comp = malloc(C * D * 4);
        int32_t *ids = malloc((size_t)n * K * 4);
        CHECK(q && actual && compact && part && raw && comp && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                ids[t * K + j] = j % 29 ? (int32_t)((j * 127u + t * 17u) % C) : -1;
        ds4_gpu_tensor *qt = upload(q, nq * 4), *rt = upload(raw, (size_t)nr * D * 4);
        ds4_gpu_tensor *ct = upload(comp, C * D * 4), *it = upload(ids, (size_t)n * K * 4);
        ds4_gpu_tensor *out = upload(NULL, nq * 4), *qp = upload(NULL, nq * 2);
        ds4_gpu_tensor *op = upload(NULL, nq * 2);
        CHECK(qt && rt && ct && it && out && qp && op);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, nq * 4));
        double max_split = 0, max_oracle = 0;
        for (uint32_t rank = 0; rank < 2; rank++) {
            for (uint32_t t = 0; t < n; t++)
                memcpy(compact + (size_t)t * H/2 * D,
                    q + ((size_t)t * H + rank * H/2) * D, H/2 * D * 4);
            CHECK(ds4_gpu_tensor_write(qp, 0, compact, nq * 2));
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
            CHECK(ds4_gpu_tensor_read(op, 0, part, nq * 2));
#ifdef __APPLE__
            if (n > 1) {
                ds4_gpu_set_quality(true); /* Existing eight-head kernel. */
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                    rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
                CHECK(!memcmp(part, ds4_gpu_tensor_contents(op), nq * 2));
                ds4_gpu_set_quality(false);
            }
#endif
            if (n == 2048 && rank == 0) {
                const bool controls[] = {true, false, false, true};
                for (unsigned pass = 0; pass < 4; pass++) {
                    ds4_gpu_set_quality(controls[pass]);
                    const double begin = monotonic_seconds();
                    for (int repeat = 0; repeat < 4; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op,
                            sinks, getpagesize(), 0, qp, rt, ct, 0, it, n, start,
                            nr, nr, 0, C, K, 128, ratio, H/2, D));
                    CHECK(ds4_gpu_synchronize());
                    fprintf(stderr, "TP indexed attention control=%u %.3f ms\n",
                        controls[pass], (monotonic_seconds() - begin) * 250);
                }
                ds4_gpu_set_quality(false);
            }
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H/2; h++) {
                    const float *got = part + ((size_t)t * H/2 + h) * D;
                    const float *ref = actual + ((size_t)t * H + rank * H/2 + h) * D;
                    for (int d = 0; d < D; d++) {
                        const double err = fabs((double)got[d] - ref[d]);
                        max_split = fmax(max_split, err);
                        CHECK(isfinite(got[d]) && err < 3e-5 * (1 + fabs(ref[d])));
                    }
                    if ((t != 0 && t != n/2 && t + 1 != n) || h % 15) continue;
                    const float *query = compact + ((size_t)t * H/2 + h) * D;
                    double logits[128 + K + 1], values[128 + K + 1];
                    const uint32_t col = (t * 71u + h * 19u) % D;
                    unsigned count = 0;
                    for (uint32_t j = 0; j < 128 + K; j++) {
                        const int32_t id = j < 128 ? (int32_t)(t + j) : ids[t * K + j - 128];
                        if (j >= 128 && (uint32_t)id >= (start + t + 1) / ratio) continue;
                        const float *key = (j < 128 ? raw : comp) + (size_t)id * D;
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += (double)query[d] * key[d];
                        logits[count] = dot / sqrt(512.0);
                        values[count++] = key[col];
                    }
                    logits[count] = sinks[rank * H/2 + h]; values[count++] = 0;
                    double max = -INFINITY, den = 0, num = 0;
                    for (unsigned j = 0; j < count; j++) max = fmax(max, logits[j]);
                    for (unsigned j = 0; j < count; j++) {
                        const double p = exp(logits[j] - max);
                        den += p; num += p * values[j];
                    }
                    const double err = fabs(got[col] - num / den);
                    max_oracle = fmax(max_oracle, err);
                    CHECK(err < 3e-5 * (1 + fabs(num / den)));
                }
            }
        }
        fprintf(stderr, "V4.1 TP attention rows=%u ratio=%u max split=%g oracle=%g: PASS\n",
            n, ratio, max_split, max_oracle);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(ct);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(qp);
        ds4_gpu_tensor_free(op);
        free(q); free(actual); free(compact); free(part); free(raw); free(comp); free(ids);
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}

#ifdef __APPLE__
/* The BF16 epilogues must equal producer + kernel_dsv41_bf16_linear bit for bit
 * on ordinary random data (the rounding is the same integer op on the same
 * fp32 value) — for the Q8_0 matvec and the weighted RMS norm (2026-09-17). */
static int check_fused_bf16(void) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const struct { uint32_t in, out; } shapes[] = {{5120, 512}, {5120, 1280}, {2304, 5120}, {5120, 2304}, {1280, 32768}};
    uint64_t max_bytes = 0, max_in = 0, max_out = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const uint64_t b = (uint64_t)shapes[s].out * shapes[s].in / 32 * sizeof(q8_block);
        if (b > max_bytes) max_bytes = b;
        if (shapes[s].in > max_in) max_in = shapes[s].in;
        if (shapes[s].out > max_out) max_out = shapes[s].out;
    }
    const uint64_t norm_off = max_bytes, model_bytes = max_bytes + 5120 * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), model_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < max_bytes / sizeof(*w); i++) {
        w[i].d = (uint16_t)(0x1800u + (uint32_t)(random_value() * 2048));
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 255) - 127);
    }
    float *nw = (float *)((uint8_t *)model + norm_off);
    for (int i = 0; i < 5120; i++) nw[i] = random_value() * 2 - 1;
    float *xh = malloc(max_in * 4);
    CHECK(xh);
    for (uint64_t i = 0; i < max_in; i++) xh[i] = random_value() * 2 - 1;
    ds4_gpu_tensor *xt = upload(xh, max_in * 4), *a = upload(NULL, max_out * 4), *b = upload(NULL, max_out * 4);
    ds4_gpu_tensor *c = upload(NULL, max_out * 4);
    CHECK(xt && a && b && c && ds4_gpu_set_model_map(model, model_bytes));
    unsetenv("DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE");
    /* The Q8_0 matvec is the one producer whose fold is chosen at run time
     * (ds4_gpu_matmul_q8_0_tensor_bf16 -> g_pre_m5_bf16_epilogue).  Off pre-M5
     * Apple silicon that wrapper's fallback IS matvec + kernel_dsv41_bf16_linear,
     * i.e. exactly what the reference side below computes, so the comparison
     * would compare a value with itself: say so and skip rather than report a
     * tautology as a proof.  The other producers dispatch their fused pipeline
     * unconditionally and are checked on every device. */
    const int fused_matvec = ds4_gpu_device_is_pre_m5_apple_silicon() != 0;
    if (!fused_matvec)
        fprintf(stderr, "Q8 matvec + BF16 epilogue: SKIPPED (not pre-M5 Apple silicon, so the wrapper "
                        "takes the separate-rounding fallback and the comparison would be a tautology)\n");
    for (size_t s = 0; fused_matvec && s < sizeof(shapes) / sizeof(*shapes); s++) {
        const uint32_t in = shapes[s].in, outn = shapes[s].out;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(xt, 0, (uint64_t)in * 4);
        ds4_gpu_tensor *av = ds4_gpu_tensor_view(a, 0, (uint64_t)outn * 4);
        ds4_gpu_tensor *bv = ds4_gpu_tensor_view(b, 0, (uint64_t)outn * 4);
        ds4_gpu_tensor *cv = ds4_gpu_tensor_view(c, 0, (uint64_t)outn * 4);
        CHECK(xv && av && bv && cv);
        CHECK(ds4_gpu_tensor_fill_f32(av, NAN, outn) && ds4_gpu_tensor_fill_f32(bv, NAN, outn) &&
              ds4_gpu_tensor_fill_f32(cv, NAN, outn));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_matmul_q8_0_tensor(av, model, model_bytes, 0, in, outn, xv, 1));
        CHECK(ds4_gpu_dsv41_quantize(av, outn, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(bv, model, model_bytes, 0, in, outn, xv, 1));
        CHECK(ds4_gpu_end_commands());
        /* ds4_gpu_begin_commands() snapshots the rollback env, so set it before
         * this one: the documented escape hatch must route the same wrapper
         * through producer + rounding kernel and land on the very same bits. */
        CHECK(setenv("DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE", "1", 1) == 0);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(cv, model, model_bytes, 0, in, outn, xv, 1));
        CHECK(ds4_gpu_end_commands());
        unsetenv("DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE");
        const float *pa = ds4_gpu_tensor_contents(av), *pb = ds4_gpu_tensor_contents(bv);
        const float *pc = ds4_gpu_tensor_contents(cv);
        for (uint32_t r = 0; r < outn; r++) {
            if (memcmp(&pa[r], &pb[r], 4))
                fprintf(stderr, "fused bf16 matvec in=%u out=%u row=%u separate=%.9g fused=%.9g\n", in, outn, r, pa[r], pb[r]);
            CHECK(!memcmp(&pa[r], &pb[r], 4));
            if (memcmp(&pb[r], &pc[r], 4))
                fprintf(stderr, "bf16 rollback matvec in=%u out=%u row=%u fused=%.9g disabled=%.9g\n", in, outn, r, pb[r], pc[r]);
            CHECK(!memcmp(&pb[r], &pc[r], 4));
        }
        fprintf(stderr, "Q8 matvec + BF16 epilogue %5u x %6u: bit-identical (fused, and with the rollback env set)\n", in, outn);
        ds4_gpu_tensor_free(xv); ds4_gpu_tensor_free(av); ds4_gpu_tensor_free(bv); ds4_gpu_tensor_free(cv);
    }
    {
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(xt, 0, 5120 * 4);
        ds4_gpu_tensor *av = ds4_gpu_tensor_view(a, 0, 5120 * 4), *bv = ds4_gpu_tensor_view(b, 0, 5120 * 4);
        CHECK(xv && av && bv);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_rms_norm_weight_tensor(av, xv, model, model_bytes, norm_off, 5120, 1e-6f));
        CHECK(ds4_gpu_dsv41_quantize(av, 5120, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_rms_norm_weight_bf16_tensor(bv, xv, model, model_bytes, norm_off, 5120, 1e-6f));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(av), ds4_gpu_tensor_contents(bv), 5120 * 4));
        fprintf(stderr, "weighted RMS norm + BF16 epilogue 5120: bit-identical\n");
        ds4_gpu_tensor_free(xv); ds4_gpu_tensor_free(av); ds4_gpu_tensor_free(bv);
    }
    {   /* elementwise producers: SwiGLU, add, HC weighted sums, HC expand4; and the attention low projection */
        enum { E = 5120, H = 4, FF = 2304, GROUPS = 8, RANK = 1024, GDIM = 4096 };
        float *hbuf = malloc((size_t)GROUPS * GDIM * 4);
        CHECK(hbuf);
        for (int i = 0; i < GROUPS * GDIM; i++) hbuf[i] = random_value() * 2 - 1;
        ds4_gpu_tensor *gate = upload(xh, FF * 4), *up = upload(xh + FF, FF * 4);
        ds4_gpu_tensor *res = upload(NULL, E * H * 4), *blk = upload(xh, E * 4), *w4 = upload(NULL, 24 * 4);
        ds4_gpu_tensor *oa = upload(NULL, E * H * 4), *ob = upload(NULL, E * H * 4);
        ds4_gpu_tensor *heads = upload(hbuf, (size_t)GROUPS * GDIM * 4);
        ds4_gpu_tensor *la = upload(NULL, GROUPS * RANK * 4), *lb = upload(NULL, GROUPS * RANK * 4);
        CHECK(gate && up && res && blk && w4 && oa && ob && heads && la && lb);
        float *pr = ds4_gpu_tensor_contents(res), *pw = ds4_gpu_tensor_contents(w4);
        for (int i = 0; i < E * H; i++) pr[i] = random_value() * 2 - 1;
        for (int i = 0; i < 24; i++) pw[i] = random_value() * 2 - 1;
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_swiglu_tensor(oa, gate, up, FF, 7.0f, 1.0f) && ds4_gpu_dsv41_quantize(oa, FF, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_swiglu_bf16_tensor(ob, gate, up, FF, 7.0f, 1.0f));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa), ds4_gpu_tensor_contents(ob), FF * 4));
        fprintf(stderr, "SwiGLU + BF16 epilogue: bit-identical\n");
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_add_tensor(oa, blk, xt, E) && ds4_gpu_dsv41_quantize(oa, E, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_add_bf16_tensor(ob, blk, xt, E));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa), ds4_gpu_tensor_contents(ob), E * 4));
        fprintf(stderr, "add + BF16 epilogue: bit-identical\n");
        /* The weighted sums infer the token count from the output tensor: one-row views. */
        ds4_gpu_tensor *oa1 = ds4_gpu_tensor_view(oa, 0, E * 4), *ob1 = ds4_gpu_tensor_view(ob, 0, E * 4);
        CHECK(oa1 && ob1);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hc_weighted_sum_tensor(oa1, res, w4, E, H) && ds4_gpu_dsv41_quantize(oa1, E, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_hc_weighted_sum_bf16_tensor(ob1, res, w4, E, H));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa1), ds4_gpu_tensor_contents(ob1), E * 4));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hc_weighted_sum_split_tensor(oa1, res, w4, E, H) && ds4_gpu_dsv41_quantize(oa1, E, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_hc_weighted_sum_split_bf16_tensor(ob1, res, w4, E, H));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa1), ds4_gpu_tensor_contents(ob1), E * 4));
        /* 3.5(a) pre-copy removal: `pre` is a copy of the first n_hc floats of the previous
         * sublayer's ffn_split, and at one token the split variant never reads its row stride,
         * so reading ffn_split directly is the same dispatch on the same four floats. */
        ds4_gpu_tensor *pre4 = upload(NULL, H * 4);
        CHECK(pre4);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_tensor_copy(pre4, 0, w4, 0, H * 4));
        CHECK(ds4_gpu_hc_weighted_sum_tensor(oa1, res, pre4, E, H));
        CHECK(ds4_gpu_hc_weighted_sum_split_tensor(ob1, res, w4, E, H));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa1), ds4_gpu_tensor_contents(ob1), E * 4));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hc_weighted_sum_bf16_tensor(oa1, res, pre4, E, H));
        CHECK(ds4_gpu_hc_weighted_sum_split_bf16_tensor(ob1, res, w4, E, H));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa1), ds4_gpu_tensor_contents(ob1), E * 4));
        ds4_gpu_tensor_free(pre4);
        fprintf(stderr, "HC weighted sum from ffn_split == from the copied pre row: bit-identical\n");
        ds4_gpu_tensor_free(oa1); ds4_gpu_tensor_free(ob1);
        fprintf(stderr, "HC weighted sums + BF16 epilogue: bit-identical\n");
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hc_expand_split_tensor(oa, blk, res, w4, E, H) && ds4_gpu_dsv41_quantize(oa, E * H, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_hc_expand_split_bf16_tensor(ob, blk, res, w4, E, H));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(oa), ds4_gpu_tensor_contents(ob), E * H * 4));
        fprintf(stderr, "HC expand4 + BF16 epilogue: bit-identical\n");
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_attention_output_low_q8_tensor(la, model, model_bytes, 0, GDIM, RANK, GROUPS, heads) &&
              ds4_gpu_dsv41_quantize(la, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_attention_output_low_q8_bf16_tensor(lb, model, model_bytes, 0, GDIM, RANK, GROUPS, heads));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(la), ds4_gpu_tensor_contents(lb), GROUPS * RANK * 4));
        fprintf(stderr, "attention low projection + BF16 epilogue: bit-identical\n");
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(res); ds4_gpu_tensor_free(blk);
        ds4_gpu_tensor_free(w4); ds4_gpu_tensor_free(oa); ds4_gpu_tensor_free(ob); ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(la); ds4_gpu_tensor_free(lb); free(hbuf);
    }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(c); free(xh);
    fprintf(stderr, "PRE_M5 BF16 epilogues (Q8 matvec%s, weighted norm, elementwise producers, attention low): PASS\n",
            fused_matvec ? " + rollback" : " SKIPPED");
    return 1;
}
#endif

int main(int argc, char **argv) {
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--fused-bf16")) {
        const int ok = ds4_gpu_init() && check_fused_bf16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-pair")) {
        const int ok = ds4_gpu_init() && check_rope_pair();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--quantize-store")) {
        const int ok = ds4_gpu_init() && check_quantize_store();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-digest")) {
        const int ok = ds4_gpu_init() && check_rope_digest();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-freqs")) {
        const int ok = ds4_gpu_init() && check_rope_freqs();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fusion-gates")) {
        /* ds4_gpu_init() first: the gates start with the Metal device name. */
        const int ok = ds4_gpu_init() && check_fusion_gates();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--embedding")) {
        const int ok = ds4_gpu_init() && check_embedding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        const int ok = ds4_gpu_init() && check_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--tp-attention")) {
        const int ok = ds4_gpu_init() && check_tp_attention();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && (!strcmp(argv[1], "--attention-output") ||
                      !strcmp(argv[1], "--attention-output-large"))) {
        const int ok = ds4_gpu_init() && check_attention_output(
            !strcmp(argv[1], "--attention-output-large"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--hc-scaled")) {
        const int ok = ds4_gpu_init() && check_hc_scaled();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--bf16-linear")) {
        const int ok = ds4_gpu_init() && check_bf16_linear();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compact-carry")) {
        const int ok = ds4_gpu_init() && check_compact_carry();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-scores")) {
        const int ok = ds4_gpu_init() && check_indexer_batch();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-topk")) {
        const int ok = ds4_gpu_init() && check_causal_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--general-topk")) {
        const int ok = ds4_gpu_init() && check_general_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-projection")) {
        const int ok = ds4_gpu_init() && check_index_projection();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    int ok = ds4_gpu_init() && check_router() && check_quantization() && check_engram() && check_rope_stride() && check_pool() &&
             check_candidates() && check_sparse_gather() && check_indexer_batch() &&
             check_embedding() && check_index_projection() && check_general_topk() && check_causal_topk() && check_compact_carry() && check_attention_output(false) &&
             check_tp_attention();
#ifdef __APPLE__
    if (ok) ok = check_rope_pair() && check_quantize_store() &&
                 check_rope_freqs() && check_fusion_gates();
#endif
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
