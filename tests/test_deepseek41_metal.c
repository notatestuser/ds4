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

/* The rollback switches are read by file-static helpers in ds4.c, which no test can reach
 * by name; the fused entry points above are called directly and are deliberately unaware of them.
 * ds4_v41_decode_fusion_gates() is the oracle: assert that each env name, spelled as ds4.c spells
 * it, disables exactly its own fusion and nothing else. A misspelt name would otherwise survive
 * every check - including the strict bench, which would then A/B the fused path against itself.
 * 3.6d (2026-09-17): the batched attention-output rollback is bit 8. It gates a ds4.c-side branch
 * rather than a kernel choice, so this oracle is the only model-free proof that its name is the
 * one the A/B harness must export; without it a typo would leave every sweep comparing the new
 * path with itself and reporting the stage clean and free.
 * 3.6d2 (2026-09-17): bit 16 is the rows<R> kernels and bit 32 the late Engram join. Both are
 * ds4.c-side branches for the same reason, and the join has no kernel at all -- a typo in its name
 * would make its A/B a pure noise measurement that nothing else could detect. */
static int check_fusion_gates(void) {
    static const char *const names[8] = {
        "DS4_METAL_DISABLE_PRE_M5_V41_PRE_COPY",
        "DS4_METAL_DISABLE_PRE_M5_V41_ROPE_PAIR",
        "DS4_METAL_DISABLE_PRE_M5_V41_QUANTIZE_STORE",
        "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT",
        "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_ROWS",
        "DS4_METAL_DISABLE_PRE_M5_V41_LATE_ENGRAM_JOIN",
        /* 3.6c (2026-09-17): bit 64 is the batched flash stage.  Same reason as bits 8/16: it
         * gates a ds4.c-side branch no kernel test can reach, so this is the only model-free proof
         * that the name the A/B harness exports is the name ds4.c reads. */
        "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_FLASH",
        /* 3.6a (2026-09-17): bit 128 is the batched RoPE/quantize/window/publish stage. Same
         * reason as bits 8/16/32/64: it gates a ds4.c-side branch no kernel test can reach, so
         * this is the only model-free proof that the name the A/B harness exports is the name
         * ds4.c reads. */
        "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_ROWS_A",
    };
    /* ds4_gpu_init() must have run: the pre-M5 test reads the Metal device name, and without it
     * every gate reads false and this whole check would pass vacuously. */
    const int pre_m5 = ds4_gpu_device_is_pre_m5_apple_silicon();
    const int base = ds4_v41_decode_fusion_gates();
    printf("fusion_gates=%d\n", base);
    CHECK(pre_m5 || base == 0);          /* nothing fuses off pre-M5 Apple silicon */
    for (int i = 0; i < (int)(sizeof(names) / sizeof(*names)); i++) {
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
    fprintf(stderr, "V4.1 decode fusion gates: pre_m5=%d live=0x%x, each of the %zu rollback envs "
                    "disables exactly its own fusion PASS\n", pre_m5, base,
            sizeof(names) / sizeof(*names));
    return 1;
}

/* 3.6a (2026-09-17): exactness witness for the hoisted quantizer body. patch3_6a moves the
 * per-element code of kernel_dsv41_quantize and kernel_dsv41_quantize_store2 into
 * dsv41_quantize_value() so the rows kernel shares it. Both of those are on the DEFAULT
 * single-token path, and the strict schedule bench cannot see a move that affects its control and
 * its candidate equally -- it A/Bs one binary against itself. This is the instrument that can:
 * an FNV-1a digest over a fixed matrix of modes and magnitudes, run once with
 * DS4_METAL_DSV41_SOURCE pointing at the pre-patch metal/dsv41.metal and once without. The two
 * digests must be equal. (Mode 0 of ds4_gpu_dsv41_quantize takes kernel_dsv41_bf16_linear, which
 * this patch does not touch; the store variant below covers mode 0 of the refactored body.) */
static int check_quantize_digest(void) {
    enum { WIDTH = 512, ROWS = 9, N = WIDTH * ROWS, SLOT = 3, SLOTS = 16, WN = SLOTS * WIDTH };
    ds4_gpu_tensor *a = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *w = upload(NULL, WN * sizeof(float));
    CHECK(a && w);
    float *x = ds4_gpu_tensor_contents(a), *window = ds4_gpu_tensor_contents(w);
    uint64_t digest = 14695981039346656037ull;
    seed = 7919;
    for (int mode = 0; mode < 4; mode++) {
        for (int magnitude = 0; magnitude < 3; magnitude++) {
            for (int store = 0; store < 2; store++) {
                for (size_t j = 0; j < N; j++)
                    x[j] = random_value() * (float)(1u << (magnitude * 5));
                for (size_t j = 0; j < WN; j++) window[j] = (float)(j % 251) - 125.0f;
                CHECK(ds4_gpu_begin_commands());
                CHECK(store ?
                    ds4_gpu_dsv41_quantize_store(a, WIDTH, ROWS, (ds4_v41_activation_format)mode,
                                                 w, (uint64_t)SLOT * WIDTH * 4u) :
                    ds4_gpu_dsv41_quantize(a, WIDTH, ROWS, (ds4_v41_activation_format)mode));
                CHECK(ds4_gpu_end_commands());
                const unsigned char *bytes = (const unsigned char *)x;
                for (size_t j = 0; j < N * sizeof(float); j++) {
                    digest ^= bytes[j];
                    digest *= 1099511628211ull;
                }
            }
        }
    }
    ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(a);
    printf("quantize_digest=%016llx\n", (unsigned long long)digest);
    fprintf(stderr, "V4.1 quantize digest over 24 fixed dispatches: %016llx\n",
            (unsigned long long)digest);
    return 1;
}

/* 3.6a: the four rows kernels against the per-row dispatch sequences they replace.
 *
 * This is the memcmp half of the exactness claim: at N in {1,2,4,8}, with the rows at different
 * absolute positions, one rows dispatch must produce byte for byte what N single-row dispatches
 * produce, and must touch nothing else -- not another row's slice, not another slot of a window,
 * not the cache of a row that is not publishing. The row table is built here exactly as
 * ds41_attention_rows builds it, so a wrong stride, a wrong slot or a missing useResource shows up
 * as a diff (or, under MTL_SHADER_VALIDATION=1, as a fault). */
static int check_rows_a(void) {
    enum { WIDTH = 512, KEY = 128, HEADS = 4, MAX_ROWS = 8, SLOTS = 128, CACHE = 256 };
    /* Positions chosen so the rows differ in every way the kernels can see: slot (pos % 128),
     * parity (the pool2 branch), and n_comp. The largest is the last position the RoPE accepts. */
    const uint32_t positions[MAX_ROWS] = {0, 1, 127, 128, 129, 254, 255, 1048575};
    const uint32_t counts[] = {1, 2, 4, 8};
    ds4_gpu_tensor *ra = upload(NULL, ((size_t)MAX_ROWS * HEADS * WIDTH + 1) * sizeof(float));
    ds4_gpu_tensor *rb = upload(NULL, ((size_t)MAX_ROWS * HEADS * WIDTH + 1) * sizeof(float));
    CHECK(ra && rb);
    float *xa = ds4_gpu_tensor_contents(ra), *xb = ds4_gpu_tensor_contents(rb);

    /* (1) RoPE. Both frequency tables, both directions, the head counts the decode step uses. */
    const uint32_t head_counts[] = {1, HEADS};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++)
    for (size_t hi = 0; hi < sizeof(head_counts) / sizeof(*head_counts); hi++)
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t heads = head_counts[hi], rows = counts[ci];
        const size_t n = (size_t)rows * heads * WIDTH;
        for (size_t j = 0; j < n; j++) xa[j] = xb[j] = bf16(random_value());
        xa[n] = xb[n] = 12345;
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope_rows(ra, WIDTH, heads, rows, positions, kind, inverse));
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *v = ds4_gpu_tensor_view(rb, (uint64_t)r * heads * WIDTH * 4u,
                                                    (uint64_t)heads * WIDTH * 4u);
            CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, heads, 1, positions[r], kind, inverse));
            ds4_gpu_tensor_free(v);
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(xa, xb, (n + 1) * sizeof(float)) && xa[n] == 12345);
    }
    CHECK(!ds4_gpu_dsv41_rope_rows(ra, WIDTH, HEADS, 0, positions, true, false));
    CHECK(!ds4_gpu_dsv41_rope_rows(ra, WIDTH, HEADS, MAX_ROWS + 1u, positions, true, false));
    CHECK(!ds4_gpu_dsv41_rope_rows(ra, WIDTH, HEADS, 1, NULL, true, false));
    CHECK(!ds4_gpu_dsv41_rope_rows(ra, 32, HEADS, 1, positions, true, false));
    CHECK(!ds4_gpu_dsv41_rope_rows(NULL, WIDTH, HEADS, 1, positions, true, false));
    {   const uint32_t too_far[MAX_ROWS] = {1048576, 0, 0, 0, 0, 0, 0, 0};
        CHECK(!ds4_gpu_dsv41_rope_rows(ra, WIDTH, HEADS, 1, too_far, true, false)); }
    fprintf(stderr, "V4.1 rows RoPE: bit-identical to per-row dispatches at N=1,2,4,8, guards PASS\n");

    /* (2) Quantize + window scatter. Each row scatters into its own session's window, so both the
     * slot it writes and every slot it must not write are compared over the whole buffer. */
    ds4_gpu_tensor *window_a[MAX_ROWS] = {0}, *window_b[MAX_ROWS] = {0};
    for (uint32_t r = 0; r < MAX_ROWS; r++) {
        window_a[r] = upload(NULL, (size_t)SLOTS * WIDTH * sizeof(float));
        window_b[r] = upload(NULL, (size_t)SLOTS * WIDTH * sizeof(float));
        CHECK(window_a[r] && window_b[r]);
    }
    for (int mode = 0; mode < 4; mode++)
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t rows = counts[ci];
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        const size_t n = (size_t)rows * WIDTH;
        for (size_t j = 0; j < n; j++)
            xa[j] = xb[j] = random_value() * (float)(1u << ((j / block) % 4));
        for (uint32_t r = 0; r < rows; r++) {
            float *pa = ds4_gpu_tensor_contents(window_a[r]);
            float *pb = ds4_gpu_tensor_contents(window_b[r]);
            CHECK(pa && pb);
            for (size_t j = 0; j < (size_t)SLOTS * WIDTH; j++)
                pa[j] = pb[j] = (float)((j + r) % 251) - 125.0f;
        }
        CHECK(ds4_gpu_v41_rows_begin(rows));
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_v41_row row = {0};
            row.window = window_a[r];
            row.pos = positions[r];
            CHECK(ds4_gpu_v41_rows_set(r, &row));
        }
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, rows,
                                                 (ds4_v41_activation_format)mode, SLOTS));
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *v = ds4_gpu_tensor_view(rb, (uint64_t)r * WIDTH * 4u, WIDTH * 4u);
            CHECK(v && ds4_gpu_dsv41_quantize(v, WIDTH, 1, (ds4_v41_activation_format)mode) &&
                  ds4_gpu_tensor_copy(window_b[r], (uint64_t)(positions[r] % SLOTS) * WIDTH * 4u,
                                      v, 0, WIDTH * 4u));
            ds4_gpu_tensor_free(v);
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(xa, xb, n * sizeof(float)));
        for (uint32_t r = 0; r < rows; r++)
            CHECK(!memcmp(ds4_gpu_tensor_contents(window_a[r]),
                          ds4_gpu_tensor_contents(window_b[r]),
                          (size_t)SLOTS * WIDTH * sizeof(float)));
    }
    /* A half-built table must not reach a dispatch: begin(2) with only row 0 set is refused, and
     * so is a 3-row dispatch against a 2-row table -- otherwise the kernel would read a stale or
     * zero address and fault. */
    CHECK(ds4_gpu_v41_rows_begin(2));
    {   ds4_gpu_v41_row row = {0}; row.window = window_a[0]; row.pos = 1;
        CHECK(ds4_gpu_v41_rows_set(0, &row));
        CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 2, DS4_V41_FP8_E8M0, SLOTS));
        CHECK(ds4_gpu_v41_rows_set(1, &row));
        CHECK(!ds4_gpu_v41_rows_set(2, &row));
        CHECK(!ds4_gpu_v41_rows_set(0, NULL)); }
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 3, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 2, DS4_V41_FP8_E8M0, 0));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 2, (ds4_v41_activation_format)4, SLOTS));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, 24, 2, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, MAX_ROWS + 1u, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(NULL, WIDTH, 2, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(!ds4_gpu_v41_rows_begin(MAX_ROWS + 1u));
    CHECK(!ds4_gpu_v41_rows_begin(0));
    fprintf(stderr, "V4.1 rows quantize + window scatter: bit-identical to quantize + copy in all "
                    "four formats at N=1,2,4,8, guards PASS\n");

    /* (3) Pooling. Mixed odd/even positions in one batch: the odd rows take the pair branch and
     * read previous_*, the even rows take the carry branch and write it. */
    ds4_gpu_tensor *pool_a = upload(NULL, (size_t)MAX_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *pool_b = upload(NULL, (size_t)MAX_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *scores = upload(NULL, (size_t)MAX_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *prev_kv_a[MAX_ROWS] = {0}, *prev_kv_b[MAX_ROWS] = {0};
    ds4_gpu_tensor *prev_sc_a[MAX_ROWS] = {0}, *prev_sc_b[MAX_ROWS] = {0};
    CHECK(pool_a && pool_b && scores);
    for (uint32_t r = 0; r < MAX_ROWS; r++) {
        prev_kv_a[r] = upload(NULL, WIDTH * sizeof(float));
        prev_kv_b[r] = upload(NULL, WIDTH * sizeof(float));
        prev_sc_a[r] = upload(NULL, WIDTH * sizeof(float));
        prev_sc_b[r] = upload(NULL, WIDTH * sizeof(float));
        CHECK(prev_kv_a[r] && prev_kv_b[r] && prev_sc_a[r] && prev_sc_b[r]);
    }
    for (uint32_t ratio = 1; ratio <= 2; ratio++)
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t rows = counts[ci];
        const size_t n = (size_t)rows * WIDTH;
        float *kv = ds4_gpu_tensor_contents(ra), *sc = ds4_gpu_tensor_contents(scores);
        float *ga = ds4_gpu_tensor_contents(pool_a), *gb = ds4_gpu_tensor_contents(pool_b);
        CHECK(kv && sc && ga && gb);
        for (size_t j = 0; j < n; j++) { kv[j] = bf16(random_value()); sc[j] = random_value(); }
        /* An even row leaves `out` untouched in the per-row path and gets kv in the rows kernel;
         * seeding the reference with kv states that intent instead of hiding it. */
        for (size_t j = 0; j < n; j++) { ga[j] = -7.5f; gb[j] = kv[j]; }
        for (uint32_t r = 0; r < rows; r++) {
            float *ka = ds4_gpu_tensor_contents(prev_kv_a[r]);
            float *kb = ds4_gpu_tensor_contents(prev_kv_b[r]);
            float *sa = ds4_gpu_tensor_contents(prev_sc_a[r]);
            float *sb = ds4_gpu_tensor_contents(prev_sc_b[r]);
            CHECK(ka && kb && sa && sb);
            for (uint32_t j = 0; j < WIDTH; j++) {
                ka[j] = kb[j] = bf16(random_value());
                sa[j] = sb[j] = random_value();
            }
        }
        CHECK(ds4_gpu_v41_rows_begin(rows));
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_v41_row row = {0};
            row.previous_kv = ratio == 2u ? prev_kv_a[r] : NULL;
            row.previous_score = ratio == 2u ? prev_sc_a[r] : NULL;
            row.pos = positions[r];
            CHECK(ds4_gpu_v41_rows_set(r, &row));
        }
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, rows, ratio));
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(pool_b, (uint64_t)r * WIDTH * 4u, WIDTH * 4u);
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(ra, (uint64_t)r * WIDTH * 4u, WIDTH * 4u);
            ds4_gpu_tensor *v = ds4_gpu_tensor_view(scores, (uint64_t)r * WIDTH * 4u, WIDTH * 4u);
            CHECK(o && k && v);
            CHECK(ratio == 2u ?
                ds4_gpu_dsv41_pool2(o, k, v, prev_kv_b[r], prev_sc_b[r], WIDTH, 1, positions[r]) :
                ds4_gpu_tensor_copy(o, 0, k, 0, WIDTH * 4u));
            ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(o);
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ga, gb, n * sizeof(float)));
        for (uint32_t r = 0; ratio == 2u && r < rows; r++) {
            CHECK(!memcmp(ds4_gpu_tensor_contents(prev_kv_a[r]),
                          ds4_gpu_tensor_contents(prev_kv_b[r]), WIDTH * sizeof(float)));
            CHECK(!memcmp(ds4_gpu_tensor_contents(prev_sc_a[r]),
                          ds4_gpu_tensor_contents(prev_sc_b[r]), WIDTH * sizeof(float)));
        }
    }
    CHECK(ds4_gpu_v41_rows_begin(2));
    for (uint32_t r = 0; r < 2; r++) {
        ds4_gpu_v41_row row = {0};
        row.previous_kv = prev_kv_a[r];
        row.previous_score = prev_sc_a[r];
        row.pos = positions[r];
        CHECK(ds4_gpu_v41_rows_set(r, &row));
    }
    CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 2, 0));
    CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 2, 3));
    CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, MAX_ROWS + 1u, 2));
    CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 4, 2));   /* table holds 2 rows */
    CHECK(!ds4_gpu_dsv41_pool2_rows(NULL, ra, scores, WIDTH, 2, 2));
    fprintf(stderr, "V4.1 rows pooling: bit-identical to per-row pool2 at ratio 1 and 2 with mixed "
                    "parities, carry included, guards PASS\n");

    /* (4) Publish scatter. Only the publishing rows may write, and only at slot n_comp - 1. */
    ds4_gpu_tensor *keys = upload(NULL, (size_t)MAX_ROWS * KEY * sizeof(float));
    ds4_gpu_tensor *cache_k_a[MAX_ROWS] = {0}, *cache_k_b[MAX_ROWS] = {0};
    ds4_gpu_tensor *cache_v_a[MAX_ROWS] = {0}, *cache_v_b[MAX_ROWS] = {0};
    CHECK(keys);
    for (uint32_t r = 0; r < MAX_ROWS; r++) {
        cache_k_a[r] = upload(NULL, (size_t)CACHE * KEY * sizeof(float));
        cache_k_b[r] = upload(NULL, (size_t)CACHE * KEY * sizeof(float));
        cache_v_a[r] = upload(NULL, (size_t)CACHE * WIDTH * sizeof(float));
        cache_v_b[r] = upload(NULL, (size_t)CACHE * WIDTH * sizeof(float));
        CHECK(cache_k_a[r] && cache_k_b[r] && cache_v_a[r] && cache_v_b[r]);
    }
    /* Positions small enough that n_comp - 1 stays inside CACHE at ratio 1 (the worst case,
     * n_comp = pos + 1), and mixed so that at ratio 2 only the odd ones publish while the even
     * ones must leave their sentinel untouched. */
    const uint32_t publish_positions[MAX_ROWS] = {1, 2, 3, 20, 21, 100, 101, 190};
    for (uint32_t ratio = 1; ratio <= 2; ratio++)
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t rows = counts[ci];
        float *k = ds4_gpu_tensor_contents(keys), *v = ds4_gpu_tensor_contents(ra);
        CHECK(k && v);
        for (size_t j = 0; j < (size_t)rows * KEY; j++) k[j] = bf16(random_value());
        for (size_t j = 0; j < (size_t)rows * WIDTH; j++) v[j] = bf16(random_value());
        for (uint32_t r = 0; r < rows; r++) {
            float *ka = ds4_gpu_tensor_contents(cache_k_a[r]);
            float *kb = ds4_gpu_tensor_contents(cache_k_b[r]);
            float *va = ds4_gpu_tensor_contents(cache_v_a[r]);
            float *vb = ds4_gpu_tensor_contents(cache_v_b[r]);
            CHECK(ka && kb && va && vb);
            for (size_t j = 0; j < (size_t)CACHE * KEY; j++) ka[j] = kb[j] = (float)(j % 97) - 48.0f;
            for (size_t j = 0; j < (size_t)CACHE * WIDTH; j++) va[j] = vb[j] = (float)(j % 89) - 44.0f;
        }
        CHECK(ds4_gpu_v41_rows_begin(rows));
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t pos = publish_positions[r];
            ds4_gpu_v41_row row = {0};
            row.compressed = cache_v_a[r];
            row.index_cache = cache_k_a[r];
            row.pos = pos;
            row.n_comp = (pos + 1u) / ratio;
            row.publish = (pos + 1u) % ratio == 0u;
            CHECK(ds4_gpu_v41_rows_set(r, &row));
        }
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, rows));
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t pos = publish_positions[r], n_comp = (pos + 1u) / ratio;
            if ((pos + 1u) % ratio) continue;
            ds4_gpu_tensor *kr = ds4_gpu_tensor_view(keys, (uint64_t)r * KEY * 4u, KEY * 4u);
            ds4_gpu_tensor *vr = ds4_gpu_tensor_view(ra, (uint64_t)r * WIDTH * 4u, WIDTH * 4u);
            CHECK(kr && vr);
            CHECK(ds4_gpu_tensor_copy(cache_k_b[r], (uint64_t)(n_comp - 1u) * KEY * 4u, kr, 0, KEY * 4u));
            CHECK(ds4_gpu_tensor_copy(cache_v_b[r], (uint64_t)(n_comp - 1u) * WIDTH * 4u, vr, 0, WIDTH * 4u));
            ds4_gpu_tensor_free(vr); ds4_gpu_tensor_free(kr);
        }
        CHECK(ds4_gpu_end_commands());
        for (uint32_t r = 0; r < rows; r++) {
            CHECK(!memcmp(ds4_gpu_tensor_contents(cache_k_a[r]),
                          ds4_gpu_tensor_contents(cache_k_b[r]), (size_t)CACHE * KEY * sizeof(float)));
            CHECK(!memcmp(ds4_gpu_tensor_contents(cache_v_a[r]),
                          ds4_gpu_tensor_contents(cache_v_b[r]), (size_t)CACHE * WIDTH * sizeof(float)));
        }
    }
    CHECK(ds4_gpu_v41_rows_begin(2));
    for (uint32_t r = 0; r < 2; r++) {
        ds4_gpu_v41_row row = {0};
        row.compressed = cache_v_a[r];
        row.index_cache = cache_k_a[r];
        row.pos = 1; row.n_comp = 1; row.publish = 0;
        CHECK(ds4_gpu_v41_rows_set(r, &row));
    }
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, WIDTH, KEY, 2));
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, MAX_ROWS + 1u));
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, 4));   /* table holds 2 rows */
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, 0, WIDTH, 2));
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(NULL, ra, KEY, WIDTH, 2));
    fprintf(stderr, "V4.1 rows publish scatter: bit-identical to the per-row copies, "
                    "non-publishing caches untouched, guards PASS\n");

    /* (5) The destination bounds. The two ds4_gpu_tensor_copy calls and the window copy this stage
     * replaces all began by refusing a write past the end of their destination; a raw GPU address
     * cannot, so the encode helpers check it on the host and this is the proof that they do -- and
     * that they do not merely refuse everything, which would pass a negative-only test while the
     * stage never ran. Each rejection below is otherwise a write into whatever follows another
     * session's cache: production has no shader validation layer to catch it. */
    CHECK(ds4_gpu_v41_rows_begin(2));
    for (uint32_t r = 0; r < 2; r++) {
        ds4_gpu_v41_row row = {0};
        row.window = window_a[r]; row.compressed = cache_v_a[r]; row.index_cache = cache_k_a[r];
        row.previous_kv = prev_kv_a[r]; row.previous_score = prev_sc_a[r];
        row.pos = positions[r]; row.n_comp = CACHE; row.publish = 1;
        CHECK(ds4_gpu_v41_rows_set(r, &row));
    }
    /* Positive controls: the last slot of each destination is accepted, on the persistent encoder
     * production uses. */
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 2, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 2, 2));
    CHECK(ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, 2));
    CHECK(ds4_gpu_end_commands());
    /* The window holds SLOTS slots of WIDTH; a dispatch claiming more slots, or a wider row than
     * the slot it would land in, is refused. */
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 2, DS4_V41_FP8_E8M0, SLOTS + 1u));
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH * 2u, 2, DS4_V41_FP8_E8M0, SLOTS));
    /* One row past the end of the caches: slot n_comp - 1 = CACHE is outside both. */
    CHECK(ds4_gpu_v41_rows_begin(2));
    for (uint32_t r = 0; r < 2; r++) {
        ds4_gpu_v41_row row = {0};
        row.compressed = cache_v_a[r]; row.index_cache = cache_k_a[r];
        row.pos = positions[r]; row.n_comp = CACHE + 1u; row.publish = 1;
        CHECK(ds4_gpu_v41_rows_set(r, &row));
    }
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, 2));
    /* publish with n_comp == 0 would wrap the slot to 0xFFFFFFFF: the row itself is refused. */
    CHECK(ds4_gpu_v41_rows_begin(1));
    {   ds4_gpu_v41_row row = {0};
        row.compressed = cache_v_a[0]; row.index_cache = cache_k_a[0];
        row.n_comp = 0; row.publish = 1;
        CHECK(!ds4_gpu_v41_rows_set(0, &row));
        row.n_comp = 1;
        CHECK(ds4_gpu_v41_rows_set(0, &row)); }
    /* A destination this layer never set resolves to address 0; no dispatch may dereference it. */
    CHECK(ds4_gpu_v41_rows_begin(1));
    {   ds4_gpu_v41_row row = {0}; row.pos = 5; row.n_comp = 1; row.publish = 1;
        CHECK(ds4_gpu_v41_rows_set(0, &row)); }
    CHECK(!ds4_gpu_dsv41_quantize_window_rows(ra, WIDTH, 1, DS4_V41_FP8_E8M0, SLOTS));
    CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 1, 2));
    CHECK(!ds4_gpu_dsv41_publish_scatter_rows(keys, ra, KEY, WIDTH, 1));
    /* ...but ratio 1 never touches previous_*, so that same table is fine for the plain copy. */
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 1, 1));
    CHECK(ds4_gpu_end_commands());
    /* A previous_kv shorter than the width: refused at ratio 2. */
    {   ds4_gpu_tensor *half = ds4_gpu_tensor_view(prev_kv_a[0], 0, (WIDTH / 2u) * 4u);
        CHECK(half);
        CHECK(ds4_gpu_v41_rows_begin(1));
        ds4_gpu_v41_row row = {0};
        row.previous_kv = half; row.previous_score = prev_sc_a[0]; row.pos = 1;
        CHECK(ds4_gpu_v41_rows_set(0, &row));
        CHECK(!ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH, 1, 2));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_pool2_rows(pool_a, ra, scores, WIDTH / 2u, 1, 2));
        CHECK(ds4_gpu_end_commands());
        ds4_gpu_tensor_free(half); }
    fprintf(stderr, "V4.1 rows table bounds: window slots, cache slots, previous_* width, the "
                    "n_comp == 0 wrap and unset addresses are all refused on the host, and the "
                    "last valid slot of each is accepted, guards PASS\n");

    for (uint32_t r = 0; r < MAX_ROWS; r++) {
        ds4_gpu_tensor_free(cache_v_b[r]); ds4_gpu_tensor_free(cache_v_a[r]);
        ds4_gpu_tensor_free(cache_k_b[r]); ds4_gpu_tensor_free(cache_k_a[r]);
        ds4_gpu_tensor_free(prev_sc_b[r]); ds4_gpu_tensor_free(prev_sc_a[r]);
        ds4_gpu_tensor_free(prev_kv_b[r]); ds4_gpu_tensor_free(prev_kv_a[r]);
        ds4_gpu_tensor_free(window_b[r]); ds4_gpu_tensor_free(window_a[r]);
    }
    ds4_gpu_tensor_free(keys);
    ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(pool_b); ds4_gpu_tensor_free(pool_a);
    ds4_gpu_tensor_free(rb); ds4_gpu_tensor_free(ra);

    /* (6) The table must not survive a cleanup/init cycle. views[] is __unsafe_unretained and
     * n_rows/filled/the capacities are plain scalars, so a table left behind would still satisfy
     * ds4_gpu_v41_rows_ready() and pass up to 40 released MTLBuffer handles to
     * -useResources:count:usage: on the next dispatch -- a GPU fault or worse, and the one stale
     * state the readiness guard cannot see. ds4_gpu_cleanup() clears it (ds4_gpu_v41_rows_reset);
     * this is the witness, and it runs last because it releases everything allocated above. */
    {   ds4_gpu_tensor *window = upload(NULL, (size_t)SLOTS * WIDTH * sizeof(float));
        CHECK(window);
        CHECK(ds4_gpu_v41_rows_begin(1));
        ds4_gpu_v41_row row = {0};
        row.window = window;
        row.pos = 3;
        CHECK(ds4_gpu_v41_rows_set(0, &row));
        ds4_gpu_tensor_free(window);
        ds4_gpu_cleanup();
        CHECK(ds4_gpu_init());
        ds4_gpu_tensor *fresh = upload(NULL, WIDTH * sizeof(float));
        CHECK(fresh);
        CHECK(!ds4_gpu_dsv41_quantize_window_rows(fresh, WIDTH, 1, DS4_V41_FP8_E8M0, SLOTS));
        ds4_gpu_tensor_free(fresh);
        fprintf(stderr, "V4.1 rows table cleared by ds4_gpu_cleanup: a dispatch after a "
                        "cleanup/init cycle is refused, guards PASS\n");
    }
    fprintf(stderr, "V4.1 batched attention rows, stage (a): PASS\n");
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
    /* 3.6d (2026-09-17): 1..8 are the decode batch widths. The V4.1 session step
     * now sends its N rows through this entry point instead of calling the
     * per-row projections N times, and nothing covered those widths here
     * before. The reference above is the per-row path, so `low` is a
     * bit-identical memcmp at every width and `out` is the 2e-5 comparison.
     * These inputs are exact by construction (weight scale 1/128 with quants
     * 0..7, x a non-negative multiple of 1/256), so no ordering can round
     * differently here: this case pins row indexing, the row bounds and the
     * BF16 boundary, and it is blind to the summation order. The order is what
     * check_attention_output_decode() below tests, on signed dense inputs. */
    const uint32_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8,
                              31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    /* Row 0 must come out IDENTICAL at every decode width on these exact inputs: a row reading
     * or writing the wrong slice once the width changes which rows share a threadgroup would
     * show up here even though a reordering cannot. */
    enum { WIDTH_PROBE = 8 };
    float *row0 = calloc(WIDTH_PROBE, OUT * sizeof(float));
    CHECK(row0);
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
        if (rows < ROWS) {
            CHECK(isnan(got[(uint64_t)rows * OUT]));
            /* 3.6d: the decode step hands this call views sized to exactly N
             * rows, so a write past row N-1 lands in the next workspace tensor. */
            CHECK(isnan(got_low[(uint64_t)rows * GROUPS * RANK]));
        }
        if (rows <= WIDTH_PROBE) {
            memcpy(row0 + (size_t)(rows - 1) * OUT, got, OUT * sizeof(float));
            /* Compared here rather than after the loop so the probe still runs when a wider
             * case fails -- rows=31 does exactly that under MTL_SHADER_VALIDATION=1, on the
             * base tree as well as this one. */
            for (uint32_t i = 0; rows > 1 && i < OUT; i++) {
                const float at_1 = row0[i], at_n = row0[(size_t)(rows - 1) * OUT + i];
                if (at_n != at_1)
                    fprintf(stderr, "width dependence rows=%u column=%u at_1=%.9g at_%u=%.9g\n",
                            rows, i, at_1, rows, at_n);
                CHECK(at_n == at_1);
            }
            if (rows > 1)
                fprintf(stderr, "V4.1 batched Q8 output, row 0 identical at rows=%u and rows=1 "
                                "(exact inputs; 0 of %u columns moved): PASS\n",
                        rows, (unsigned)OUT);
        }
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
    free(x); free(reference); free(reference_low); free(model); free(row0);
    return 1;
}

#ifdef __APPLE__
/* 3.6d (2026-09-17): the batched decode step's attention-output stage, reproduced end to end
 * against the per-row path it replaces, on inputs that can tell two summation orders apart.
 *
 * check_attention_output() above cannot tell them apart. Its weights are all +1/128 x quants 0..7
 * and its activations are non-negative multiples of 1/256, so every partial sum is exact and every
 * order gives the same float; it pins indexing, bounds and the BF16 boundary and nothing else.
 * That is how the first version of this stage shipped green here and then produced different
 * tokens from the per-row loop on the real decode step. Here the quants are signed (-127..127) and
 * the activations are full-mantissa signed floats, so cancellation puts the summation order into
 * the low bits, exactly as the model's own data does.
 *
 * REFERENCE = what ds41_attention_output() runs per row on the decode path at tp_world == 1, on
 * views of one row-contiguous workspace buffer, exactly as the loop does through g->rows_view[i]:
 *     ds4_gpu_attention_output_low_q8_bf16_tensor(low_i, ..., heads_i)   (ds41_attention_low)
 *     ds4_gpu_matmul_q8_0_tensor_bf16(block_i, ..., low_i, 1)            (ds41_matmul, prerounded)
 *
 * CANDIDATES = the two things the batched step can run for all rows at once, each followed by the
 * step's own ds4_gpu_dsv41_quantize(block, DS4_N_EMBD, rows, BF16):
 *   row-exact (the default): ds4_gpu_dsv41_attention_output_low_batch() + the row-exact batched Q8
 *          matvec ds41_matmul_batch() picks for 2..8 Q8_0 rows. Every row must be BIT-IDENTICAL at
 *          every width.
 *   mv_ext (DS4_METAL_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_MV_EXT): ds4_gpu_dsv41_attention_output_batch(),
 *          whose out_b goes through kernel_mul_mv_ext_q8_0_f32_r1_N -- one weight chunk dotted
 *          against r1ptg rows, reduced with simd_shuffle_down over nxpsg lanes (16 at 2 rows,
 *          8 at 3+) instead of simd_sum + the threadgroup tree.
 * Both are run and reported at every width; the one ds4.c will actually run is the one this test
 * fails on, so `--attention-output-decode` passes by default and fails under the mv_ext env -- the
 * model-free reproduction of the --verify and oracle failures the first revision measured.
 *
 * Both are also checked against a double-precision oracle at a few columns. That is not a bit gate
 * -- `block` is BF16-rounded, so 2^-9 relative is the floor for either -- it is what separates "a
 * different order" from "a wrong dot product", and it is why this test can say the mv_ext kernel is
 * correct and still unusable here.
 *
 * One thing a kernel memcmp alone would still miss, so it is checked first: the SELECTION. This
 * test calls ds4_gpu_matmul_q8_0_decode_rows_exact_tensor() itself, while the decode step calls
 * ds41_matmul_batch(), whose fall-through is the mul_mv_ext path. The oracle
 * ds4_v41_decode_batch_out_b_row_exact() asks ds4.c which one it would pick for this shape at
 * every width, so a change to DS4_TP_BATCH_MAX_ROWS or to that predicate fails here rather than
 * silently reinstating the first revision's defect.
 *
 * And one selector on the out_a half, the same worry on the other side: the batched encode is
 * picked by ds4_gpu_attention_output_q8_batch_impl's use_direct_low, `n_tokens < 32 &&
 * getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == NULL`, while the single-row
 * ds4_gpu_attention_output_low_q8_impl() this stage replaces reads no environment at all -- 3.6d
 * is what first puts a decode step behind that switch. A decode step is at most 8 rows, so only
 * the env half can fire, and when it does the encode becomes kernel_mul_mv_id_q8_0_f32 over a
 * group-ids buffer instead of kernel_dsv4_attn_out_low_q8_0_f32. So the widths are walked a third
 * time with that env exported and `low` must still be bit-identical, rather than that being
 * argued from the two kernels wrapping the same kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>. */
static int check_attention_output_decode(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120, ROWS = 8 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t hb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000;  /* exactly 1/128, so the oracle below needs no half converter */
        for (int j = 0; j < 32; j++)
            w[i].qs[j] = (int8_t)((((int)(random_value() * 8192) & 0x7fffffff) % 255) - 127);
    }
    float *x = malloc(hb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    /* Full-mantissa signed activations: the real `heads` are BF16-rounded and then inverse-RoPEd,
     * so what reaches out_a is an arbitrary float, not a short binary fraction. */
    for (uint64_t i = 0; i < hb / 4; i++) x[i] = random_value() * 0.3141592653589793f;
    ds4_gpu_tensor *heads = upload(x, hb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(heads && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));

    /* THE SELECTOR, not only the kernel. Everything below calls
     * ds4_gpu_matmul_q8_0_decode_rows_exact_tensor() directly, but the decode step calls
     * ds41_matmul_batch(), whose four-way dispatch has three other outcomes -- and whose final
     * `else` is metal_graph_matmul_plain_tensor() -> ds4_gpu_matmul_q8_0_tensor(), i.e.
     * kernel_mul_mv_ext_q8_0_f32_r1_N for 2..16 rows of a 128-aligned K. That is exactly the
     * kernel the first revision of this stage shipped by accident. So ask ds4.c which out_b it
     * would select for this shape at every width the stage can take, before certifying the
     * kernel: a change to DS4_TP_BATCH_MAX_ROWS or to that predicate fails here instead of
     * silently reinstating the defect with this test still green. rows == 1 must answer NO --
     * ds41_graph_step_batch() requires count >= 2, so width 1 below is kernel coverage only. */
    for (uint32_t rows = 1; rows <= ROWS; rows++)
        CHECK(ds4_v41_decode_batch_out_b_row_exact(rows, OUT) == (rows >= 2 ? 1 : 0));
    fprintf(stderr, "V4.1 decode output stage: ds41_matmul_batch selects the row-exact Q8 out_b "
                    "for the %u-wide out_b at rows=2..%u, and not at rows=1: PASS\n",
            (unsigned)OUT, (unsigned)ROWS);

    const int step_mv_ext = getenv("DS4_METAL_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_MV_EXT") != NULL;

    /* The per-row reference does not depend on how many rows the step carries, so it is built
     * once for all ROWS rows and each width compares its first `rows` of them. */
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *hr = ds4_gpu_tensor_view(heads, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *br = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(hr && lr && br);
        CHECK(ds4_gpu_attention_output_low_q8_bf16_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, hr));
        CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(br, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(hr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(br);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));

    for (int variant = 0; variant < 2; variant++) {
        const int mv_ext = variant == 1;
        /* The gate follows ds4.c: only the variant the decode step will run has to be exact. */
        const int gated = mv_ext == step_mv_ext;
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
            ds4_gpu_tensor *hv = ds4_gpu_tensor_view(heads, 0, (uint64_t)rows * GROUPS * GROUP * 4);
            ds4_gpu_tensor *lv = ds4_gpu_tensor_view(low, 0, (uint64_t)rows * GROUPS * RANK * 4);
            ds4_gpu_tensor *bv = ds4_gpu_tensor_view(out, 0, (uint64_t)rows * OUT * 4);
            CHECK(hv && lv && bv);
            /* One command batch, as the layer is encoded in the real step. */
            CHECK(ds4_gpu_begin_commands());
            if (mv_ext) {
                CHECK(ds4_gpu_dsv41_attention_output_batch(bv, lv, model, a_bytes + b_bytes,
                    0, a_bytes, hv, rows));
            } else {
                CHECK(ds4_gpu_dsv41_attention_output_low_batch(lv, model, a_bytes + b_bytes,
                    0, hv, rows));
                /* The kernel (A) above proved ds41_matmul_batch() selects at these widths. */
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(bv, model, a_bytes + b_bytes,
                    a_bytes, GROUPS * RANK, OUT, lv, rows));
            }
            /* ds41_graph_step_batch's own epilogue over the projected block. */
            CHECK(ds4_gpu_dsv41_quantize(bv, OUT, rows, DS4_V41_BF16));
            CHECK(ds4_gpu_end_commands());
            ds4_gpu_tensor_free(hv); ds4_gpu_tensor_free(lv); ds4_gpu_tensor_free(bv);

            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(out);
            CHECK(got_low && got);
            uint64_t low_differ = 0, out_differ = 0;
            double worst = 0, worst_rel = 0;
            for (uint64_t i = 0; i < (uint64_t)rows * GROUPS * RANK; i++)
                if (memcmp(&got_low[i], &reference_low[i], 4)) low_differ++;
            for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
                CHECK(isfinite(got[i]));
                if (!memcmp(&got[i], &reference[i], 4)) continue;
                out_differ++;
                const double d = fabs((double)got[i] - (double)reference[i]);
                if (d > worst) worst = d;
                if (d / (1 + fabs((double)reference[i])) > worst_rel)
                    worst_rel = d / (1 + fabs((double)reference[i]));
            }
            /* Nothing may be written past row rows-1: the decode step hands these calls views
             * sized to exactly N rows, so an overrun lands in the next workspace tensor. */
            if (rows < ROWS) {
                CHECK(isnan(got_low[(uint64_t)rows * GROUPS * RANK]));
                CHECK(isnan(got[(uint64_t)rows * OUT]));
            }
            /* Independent double sums over reference_low, at both ends and in the interior.
             * Loose by design: `block` is BF16-rounded, so neither variant can be closer than
             * 2^-9 relative. This says the arithmetic is right; the memcmp says it is the same. */
            const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
            for (uint32_t t = 0; t < rows; t++) {
                for (size_t ci = 0; ci < sizeof(columns) / sizeof(*columns); ci++) {
                    const uint32_t col = columns[ci];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * (GROUPS * RANK / 32);
                    double sum = 0;
                    for (uint32_t k = 0; k < GROUPS * RANK; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            (double)reference_low[(size_t)t * GROUPS * RANK + k];
                    if (fabs((double)got[(size_t)t * OUT + col] - sum) > 8e-3 * (1 + fabs(sum)))
                        fprintf(stderr, "decode output stage oracle rows=%u variant=%s row=%u "
                                "col=%u actual=%.9g oracle=%.9g\n", rows,
                                mv_ext ? "mv_ext" : "row-exact", t, col,
                                got[(size_t)t * OUT + col], sum);
                    CHECK(fabs((double)got[(size_t)t * OUT + col] - sum) <= 8e-3 * (1 + fabs(sum)));
                }
            }
            fprintf(stderr, "V4.1 decode output stage rows=%u %-9s: low %llu/%llu differ, "
                            "block %llu/%llu differ (worst %.3g abs, %.3g rel)%s\n",
                    rows, mv_ext ? "mv_ext" : "row-exact",
                    (unsigned long long)low_differ,
                    (unsigned long long)((uint64_t)rows * GROUPS * RANK),
                    (unsigned long long)out_differ, (unsigned long long)((uint64_t)rows * OUT),
                    worst, worst_rel, gated ? "   <- the decode step's default" : "");
            /* out_a is bit-identical either way: the batch call runs the same (group, row)
             * threadgroups, and its separate BF16 rounding is the same round-to-nearest-even
             * as the one folded into the single-row matvec's store. */
            CHECK(low_differ == 0);
            if (gated && out_differ)
                fprintf(stderr, "V4.1 decode output stage rows=%u %s: NOT bit-identical to the "
                        "per-row path -- a batched decode step would not reproduce the tokens "
                        "the per-row loop emits\n", rows, mv_ext ? "mv_ext" : "row-exact");
            if (gated) CHECK(out_differ == 0);
        }
        fprintf(stderr, "V4.1 decode output stage, widths 1..%u, %s%s: %s\n", (unsigned)ROWS,
                mv_ext ? "mv_ext" : "row-exact", gated ? " (the decode step's default)" : "",
                gated ? "bit-identical to the per-row path PASS" : "reported only");
    }

    /* The out_a selector (see the header): the only env this stage newly exposes a decode step to
     * is DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT, which picks kernel_mul_mv_id_q8_0_f32 over a
     * group-ids buffer instead of kernel_dsv4_attn_out_low_q8_0_f32. out_b does not read it, so
     * only out_a is rerun, against the same per-row `low` reference as above. */
    CHECK(setenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT", "1", 1) == 0);
    for (uint32_t rows = 1; rows <= ROWS; rows++) {
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        ds4_gpu_tensor *hv = ds4_gpu_tensor_view(heads, 0, (uint64_t)rows * GROUPS * GROUP * 4);
        ds4_gpu_tensor *lv = ds4_gpu_tensor_view(low, 0, (uint64_t)rows * GROUPS * RANK * 4);
        CHECK(hv && lv);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_attention_output_low_batch(lv, model, a_bytes + b_bytes, 0, hv, rows));
        CHECK(ds4_gpu_end_commands());
        ds4_gpu_tensor_free(hv); ds4_gpu_tensor_free(lv);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(got_low);
        uint64_t low_differ = 0;
        for (uint64_t i = 0; i < (uint64_t)rows * GROUPS * RANK; i++)
            if (memcmp(&got_low[i], &reference_low[i], 4)) low_differ++;
        if (rows < ROWS) CHECK(isnan(got_low[(uint64_t)rows * GROUPS * RANK]));
        fprintf(stderr, "V4.1 decode output stage rows=%u out_a, ATTN_OUT_LOW_DIRECT disabled: "
                        "low %llu/%llu differ\n", rows, (unsigned long long)low_differ,
                (unsigned long long)((uint64_t)rows * GROUPS * RANK));
        CHECK(low_differ == 0);
    }
    CHECK(unsetenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == 0);
    fprintf(stderr, "V4.1 decode output stage, widths 1..%u, out_a with "
                    "DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT exported: bit-identical to the "
                    "per-row path PASS\n", (unsigned)ROWS);

    ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}
#endif

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
    /* 3.6a (2026-09-17): 2..8 are the decode batch widths. ds41_project_rows sends a batched
     * decode step's compressor and indexer projections through here, and the memcmp below against
     * N single-row dispatches is what makes stage (a) bit-identical per row rather than merely
     * "batched == sequential"; nothing exercised those widths before. */
    const uint32_t counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 31, 64, 65, 513, 2048, MAX_ROWS};
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
/* PRE_M5 3.6c (2026-09-17): the N-row decode FlashAttention against N single-row dispatches, by
 * memcmp.  This is the stage's exactness proof and it needs no model: the claim is that
 * ds4_gpu_attention_decode_heads_rows_tensor() computes row r out of row r's own window ring,
 * gathered compressed rows and query exactly as ds4_gpu_dsv41_gather_kv() +
 * ds4_gpu_attention_decode_heads_tensor() do, whatever the other rows in the batch are doing.
 *
 * What the cases are chosen to hit:
 *   - N = 1, which compares the batched encoder against the single-row one on the SAME key count,
 *     including key counts that are not multiples of 32.  Those are exactly the cases where the
 *     single-row path compiles the vector kernel with has_kvpad = true and takes its tail block out
 *     of the pad buffer while the rows path pads the slab instead: if that substitution were not
 *     bit-exact, this is where it shows.
 *   - rows at very different key counts in one batch (1 key next to 640), so a short row walks many
 *     fully masked blocks that must contribute nothing.
 *   - ring wrap (raw_start != 0) on BOTH reference encoders, not just the gathered one.  Wrap needs
 *     a full window, so a wrapped row always has n_raw == raw_cap == 128; pos_raw[0] = 200 puts a
 *     wrapped raw-only row in every width including N = 1, and 128 / 333 add two more at N >= 4.
 *     That is the exact shape layers 0 and 1 present on every decode step past position 127
 *     (ds4_expected_layer_compress_ratio() gives them ratio 0, so they are raw-only forever), and
 *     it is the case where the reference stages the ring as a TAIL copy plus a HEAD copy --
 *     ds4_gpu_encode_copy_raw_ring_to_f16(), two ds4_gpu_encode_cpy_f32_f16_1d dispatches -- while
 *     kernel_dsv41_flash_stage_rows wraps in one pass.  The PASS line prints keys@raw_start per
 *     row, so the log shows which cases actually wrapped.
 *   - all three shapes a V4.1 decode layer can present: raw only (layers 0-1, ratio 0, and any row
 *     whose pos + 1 < ratio), gathered, and a batch mixing the two.  Note the raw-only ENCODER is
 *     also exercised at N = 1 with a key count that is not a multiple of 32 by pos_mixed[0] = 0
 *     (ratio 2 gives n_comp = 0 there, so that row is raw-only with a single key), which is why
 *     pos_raw can spend its first slot on the wrapped case.
 */
static int check_flash_rows(void) {
    enum { D = 512, H = 64, RAW = 128, COMP = 1024, TOPK = 512 };
    static const uint32_t widths[] = {1, 2, 4, 8};
    /* pos values -> n_raw = min(pos+1,128), raw_start = (pos+1-n_raw)%128, and with ratio 2,
     * n_comp = (pos+1)/2 and attended = min(n_comp, 512).  0 gives a raw-only row. */
    static const uint32_t pos_mixed[8]    = {0, 5, 63, 127, 128, 200, 511, 2047};
    static const uint32_t pos_gathered[8] = {3, 9, 65, 129, 333, 640, 1500, 2047};
    /* Raw only, so n_keys == n_raw.  raw_start is 73, 0, 0, 1, 0, 78, 0, 0 in this order: the
     * wrapped full-window rows (pos 200/128/333) are at slots 0 and 3/5 so that N = 1 and N = 2
     * already carry one, and the rest keep key counts that are and are not multiples of 32
     * (128, 18, 32, 128, 34, 128, 101, 128). */
    static const uint32_t pos_raw[8]      = {200, 17, 31, 128, 33, 333, 100, 2047};
    static const char *const variant_name[3] = {"mixed", "gathered", "raw only"};
    const size_t page = (size_t)getpagesize();
    float *sinks = NULL;
    CHECK(page >= H * sizeof(float));
    CHECK(posix_memalign((void **)&sinks, page, page) == 0);
    memset(sinks, 0, page);
    for (int h = 0; h < H; h++) sinks[h] = random_value() / 2;
    CHECK(ds4_gpu_set_model_map(sinks, page));
    ds4_gpu_set_quality(false);

    for (unsigned w = 0; w < sizeof(widths) / sizeof(*widths); w++) {
        const uint32_t n = widths[w];
        for (unsigned variant = 0; variant < 3; variant++) {
            const uint32_t *pos_table = variant == 0 ? pos_mixed :
                                        variant == 1 ? pos_gathered : pos_raw;
            const uint32_t ratio = variant == 2 ? 0u : 2u;
            ds4_gpu_tensor *window[8] = {0}, *comp[8] = {0}, *idt[8] = {0}, *sel[8] = {0};
            ds4_gpu_tensor *q = NULL, *ref = NULL, *got = NULL;
            ds4_gpu_v41_flash_row desc[8];
            float *raw_host = malloc((size_t)RAW * D * sizeof(float));
            float *comp_host = malloc((size_t)COMP * D * sizeof(float));
            float *q_host = malloc((size_t)n * H * D * sizeof(float));
            float *ref_host = malloc((size_t)n * H * D * sizeof(float));
            float *got_host = malloc((size_t)n * H * D * sizeof(float));
            int32_t *ids_host = malloc((size_t)TOPK * sizeof(int32_t));
            int ok = raw_host && comp_host && q_host && ref_host && got_host && ids_host;
            CHECK(ok);
            for (size_t i = 0; i < (size_t)n * H * D; i++) q_host[i] = bf16(random_value() / 4);
            q = upload(q_host, (size_t)n * H * D * sizeof(float));
            ref = upload(NULL, (size_t)n * H * D * sizeof(float));
            got = upload(NULL, (size_t)n * H * D * sizeof(float));
            CHECK(q && ref && got);

            for (uint32_t r = 0; r < n; r++) {
                const uint32_t pos = pos_table[r];
                const uint32_t n_comp = ratio ? (pos + 1u) / ratio : 0u;
                const uint32_t n_raw = pos + 1u < (uint32_t)RAW ? pos + 1u : (uint32_t)RAW;
                const uint32_t attended = n_comp < (uint32_t)TOPK ? n_comp : (uint32_t)TOPK;
                CHECK(n_comp <= (uint32_t)COMP);
                for (size_t i = 0; i < (size_t)RAW * D; i++)
                    raw_host[i] = bf16(random_value() / 4);
                for (size_t i = 0; i < (size_t)COMP * D; i++)
                    comp_host[i] = bf16(random_value() / 4);
                for (uint32_t j = 0; j < attended; j++)
                    ids_host[j] = (int32_t)((j * 37u + r * 11u + 1u) % (n_comp ? n_comp : 1u));
                window[r] = upload(raw_host, (size_t)RAW * D * sizeof(float));
                comp[r] = upload(comp_host, (size_t)COMP * D * sizeof(float));
                idt[r] = attended ? upload(ids_host, (size_t)attended * sizeof(int32_t)) : NULL;
                sel[r] = attended ? upload(NULL, (size_t)attended * D * sizeof(float)) : NULL;
                CHECK(window[r] && comp[r] && (!attended || (idt[r] && sel[r])));
                desc[r].raw_kv = window[r];
                desc[r].comp_kv = attended ? comp[r] : NULL;
                desc[r].comp_ids = attended ? idt[r] : NULL;
                desc[r].n_raw = n_raw;
                desc[r].raw_cap = RAW;
                desc[r].raw_start = (pos + 1u - n_raw) % (uint32_t)RAW;
                desc[r].source_rows = n_comp;
                desc[r].attended = attended;

                /* The reference: this row's own two dispatches, into its slice of `ref`. */
                ds4_gpu_tensor *qv = ds4_gpu_tensor_view(q, (uint64_t)r * H * D * 4,
                                                         (uint64_t)H * D * 4);
                ds4_gpu_tensor *hv = ds4_gpu_tensor_view(ref, (uint64_t)r * H * D * 4,
                                                         (uint64_t)H * D * 4);
                CHECK(qv && hv);
                if (attended)
                    CHECK(ds4_gpu_dsv41_gather_kv(sel[r], comp[r], idt[r], n_comp, attended));
                CHECK(ds4_gpu_attention_decode_heads_tensor(hv, sinks, page, 0, qv, window[r],
                    n_raw, RAW, desc[r].raw_start, attended ? sel[r] : NULL, 0, attended,
                    NULL, 0, H, D));
                ds4_gpu_tensor_free(qv);
                ds4_gpu_tensor_free(hv);
            }
            CHECK(ds4_gpu_tensor_read(ref, 0, ref_host, (size_t)n * H * D * sizeof(float)));

            /* The candidate: one stage dispatch per row, one vec with ne03 = N, one reduce. */
            CHECK(ds4_gpu_attention_decode_heads_rows_tensor(got, sinks, page, 0, q, desc,
                                                             n, H, D) == 1);
            CHECK(ds4_gpu_tensor_read(got, 0, got_host, (size_t)n * H * D * sizeof(float)));
            for (uint32_t r = 0; r < n; r++) {
                const float *a = ref_host + (size_t)r * H * D;
                const float *b = got_host + (size_t)r * H * D;
                if (memcmp(a, b, (size_t)H * D * sizeof(float))) {
                    size_t i = 0;
                    while (i < (size_t)H * D && a[i] == b[i]) i++;
                    fprintf(stderr, "V4.1 flash rows=%u %s: row %u differs at %zu "
                                    "(%.9g vs %.9g), n_raw=%u raw_start=%u attended=%u\n",
                            n, variant_name[variant], r, i,
                            i < (size_t)H * D ? a[i] : 0.0, i < (size_t)H * D ? b[i] : 0.0,
                            desc[r].n_raw, desc[r].raw_start, desc[r].attended);
                    CHECK(0);
                }
                CHECK(isfinite(a[0]));
            }
            /* keys@raw_start per row: raw_start != 0 is the wrapped ring, which the reference
             * stages as two copies and the new kernel in one pass, so the log has to show that
             * the wrapped cases really ran -- on the raw-only batches as well as the gathered. */
            fprintf(stderr, "V4.1 flash rows=%u (%s, keys", n, variant_name[variant]);
            for (uint32_t r = 0; r < n; r++)
                fprintf(stderr, " %u@%u", desc[r].n_raw + desc[r].attended, desc[r].raw_start);
            fprintf(stderr, "): bit-identical to %u single-row dispatches PASS\n", n);

            for (uint32_t r = 0; r < n; r++) {
                ds4_gpu_tensor_free(window[r]); ds4_gpu_tensor_free(comp[r]);
                ds4_gpu_tensor_free(idt[r]); ds4_gpu_tensor_free(sel[r]);
            }
            ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(got);
            free(raw_host); free(comp_host); free(q_host);
            free(ref_host); free(got_host); free(ids_host);
        }
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}

#ifdef __APPLE__
/* PRE_M5 3.6c (2026-09-18): the ds4.c half of the stage -- which cache each row of the N-row
 * dispatch is pointed at and with which key counts, not whether the dispatch itself is exact.
 *
 * check_flash_rows() above is the exactness proof, but it builds its OWN descriptors and calls
 * the entry point directly, so it passes unchanged if ds41_batch_attention_flash() hands that
 * entry point row 0's window for every row, the wrong compressed owner for a layer, the
 * workspace's caches where the session's belong, or a position off by one.  That mapping runs
 * only inside a batched decode step, so the only other thing that would notice is a model run,
 * and the model runs belong to the measurement pass.  ds4_v41_batch_attention_flash_desc() runs
 * the real per-row mapping over synthetic graphs whose tensor pointers are sentinels and reports
 * which structure, which row and which array index each descriptor pointer came from.
 *
 * TOP_K mirrors DS4_N_INDEXER_TOP_K and MAX_ROWS mirrors DS4_TP_BATCH_MAX_ROWS (ds4.c and
 * ds4_tp.h are not included here, as check_attn_out_path_cases() does the same): if either
 * moves, this fails, which is the intent. */
static int check_flash_rows_desc_row(unsigned rows, unsigned il, unsigned ratio,
                                     const unsigned *positions, unsigned row) {
    enum { MAX_ROWS = 8, TOP_K = 512, RAW_CAP = 128 };
    const unsigned pos = positions[row];
    const unsigned owner = il < 8 ? 0u : il < 14 ? 1u : il < 20 ? 2u : 3u;
    const unsigned n_comp = ratio ? (pos + 1u) / ratio : 0u;
    const unsigned n_raw = pos + 1u < RAW_CAP ? pos + 1u : RAW_CAP;
    ds4_v41_flash_desc_probe p;
    CHECK(rows <= MAX_ROWS && row < rows);
    memset(&p, 0xa5, sizeof p);
    CHECK(ds4_v41_batch_attention_flash_desc(rows, positions, il, ratio, row, &p) == 1);
    /* The row's own session, at this layer -- never another row's, never the workspace's. */
    CHECK(p.raw_kv_kind == 1 && p.raw_kv_row == (int)row && p.raw_kv_index == (int)il);
    if (n_comp) {
        /* The compressed cache is the session's and is picked by the layer's owner; the ids are
         * the WORKSPACE row view's, which is where the per-row front wrote them. */
        CHECK(p.comp_kv_kind == 2 && p.comp_kv_row == (int)row && p.comp_kv_index == (int)owner);
        CHECK(p.comp_ids_kind == 3 && p.comp_ids_row == (int)row && p.comp_ids_index == 0);
    } else {
        /* A raw-only row hands the dispatch no compressed slab and no ids at all. */
        CHECK(p.comp_kv_kind == 0 && p.comp_ids_kind == 0);
    }
    CHECK(p.n_raw == n_raw && p.raw_cap == RAW_CAP);
    CHECK(p.raw_start == (pos + 1u - n_raw) % RAW_CAP);
    CHECK(p.source_rows == n_comp);
    CHECK(p.attended == (n_comp < TOP_K ? n_comp : TOP_K));
    return 1;
}

static int check_flash_rows_desc(void) {
    enum { MAX_ROWS = 8 };
    /* check_flash_rows()'s own position tables, so both halves of the stage answer for the same
     * rows, except that the gathered one spends its last two slots on 1023 and 4095: n_comp there
     * is exactly TOP_K and four times TOP_K, which is where `attended` has to clamp and which
     * check_flash_rows() cannot run (its compressed cache is 1024 rows). */
    static const unsigned pos_mixed[MAX_ROWS]    = {0, 5, 63, 127, 128, 200, 511, 2047};
    static const unsigned pos_gathered[MAX_ROWS] = {3, 9, 65, 129, 333, 640, 1023, 4095};
    static const unsigned pos_raw[MAX_ROWS]      = {200, 17, 31, 128, 33, 333, 100, 2047};
    /* Both sides of every step of the owner ladder, plus the last layer window[] holds. */
    static const unsigned layers[] = {0, 1, 7, 8, 13, 14, 19, 20, 39};
    static const unsigned widths[] = {1, 2, 4, 8};
    unsigned cases = 0;
    for (unsigned w = 0; w < sizeof(widths) / sizeof(*widths); w++)
        for (unsigned variant = 0; variant < 3; variant++) {
            const unsigned *pos = variant == 0 ? pos_mixed :
                                  variant == 1 ? pos_gathered : pos_raw;
            const unsigned ratio = variant == 2 ? 0u : 2u;
            for (unsigned li = 0; li < sizeof(layers) / sizeof(*layers); li++)
                for (unsigned r = 0; r < widths[w]; r++) {
                    CHECK(check_flash_rows_desc_row(widths[w], layers[li], ratio, pos, r));
                    cases++;
                }
        }
    /* Refusals: nothing a batched step cannot present gets a descriptor. */
    {
        static const unsigned pos[MAX_ROWS] = {0, 1, 2, 3, 4, 5, 6, 7};
        ds4_v41_flash_desc_probe p;
        CHECK(ds4_v41_batch_attention_flash_desc(MAX_ROWS + 1u, pos, 0, 2, 0, &p) == 0);
        CHECK(ds4_v41_batch_attention_flash_desc(0, pos, 0, 2, 0, &p) == 0);
        CHECK(ds4_v41_batch_attention_flash_desc(2, pos, 0, 2, 2, &p) == 0);
        CHECK(ds4_v41_batch_attention_flash_desc(2, NULL, 0, 2, 0, &p) == 0);
        CHECK(ds4_v41_batch_attention_flash_desc(2, pos, 40, 2, 0, &p) == 0);
        CHECK(ds4_v41_batch_attention_flash_desc(2, pos, 0, 2, 0, NULL) == 0);
    }
    fprintf(stderr, "V4.1 batched flash descriptors: %u rows at N = 1/2/4/8 over mixed, gathered "
                    "and raw-only batches, each pointed at its OWN session's window[il] and "
                    "compressed[owner] and at the workspace's selected_comp, with n_raw, "
                    "raw_start, source_rows and attended as the per-row path computes them: "
                    "PASS\n", cases);
    return 1;
}
#endif

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

#ifdef __APPLE__
/* The block scale of a Q8_0 block, for the CPU-side order-sensitivity proof in
 * check_attn_out_rows_geometry.  Normals and zero cover everything these tests build. */
static float half_bits_to_float(uint16_t h) {
    const int exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    if (exp == 0) return sign * ldexpf((float)man, -24);
    return sign * ldexpf((float)(man | 0x400), exp - 25);
}

/* 3.6d2 (2026-09-17): the rows-templated attention-output kernels against R SINGLE-ROW dispatches.
 *
 * check_attention_output above deliberately uses inputs whose every partial sum is exact in fp32
 * (weight scale 1/128, |q| < 8, x a multiple of 1/256), which is what lets it compare the batched
 * entry point with the per-row reference at all -- but it also means no reordering could ever be
 * seen there. This mode is the opposite: full-range quants, varied block scales and random f32
 * activations, with a CPU proof that summing one row's blocks forwards and backwards in fp32
 * really does disagree. On that data a memcmp against the per-row path is a statement about the
 * KERNEL, not about the data, and it is the claim the whole stage rests on -- that
 * kernel_mul_mv_q8_0_f32_nrows_impl walks K and reduces exactly as the single-row kernel does, so
 * a row's logits do not depend on how many other sessions share its step.
 *
 * Both epilogue variants are covered, because both exist in the engine: the fused BF16 store
 * (ds41_out_b_prerounded / ds4_gpu_attention_output_low_q8_bf16_tensor) and the plain f32 store.
 * The references are the real single-row producers -- per-row bf16 calls for the fused variant,
 * and for the plain one ds4_gpu_matmul_q8_0_decode_rows_exact_tensor, which is literally R
 * single-row matvec dispatches in one grid.
 *
 * The NaN fill before each call is the bounds check: the batched decode step hands these calls
 * views sized to exactly N rows, so a write past row N-1 lands in the next workspace tensor.
 *
 * Geometry is a parameter because the k-split is not constant: every single-row Q8_0 matvec
 * switches from nsg 4 to nsg 8 above 65536 output rows, and nsg decides which blocks land in
 * which simdgroup partial, so it is part of the reduction tree the rows kernels have to
 * reproduce. check_attn_out_rows() runs V4.1's own shape and one above that threshold. */
static int check_attn_out_rows_geometry(uint32_t group_dim, uint32_t rank, uint32_t groups,
                                        uint32_t out_dim, int negatives, const char *label) {
    enum { MAXR = 8 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t low_dim = (uint64_t)groups * rank;
    const uint64_t a_bytes = low_dim * group_dim / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)out_dim * low_dim / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)MAXR * groups * group_dim * 4;
    const uint64_t lb = (uint64_t)MAXR * low_dim * 4;
    const uint64_t ob = (uint64_t)MAXR * out_dim * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = (uint16_t)(0x1800u + ((uint32_t)(int)(random_value() * 8192) & 0x7ffu));
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 127);
    }
    float *x = malloc(xb), *reference_low = malloc(lb), *reference_out = malloc(ob);
    CHECK(x && reference_low && reference_out);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();

    {   /* The data must really be order-sensitive, or every memcmp below holds vacuously: one
         * out_a row's block contributions summed forwards and backwards in fp32 must disagree.
         * The accumulators are volatile so -ffast-math cannot reassociate the two loops. */
        int order_matters = 0;
        for (uint32_t r = 0; r < rank && !order_matters; r++) {
            const q8_block *row = w + (uint64_t)r * (group_dim / 32);
            volatile float fwd = 0, rev = 0;
            for (uint32_t b = 0; b < group_dim / 32; b++) {
                float acc = 0;
                for (int j = 0; j < 32; j++) acc += row[b].qs[j] * x[b * 32 + j];
                fwd = fwd + acc * half_bits_to_float(row[b].d);
            }
            for (uint32_t b = group_dim / 32; b-- > 0; ) {
                float acc = 0;
                for (int j = 0; j < 32; j++) acc += row[b].qs[j] * x[b * 32 + j];
                rev = rev + acc * half_bits_to_float(row[b].d);
            }
            const float f = fwd, v = rev;
            order_matters = memcmp(&f, &v, 4) != 0;
        }
        CHECK(order_matters);
    }

    ds4_gpu_tensor *xt = upload(x, xb);
    ds4_gpu_tensor *low = upload(NULL, lb), *out = upload(NULL, ob);
    ds4_gpu_tensor *low_ref = upload(NULL, lb), *out_ref = upload(NULL, ob);
    CHECK(xt && low && out && low_ref && out_ref &&
          ds4_gpu_set_model_map(model, a_bytes + b_bytes));

    /* The V4.1 wrapper hard-codes its own geometry, so it can only check itself. */
    const int v41_geometry = group_dim == 4096 && rank == 1024 && groups == 8 && out_dim == 5120;

    for (unsigned fused = 0; fused < 2; fused++) {
        for (uint32_t rows = 2; rows <= MAXR; rows++) {
            /* Reference: the per-row producers the batched step replaces, one dispatch a row. */
            CHECK(ds4_gpu_tensor_fill_f32(low_ref, NAN, lb / 4));
            CHECK(ds4_gpu_tensor_fill_f32(out_ref, NAN, ob / 4));
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t i = 0; i < rows; i++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt,
                    (uint64_t)i * groups * group_dim * 4, (uint64_t)groups * group_dim * 4);
                ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low_ref,
                    (uint64_t)i * low_dim * 4, low_dim * 4);
                CHECK(xr && lr);
                CHECK(fused ?
                    ds4_gpu_attention_output_low_q8_bf16_tensor(lr, model, a_bytes + b_bytes,
                        0, group_dim, rank, groups, xr) :
                    ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
                        0, group_dim, rank, groups, xr));
                if (fused) {
                    ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out_ref,
                        (uint64_t)i * out_dim * 4, (uint64_t)out_dim * 4);
                    CHECK(yr);
                    CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(yr, model, a_bytes + b_bytes, a_bytes,
                        low_dim, out_dim, lr, 1));
                    ds4_gpu_tensor_free(yr);
                }
                ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr);
            }
            /* The plain out_b reference is R single-row matvec dispatches in one grid
             * (grid (out/nr0, rows, 1), the single-row geometry per row). */
            if (!fused)
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out_ref, model,
                    a_bytes + b_bytes, a_bytes, low_dim, out_dim, low_ref, rows));
            CHECK(ds4_gpu_end_commands());
            memcpy(reference_low, ds4_gpu_tensor_contents(low_ref), (size_t)(rows * low_dim * 4));
            memcpy(reference_out, ds4_gpu_tensor_contents(out_ref), (size_t)rows * out_dim * 4);

            /* Candidate: one dispatch a projection for all `rows` rows. */
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_attention_output_q8_rows_tensor(out, low, model, a_bytes + b_bytes,
                0, a_bytes, group_dim, rank, groups, out_dim, xt, rows, (int)fused));
            CHECK(ds4_gpu_end_commands());

            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(out);
            for (uint64_t i = 0; i < (uint64_t)rows * low_dim; i++) {
                if (memcmp(&got_low[i], &reference_low[i], 4))
                    fprintf(stderr, "attn out rows low %s fused=%u rows=%u index=%llu row=%llu "
                                    "actual=%.9g reference=%.9g\n",
                            label, fused, rows, (unsigned long long)i,
                            (unsigned long long)(i / low_dim), got_low[i], reference_low[i]);
                CHECK(!memcmp(&got_low[i], &reference_low[i], 4));
            }
            for (uint64_t i = 0; i < (uint64_t)rows * out_dim; i++) {
                if (memcmp(&got[i], &reference_out[i], 4))
                    fprintf(stderr, "attn out rows out %s fused=%u rows=%u index=%llu row=%llu "
                                    "actual=%.9g reference=%.9g\n",
                            label, fused, rows, (unsigned long long)i,
                            (unsigned long long)(i / out_dim), got[i], reference_out[i]);
                CHECK(!memcmp(&got[i], &reference_out[i], 4));
            }
            /* Nothing past the step's last row. */
            if (rows < MAXR) {
                CHECK(isnan(got_low[(uint64_t)rows * low_dim]));
                CHECK(isnan(got[(uint64_t)rows * out_dim]));
            }
            /* And once more through ds4_gpu_dsv41_attention_output_rows(), the V4.1 wrapper the
             * batched decode step actually calls: same geometry, round_bf16 fixed to 1. */
            if (fused && v41_geometry) {
                CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
                CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_dsv41_attention_output_rows(out, low, model, a_bytes + b_bytes,
                    0, a_bytes, xt, rows));
                CHECK(ds4_gpu_end_commands());
                CHECK(!memcmp(ds4_gpu_tensor_contents(low), reference_low,
                              (size_t)(rows * low_dim * 4)));
                CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference_out,
                              (size_t)rows * out_dim * 4));
            }
            fprintf(stderr, "V4.1 attention output rows=%u %s (%s epilogue): bit-identical to %u "
                            "single-row dispatches, low and out: PASS\n",
                    rows, label, fused ? "fused BF16" : "plain", rows);
        }
    }

    if (negatives) {
        /* Row counts the decode step never produces are refused, so the caller falls back rather
         * than running a kernel whose template does not cover them. */
        CHECK(!ds4_gpu_attention_output_q8_rows_tensor(out, low, model, a_bytes + b_bytes,
            0, a_bytes, group_dim, rank, groups, out_dim, xt, 1, 1));
        CHECK(!ds4_gpu_attention_output_q8_rows_tensor(out, low, model, a_bytes + b_bytes,
            0, a_bytes, group_dim, rank, groups, out_dim, xt, MAXR + 1, 1));
        /* An odd output extent would make the NR0 2 impl read a weight row past the tensor. */
        CHECK(!ds4_gpu_attention_output_q8_rows_tensor(out, low, model, a_bytes + b_bytes,
            0, a_bytes, group_dim, rank, groups, out_dim - 1, xt, 2, 1));
        /* Weights outside the mapped model. */
        CHECK(!ds4_gpu_attention_output_q8_rows_tensor(out, low, model, a_bytes + b_bytes - 1,
            0, a_bytes, group_dim, rank, groups, out_dim, xt, 2, 1));
    }

    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low_ref); ds4_gpu_tensor_free(out_ref);
    ds4_gpu_cleanup();
    free(x); free(reference_low); free(reference_out); free(model);
    return 1;
}

static int check_attn_out_rows(void) {
    /* V4.1's own shape: out_a 8 groups x 1024 rank over 4096, out_b 8192 -> 5120. */
    if (!check_attn_out_rows_geometry(4096, 1024, 8, 5120, 1, "8x1024x4096->5120")) return 0;
    /* And one whose out_dim is above 65536, where ds4_gpu_matmul_q8_0_tensor_impl() and
     * ds4_gpu_matmul_q8_0_decode_rows_exact_tensor() -- the two single-row producers compared
     * against above -- switch their k-split from nsg 4 to nsg 8. nsg is not an occupancy knob:
     * it sets ib0 = sgitg*NQ + ix and the NSG*NQ stride, so it decides which blocks are summed
     * into which simdgroup partial before the final simd_sum. A rows dispatch that kept nsg 4
     * here would differ by ULPs, and no V4.1-shaped case could ever see it. K = 2048 (64 blocks)
     * is the smallest K at which the two splits really do differ: at 32 blocks or fewer the extra
     * simdgroups own nothing and both trees reduce the same partials. */
    if (!check_attn_out_rows_geometry(4096, 1024, 2, 65538, 0, "2x1024x4096->65538 (nsg 8)"))
        return 0;
    return 1;
}

/* The env names ds4_v41_batch_attention_output_path() answers for, saved and restored around the
 * cases below so this check is independent of whatever the caller exported. */
static const char *const attn_out_path_envs[3] = {
    "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT",
    "DS4_METAL_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_MV_EXT",
    "DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_ROWS",
};

/* 3.6d2 (2026-09-18): which projection the step SELECTS, not which kernels are correct.
 *
 * --fusion-gates proves each rollback name is spelled the way the predicate that owns it spells
 * it.  It does not prove the predicate routes anything, and for this stage nothing else does
 * either: the entry points --attn-out-rows drives (ds4_gpu_attention_output_q8_rows_tensor,
 * ds4_gpu_dsv41_attention_output_rows) read no environment at all, so running that mode with
 * DS4_METAL_DISABLE_PRE_M5_V41_BATCH_ATTENTION_OUTPUT_ROWS exported executes exactly the same
 * code and is not evidence about the rollback; and the pair the rollback returns to is
 * bit-identical to the rows<R> kernels, so no exactness check can separate them either.  This is
 * the ds4.c half: the widths the rows kernels claim, that the rollback moves every one of them
 * onto 3.6d's pair, that 3.6d's own rollback removes the stage, and that the opt-in mv_ext
 * variant still outranks the rows kernels as 3.6d gave it.
 *
 * MAX_ROWS mirrors DS4_TP_BATCH_MAX_ROWS (ds4_tp.h, not included here, as check_attention_output_
 * decode does the same): if it moves, the width above it stops answering 1 and this fails, which
 * is the intent. */
static int check_attn_out_path_cases(void) {
    enum { MAX_ROWS = 8 };
    const int pre_m5 = ds4_gpu_device_is_pre_m5_apple_silicon();
    if (!pre_m5) {
        /* Nothing here is live off pre-M5 Apple silicon: every width takes the per-row loop, and
         * saying so is the whole check rather than a vacuous pass. */
        for (unsigned r = 1; r <= MAX_ROWS + 1u; r++)
            CHECK(ds4_v41_batch_attention_output_path(r) == 0);
        fprintf(stderr, "V4.1 batched attention output path: not pre-M5 Apple silicon, so every "
                        "width 1..%u takes the per-row loop: PASS\n", MAX_ROWS + 1u);
        return 1;
    }
    /* Default: the rows<R> kernels for exactly the widths their 2/4/8 templates cover, 3.6d's
     * pair outside them.  Width 1 never reaches the batched step at all (count >= 2). */
    CHECK(ds4_v41_batch_attention_output_path(1) == 1);
    for (unsigned r = 2; r <= MAX_ROWS; r++)
        CHECK(ds4_v41_batch_attention_output_path(r) == 3);
    CHECK(ds4_v41_batch_attention_output_path(MAX_ROWS + 1u) == 1);
    /* THE LIVENESS PROOF: the rollback must move every covered width off the rows kernels and
     * onto 3.6d's pair, and put them back when it is cleared. */
    CHECK(setenv(attn_out_path_envs[2], "1", 1) == 0);
    for (unsigned r = 1; r <= MAX_ROWS + 1u; r++)
        CHECK(ds4_v41_batch_attention_output_path(r) == 1);
    CHECK(unsetenv(attn_out_path_envs[2]) == 0);
    CHECK(ds4_v41_batch_attention_output_path(2) == 3);
    /* 3.6d's own rollback takes the whole stage out at every width, rows kernels included. */
    CHECK(setenv(attn_out_path_envs[0], "1", 1) == 0);
    for (unsigned r = 1; r <= MAX_ROWS + 1u; r++)
        CHECK(ds4_v41_batch_attention_output_path(r) == 0);
    CHECK(unsetenv(attn_out_path_envs[0]) == 0);
    /* And the opt-in, non-exact mv_ext variant still wins, at every width, exactly as under
     * 3.6d: the rows kernels must not have quietly taken its place. */
    CHECK(setenv(attn_out_path_envs[1], "1", 1) == 0);
    for (unsigned r = 1; r <= MAX_ROWS + 1u; r++)
        CHECK(ds4_v41_batch_attention_output_path(r) == 2);
    CHECK(unsetenv(attn_out_path_envs[1]) == 0);
    CHECK(ds4_v41_batch_attention_output_path(2) == 3);
    fprintf(stderr, "V4.1 batched attention output path: rows<R> at widths 2..%u and 3.6d's pair "
                    "outside them, %s moves every width back to the pair, the stage rollback "
                    "removes it and mv_ext still outranks it: PASS\n",
            MAX_ROWS, attn_out_path_envs[2]);
    return 1;
}

static int check_attn_out_path(void) {
    char *saved[3] = { NULL, NULL, NULL };
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        const char *v = getenv(attn_out_path_envs[i]);
        if (v) {
            saved[i] = strdup(v);
            CHECK(saved[i] != NULL);
        }
        CHECK(unsetenv(attn_out_path_envs[i]) == 0);
    }
    ok = check_attn_out_path_cases();
    for (int i = 0; i < 3; i++) {
        if (saved[i]) {
            setenv(attn_out_path_envs[i], saved[i], 1);
            free(saved[i]);
        } else {
            unsetenv(attn_out_path_envs[i]);
        }
    }
    return ok;
}

static void test_pre_commit_hook(void *ctx) { (*(unsigned *)ctx)++; }

/* 3.6d2 (2026-09-17): the pre-commit hook, on its own.
 *
 * The batched decode step stops joining its Engram reader threads before layer 1 is ENCODED and
 * waits until something is COMMITTED -- which is what the invariant in ds4.c has always said.
 * Whether that is safe now rests entirely on this mechanism, because the step cannot enumerate
 * its callees: ds4_gpu_routed_moe_batch_tensor() ends and restarts the command batch in every
 * layer under DS4_METAL_Q4_TABLE_RESIDENCY_SET, several stage profiles do the same, and the
 * selected-id readback commits from inside the attention stage. A hook that silently stopped
 * firing -- or fired on encode instead of on commit, or survived being cleared -- would turn that
 * into a data race on the row buffers, which shows up as nondeterministic logits and not as a
 * clean failure. Hence an explicit test of the three properties the step relies on: it runs on
 * every commit, it does not run on encode, and NULL clears it. */
static int check_pre_commit_hook(void) {
    unsigned calls = 0;
    ds4_gpu_tensor *t = upload(NULL, 4096);
    CHECK(t);

    ds4_gpu_set_pre_commit_hook(test_pre_commit_hook, &calls);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_fill_f32(t, 1.0f, 1024));
    CHECK(calls == 0);                      /* encoding is not committing */
    CHECK(ds4_gpu_flush_commands());
    CHECK(calls == 1);                      /* the step's own in-loop flush */
    CHECK(ds4_gpu_tensor_fill_f32(t, 2.0f, 1024));
    CHECK(calls == 1);
    CHECK(ds4_gpu_end_commands());
    CHECK(calls == 2);                      /* and every other way a batch ends */

    ds4_gpu_set_pre_commit_hook(NULL, NULL);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_fill_f32(t, 3.0f, 1024));
    CHECK(ds4_gpu_end_commands());
    CHECK(calls == 2);                      /* cleared means cleared */

    ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    fprintf(stderr, "V4.1 pre-commit hook: once per commit (flush and end), never on encode, "
                    "and NULL clears it: PASS\n");
    return 1;
}
#endif

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
    {   /* 3.6a (2026-09-17): 5120 is the residual norm; 512 (attn_compressor_norm) and 128
         * (indexer_k_norm) are the two widths the V4.1 publish tail asks for, and the batched
         * decode step replaces N of those calls with one rows dispatch. */
        static const uint32_t norm_widths[] = {5120, 512, 128};
        for (size_t wi = 0; wi < sizeof(norm_widths) / sizeof(*norm_widths); wi++) {
            const uint32_t n = norm_widths[wi];
            ds4_gpu_tensor *xv = ds4_gpu_tensor_view(xt, 0, n * 4);
            ds4_gpu_tensor *av = ds4_gpu_tensor_view(a, 0, n * 4), *bv = ds4_gpu_tensor_view(b, 0, n * 4);
            CHECK(xv && av && bv);
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_rms_norm_weight_tensor(av, xv, model, model_bytes, norm_off, n, 1e-6f));
            CHECK(ds4_gpu_dsv41_quantize(av, n, 1, DS4_V41_BF16));
            CHECK(ds4_gpu_rms_norm_weight_bf16_tensor(bv, xv, model, model_bytes, norm_off, n, 1e-6f));
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(ds4_gpu_tensor_contents(av), ds4_gpu_tensor_contents(bv), n * 4));
            fprintf(stderr, "weighted RMS norm + BF16 epilogue %u: bit-identical\n", n);
            ds4_gpu_tensor_free(xv); ds4_gpu_tensor_free(av); ds4_gpu_tensor_free(bv);
        }
        /* 3.6a: and the substitution itself. ds41_norm_batch() is
         * ds4_gpu_rms_norm_weight_rows_tensor + one batched BF16 quantize; the calls it replaces
         * are N x ds41_norm(), which on this device is the fused single-row kernel. Byte for byte,
         * per row, at the two widths and every decode batch width. Nothing pinned rows > 1 before:
         * the rows entry point is the same pipeline with dispatchThreadgroups(rows,1,1), and this
         * is the memcmp that says so. */
        {   static const uint32_t rows_widths[] = {512, 128};
            static const uint32_t row_counts[] = {1, 2, 4, 8};
            for (size_t wi = 0; wi < sizeof(rows_widths) / sizeof(*rows_widths); wi++)
            for (size_t ri = 0; ri < sizeof(row_counts) / sizeof(*row_counts); ri++) {
                const uint32_t n = rows_widths[wi], rows = row_counts[ri];
                const uint64_t bytes = (uint64_t)n * rows * 4;
                ds4_gpu_tensor *xv = ds4_gpu_tensor_view(xt, 0, bytes);
                ds4_gpu_tensor *av = ds4_gpu_tensor_view(a, 0, bytes);
                ds4_gpu_tensor *bv = ds4_gpu_tensor_view(b, 0, bytes);
                CHECK(xv && av && bv);
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_rms_norm_weight_rows_tensor(av, xv, model, model_bytes, norm_off,
                                                          n, rows, 1e-6f));
                CHECK(ds4_gpu_dsv41_quantize(av, n, rows, DS4_V41_BF16));
                for (uint32_t r = 0; r < rows; r++) {
                    ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * n * 4, n * 4);
                    ds4_gpu_tensor *br = ds4_gpu_tensor_view(b, (uint64_t)r * n * 4, n * 4);
                    CHECK(xr && br);
                    CHECK(ds4_gpu_rms_norm_weight_bf16_tensor(br, xr, model, model_bytes,
                                                              norm_off, n, 1e-6f));
                    ds4_gpu_tensor_free(br); ds4_gpu_tensor_free(xr);
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(!memcmp(ds4_gpu_tensor_contents(av), ds4_gpu_tensor_contents(bv), (size_t)bytes));
                fprintf(stderr, "weighted RMS norm rows n=%u rows=%u: bit-identical to %u fused "
                                "single-row norms\n", n, rows, rows);
                ds4_gpu_tensor_free(bv); ds4_gpu_tensor_free(av); ds4_gpu_tensor_free(xv);
            }
            fprintf(stderr, "V4.1 batched norm substitution (ds41_norm_batch vs N x ds41_norm) "
                            "at n=512,128 and N=1,2,4,8: PASS\n");
        }
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

/* 3.4: the concurrent-dispatch section for the V4.1 MoE layer.  Synthetic Q4_K
 * routed experts plus synthetic Q8_0 shared-expert weights at matching widths;
 * no model file and no GGUF.  The section replaces four serial calls -- two Q8
 * matvecs with the BF16 epilogue, the BF16 SwiGLU and the Q8 down matvec --
 * with three levels of one concurrent encoder.
 *
 * The same body runs on both routes the section arms, because a route is only
 * covered if the arm point inside it is exercised:
 *
 *   DS4_GPU_V41_PAR_ROUTE_ID     16 experts at the real V4.1 decode shapes
 *                                (in 5120, mid 2304, out 5120) with
 *                                force_resident = true.  n_total_expert is 16
 *                                and the expert tensors are far below the
 *                                2 GiB floor, so the grouped / group6 /
 *                                group8 / group24 / exact / table / addr /
 *                                gather / slots paths are all excluded and the
 *                                routed call takes the generic fused
 *                                pair-SwiGLU + direct sum6 branch -- the route
 *                                the resident V4.1 decode takes.
 *
 *   DS4_GPU_V41_PAR_ROUTE_SLOTS  128 experts at reduced shapes (in 512,
 *                                mid 256, out 512) with force_resident = false,
 *                                the selected-expert-view opt-in and the
 *                                test-only tensor-size override, which is the
 *                                only combination that turns
 *                                use_q4_selected_slots on: every other Q4 path
 *                                needs n_total_expert == 384 or its own enable
 *                                variable.  The routed call then takes the
 *                                selected-slots pair-SwiGLU + sum6 branch.
 *                                This route is NOT reachable from the shipping
 *                                V4.1 decode and this mode is not a
 *                                production-representative configuration: (a)
 *                                ds41_moe_partial passes force_resident =
 *                                !g->streaming and only plans a section when
 *                                !g->streaming, so force_resident is always
 *                                true there while use_q4_selected_slots needs
 *                                it false; and (b) the mode has to set the
 *                                selected-id override to keep the plan alive
 *                                across the route's host-side id readback,
 *                                and no ds41_* path ever sets that override
 *                                (see the comment at the setenv below).  What
 *                                it buys is coverage of the section's encode
 *                                schedule against the slots6 encoders, so the
 *                                schedule stays correct if that route ever
 *                                does become reachable.
 *
 * Five modes per route, each compared byte for byte against a serial reference
 * taken on the SAME routed route:
 *   0  serial baseline, pair-SwiGLU fusion on          -> reference A
 *   1  section armed, resource barriers                -> == A, joined, armed
 *   2  section armed, scope barriers                   -> == A, joined, armed
 *   3  serial baseline, fusion OFF (no section)        -> reference B
 *   4  section armed, fusion OFF                       -> == B, NOT joined and
 *                                                         NEVER armed
 * Mode 4 is the regression test for the arm points: with the fusion off the
 * routed call leaves both of them (the generic arm sits inside
 * `else if (fuse_pair_swiglu)`, and use_q4_selected_slots requires the fusion
 * too) and encodes a gate matvec, an up matvec and a dependent activation
 * dispatch with no barrier between them, so if the section had turned the
 * encoder concurrent before knowing the route, the routed mid/out would be
 * computed from unwritten gate/up -- the buffers are pre-filled with NaN to
 * make that loud.  The per-route armed counter additionally proves that no
 * concurrent encoder was opened at all, which is the structural version of the
 * same claim, and on modes 1 and 2 that the section armed on the route under
 * test and never on the other one.  Modes 0 and 1 differ only in whether
 * start() was called, and nothing start() does is read by any route predicate,
 * so mode 1's armed route also identifies mode 0's reference. */
static int check_v41_parallel_ffn_route(unsigned route) {
    typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    enum { SEL = 6, Q4_K_TYPE = 12, MODES = 5 };
    const int slots = route == DS4_GPU_V41_PAR_ROUTE_SLOTS;
    /* The slots route needs n_total_expert >= 128, so it trades expert width
     * for expert count and keeps the synthetic model at ~29 MiB. */
    const uint32_t D = slots ? 512u : 5120u;
    const uint32_t H = slots ? 256u : 2304u;
    const uint32_t E = slots ? 128u : 16u;
    const char *const route_name = slots ? "slots6" : "id";
    const uint64_t row = D / 256 * sizeof(q4_block);
    const uint64_t down_row = H / 256 * sizeof(q4_block);
    const uint64_t expert = (uint64_t)H * row;
    const uint64_t tensor = (uint64_t)E * expert;
    const uint64_t q4_bytes = 3 * tensor;
    const uint64_t shexp_gate_bytes = (uint64_t)H * (D / 32) * sizeof(q8_block);
    const uint64_t shexp_down_bytes = (uint64_t)D * (H / 32) * sizeof(q8_block);
    const uint64_t shexp_gate_off = q4_bytes;
    const uint64_t shexp_up_off = shexp_gate_off + shexp_gate_bytes;
    const uint64_t shexp_down_off = shexp_up_off + shexp_gate_bytes;
    const uint64_t bytes = shexp_down_off + shexp_down_bytes;
    CHECK((uint64_t)D * down_row == expert);

    unsetenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS");
    unsetenv("DS4_METAL_TEST_Q4_SELECTED_MIN_TENSOR_MB");
    if (slots) {
        /* Opt in to the selected-expert views off the streaming path, and
         * lower the 2 GiB routed-tensor floor to 1 MiB so a synthetic model
         * can reach the route at all.  Both are cleared again at the end. */
        setenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS", "1", 1);
        setenv("DS4_METAL_TEST_Q4_SELECTED_MIN_TENSOR_MB", "1", 1);
    }

    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    uint32_t r = 0x9e3779b9u;
    uint32_t *words = model;
    for (uint64_t i = 0; i < bytes / 4; i++) {
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        words[i] = r;
    }
    q4_block *blocks = model;
    for (uint64_t i = 0; i < q4_bytes / sizeof(*blocks); i++) {
        blocks[i].d = 0x1800;
        blocks[i].dmin = 0x1800;
    }
    q8_block *q8 = (q8_block *)((char *)model + shexp_gate_off);
    for (uint64_t i = 0; i < (bytes - q4_bytes) / sizeof(*q8); i++) {
        q8[i].d = 0x2000;
        for (int j = 0; j < 32; j++) q8[i].qs[j] = (int)(random_value() * 8192) % 8;
    }

    float *xh = malloc(D * 4);
    CHECK(xh);
    for (uint32_t i = 0; i < D; i++) xh[i] = random_value() / 8.0f;
    int32_t ids[SEL] = {0, 3, 5, 9, 12, 15};
    float weights[SEL];
    for (int s = 0; s < SEL; s++) weights[s] = (s + 1) / 21.0f;

    ds4_gpu_tensor *xt = upload(xh, D * 4);
    ds4_gpu_tensor *it = upload(ids, sizeof(ids));
    ds4_gpu_tensor *wt = upload(weights, sizeof(weights));
    ds4_gpu_tensor *gate = upload(NULL, (size_t)SEL * H * 4);
    ds4_gpu_tensor *up = upload(NULL, (size_t)SEL * H * 4);
    ds4_gpu_tensor *mid = upload(NULL, (size_t)SEL * H * 4);
    ds4_gpu_tensor *down = upload(NULL, (size_t)SEL * D * 4);
    ds4_gpu_tensor *out = upload(NULL, D * 4);
    ds4_gpu_tensor *sgate = upload(NULL, H * 4);
    ds4_gpu_tensor *sup = upload(NULL, H * 4);
    ds4_gpu_tensor *smid = upload(NULL, H * 4);
    ds4_gpu_tensor *sout = upload(NULL, D * 4);
    CHECK(xt && it && wt && gate && up && mid && down && out &&
          sgate && sup && smid && sout);
    CHECK(ds4_gpu_set_model_map(model, bytes));

    const size_t pair_bytes = (size_t)SEL * H * 4, out_bytes = D * 4;
    const size_t sh_bytes = H * 4;
    /* Two routed references: [0] with the pair-SwiGLU fusion on (modes 0-2),
     * [1] with it off (modes 3-4).  The shared expert is the same chain in
     * every mode, so one reference covers it. */
    float *ref_gate[2], *ref_up[2], *ref_mid[2], *ref_out[2];
    for (int i = 0; i < 2; i++) {
        ref_gate[i] = malloc(pair_bytes); ref_up[i] = malloc(pair_bytes);
        ref_mid[i] = malloc(pair_bytes); ref_out[i] = malloc(out_bytes);
        CHECK(ref_gate[i] && ref_up[i] && ref_mid[i] && ref_out[i]);
    }
    float *ref_sgate = malloc(sh_bytes), *ref_sup = malloc(sh_bytes);
    float *ref_smid = malloc(sh_bytes), *ref_sout = malloc(out_bytes);
    CHECK(ref_sgate && ref_sup && ref_smid && ref_sout);

    /* ds4_gpu_begin_commands only arms the section on pre-M5 Apple silicon (it
     * needs the BF16 epilogue).  Elsewhere every mode runs the serial chain and
     * the comparisons hold trivially; only the "the section really ran" checks
     * are conditioned on the device. */
    unsetenv("DS4_METAL_DISABLE_PRE_M5_V41_BF16_EPILOGUE");
    const int section_dev = ds4_gpu_device_is_pre_m5_apple_silicon() != 0;
    if (!section_dev)
        fprintf(stderr, "V4.1 parallel FFN: section unavailable on this device "
                        "(not pre-M5 Apple silicon)\n");
    static const char *const mode_name[MODES] = {
        "serial baseline",
        "section, resource barriers",
        "section, scope barriers",
        "serial baseline, fusion off",
        "section armed, fusion off (not admitted)"};
    /* mode:                         0      1      2      3      4   */
    static const int mode_section[MODES] = {0,     1,     1,     0,     1};
    static const int mode_nofuse[MODES]  = {0,     0,     0,     1,     1};
    static const int mode_ref[MODES]     = {0,     0,     0,     1,     1};
    int ok = 1;
    for (int mode = 0; mode < MODES && ok; mode++) {
        unsetenv("DS4_METAL_V41_PARALLEL_FFN_SCOPE_BARRIER");
        unsetenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
        if (mode == 2) setenv("DS4_METAL_V41_PARALLEL_FFN_SCOPE_BARRIER", "1", 1);
        if (mode_nofuse[mode]) setenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION", "1", 1);
        CHECK(ds4_gpu_tensor_fill_f32(gate, NAN, (uint64_t)SEL * H) &&
              ds4_gpu_tensor_fill_f32(up, NAN, (uint64_t)SEL * H) &&
              ds4_gpu_tensor_fill_f32(mid, NAN, (uint64_t)SEL * H) &&
              ds4_gpu_tensor_fill_f32(out, NAN, D) &&
              ds4_gpu_tensor_fill_f32(sgate, NAN, H) &&
              ds4_gpu_tensor_fill_f32(sup, NAN, H) &&
              ds4_gpu_tensor_fill_f32(smid, NAN, H) &&
              ds4_gpu_tensor_fill_f32(sout, NAN, D));

        const uint32_t armed_before_id =
            ds4_gpu_test_v41_parallel_ffn_armed_count(DS4_GPU_V41_PAR_ROUTE_ID);
        const uint32_t armed_before_slots =
            ds4_gpu_test_v41_parallel_ffn_armed_count(DS4_GPU_V41_PAR_ROUTE_SLOTS);
        CHECK(ds4_gpu_begin_commands());
        int par = 0;
        if (mode_section[mode]) {
            par = ds4_gpu_dsv41_parallel_ffn_start(sgate, sup, smid, sout,
                                                   model, bytes,
                                                   shexp_gate_off, shexp_up_off,
                                                   shexp_down_off, D, H, xt, 7.0f);
            if (par != section_dev)
                fprintf(stderr, "V4.1 parallel FFN %s mode=%d: start returned %d, expected %d\n",
                        route_name, mode, par, section_dev);
            CHECK(par == section_dev);
        }
        if (!par) {
            /* Exactly what ds41_moe_partial encodes without the section. */
            CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sgate, model, bytes, shexp_gate_off,
                                                  D, H, xt, 1));
            CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sup, model, bytes, shexp_up_off,
                                                  D, H, xt, 1));
            CHECK(ds4_gpu_swiglu_bf16_tensor(smid, sgate, sup, H, 7.0f, 1.0f));
            CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sout, model, bytes, shexp_down_off,
                                                  H, D, smid, 1));
        }
        /* The selected-slots route reads the expert ids on the host.  Without
         * an override that is a command-buffer boundary plus a readback in the
         * middle of the routed call, which would drop the planned section (by
         * design: end_commands calls ds4_gpu_parallel_ffn_reset_state, which
         * clears g_v41_par) and prove nothing.
         *
         * This IS a test shortcut, and it is the second of the two reasons the
         * slots arm point cannot be reached by the V4.1 decode (the first is
         * force_resident; see check_v41_parallel_ffn_route's header).  The
         * only non-test callers of ds4_gpu_routed_moe_set_selected_override
         * are V4/GLM graph paths in ds4.c -- metal_graph_decode_set_hash_selected_override,
         * metal_graph_decode_cpu_router, metal_graph_decode_selected_readahead_override,
         * metal_graph_selected_async_load_finish and
         * metal_graph_encode_decode_layer_phase -- and none of them plans a
         * section; ds41_moe_partial, which does, selects on the GPU via
         * ds4_gpu_router_select_tensor and never sets the override.  So what
         * this mode validates is the section's encode schedule on the slots
         * encoders, not a configuration any shipping path produces. */
        if (slots)
            CHECK(ds4_gpu_routed_moe_set_selected_override(ids, SEL));
        CHECK(ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, down, model, bytes,
                                            0, tensor, 2 * tensor,
                                            Q4_K_TYPE, Q4_K_TYPE,
                                            expert, row, expert, down_row,
                                            D, H, D, it, wt, E, SEL, 7.0f, xt,
                                            NULL, 0, !slots));
        int joined = 0;
        if (par) {
            joined = ds4_gpu_dsv41_parallel_ffn_finish();
            if (!joined) {
                /* The section encoded nothing (or what it encoded is fenced):
                 * the caller's whole serial chain, exactly as ds41_moe_partial
                 * runs it. */
                CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sgate, model, bytes, shexp_gate_off,
                                                      D, H, xt, 1));
                CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sup, model, bytes, shexp_up_off,
                                                      D, H, xt, 1));
                CHECK(ds4_gpu_swiglu_bf16_tensor(smid, sgate, sup, H, 7.0f, 1.0f));
                CHECK(ds4_gpu_matmul_q8_0_tensor_bf16(sout, model, bytes, shexp_down_off,
                                                      H, D, smid, 1));
            }
        }
        CHECK(ds4_gpu_end_commands());

        const int want_joined = (mode == 1 || mode == 2) && section_dev;
        if (joined != want_joined)
            fprintf(stderr, "V4.1 parallel FFN %s mode=%d (%s): joined=%d expected=%d\n",
                    route_name, mode, mode_name[mode], joined, want_joined);
        CHECK(joined == want_joined);
        const uint32_t armed_id =
            ds4_gpu_test_v41_parallel_ffn_armed_count(DS4_GPU_V41_PAR_ROUTE_ID) -
            armed_before_id;
        const uint32_t armed_slots =
            ds4_gpu_test_v41_parallel_ffn_armed_count(DS4_GPU_V41_PAR_ROUTE_SLOTS) -
            armed_before_slots;
        const uint32_t armed = slots ? armed_slots : armed_id;
        const uint32_t armed_other = slots ? armed_id : armed_slots;

        if (mode == 0 || mode == 3) {
            const int ri = mode_ref[mode];
            memcpy(ref_gate[ri], ds4_gpu_tensor_contents(gate), pair_bytes);
            memcpy(ref_up[ri], ds4_gpu_tensor_contents(up), pair_bytes);
            memcpy(ref_mid[ri], ds4_gpu_tensor_contents(mid), pair_bytes);
            memcpy(ref_out[ri], ds4_gpu_tensor_contents(out), out_bytes);
            /* The baseline must not be all-NaN: prove the kernels really ran. */
            CHECK(ref_out[ri][0] == ref_out[ri][0] &&
                  ref_mid[ri][0] == ref_mid[ri][0]);
        }
        if (mode == 0) {
            memcpy(ref_sgate, ds4_gpu_tensor_contents(sgate), sh_bytes);
            memcpy(ref_sup, ds4_gpu_tensor_contents(sup), sh_bytes);
            memcpy(ref_smid, ds4_gpu_tensor_contents(smid), sh_bytes);
            memcpy(ref_sout, ds4_gpu_tensor_contents(sout), out_bytes);
            CHECK(ref_sout[0] == ref_sout[0] && ref_smid[0] == ref_smid[0]);
        } else {
            if (memcmp(ds4_gpu_tensor_contents(sgate), ref_sgate, sh_bytes) ||
                memcmp(ds4_gpu_tensor_contents(sup), ref_sup, sh_bytes) ||
                memcmp(ds4_gpu_tensor_contents(smid), ref_smid, sh_bytes) ||
                memcmp(ds4_gpu_tensor_contents(sout), ref_sout, out_bytes)) {
                fprintf(stderr, "V4.1 parallel FFN %s mode=%d (%s): shared-expert mismatch\n",
                        route_name, mode, mode_name[mode]);
                ok = 0;
            }
            const int ri = mode_ref[mode];
            if (ok && mode != 3 &&
                (memcmp(ds4_gpu_tensor_contents(gate), ref_gate[ri], pair_bytes) ||
                 memcmp(ds4_gpu_tensor_contents(up), ref_up[ri], pair_bytes) ||
                 memcmp(ds4_gpu_tensor_contents(mid), ref_mid[ri], pair_bytes) ||
                 memcmp(ds4_gpu_tensor_contents(out), ref_out[ri], out_bytes))) {
                fprintf(stderr, "V4.1 parallel FFN %s mode=%d (%s): routed mismatch "
                                "vs reference %d\n",
                        route_name, mode, mode_name[mode], ri);
                ok = 0;
            }
        }
        /* The concurrent encoder must be opened for the admitted route and for
         * nothing else: mode 4 arms the section on a route that encodes a gate
         * matvec, an up matvec and a dependent activation dispatch with no
         * barrier between them, so a single concurrent encoder there is the
         * bug -- checked after the memcmps so a regression reports both the
         * structural cause and the corrupted output.  armed_other also pins
         * the route: the section must have opened its encoder inside the
         * branch under test and not the other one. */
        if (armed != (uint32_t)want_joined || armed_other != 0) {
            fprintf(stderr, "V4.1 parallel FFN %s mode=%d (%s): concurrent encoders "
                            "opened id=%u slots6=%u, expected %s=%d and the other 0\n",
                    route_name, mode, mode_name[mode], armed_id, armed_slots,
                    route_name, want_joined);
            ok = 0;
        }
        if (ok)
            fprintf(stderr, "V4.1 parallel FFN %-6s %-40s joined=%d armed=%u: bit-identical\n",
                    route_name, mode_name[mode], joined, armed);
    }
    unsetenv("DS4_METAL_V41_PARALLEL_FFN_SCOPE_BARRIER");
    unsetenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
    unsetenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS");
    unsetenv("DS4_METAL_TEST_Q4_SELECTED_MIN_TENSOR_MB");

    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(sgate); ds4_gpu_tensor_free(sup);
    ds4_gpu_tensor_free(smid); ds4_gpu_tensor_free(sout);
    /* Release Metal before the mapped block: the model bytes are wrapped by
     * cached no-copy buffer views (and a residency set) that only go away in
     * cleanup.  ds4_gpu_cleanup is idempotent, so main()'s call is a no-op. */
    ds4_gpu_cleanup();
    for (int i = 0; i < 2; i++) {
        free(ref_gate[i]); free(ref_up[i]); free(ref_mid[i]); free(ref_out[i]);
    }
    free(ref_sgate); free(ref_sup); free(ref_smid); free(ref_sout);
    free(xh); free(model);
    CHECK(ok);
    fprintf(stderr, "PRE_M5 V4.1 concurrent MoE section (%s route), 5 modes, "
                    "bit-identical: PASS\n", route_name);
    return 1;
}

static int check_v41_parallel_ffn(void) {
    return check_v41_parallel_ffn_route(DS4_GPU_V41_PAR_ROUTE_ID);
}

static int check_v41_parallel_ffn_slots(void) {
    return check_v41_parallel_ffn_route(DS4_GPU_V41_PAR_ROUTE_SLOTS);
}
#endif

int main(int argc, char **argv) {
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--fused-bf16")) {
        const int ok = ds4_gpu_init() && check_fused_bf16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--v41-parallel-ffn")) {
        const int ok = ds4_gpu_init() && check_v41_parallel_ffn();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--v41-parallel-ffn-slots")) {
        const int ok = ds4_gpu_init() && check_v41_parallel_ffn_slots();
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
    if (argc == 2 && !strcmp(argv[1], "--quantize-digest")) {
        const int ok = ds4_gpu_init() && check_quantize_digest();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rows-a")) {
        const int ok = ds4_gpu_init() && check_rows_a();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fusion-gates")) {
        /* ds4_gpu_init() first: the gates start with the Metal device name. */
        const int ok = ds4_gpu_init() && check_fusion_gates();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--attention-output-decode")) {
        const int ok = ds4_gpu_init() && check_attention_output_decode();
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
    if (argc == 2 && !strcmp(argv[1], "--flash-rows")) {
        const int ok = ds4_gpu_init() && check_flash_rows();
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
    if (argc == 2 && !strcmp(argv[1], "--attn-out-rows")) {
        const int ok = ds4_gpu_init() && check_attn_out_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--flash-rows-desc")) {
        /* No device and no model: the mapping is pure ds4.c. */
        return check_flash_rows_desc() ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--attn-out-path")) {
        /* ds4_gpu_init() first: the gates start with the Metal device name. */
        const int ok = ds4_gpu_init() && check_attn_out_path();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--pre-commit-hook")) {
        const int ok = ds4_gpu_init() && check_pre_commit_hook();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
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
             check_flash_rows() && check_tp_attention();
#ifdef __APPLE__
    if (ok) ok = check_rope_pair() && check_quantize_store() &&
                 check_rope_freqs() && check_rows_a() && check_fusion_gates() && check_attn_out_path() &&
                 check_flash_rows_desc() &&
                 check_attention_output_decode() && check_attn_out_rows() &&
                 check_pre_commit_hook();
#endif
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
