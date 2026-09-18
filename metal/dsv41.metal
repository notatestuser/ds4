// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

// One lane's adjacent pair. kernel_dsv41_rope and kernel_dsv41_rope2 both go
// through this, so the two dispatch shapes run the identical arithmetic in the
// identical order; only which tensor and head a threadgroup owns differs.
static inline void dsv41_rope_lane(device float *x, float frequency,
                                   uint width, uint heads, uint head, uint row,
                                   uint start, uint stride, uint inverse, uint lane) {
    const float theta = float(start + row * stride) * frequency;
    const float c = precise::cos(theta);
    const float s = inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)row * heads + head) * width +
                    width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_rope_lane(x, args.frequencies[lane], args.width, args.heads,
                    group.x, group.y, args.start, args.stride, args.inverse, lane);
}

struct ds4_metal_args_dsv41_rope2 {
    uint width, heads0, heads1, start, inverse;
    float frequencies[32];
};

// PRE_M5 (2026-09-17): the decode step's q RoPE and its single kv RoPE share a
// position and a frequency table, so one dispatch of heads0 + heads1 groups
// covers both. Each threadgroup picks its tensor from its index and runs the
// single-row case (row 0, where the stride is never read) of dsv41_rope_lane.
kernel void kernel_dsv41_rope2(
        constant ds4_metal_args_dsv41_rope2 &args,
        device float *x0,
        device float *x1,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const bool first = group < args.heads0;
    dsv41_rope_lane(first ? x0 : x1, args.frequencies[lane], args.width,
                    first ? args.heads0 : args.heads1,
                    first ? group : group - args.heads0,
                    0u, args.start, 1u, args.inverse, lane);
}

// PRE_M5 3.6a (2026-09-17): the per-element body of the V4.1 activation quantizers, hoisted so
// kernel_dsv41_quantize, kernel_dsv41_quantize_store2 and kernel_dsv41_quantize_window_rows run
// the identical arithmetic in the identical order and differ only in where the rounded value is
// stored - the treatment kernel_dsv41_rope and kernel_dsv41_rope2 already give dsv41_rope_lane.
// `value` must already be the lane's masked input (0 for lanes outside the block): simd_max below
// reduces over the whole simdgroup, so every lane has to reach it with the value the single-row
// kernel gave it. tests/test_deepseek41_metal --quantize-digest is the witness that this hoist
// moved nothing: run it against the pre-patch metal/dsv41.metal (DS4_METAL_DSV41_SOURCE) and
// against this one, and the two digests must be equal.
static inline float dsv41_quantize_value(float value, uint mode) {
    const float amax = simd_max(abs(value));
    float result = value;
    if (mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (mode == 2u || mode == 3u) {
        const float scale = mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    return dsv41_bf16(result);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float stored = dsv41_quantize_value(value, args.mode);
    if (valid) x[index] = stored;
}

// PRE_M5 (2026-09-17): the decode step quantizes its 512-float KV row and then
// copies that row into the layer's 128-slot sliding window. This is the body of
// kernel_dsv41_quantize with a second store of the same rounded value, so the
// window receives exactly the bytes the copy used to move.
kernel void kernel_dsv41_quantize_store2(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        device float *dst,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float stored = dsv41_quantize_value(value, args.mode);
    if (valid) {
        x[index] = stored;
        dst[index] = stored;
    }
}

// PRE_M5 3.6a (2026-09-17): one decode row of a V4.1 session batch, as the rows kernels see it.
// The five addresses are GPU addresses of session-private cache tensors, resolved on the host by
// ds4_gpu_v41_rows_set() the way ds4_gpu_encode_touch_table resolves its model ranges; every
// buffer behind one is passed to useResource on the encoder that dereferences it, so a missing one
// is an immediate GPU fault that MTL_DEBUG_LAYER=1 reports. 0 means "this layer does not use it",
// and each encode helper refuses a dispatch whose rows would dereference a 0, so a kernel never
// sees one. The record carries no sizes: the destinations below are indexed by (pos % slots),
// (n_comp - 1) and the width, all host-known, so ds4_metal.m checks each row against the
// destination tensor's own length -- the bound ds4_gpu_tensor_copy used to check -- before the
// dispatch is encoded. 64 B a row, 8 rows = 512 B, uploaded by value with setBytes;
// ds4_gpu_v41_row_gpu in ds4_metal.m mirrors this exactly.
struct ds4_metal_dsv41_row {
    ulong window, compressed, index_cache, previous_kv, previous_score;
    uint pos, n_comp, publish, reserved0, reserved1, reserved2;
};

// PRE_M5 3.6a: the decode step's RoPE for all N rows of a session batch in one dispatch. Each row
// rotates at its own absolute position, so the position comes from a table rather than from
// `start + row * stride`; passing stride 0 makes dsv41_rope_lane's float(start + row * stride)
// exactly float(positions[row]), which is the single-row call's theta, and the element index is
// the one that row's own view would have produced. Same device function, same frequencies, same
// order, so each row is bit-identical to the per-row dispatch it replaces.
kernel void kernel_dsv41_rope_rows(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        constant const uint *positions,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_rope_lane(x, args.frequencies[lane], args.width, args.heads,
                    group.x, group.y, positions[group.y], 0u, args.inverse, lane);
}

struct ds4_metal_args_dsv41_quantize_window {
    uint width, rows, mode, slots;
};

// PRE_M5 3.6a: quantize the N KV rows of a session batch and scatter each rounded row into its own
// session's sliding-window slot, in one dispatch. This is kernel_dsv41_quantize_store2 with the
// second destination taken from the row table instead of a bound buffer: one threadgroup per
// (block, row), the same simd_max over the same 32 lanes of the same row, and the window receives
// exactly the bytes ds4_gpu_tensor_copy moved.
kernel void kernel_dsv41_quantize_window_rows(
        constant ds4_metal_args_dsv41_quantize_window &args,
        device float *x,
        constant const ds4_metal_dsv41_row *tab,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float stored = dsv41_quantize_value(value, args.mode);
    if (valid) {
        x[index] = stored;
        device float *window = (device float *)tab[group.y].window;
        window[(ulong)(tab[group.y].pos % args.slots) * args.width + column] = stored;
    }
}

struct ds4_metal_args_dsv41_pool_rows {
    uint width, rows, ratio;
};

// PRE_M5 3.6a: the compressor pooling of all N rows of a session batch in one dispatch, with each
// row's previous_kv/previous_score carry taken from the row table. Row r runs exactly the branch
// ds4_gpu_dsv41_pool2 runs for a single decode row at position tab[r].pos:
//   ratio 2, odd pos  -> the pairs=1/tail=1 case, where a = -1 puts ka/sa in previous_* and sb/kv
//                        in the row's own slot: the identical expression in the identical order;
//   ratio 2, even pos -> the last_even carry, which stored kv and scores into previous_* with two
//                        ds4_gpu_tensor_copy calls (a float copy moves the same bytes);
//   ratio 1           -> the plain pool_kv -> latent copy.
// Even-position rows also get out = kv. The per-row path leaves their latent stale and nothing
// reads it (kernel_dsv41_publish_scatter_rows skips publish == 0), but writing it keeps the
// batched publish tail off uninitialised memory under shader validation.
kernel void kernel_dsv41_pool2_rows(
        constant ds4_metal_args_dsv41_pool_rows &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        constant const ds4_metal_dsv41_row *tab,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong i = (ulong)index.y * args.width + index.x;
    if (args.ratio != 2u || !(tab[index.y].pos & 1u)) {
        out[i] = kv[i];
        if (args.ratio == 2u) {
            ((device float *)tab[index.y].previous_kv)[index.x] = kv[i];
            ((device float *)tab[index.y].previous_score)[index.x] = scores[i];
        }
        return;
    }
    const float ka = ((device const float *)tab[index.y].previous_kv)[index.x];
    const float sa = ((device const float *)tab[index.y].previous_score)[index.x];
    const float sb = scores[i], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[i] = dsv41_bf16((ka * ea + kv[i] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_publish_rows {
    uint key_width, value_width, rows;
};

// PRE_M5 3.6a: publish the compressed key and latent of every publishing row into that row's own
// index_cache and compressed caches, at slot n_comp - 1. Replaces the two ds4_gpu_tensor_copy
// calls at the end of ds41_attention_publish; rows with publish == 0 are skipped and their caches
// are not touched at all.
kernel void kernel_dsv41_publish_scatter_rows(
        constant ds4_metal_args_dsv41_publish_rows &args,
        device const float *index_k,
        device const float *latent,
        constant const ds4_metal_dsv41_row *tab,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.value_width || index.y >= args.rows) return;
    if (!tab[index.y].publish) return;
    const ulong slot = (ulong)(tab[index.y].n_comp - 1u);
    if (index.x < args.key_width)
        ((device float *)tab[index.y].index_cache)[slot * args.key_width + index.x] =
            index_k[(ulong)index.y * args.key_width + index.x];
    ((device float *)tab[index.y].compressed)[slot * args.value_width + index.x] =
        latent[(ulong)index.y * args.value_width + index.x];
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif
