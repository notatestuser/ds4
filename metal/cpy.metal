struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;
template [[host_name("kernel_cpy_f16_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<half, half>;

// Contiguous 1D conversions avoid the generic tensor-index reconstruction
// above. Packed vector types retain scalar alignment, so tensor views whose
// offsets are float/half aligned do not need additional 16/8-byte alignment.
// Bind scalar pointers so Metal also accepts buffers containing fewer than
// four elements. Vectorize only complete groups; the tail stays scalar.
kernel void kernel_cpy_contig_f32_f16_4(
        constant uint & n,
        device const float * src,
        device       half  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const float4 value = float4(((device const packed_float4 *)src)[gid]);
        ((device packed_half4 *)dst)[gid] = packed_half4(half4(value));
        return;
    }

    for (uint lane = 0; lane < remaining; ++lane) {
        dst[i + lane] = half(src[i + lane]);
    }
}

kernel void kernel_cpy_contig_f16_f32_4(
        constant uint & n,
        device const half  * src,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const half4 value = half4(((device const packed_half4 *)src)[gid]);
        ((device packed_float4 *)dst)[gid] = packed_float4(float4(value));
        return;
    }

    for (uint lane = 0; lane < remaining; ++lane) {
        dst[i + lane] = float(src[i + lane]);
    }
}

// Bitwise F16 transport for cache staging. Use ushort rather than half so NaN
// payloads and every other binary16 encoding pass through unchanged.
kernel void kernel_cpy_contig_f16_f16_bits_4(
        constant uint & n,
        device const ushort * src,
        device       ushort * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        ((device packed_ushort4 *)dst)[gid] = ((device const packed_ushort4 *)src)[gid];
        return;
    }

    for (uint lane = 0; lane < remaining; ++lane) {
        dst[i + lane] = src[i + lane];
    }
}

struct ds4_metal_args_flash_kv_stage_f16 {
    uint raw_cap;
    uint raw_start;
    uint n_raw;
    uint n_comp;
    uint pad_rows;
    uint shared_pad;
};

// Decode-time gathered attention consumes a logical raw-cache ring followed
// by an already-F16 compressed cache. Pack both regions into the contiguous
// F16 FlashAttention scratch in one dispatch. The raw conversion expression
// and compressed ushort4 transport exactly match the standalone copy kernels.
kernel void kernel_dsv4_flash_kv_stage_f16(
        constant ds4_metal_args_flash_kv_stage_f16 & args,
        device const char * raw_src,
        device const char * comp_src,
        device       char * dst,
        device const char * mask_src,
        device       char * pad_dst,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint row_vecs = 128;
    const uint raw_vecs = args.n_raw * row_vecs;
    const uint n_keys = args.n_raw + args.n_comp;
    const uint total_vecs = n_keys * row_vecs;

    if (gid < raw_vecs) {
        const uint logical_row = gid >> 7;
        const uint col = gid & 127u;
        uint physical_row = args.raw_start + logical_row;
        if (physical_row >= args.raw_cap) {
            physical_row -= args.raw_cap;
        }
        device const packed_float4 *raw =
            (device const packed_float4 *)raw_src;
        device packed_half4 *dst_half = (device packed_half4 *)dst;
        const float4 value =
            float4(raw[physical_row * row_vecs + col]);
        dst_half[gid] = packed_half4(half4(value));
        return;
    }

    if (gid < total_vecs) {
        device const packed_ushort4 *comp =
            (device const packed_ushort4 *)comp_src;
        device packed_ushort4 *dst_bits = (device packed_ushort4 *)dst;
        dst_bits[gid] = comp[gid - raw_vecs];
        return;
    }

    // The vector FlashAttention kernel redirects its final partial block to
    // a compact K/V/mask buffer. When requested, append those writes to this
    // dispatch so gathered decode does not need a standalone pad dispatch.
    const uint pad_rows = args.pad_rows;
    const uint pad_vecs = pad_rows * row_vecs;
    uint pad_gid = gid - total_vecs;
    if (pad_gid < pad_vecs) {
        const uint row = pad_gid >> 7;
        const uint col = pad_gid & 127u;
        const uint valid_rows = n_keys % pad_rows;
        device packed_half4 *pad_half = (device packed_half4 *)pad_dst;
        device packed_ushort4 *pad_bits = (device packed_ushort4 *)pad_dst;
        if (row >= valid_rows) {
            const packed_half4 zero = packed_half4(half4(0.0h));
            pad_half[pad_gid] = zero;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = zero;
            }
            return;
        }

        const uint logical_row = n_keys - valid_rows + row;
        if (logical_row < args.n_raw) {
            uint physical_row = args.raw_start + logical_row;
            if (physical_row >= args.raw_cap) {
                physical_row -= args.raw_cap;
            }
            device const packed_float4 *raw =
                (device const packed_float4 *)raw_src;
            const float4 value =
                float4(raw[physical_row * row_vecs + col]);
            const packed_half4 value_half = packed_half4(half4(value));
            pad_half[pad_gid] = value_half;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = value_half;
            }
        } else {
            device const packed_ushort4 *comp =
                (device const packed_ushort4 *)comp_src;
            const packed_ushort4 value_bits =
                comp[(logical_row - args.n_raw) * row_vecs + col];
            pad_bits[pad_gid] = value_bits;
            if (!args.shared_pad) {
                pad_bits[pad_vecs + pad_gid] = value_bits;
            }
        }
        return;
    }

    pad_gid -= pad_vecs;
    if (pad_gid < pad_rows) {
        const uint valid_rows = n_keys % pad_rows;
        device const ushort *mask_bits = (device const ushort *)mask_src;
        device ushort *pad_mask_bits =
            (device ushort *)pad_dst + 2u * pad_vecs * 4u;
        pad_mask_bits[pad_gid] = pad_gid < valid_rows
            ? mask_bits[n_keys - valid_rows + pad_gid]
            : 0xfbffu;
    }
}


// PRE_M5 3.6c (2026-09-17): stage one decode row of a batched step into its slice of the shared
// FlashAttention K/V slab, and write that row's mask.  This is ds4_gpu_dsv41_gather_kv (get_rows
// f32 into selected_kv), the raw-ring f32->f16 copy, the selected_kv f32->f16 copy and
// kernel_flash_attn_ext_pad, in one dispatch, writing straight into a slab padded to the BATCH's
// key count so the vector kernel can run all N rows at once with ne03 = N.
//
// Bit-exactness against the four dispatches it replaces:
//   * raw region -- the same ring arithmetic and the same conversion expression as
//     kernel_dsv4_flash_kv_stage_f16 above and kernel_cpy_contig_f32_f16_4;
//   * gathered region -- get_rows copies compressed[id] verbatim and the f32->f16 copy then applies
//     packed_half4(half4(float4(...))); doing both in one read gives the same bits;
//   * padding -- half4(0) for K/V and 0xfbff (-MAXHALF) for the mask, the values
//     kernel_flash_attn_ext_pad writes for the tail of a partial block, extended to the batch's
//     n_keys_max so every block past this row's keys is skipped by the kernel's fully-masked test.
// n_keys_max is a multiple of 32 (the kernel's C), so no partial block survives and the vector
// kernel runs with has_kvpad = false.
struct ds4_metal_args_flash_stage_rows {
    uint n_keys_max;   // padded key count, a multiple of 32; the slab stride of every row
    uint n_raw;        // this row's sliding-window keys
    uint raw_start;    // first physical row of the window ring
    uint raw_cap;      // window ring capacity (128 in V4.1 decode)
    uint attended;     // gathered compressed keys (0 = raw only)
    uint source_rows;  // rows available in `comp_src`; ids are indices into it
};

kernel void kernel_dsv41_flash_stage_rows(
        constant ds4_metal_args_flash_stage_rows & args,
        device const char  * raw_src,
        device const char  * comp_src,
        device const int   * ids,
        device       char  * dst,
        device       char  * mask,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint row_vecs = 128;   // 512 floats = 128 packed-4 vectors
    const uint n_keys = args.n_raw + args.attended;
    const uint total_vecs = args.n_keys_max * row_vecs;

    if (gid < total_vecs) {
        const uint key = gid >> 7;
        const uint col = gid & 127u;
        device packed_half4 *dst_half = (device packed_half4 *)dst;
        if (key < args.n_raw) {
            uint physical_row = args.raw_start + key;
            if (physical_row >= args.raw_cap) {
                physical_row -= args.raw_cap;
            }
            device const packed_float4 *raw = (device const packed_float4 *)raw_src;
            const float4 value = float4(raw[physical_row * row_vecs + col]);
            dst_half[gid] = packed_half4(half4(value));
        } else if (key < n_keys) {
            const uint id = (uint)ids[key - args.n_raw];
            device const packed_float4 *comp = (device const packed_float4 *)comp_src;
            const float4 value =
                float4(comp[(id < args.source_rows ? id : 0u) * row_vecs + col]);
            dst_half[gid] = packed_half4(half4(value));
        } else {
            dst_half[gid] = packed_half4(half4(0.0h));
        }
        return;
    }

    const uint mask_gid = gid - total_vecs;
    if (mask_gid < args.n_keys_max) {
        device ushort *mask_bits = (device ushort *)mask;
        mask_bits[mask_gid] = mask_gid < n_keys ? 0u : 0xfbffu;
    }
}


// Tiled-row expansion of a small table: dst row t = src row (pos0 + t) % ratio.
// Replaces the per-segment copies the compressor store used to encode (one
// dispatch per prefill layer instead of n_tokens/ratio single-threadgroup
// copies).  Values are identical to the contiguous copies.
struct ds4_cpy_tile_rows_args {
    uint n_total;   // n_tokens * width
    uint width;
    uint ratio;
    uint pos0;
};

kernel void kernel_cpy_tile_rows_f16_f32(
        constant ds4_cpy_tile_rows_args & args,
        device const half  * src,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_total) return;
    const uint t = gid / args.width;
    const uint e = gid - t * args.width;
    const uint r = (args.pos0 + t) % args.ratio;
    dst[gid] = (float)src[r * args.width + e];
}

kernel void kernel_cpy_tile_rows_f32_f32(
        constant ds4_cpy_tile_rows_args & args,
        device const float * src,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_total) return;
    const uint t = gid / args.width;
    const uint e = gid - t * args.width;
    const uint r = (args.pos0 + t) % args.ratio;
    dst[gid] = src[r * args.width + e];
}
