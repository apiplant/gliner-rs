#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

struct RelativeAttentionParams {
    uint dims[4]; // batch, heads, tokens, relative table width
    ulong strides[6][4]; // content, qRel, kRel, c2p, p2c, pair mask
};

kernel void gliner_relative_attention_f32(
    device const float *content [[buffer(0)]],
    device const float *qrel [[buffer(1)]],
    device const float *krel [[buffer(2)]],
    device const uint *c2p [[buffer(3)]],
    device const uint *p2c [[buffer(4)]],
    device const uchar *mask [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant RelativeAttentionParams &p [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    const uint heads = p.dims[1], tokens = p.dims[2];
    const uint count = p.dims[0] * heads * tokens * tokens;
    if (tid >= count) return;
    const uint j = tid % tokens;
    const uint i = (tid / tokens) % tokens;
    const uint h = (tid / (tokens * tokens)) % heads;
    const uint b = tid / (heads * tokens * tokens);
    const ulong mask_offset = b*p.strides[5][0] + i*p.strides[5][2] + j*p.strides[5][3];
    if (mask[mask_offset] == 0) {
        output[tid] = as_type<float>(0xff7fffffu);
        return;
    }
    const uint ci = c2p[i*p.strides[3][2] + j*p.strides[3][3]];
    const uint pi = p2c[j*p.strides[4][2] + i*p.strides[4][3]];
    const ulong content_offset = b*p.strides[0][0] + h*p.strides[0][1] + i*p.strides[0][2] + j*p.strides[0][3];
    const ulong q_offset = b*p.strides[1][0] + h*p.strides[1][1] + i*p.strides[1][2] + ci*p.strides[1][3];
    const ulong k_offset = b*p.strides[2][0] + h*p.strides[2][1] + j*p.strides[2][2] + pi*p.strides[2][3];
    // Keep both additions independently rounded, as in Candle's two binary
    // kernels. Its affine kernel explicitly uses fma(x, scale, 0).
    const float first_sum = content[content_offset] + qrel[q_offset];
    const float second_sum = first_sum + krel[k_offset];
    output[tid] = fma(second_sum, scale, 0.0f);
}
