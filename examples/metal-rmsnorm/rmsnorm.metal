#include <metal_stdlib>
using namespace metal;

// Simple reference implementation: one thread owns a row. The editable task
// is this file; preserve the host's buffers, dispatch and float32 contract.
kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* weight [[buffer(1)]],
                    device float* output [[buffer(2)]],
                    constant uint& rows [[buffer(3)]],
                    constant uint& columns [[buffer(4)]],
                    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const uint start = row * columns;
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        const float value = x[start + column];
        sum += value * value;
    }
    const float scale = rsqrt(sum / float(columns) + 1e-5f);
    for (uint column = 0; column < columns; ++column)
        output[start + column] = x[start + column] * scale * weight[column];
}
