// VectorStoreKit: Pooling Operations Metal Shaders
//
// Max pooling, average pooling, and global pooling operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Max Pooling

/// 2D Max Pooling forward pass
kernel void maxpool2d_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device uint* indices [[buffer(2)]],  // Store max indices for backward pass
    constant uint& batch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& inputHeight [[buffer(5)]],
    constant uint& inputWidth [[buffer(6)]],
    constant uint& outputHeight [[buffer(7)]],
    constant uint& outputWidth [[buffer(8)]],
    constant uint& poolHeight [[buffer(9)]],
    constant uint& poolWidth [[buffer(10)]],
    constant uint& strideHeight [[buffer(11)]],
    constant uint& strideWidth [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint outX = gid.x;
    uint outY = gid.y;
    uint n = gid.z / channels;  // batch index
    uint c = gid.z % channels;  // channel index
    
    if (outX >= outputWidth || outY >= outputHeight || n >= batch) return;
    
    // Calculate input region
    uint inYStart = outY * strideHeight;
    uint inXStart = outX * strideWidth;
    
    // Find maximum value and its index
    float maxVal = -INFINITY;
    uint maxIdx = 0;
    
    for (uint py = 0; py < poolHeight; py++) {
        for (uint px = 0; px < poolWidth; px++) {
            uint inY = inYStart + py;
            uint inX = inXStart + px;
            
            if (inY < inputHeight && inX < inputWidth) {
                uint inputIdx = ((n * channels + c) * inputHeight + inY) * inputWidth + inX;
                float val = input[inputIdx];
                
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = inputIdx;
                }
            }
        }
    }
    
    uint outputIdx = ((n * channels + c) * outputHeight + outY) * outputWidth + outX;
    output[outputIdx] = maxVal;
    
    if (indices) {
        indices[outputIdx] = maxIdx;
    }
}

/// 2D Max Pooling backward pass
kernel void maxpool2d_backward(
    constant float* gradOutput [[buffer(0)]],
    constant uint* indices [[buffer(1)]],
    device atomic_float* gradInput [[buffer(2)]],  // Use atomic for accumulation
    constant uint& batch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& inputHeight [[buffer(5)]],
    constant uint& inputWidth [[buffer(6)]],
    constant uint& outputHeight [[buffer(7)]],
    constant uint& outputWidth [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint outX = gid.x;
    uint outY = gid.y;
    uint n = gid.z / channels;
    uint c = gid.z % channels;
    
    if (outX >= outputWidth || outY >= outputHeight || n >= batch) return;
    
    uint outputIdx = ((n * channels + c) * outputHeight + outY) * outputWidth + outX;
    float grad = gradOutput[outputIdx];
    uint maxIdx = indices[outputIdx];
    
    // Accumulate gradient at the max index
    atomic_fetch_add_explicit(&gradInput[maxIdx], grad, memory_order_relaxed);
}

// MARK: - Average Pooling

/// 2D Average Pooling forward pass
kernel void avgpool2d_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& inputHeight [[buffer(4)]],
    constant uint& inputWidth [[buffer(5)]],
    constant uint& outputHeight [[buffer(6)]],
    constant uint& outputWidth [[buffer(7)]],
    constant uint& poolHeight [[buffer(8)]],
    constant uint& poolWidth [[buffer(9)]],
    constant uint& strideHeight [[buffer(10)]],
    constant uint& strideWidth [[buffer(11)]],
    constant uint& includePadding [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint outX = gid.x;
    uint outY = gid.y;
    uint n = gid.z / channels;
    uint c = gid.z % channels;
    
    if (outX >= outputWidth || outY >= outputHeight || n >= batch) return;
    
    // Calculate input region
    uint inYStart = outY * strideHeight;
    uint inXStart = outX * strideWidth;
    
    float sum = 0.0f;
    uint count = 0;
    
    for (uint py = 0; py < poolHeight; py++) {
        for (uint px = 0; px < poolWidth; px++) {
            uint inY = inYStart + py;
            uint inX = inXStart + px;
            
            if (inY < inputHeight && inX < inputWidth) {
                uint inputIdx = ((n * channels + c) * inputHeight + inY) * inputWidth + inX;
                sum += input[inputIdx];
                count++;
            }
        }
    }
    
    // Compute average
    float divisor = includePadding ? float(poolHeight * poolWidth) : float(count);
    uint outputIdx = ((n * channels + c) * outputHeight + outY) * outputWidth + outX;
    output[outputIdx] = sum / divisor;
}

/// 2D Average Pooling backward pass
kernel void avgpool2d_backward(
    constant float* gradOutput [[buffer(0)]],
    device atomic_float* gradInput [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& inputHeight [[buffer(4)]],
    constant uint& inputWidth [[buffer(5)]],
    constant uint& outputHeight [[buffer(6)]],
    constant uint& outputWidth [[buffer(7)]],
    constant uint& poolHeight [[buffer(8)]],
    constant uint& poolWidth [[buffer(9)]],
    constant uint& strideHeight [[buffer(10)]],
    constant uint& strideWidth [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint outX = gid.x;
    uint outY = gid.y;
    uint n = gid.z / channels;
    uint c = gid.z % channels;
    
    if (outX >= outputWidth || outY >= outputHeight || n >= batch) return;
    
    uint outputIdx = ((n * channels + c) * outputHeight + outY) * outputWidth + outX;
    float grad = gradOutput[outputIdx];
    
    // Calculate input region
    uint inYStart = outY * strideHeight;
    uint inXStart = outX * strideWidth;
    
    // Count valid positions for proper gradient scaling
    uint count = 0;
    for (uint py = 0; py < poolHeight; py++) {
        for (uint px = 0; px < poolWidth; px++) {
            uint inY = inYStart + py;
            uint inX = inXStart + px;
            if (inY < inputHeight && inX < inputWidth) {
                count++;
            }
        }
    }
    
    float gradScale = grad / float(count);
    
    // Distribute gradient
    for (uint py = 0; py < poolHeight; py++) {
        for (uint px = 0; px < poolWidth; px++) {
            uint inY = inYStart + py;
            uint inX = inXStart + px;
            
            if (inY < inputHeight && inX < inputWidth) {
                uint inputIdx = ((n * channels + c) * inputHeight + inY) * inputWidth + inX;
                atomic_fetch_add_explicit(&gradInput[inputIdx], gradScale, memory_order_relaxed);
            }
        }
    }
}

// MARK: - Global Average Pooling

/// Global Average Pooling forward pass
kernel void global_avgpool2d_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch * channels) return;
    
    uint n = gid / channels;
    uint c = gid % channels;
    
    float sum = 0.0f;
    uint spatialSize = height * width;
    uint channelOffset = (n * channels + c) * spatialSize;
    
    for (uint i = 0; i < spatialSize; i++) {
        sum += input[channelOffset + i];
    }
    
    output[gid] = sum / float(spatialSize);
}

/// Global Average Pooling backward pass
kernel void global_avgpool2d_backward(
    constant float* gradOutput [[buffer(0)]],
    device float* gradInput [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint spatialSize = height * width;
    uint totalThreads = batch * channels * spatialSize;
    
    if (gid >= totalThreads) return;
    
    uint n = gid / (channels * spatialSize);
    uint c = (gid / spatialSize) % channels;
    uint spatial = gid % spatialSize;
    
    float grad = gradOutput[n * channels + c] / float(spatialSize);
    gradInput[gid] = grad;
}

// MARK: - Utility functions

/// Clear buffer (set to zero)
kernel void clear_buffer(
    device float* buffer [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    buffer[gid] = 0.0f;
}