// VectorStoreKit: Convolution Metal Shaders
//
// High-performance convolution operations for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - 2D Convolution Forward

/// Standard 2D convolution
kernel void conv2d_forward(
    constant float* input [[buffer(0)]],     // [N, C_in, H_in, W_in]
    constant float* weights [[buffer(1)]],   // [C_out, C_in, K_h, K_w]
    constant float* bias [[buffer(2)]],      // [C_out] (optional)
    device float* output [[buffer(3)]],      // [N, C_out, H_out, W_out]
    constant uint4& input_shape [[buffer(4)]],   // [N, C_in, H_in, W_in]
    constant uint4& weight_shape [[buffer(5)]],  // [C_out, C_in, K_h, K_w]
    constant uint4& output_shape [[buffer(6)]],  // [N, C_out, H_out, W_out]
    constant uint2& stride [[buffer(7)]],        // [stride_h, stride_w]
    constant uint2& padding [[buffer(8)]],       // [pad_h, pad_w]
    constant uint2& dilation [[buffer(9)]],      // [dilation_h, dilation_w]
    constant uint& groups [[buffer(10)]],
    constant bool& use_bias [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Get output coordinates
    uint out_w = gid.x;
    uint out_h = gid.y;
    uint out_c = gid.z % output_shape[1];
    uint batch = gid.z / output_shape[1];
    
    // Bounds check
    if (out_w >= output_shape[3] || out_h >= output_shape[2] || 
        out_c >= output_shape[1] || batch >= output_shape[0]) {
        return;
    }
    
    // Calculate input channel range for this group
    uint channels_per_group = input_shape[1] / groups;
    uint group = out_c / (output_shape[1] / groups);
    uint in_channel_start = group * channels_per_group;
    uint in_channel_end = in_channel_start + channels_per_group;
    
    float sum = 0.0f;
    
    // Perform convolution
    for (uint in_c = in_channel_start; in_c < in_channel_end; in_c++) {
        for (uint k_h = 0; k_h < weight_shape[2]; k_h++) {
            for (uint k_w = 0; k_w < weight_shape[3]; k_w++) {
                // Calculate input position
                int in_h = int(out_h * stride.x) - int(padding.x) + int(k_h * dilation.x);
                int in_w = int(out_w * stride.y) - int(padding.y) + int(k_w * dilation.y);
                
                // Check bounds
                if (in_h >= 0 && in_h < int(input_shape[2]) && 
                    in_w >= 0 && in_w < int(input_shape[3])) {
                    
                    // Calculate indices
                    uint input_idx = batch * input_shape[1] * input_shape[2] * input_shape[3] +
                                    in_c * input_shape[2] * input_shape[3] +
                                    in_h * input_shape[3] + in_w;
                    
                    uint weight_idx = out_c * channels_per_group * weight_shape[2] * weight_shape[3] +
                                     (in_c - in_channel_start) * weight_shape[2] * weight_shape[3] +
                                     k_h * weight_shape[3] + k_w;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    // Add bias if needed
    if (use_bias) {
        sum += bias[out_c];
    }
    
    // Calculate output index and store result
    uint output_idx = batch * output_shape[1] * output_shape[2] * output_shape[3] +
                     out_c * output_shape[2] * output_shape[3] +
                     out_h * output_shape[3] + out_w;
    
    output[output_idx] = sum;
}

// MARK: - 2D Convolution Backward (Input Gradient)

/// Compute gradients with respect to input (transposed convolution)
kernel void conv2d_backward_input(
    constant float* grad_output [[buffer(0)]],  // [N, C_out, H_out, W_out]
    constant float* weights [[buffer(1)]],      // [C_out, C_in, K_h, K_w]
    device float* grad_input [[buffer(2)]],     // [N, C_in, H_in, W_in]
    constant uint4& grad_output_shape [[buffer(3)]],
    constant uint4& weight_shape [[buffer(4)]],
    constant uint4& grad_input_shape [[buffer(5)]],
    constant uint2& stride [[buffer(6)]],
    constant uint2& padding [[buffer(7)]],
    constant uint2& dilation [[buffer(8)]],
    constant uint& groups [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Get input coordinates
    uint in_w = gid.x;
    uint in_h = gid.y;
    uint in_c = gid.z % grad_input_shape[1];
    uint batch = gid.z / grad_input_shape[1];
    
    // Bounds check
    if (in_w >= grad_input_shape[3] || in_h >= grad_input_shape[2] || 
        in_c >= grad_input_shape[1] || batch >= grad_input_shape[0]) {
        return;
    }
    
    // Calculate output channel range for this group
    uint channels_per_group_out = grad_output_shape[1] / groups;
    uint channels_per_group_in = grad_input_shape[1] / groups;
    uint group = in_c / channels_per_group_in;
    uint out_channel_start = group * channels_per_group_out;
    uint out_channel_end = out_channel_start + channels_per_group_out;
    
    float sum = 0.0f;
    
    // Accumulate gradients
    for (uint out_c = out_channel_start; out_c < out_channel_end; out_c++) {
        for (uint k_h = 0; k_h < weight_shape[2]; k_h++) {
            for (uint k_w = 0; k_w < weight_shape[3]; k_w++) {
                // Calculate output position that affects this input
                int out_h = (int(in_h) + int(padding.x) - int(k_h * dilation.x));
                int out_w = (int(in_w) + int(padding.y) - int(k_w * dilation.y));
                
                // Check if this output position is valid and aligns with stride
                if (out_h >= 0 && out_h % int(stride.x) == 0 && 
                    out_w >= 0 && out_w % int(stride.y) == 0) {
                    
                    out_h /= int(stride.x);
                    out_w /= int(stride.y);
                    
                    if (out_h < int(grad_output_shape[2]) && out_w < int(grad_output_shape[3])) {
                        // Calculate indices
                        uint grad_output_idx = batch * grad_output_shape[1] * grad_output_shape[2] * grad_output_shape[3] +
                                             out_c * grad_output_shape[2] * grad_output_shape[3] +
                                             out_h * grad_output_shape[3] + out_w;
                        
                        // Flip kernel for transposed convolution
                        uint weight_idx = out_c * channels_per_group_in * weight_shape[2] * weight_shape[3] +
                                         (in_c - group * channels_per_group_in) * weight_shape[2] * weight_shape[3] +
                                         k_h * weight_shape[3] + k_w;
                        
                        sum += grad_output[grad_output_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }
    
    // Store gradient
    uint grad_input_idx = batch * grad_input_shape[1] * grad_input_shape[2] * grad_input_shape[3] +
                         in_c * grad_input_shape[2] * grad_input_shape[3] +
                         in_h * grad_input_shape[3] + in_w;
    
    grad_input[grad_input_idx] = sum;
}

// MARK: - 2D Convolution Backward (Weight Gradient)

/// Compute gradients with respect to weights
kernel void conv2d_backward_weight(
    constant float* input [[buffer(0)]],         // [N, C_in, H_in, W_in]
    constant float* grad_output [[buffer(1)]],   // [N, C_out, H_out, W_out]
    device float* grad_weights [[buffer(2)]],    // [C_out, C_in, K_h, K_w]
    constant uint4& input_shape [[buffer(3)]],
    constant uint4& grad_output_shape [[buffer(4)]],
    constant uint4& weight_shape [[buffer(5)]],
    constant uint2& stride [[buffer(6)]],
    constant uint2& padding [[buffer(7)]],
    constant uint2& dilation [[buffer(8)]],
    constant uint& groups [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Get weight coordinates
    uint k_w = gid.x % weight_shape[3];
    uint k_h = (gid.x / weight_shape[3]) % weight_shape[2];
    uint in_c = gid.y;
    uint out_c = gid.z;
    
    // Bounds check
    if (k_w >= weight_shape[3] || k_h >= weight_shape[2] || 
        in_c >= weight_shape[1] || out_c >= weight_shape[0]) {
        return;
    }
    
    float sum = 0.0f;
    
    // Accumulate over batch and spatial dimensions
    for (uint batch = 0; batch < input_shape[0]; batch++) {
        for (uint out_h = 0; out_h < grad_output_shape[2]; out_h++) {
            for (uint out_w = 0; out_w < grad_output_shape[3]; out_w++) {
                // Calculate input position
                int in_h = int(out_h * stride.x) - int(padding.x) + int(k_h * dilation.x);
                int in_w = int(out_w * stride.y) - int(padding.y) + int(k_w * dilation.y);
                
                // Check bounds
                if (in_h >= 0 && in_h < int(input_shape[2]) && 
                    in_w >= 0 && in_w < int(input_shape[3])) {
                    
                    // Calculate indices
                    uint input_idx = batch * input_shape[1] * input_shape[2] * input_shape[3] +
                                    in_c * input_shape[2] * input_shape[3] +
                                    in_h * input_shape[3] + in_w;
                    
                    uint grad_output_idx = batch * grad_output_shape[1] * grad_output_shape[2] * grad_output_shape[3] +
                                          out_c * grad_output_shape[2] * grad_output_shape[3] +
                                          out_h * grad_output_shape[3] + out_w;
                    
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }
    
    // Store gradient
    uint weight_idx = out_c * weight_shape[1] * weight_shape[2] * weight_shape[3] +
                     in_c * weight_shape[2] * weight_shape[3] +
                     k_h * weight_shape[3] + k_w;
    
    grad_weights[weight_idx] = sum;
}

// MARK: - 2D Convolution Backward (Bias Gradient)

/// Compute gradients with respect to bias
kernel void conv2d_backward_bias(
    constant float* grad_output [[buffer(0)]],   // [N, C_out, H_out, W_out]
    device float* grad_bias [[buffer(1)]],       // [C_out]
    constant uint4& grad_output_shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint out_c = gid;
    
    if (out_c >= grad_output_shape[1]) {
        return;
    }
    
    float sum = 0.0f;
    
    // Sum over batch and spatial dimensions
    for (uint batch = 0; batch < grad_output_shape[0]; batch++) {
        for (uint h = 0; h < grad_output_shape[2]; h++) {
            for (uint w = 0; w < grad_output_shape[3]; w++) {
                uint idx = batch * grad_output_shape[1] * grad_output_shape[2] * grad_output_shape[3] +
                          out_c * grad_output_shape[2] * grad_output_shape[3] +
                          h * grad_output_shape[3] + w;
                
                sum += grad_output[idx];
            }
        }
    }
    
    grad_bias[out_c] = sum;
}