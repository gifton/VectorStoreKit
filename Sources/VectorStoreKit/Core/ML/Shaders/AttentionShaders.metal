// VectorStoreKit: Attention Operations Metal Shaders
//
// Multi-head attention operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Attention Score Computation

/// Compute attention scores: Q @ K^T * scale
kernel void attention_scores(
    constant float* queries [[buffer(0)]],      // [batch*heads, seq_len, head_dim]
    constant float* keys [[buffer(1)]],         // [batch*heads, seq_len, head_dim]
    device float* scores [[buffer(2)]],         // [batch*heads, seq_len, seq_len]
    constant uint& batchHeads [[buffer(3)]],
    constant uint& seqLen [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;  // query position
    uint col = gid.y;  // key position
    uint head = gid.z; // batch*head index
    
    if (row >= seqLen || col >= seqLen || head >= batchHeads) return;
    
    // Compute dot product between query[row] and key[col]
    float sum = 0.0f;
    uint qOffset = head * seqLen * headDim + row * headDim;
    uint kOffset = head * seqLen * headDim + col * headDim;
    
    for (uint i = 0; i < headDim; i++) {
        sum += queries[qOffset + i] * keys[kOffset + i];
    }
    
    // Apply scale and store
    uint scoreIdx = head * seqLen * seqLen + row * seqLen + col;
    scores[scoreIdx] = sum * scale;
}

// MARK: - Softmax

/// Apply softmax to attention scores along last dimension
kernel void softmax_2d(
    constant float* input [[buffer(0)]],    // [num_matrices, rows, cols]
    device float* output [[buffer(1)]],
    constant uint& numMatrices [[buffer(2)]],
    constant uint& size [[buffer(3)]],      // size of last dimension (cols)
    uint gid [[thread_position_in_grid]]
) {
    uint matrixRow = gid;  // Combined matrix and row index
    uint matrix = matrixRow / size;
    uint row = matrixRow % size;
    
    if (matrix >= numMatrices) return;
    
    uint offset = matrix * size * size + row * size;
    
    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (uint i = 0; i < size; i++) {
        maxVal = max(maxVal, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        float expVal = exp(input[offset + i] - maxVal);
        output[offset + i] = expVal;
        sum += expVal;
    }
    
    // Normalize
    for (uint i = 0; i < size; i++) {
        output[offset + i] /= sum;
    }
}

// MARK: - Weighted Sum

/// Compute attention output: weights @ values
kernel void attention_weighted_sum(
    constant float* weights [[buffer(0)]],   // [batch*heads, seq_len, seq_len]
    constant float* values [[buffer(1)]],    // [batch*heads, seq_len, head_dim]
    device float* output [[buffer(2)]],      // [batch*heads, seq_len, head_dim]
    constant uint& batchHeads [[buffer(3)]],
    constant uint& seqLen [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;      // sequence position
    uint dim = gid.y;      // head dimension
    uint head = gid.z;     // batch*head index
    
    if (pos >= seqLen || dim >= headDim || head >= batchHeads) return;
    
    // Compute weighted sum over sequence dimension
    float sum = 0.0f;
    uint weightOffset = head * seqLen * seqLen + pos * seqLen;
    uint valueOffset = head * seqLen * headDim;
    
    for (uint i = 0; i < seqLen; i++) {
        float weight = weights[weightOffset + i];
        float value = values[valueOffset + i * headDim + dim];
        sum += weight * value;
    }
    
    uint outputIdx = head * seqLen * headDim + pos * headDim + dim;
    output[outputIdx] = sum;
}

// MARK: - Reshape Operations

/// Reshape tensor for multi-head attention
kernel void reshape_for_heads(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batchSize [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& numHeads [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    constant uint& batchFirst [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batchSize * seqLen * numHeads * headDim) return;
    
    // Compute indices
    uint totalDim = numHeads * headDim;
    uint b, s, h, d;
    
    if (batchFirst) {
        // Input: [batch, seq_len, embed_dim]
        b = gid / (seqLen * totalDim);
        s = (gid / totalDim) % seqLen;
        uint embedIdx = gid % totalDim;
        h = embedIdx / headDim;
        d = embedIdx % headDim;
    } else {
        // Input: [seq_len, batch, embed_dim]
        s = gid / (batchSize * totalDim);
        b = (gid / totalDim) % batchSize;
        uint embedIdx = gid % totalDim;
        h = embedIdx / headDim;
        d = embedIdx % headDim;
    }
    
    // Output: [batch * num_heads, seq_len, head_dim]
    uint outputIdx = (b * numHeads + h) * seqLen * headDim + s * headDim + d;
    output[outputIdx] = input[gid];
}

/// Reshape from multi-head format back to original
kernel void reshape_from_heads(
    constant float* input [[buffer(0)]],     // [batch * num_heads, seq_len, head_dim]
    device float* output [[buffer(1)]],
    constant uint& batchSize [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& numHeads [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    constant uint& batchFirst [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batchSize * seqLen * numHeads * headDim) return;
    
    // Input indices: [batch * num_heads, seq_len, head_dim]
    uint bh = gid / (seqLen * headDim);
    uint s = (gid / headDim) % seqLen;
    uint d = gid % headDim;
    
    uint b = bh / numHeads;
    uint h = bh % numHeads;
    
    // Output index depends on format
    uint outputIdx;
    if (batchFirst) {
        // Output: [batch, seq_len, embed_dim]
        outputIdx = b * seqLen * numHeads * headDim + s * numHeads * headDim + h * headDim + d;
    } else {
        // Output: [seq_len, batch, embed_dim]
        outputIdx = s * batchSize * numHeads * headDim + b * numHeads * headDim + h * headDim + d;
    }
    
    output[outputIdx] = input[gid];
}

// MARK: - Backward Pass Operations

/// Compute gradient w.r.t attention weights: gradOutput @ V^T
kernel void attention_grad_weights(
    constant float* gradOutput [[buffer(0)]],  // [batch*heads, seq_len, head_dim]
    constant float* values [[buffer(1)]],      // [batch*heads, seq_len, head_dim]
    device float* gradWeights [[buffer(2)]],   // [batch*heads, seq_len, seq_len]
    constant uint& batchHeads [[buffer(3)]],
    constant uint& seqLen [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;  // output position
    uint col = gid.y;  // value position
    uint head = gid.z; // batch*head index
    
    if (row >= seqLen || col >= seqLen || head >= batchHeads) return;
    
    // Compute dot product between gradOutput[row] and values[col]
    float sum = 0.0f;
    uint gradOffset = head * seqLen * headDim + row * headDim;
    uint valOffset = head * seqLen * headDim + col * headDim;
    
    for (uint i = 0; i < headDim; i++) {
        sum += gradOutput[gradOffset + i] * values[valOffset + i];
    }
    
    uint weightIdx = head * seqLen * seqLen + row * seqLen + col;
    gradWeights[weightIdx] = sum;
}

/// Compute gradient w.r.t values: weights^T @ gradOutput
kernel void attention_grad_values(
    constant float* weights [[buffer(0)]],     // [batch*heads, seq_len, seq_len]
    constant float* gradOutput [[buffer(1)]],  // [batch*heads, seq_len, head_dim]
    device float* gradValues [[buffer(2)]],    // [batch*heads, seq_len, head_dim]
    constant uint& batchHeads [[buffer(3)]],
    constant uint& seqLen [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;      // sequence position
    uint dim = gid.y;      // head dimension
    uint head = gid.z;     // batch*head index
    
    if (pos >= seqLen || dim >= headDim || head >= batchHeads) return;
    
    // Compute weighted sum over sequence dimension
    float sum = 0.0f;
    uint weightOffset = head * seqLen * seqLen + pos;  // column pos in weights
    
    for (uint i = 0; i < seqLen; i++) {
        float weight = weights[weightOffset + i * seqLen];  // weights[i][pos]
        float grad = gradOutput[head * seqLen * headDim + i * headDim + dim];
        sum += weight * grad;
    }
    
    uint outputIdx = head * seqLen * headDim + pos * headDim + dim;
    gradValues[outputIdx] = sum;
}

/// Backward pass through softmax
kernel void softmax_backward(
    constant float* gradWeights [[buffer(0)]],  // gradient w.r.t softmax output
    constant float* weights [[buffer(1)]],      // softmax output (attention weights)
    device float* gradScores [[buffer(2)]],     // gradient w.r.t softmax input
    constant uint& numMatrices [[buffer(3)]],
    constant uint& size [[buffer(4)]],         // size of each row
    uint gid [[thread_position_in_grid]]
) {
    uint matrixRow = gid;
    uint matrix = matrixRow / size;
    uint row = matrixRow % size;
    
    if (matrix >= numMatrices) return;
    
    uint offset = matrix * size * size + row * size;
    
    // For softmax backward: grad_input[i] = sum_j(grad_output[j] * output[j] * (delta_ij - output[i]))
    // where delta_ij is 1 if i==j, 0 otherwise
    
    for (uint i = 0; i < size; i++) {
        float sum = 0.0f;
        float output_i = weights[offset + i];
        
        for (uint j = 0; j < size; j++) {
            float grad_j = gradWeights[offset + j];
            float output_j = weights[offset + j];
            
            if (i == j) {
                sum += grad_j * output_j * (1.0f - output_i);
            } else {
                sum += grad_j * output_j * (-output_i);
            }
        }
        
        gradScores[offset + i] = sum;
    }
}

/// Compute gradients w.r.t Q and K
kernel void attention_grad_qk(
    constant float* gradScores [[buffer(0)]],   // [batch*heads, seq_len, seq_len]
    constant float* queries [[buffer(1)]],      // [batch*heads, seq_len, head_dim]
    constant float* keys [[buffer(2)]],         // [batch*heads, seq_len, head_dim]
    device float* gradQ [[buffer(3)]],          // [batch*heads, seq_len, head_dim]
    device float* gradK [[buffer(4)]],          // [batch*heads, seq_len, head_dim]
    constant float& scale [[buffer(5)]],
    constant uint& batchHeads [[buffer(6)]],
    constant uint& seqLen [[buffer(7)]],
    constant uint& headDim [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;      // sequence position
    uint dim = gid.y;      // head dimension
    uint head = gid.z;     // batch*head index
    
    if (pos >= seqLen || dim >= headDim || head >= batchHeads) return;
    
    // Compute gradQ[pos][dim] = sum over k positions: gradScores[pos][k] * keys[k][dim] * scale
    float gradQSum = 0.0f;
    uint scoresRowOffset = head * seqLen * seqLen + pos * seqLen;
    
    for (uint k = 0; k < seqLen; k++) {
        float gradScore = gradScores[scoresRowOffset + k];
        float keyVal = keys[head * seqLen * headDim + k * headDim + dim];
        gradQSum += gradScore * keyVal * scale;
    }
    
    // Compute gradK[pos][dim] = sum over q positions: gradScores[q][pos] * queries[q][dim] * scale
    float gradKSum = 0.0f;
    
    for (uint q = 0; q < seqLen; q++) {
        uint scoresIdx = head * seqLen * seqLen + q * seqLen + pos;
        float gradScore = gradScores[scoresIdx];
        float queryVal = queries[head * seqLen * headDim + q * headDim + dim];
        gradKSum += gradScore * queryVal * scale;
    }
    
    uint outputIdx = head * seqLen * headDim + pos * headDim + dim;
    gradQ[outputIdx] = gradQSum;
    gradK[outputIdx] = gradKSum;
}