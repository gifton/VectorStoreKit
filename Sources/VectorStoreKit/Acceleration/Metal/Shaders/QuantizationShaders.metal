// VectorStoreKit: Quantization Shaders
//
// GPU kernels for vector quantization operations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Scalar Quantization

kernel void scalarQuantize(
    constant float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float value = input[id];
    float quantized = round((value - offset) * scale);
    
    // Clamp to 8-bit range
    quantized = clamp(quantized, 0.0f, 255.0f);
    output[id] = static_cast<uchar>(quantized);
}

kernel void scalarDequantize(
    constant uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float value = static_cast<float>(input[id]);
    output[id] = (value / scale) + offset;
}

// MARK: - Product Quantization

kernel void productQuantize(
    constant float* vectors [[buffer(0)]],
    constant float* codebook [[buffer(1)]],
    device uchar* codes [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numSubspaces [[buffer(4)]],
    constant uint& subspaceDimension [[buffer(5)]],
    constant uint& codebookSize [[buffer(6)]],
    uint2 id [[thread_position_in_grid]] // x: vector index, y: subspace index
) {
    uint vectorIdx = id.x;
    uint subspaceIdx = id.y;
    
    if (subspaceIdx >= numSubspaces) return;
    
    uint vectorOffset = vectorIdx * vectorDimension + subspaceIdx * subspaceDimension;
    uint codebookOffset = subspaceIdx * codebookSize * subspaceDimension;
    
    float minDistance = INFINITY;
    uint bestCode = 0;
    
    // Find nearest codebook entry
    for (uint code = 0; code < codebookSize; ++code) {
        float distance = 0.0;
        
        for (uint dim = 0; dim < subspaceDimension; ++dim) {
            float diff = vectors[vectorOffset + dim] - 
                        codebook[codebookOffset + code * subspaceDimension + dim];
            distance += diff * diff;
        }
        
        if (distance < minDistance) {
            minDistance = distance;
            bestCode = code;
        }
    }
    
    codes[vectorIdx * numSubspaces + subspaceIdx] = static_cast<uchar>(bestCode);
}

kernel void productDequantize(
    constant uchar* codes [[buffer(0)]],
    constant float* codebook [[buffer(1)]],
    device float* vectors [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numSubspaces [[buffer(4)]],
    constant uint& subspaceDimension [[buffer(5)]],
    uint2 id [[thread_position_in_grid]] // x: vector index, y: dimension index
) {
    uint vectorIdx = id.x;
    uint dimIdx = id.y;
    
    if (dimIdx >= vectorDimension) return;
    
    uint subspaceIdx = dimIdx / subspaceDimension;
    uint subspaceDimIdx = dimIdx % subspaceDimension;
    
    uint code = codes[vectorIdx * numSubspaces + subspaceIdx];
    uint codebookOffset = subspaceIdx * 256 * subspaceDimension + 
                         code * subspaceDimension + subspaceDimIdx;
    
    vectors[vectorIdx * vectorDimension + dimIdx] = codebook[codebookOffset];
}

// MARK: - Binary Quantization

kernel void binaryQuantize(
    constant float* vectors [[buffer(0)]],
    device uint* binaryVectors [[buffer(1)]],
    constant uint& vectorDimension [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint vectorIdx = id;
    uint numWords = (vectorDimension + 31) / 32;
    
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint word = 0;
        
        for (uint bit = 0; bit < 32 && wordIdx * 32 + bit < vectorDimension; ++bit) {
            uint dimIdx = wordIdx * 32 + bit;
            float value = vectors[vectorIdx * vectorDimension + dimIdx];
            
            if (value > 0.0) {
                word |= (1u << bit);
            }
        }
        
        binaryVectors[vectorIdx * numWords + wordIdx] = word;
    }
}

kernel void binaryHammingDistance(
    constant uint* queryBinary [[buffer(0)]],
    constant uint* candidateBinary [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& numWords [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint candidateIdx = id;
    uint distance = 0;
    
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint queryWord = queryBinary[wordIdx];
        uint candidateWord = candidateBinary[candidateIdx * numWords + wordIdx];
        uint xor_result = queryWord ^ candidateWord;
        
        // Count set bits (Hamming distance)
        distance += popcount(xor_result);
    }
    
    distances[candidateIdx] = static_cast<float>(distance);
}

// MARK: - Quantization Statistics

kernel void computeQuantizationStats(
    constant float* original [[buffer(0)]],
    constant float* quantized [[buffer(1)]],
    device float* mse [[buffer(2)]],
    device float* psnr [[buffer(3)]],
    constant uint& vectorDimension [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint vectorIdx = id;
    uint offset = vectorIdx * vectorDimension;
    
    float sumSquaredError = 0.0;
    float maxValue = 0.0;
    
    for (uint dim = 0; dim < vectorDimension; ++dim) {
        float orig = original[offset + dim];
        float quant = quantized[offset + dim];
        float error = orig - quant;
        
        sumSquaredError += error * error;
        maxValue = max(maxValue, abs(orig));
    }
    
    float meanSquaredError = sumSquaredError / float(vectorDimension);
    mse[vectorIdx] = meanSquaredError;
    
    // Calculate PSNR (Peak Signal-to-Noise Ratio)
    if (meanSquaredError > 0.0) {
        psnr[vectorIdx] = 20.0 * log10(maxValue / sqrt(meanSquaredError));
    } else {
        psnr[vectorIdx] = INFINITY;
    }
}
