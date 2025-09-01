// VectorStoreKit: Quantization Shaders
//
// GPU kernels for vector quantization operations
// These shaders implement compression techniques to reduce memory usage
// while preserving vector similarity properties

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Scalar Quantization

/// Quantize floating-point values to 8-bit unsigned integers
/// This provides 4x memory reduction (float32 -> uint8)
///
/// Quantization formula: q = round((x - offset) * scale)
/// where scale and offset are computed to map the value range to [0, 255]
///
/// Use cases:
/// - Reducing memory footprint of large vector databases
/// - Faster similarity search with reduced precision
/// - Mobile and edge deployment with memory constraints
///
/// Accuracy considerations:
/// - 8-bit quantization typically preserves 99%+ recall
/// - Works best with normalized or bounded value ranges
/// - Consider keeping original vectors for high-precision needs
///
/// @param input Original floating-point values
/// @param output Quantized 8-bit values
/// @param scale Quantization scale factor (precomputed)
/// @param offset Quantization offset (precomputed)
/// @param id Thread index = element index
kernel void scalarQuantize(
    constant float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float value = input[id];
    
    // Apply affine transformation to map to quantization range
    float quantized = round((value - offset) * scale);
    
    // Clamp to 8-bit range [0, 255] to handle outliers
    quantized = clamp(quantized, 0.0f, 255.0f);
    
    // Cast to unsigned char for storage
    output[id] = static_cast<uchar>(quantized);
}

/// Dequantize 8-bit values back to floating-point
/// Inverse operation of scalar quantization
///
/// Dequantization formula: x = (q / scale) + offset
/// This reconstructs approximate original values
///
/// Reconstruction error:
/// - Maximum error = 0.5 / scale (quantization step size)
/// - Error uniformly distributed if values were uniformly distributed
/// - Consider error accumulation in iterative algorithms
///
/// @param input Quantized 8-bit values
/// @param output Reconstructed floating-point values
/// @param scale Quantization scale factor (same as used in quantization)
/// @param offset Quantization offset (same as used in quantization)
/// @param id Thread index = element index
kernel void scalarDequantize(
    constant uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // Convert 8-bit value to float
    float value = static_cast<float>(input[id]);
    
    // Apply inverse transformation to reconstruct original scale
    output[id] = (value / scale) + offset;
}

// MARK: - Product Quantization

/// Product Quantization: Advanced vector compression technique
/// Divides vectors into subspaces and quantizes each independently
///
/// Algorithm overview:
/// 1. Split d-dimensional vector into m subspaces of d/m dimensions
/// 2. Learn separate codebook for each subspace (typically 256 entries)
/// 3. Encode each subspace as index to nearest codebook entry
/// 4. Store m bytes per vector instead of d floats
///
/// Compression ratio: d * 4 bytes -> m bytes (typically 32x-128x)
///
/// Benefits:
/// - Massive compression with reasonable accuracy
/// - Fast approximate distance computation
/// - Asymmetric distance computation supported
///
/// Thread organization:
/// - 2D grid: x = vector index, y = subspace index
/// - Each thread handles one subspace of one vector
///
/// @param vectors Input vectors to quantize
/// @param codebook Learned codebook entries [subspace][code][dimension]
/// @param codes Output codes (indices into codebook)
/// @param vectorDimension Total dimension of input vectors
/// @param numSubspaces Number of subspaces (m)
/// @param subspaceDimension Dimensions per subspace (d/m)
/// @param codebookSize Number of codes per subspace (typically 256)
/// @param id 2D thread index (vector, subspace)
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
    
    // Calculate offsets for this vector's subspace
    uint vectorOffset = vectorIdx * vectorDimension + subspaceIdx * subspaceDimension;
    uint codebookOffset = subspaceIdx * codebookSize * subspaceDimension;
    
    float minDistance = INFINITY;
    uint bestCode = 0;
    
    // Exhaustive search for nearest codebook entry
    // For 256 codes, this is still efficient on GPU
    for (uint code = 0; code < codebookSize; ++code) {
        float distance = 0.0;
        
        // Compute squared Euclidean distance to codebook entry
        for (uint dim = 0; dim < subspaceDimension; ++dim) {
            float diff = vectors[vectorOffset + dim] - 
                        codebook[codebookOffset + code * subspaceDimension + dim];
            distance += diff * diff;
        }
        
        // Track nearest codebook entry
        if (distance < minDistance) {
            minDistance = distance;
            bestCode = code;
        }
    }
    
    // Store index of nearest codebook entry
    codes[vectorIdx * numSubspaces + subspaceIdx] = static_cast<uchar>(bestCode);
}

/// Reconstruct vectors from Product Quantization codes
/// Maps codes back to codebook entries to approximate original vectors
///
/// Thread organization:
/// - 2D grid: x = vector index, y = dimension index
/// - Each thread reconstructs one dimension of one vector
///
/// Reconstruction process:
/// 1. Determine which subspace contains this dimension
/// 2. Look up the code for that subspace
/// 3. Copy corresponding value from codebook
///
/// @param codes Quantization codes for each subspace
/// @param codebook Learned codebook entries
/// @param vectors Output reconstructed vectors
/// @param vectorDimension Total vector dimension
/// @param numSubspaces Number of subspaces
/// @param subspaceDimension Dimensions per subspace
/// @param id 2D thread index (vector, dimension)
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
    
    // Determine which subspace this dimension belongs to
    uint subspaceIdx = dimIdx / subspaceDimension;
    uint subspaceDimIdx = dimIdx % subspaceDimension;
    
    // Look up the code for this subspace
    uint code = codes[vectorIdx * numSubspaces + subspaceIdx];
    
    // Calculate offset into codebook (assumes 256 codes per subspace)
    uint codebookOffset = subspaceIdx * 256 * subspaceDimension + 
                         code * subspaceDimension + subspaceDimIdx;
    
    // Copy value from codebook
    vectors[vectorIdx * vectorDimension + dimIdx] = codebook[codebookOffset];
}

// MARK: - Binary Quantization

/// Binary quantization: Extreme compression to 1 bit per dimension
/// Each dimension is encoded as 0 or 1 based on sign or threshold
///
/// Compression ratio: 32x (float32 -> 1 bit)
///
/// Algorithm:
/// - Positive values -> 1
/// - Negative/zero values -> 0
/// - Pack 32 bits into each uint32
///
/// Use cases:
/// - Extremely large databases where memory is critical
/// - Initial filtering before more precise reranking
/// - Binary features (presence/absence)
///
/// Distance computation:
/// - Use Hamming distance (XOR + popcount)
/// - Very fast on modern hardware
/// - Approximates angular distance for high dimensions
///
/// @param vectors Input floating-point vectors
/// @param binaryVectors Output packed binary vectors
/// @param vectorDimension Number of dimensions
/// @param id Thread index = vector index
kernel void binaryQuantize(
    constant float* vectors [[buffer(0)]],
    device uint* binaryVectors [[buffer(1)]],
    constant uint& vectorDimension [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint vectorIdx = id;
    // Calculate number of 32-bit words needed
    uint numWords = (vectorDimension + 31) / 32;
    
    // Process each 32-bit word
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint word = 0;
        
        // Pack 32 dimensions into one word
        for (uint bit = 0; bit < 32 && wordIdx * 32 + bit < vectorDimension; ++bit) {
            uint dimIdx = wordIdx * 32 + bit;
            float value = vectors[vectorIdx * vectorDimension + dimIdx];
            
            // Set bit if value is positive
            // Can also use other thresholds (e.g., median)
            if (value > 0.0) {
                word |= (1u << bit);
            }
        }
        
        // Store packed bits
        binaryVectors[vectorIdx * numWords + wordIdx] = word;
    }
}

/// Compute Hamming distance between binary vectors
/// Counts the number of differing bits between two binary vectors
///
/// Algorithm:
/// 1. XOR corresponding words (1 where bits differ)
/// 2. Count set bits using popcount
/// 3. Sum across all words
///
/// Performance:
/// - XOR: Single cycle operation
/// - popcount: Hardware accelerated on modern GPUs
/// - Very memory efficient
///
/// Relationship to other distances:
/// - For random vectors: Hamming ≈ 2 * arccos(cosine) * d / π
/// - Preserves nearest neighbor ordering well
///
/// @param queryBinary Binary query vector (packed)
/// @param candidateBinary Binary candidate vectors (packed)
/// @param distances Output Hamming distances
/// @param numWords Number of 32-bit words per vector
/// @param id Thread index = candidate index
kernel void binaryHammingDistance(
    constant uint* queryBinary [[buffer(0)]],
    constant uint* candidateBinary [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& numWords [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint candidateIdx = id;
    uint distance = 0;
    
    // Process each 32-bit word
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint queryWord = queryBinary[wordIdx];
        uint candidateWord = candidateBinary[candidateIdx * numWords + wordIdx];
        
        // XOR gives 1 where bits differ
        uint xor_result = queryWord ^ candidateWord;
        
        // Count set bits (Hamming distance)
        // popcount is a single instruction on most GPUs
        distance += popcount(xor_result);
    }
    
    // Convert to float for consistency with other distance functions
    distances[candidateIdx] = static_cast<float>(distance);
}

// MARK: - Quantization Statistics

/// Compute quantization quality metrics: MSE and PSNR
/// These metrics help evaluate the quality/compression trade-off
///
/// Metrics computed:
/// 1. MSE (Mean Squared Error): Average squared difference
///    - Lower is better
///    - 0 means perfect reconstruction
///
/// 2. PSNR (Peak Signal-to-Noise Ratio): Logarithmic quality measure
///    - Higher is better (measured in dB)
///    - >40 dB: Excellent quality
///    - 30-40 dB: Good quality
///    - <30 dB: Poor quality
///
/// Use these metrics to:
/// - Choose quantization parameters
/// - Compare different quantization methods
/// - Determine if quality is acceptable
///
/// @param original Original vectors before quantization
/// @param quantized Reconstructed vectors after dequantization
/// @param mse Output mean squared error per vector
/// @param psnr Output peak signal-to-noise ratio per vector
/// @param vectorDimension Number of dimensions
/// @param id Thread index = vector index
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
    
    // Compute error statistics for this vector
    for (uint dim = 0; dim < vectorDimension; ++dim) {
        float orig = original[offset + dim];
        float quant = quantized[offset + dim];
        float error = orig - quant;
        
        // Accumulate squared error
        sumSquaredError += error * error;
        
        // Track maximum absolute value (for PSNR)
        maxValue = max(maxValue, abs(orig));
    }
    
    // Mean Squared Error
    float meanSquaredError = sumSquaredError / float(vectorDimension);
    mse[vectorIdx] = meanSquaredError;
    
    // Peak Signal-to-Noise Ratio (in dB)
    // PSNR = 20 * log10(MAX / RMSE)
    if (meanSquaredError > 0.0) {
        psnr[vectorIdx] = 20.0 * log10(maxValue / sqrt(meanSquaredError));
    } else {
        // Perfect reconstruction
        psnr[vectorIdx] = INFINITY;
    }
}
