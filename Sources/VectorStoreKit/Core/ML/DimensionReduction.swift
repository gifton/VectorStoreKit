// VectorStoreKit: Dimension Reduction
//
// Metal-accelerated dimension reduction techniques
//

import Foundation
import Accelerate
@preconcurrency import Metal

/// Metal-accelerated dimension reduction operations
public actor DimensionReduction {
    private let metalPipeline: MetalMLPipeline
    private let device: MTLDevice
    private var pcaShaders: MetalComputePipelines?
    
    // Cached PCA components
    private var cachedComponents: PCACachedComponents?
    
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.device = await metalPipeline.device
        
        // Load PCA shaders
        try await loadPCAShaders()
    }
    
    // MARK: - PCA Implementation
    
    /// Perform PCA using Metal-accelerated SVD
    public func pca(
        _ vectors: [Vector],
        targetDim: Int,
        whiten: Bool = false
    ) async throws -> PCAResult {
        guard !vectors.isEmpty else {
            throw DimensionReductionError.emptyInput
        }
        
        let n = vectors.count
        let d = vectors.first!.count
        
        guard targetDim < d else {
            throw DimensionReductionError.invalidTargetDimension(
                "Target dimension \(targetDim) must be less than original dimension \(d)"
            )
        }
        
        // Step 1: Flatten vectors and compute mean
        let flatData = vectors.flatMap { $0 }
        let mean = try await computeMean(data: flatData, n: n, d: d)
        
        // Step 2: Center the data
        var centeredData = flatData
        try await centerData(&centeredData, mean: mean, n: n, d: d)
        
        // Step 3: Compute covariance matrix using Metal
        let covariance = try await computeCovarianceMatrix(
            data: centeredData,
            n: n,
            d: d
        )
        
        // Step 4: Compute eigenvectors using Accelerate framework
        let eigenDecomposition = try computeEigenDecomposition(
            covariance: covariance,
            dimension: d,
            numComponents: targetDim
        )
        
        // Step 5: Project data onto principal components
        let projectedData = try await projectData(
            centeredData: centeredData,
            components: eigenDecomposition.eigenvectors,
            n: n,
            d: d,
            k: targetDim
        )
        
        // Step 6: Optionally whiten the data
        let finalData: [Float]
        if whiten {
            finalData = try whitenData(
                projectedData,
                eigenvalues: eigenDecomposition.eigenvalues,
                n: n,
                k: targetDim
            )
        } else {
            finalData = projectedData
        }
        
        // Convert back to vectors
        let transformedVectors = (0..<n).map { i in
            Vector(Array(finalData[(i * targetDim)..<((i + 1) * targetDim)]))
        }
        
        // Cache components for future transformations
        cachedComponents = PCACachedComponents(
            mean: mean,
            components: eigenDecomposition.eigenvectors,
            eigenvalues: eigenDecomposition.eigenvalues,
            originalDim: d,
            reducedDim: targetDim,
            whiten: whiten
        )
        
        // Compute explained variance ratio
        let totalVariance = eigenDecomposition.allEigenvalues.reduce(0, +)
        let explainedVariance = eigenDecomposition.eigenvalues.map { $0 / totalVariance }
        
        return PCAResult(
            transformedVectors: transformedVectors,
            components: eigenDecomposition.eigenvectors,
            explainedVarianceRatio: explainedVariance,
            mean: mean
        )
    }
    
    /// Transform new vectors using cached PCA components
    public func transformWithPCA(_ vectors: [Vector]) async throws -> [Vector] {
        guard let cached = cachedComponents else {
            throw DimensionReductionError.componentNotTrained
        }
        
        guard vectors.first?.count == cached.originalDim else {
            throw DimensionReductionError.dimensionMismatch(
                expected: cached.originalDim,
                actual: vectors.first?.count ?? 0
            )
        }
        
        let n = vectors.count
        
        // Center the data
        var flatData = vectors.flatMap { $0 }
        for i in 0..<n {
            for j in 0..<cached.originalDim {
                flatData[i * cached.originalDim + j] -= cached.mean[j]
            }
        }
        
        // Project onto components
        let projected = try await projectData(
            centeredData: flatData,
            components: cached.components,
            n: n,
            d: cached.originalDim,
            k: cached.reducedDim
        )
        
        // Optionally whiten
        let finalData: [Float]
        if cached.whiten {
            finalData = try whitenData(
                projected,
                eigenvalues: cached.eigenvalues,
                n: n,
                k: cached.reducedDim
            )
        } else {
            finalData = projected
        }
        
        // Convert back to vectors
        return (0..<n).map { i in
            Vector(Array(finalData[(i * cached.reducedDim)..<((i + 1) * cached.reducedDim)]))
        }
    }
    
    // MARK: - Random Projection
    
    /// Johnson-Lindenstrauss random projection
    public func randomProjection(
        _ vectors: [Vector],
        targetDim: Int,
        seed: UInt64 = 42
    ) async throws -> RandomProjectionResult {
        guard !vectors.isEmpty else {
            throw DimensionReductionError.emptyInput
        }
        
        let originalDim = vectors.first!.count
        
        // Generate random projection matrix
        var generator = RandomNumberGenerator(seed: seed)
        let projectionMatrix = generateGaussianRandomMatrix(
            rows: targetDim,
            cols: originalDim,
            generator: &generator
        )
        
        // Scale by 1/sqrt(targetDim) for variance preservation
        let scale = 1.0 / sqrt(Float(targetDim))
        let scaledMatrix = projectionMatrix.map { $0 * scale }
        
        // Project vectors
        let projectedVectors = try await projectWithMatrix(
            vectors: vectors,
            matrix: scaledMatrix,
            targetDim: targetDim
        )
        
        return RandomProjectionResult(
            transformedVectors: projectedVectors,
            projectionMatrix: scaledMatrix
        )
    }
    
    // MARK: - Private Methods
    
    private func loadPCAShaders() async throws {
        let library = await metalPipeline.getShaderLibrary()
        
        var shaders = MetalComputePipelines()
        shaders.computeMean = try await library.makeComputePipeline(functionName: "compute_mean")
        shaders.centerData = try await library.makeComputePipeline(functionName: "center_data")
        shaders.computeCovariance = try await library.makeComputePipeline(functionName: "compute_covariance")
        shaders.projectData = try await library.makeComputePipeline(functionName: "project_data")
        
        self.pcaShaders = shaders
    }
    
    private func computeMean(data: [Float], n: Int, d: Int) async throws -> [Float] {
        guard let shaders = pcaShaders else {
            throw DimensionReductionError.shadersNotLoaded
        }
        
        // Allocate buffers
        let dataBuffer = try await metalPipeline.allocateBuffer(size: n * d)
        let meanBuffer = try await metalPipeline.allocateBuffer(size: d)
        
        // Copy data
        dataBuffer.buffer.contents().copyMemory(
            from: data,
            byteCount: data.count * MemoryLayout<Float>.stride
        )
        
        // Execute compute
        try await executeKernel(
            pipeline: shaders.computeMean,
            buffers: [dataBuffer.buffer, meanBuffer.buffer],
            constants: [n, d],
            gridSize: MTLSize(width: d, height: 1, depth: 1)
        )
        
        // Read result
        let meanPtr = meanBuffer.buffer.contents().bindMemory(to: Float.self, capacity: d)
        let mean = Array(UnsafeBufferPointer(start: meanPtr, count: d))
        
        // Release buffers
        await metalPipeline.releaseBuffer(dataBuffer)
        await metalPipeline.releaseBuffer(meanBuffer)
        
        return mean
    }
    
    private func centerData(_ data: inout [Float], mean: [Float], n: Int, d: Int) async throws {
        guard let shaders = pcaShaders else {
            throw DimensionReductionError.shadersNotLoaded
        }
        
        let dataBuffer = try await metalPipeline.allocateBuffer(size: n * d)
        let meanBuffer = try await metalPipeline.allocateBuffer(size: d)
        
        // Copy data
        dataBuffer.buffer.contents().copyMemory(
            from: data,
            byteCount: data.count * MemoryLayout<Float>.stride
        )
        meanBuffer.buffer.contents().copyMemory(
            from: mean,
            byteCount: mean.count * MemoryLayout<Float>.stride
        )
        
        // Execute centering
        try await executeKernel(
            pipeline: shaders.centerData,
            buffers: [dataBuffer.buffer, meanBuffer.buffer],
            constants: [n, d],
            gridSize: MTLSize(width: d, height: n, depth: 1)
        )
        
        // Read result
        dataBuffer.buffer.contents().copyMemory(
            to: &data,
            byteCount: data.count * MemoryLayout<Float>.stride
        )
        
        await metalPipeline.releaseBuffer(dataBuffer)
        await metalPipeline.releaseBuffer(meanBuffer)
    }
    
    private func computeCovarianceMatrix(data: [Float], n: Int, d: Int) async throws -> [Float] {
        guard let shaders = pcaShaders else {
            throw DimensionReductionError.shadersNotLoaded
        }
        
        let dataBuffer = try await metalPipeline.allocateBuffer(size: n * d)
        let covBuffer = try await metalPipeline.allocateBuffer(size: d * d)
        
        dataBuffer.buffer.contents().copyMemory(
            from: data,
            byteCount: data.count * MemoryLayout<Float>.stride
        )
        
        try await executeKernel(
            pipeline: shaders.computeCovariance,
            buffers: [dataBuffer.buffer, covBuffer.buffer],
            constants: [n, d],
            gridSize: MTLSize(width: d, height: d, depth: 1)
        )
        
        let covPtr = covBuffer.buffer.contents().bindMemory(to: Float.self, capacity: d * d)
        let covariance = Array(UnsafeBufferPointer(start: covPtr, count: d * d))
        
        await metalPipeline.releaseBuffer(dataBuffer)
        await metalPipeline.releaseBuffer(covBuffer)
        
        return covariance
    }
    
    private func computeEigenDecomposition(
        covariance: [Float],
        dimension: Int,
        numComponents: Int
    ) throws -> EigenDecomposition {
        // Use Accelerate's LAPACK for eigendecomposition
        var n = __CLPK_integer(dimension)
        var jobz = Int8(86) // 'V' - compute eigenvalues and eigenvectors
        var uplo = Int8(85) // 'U' - upper triangle
        
        // Copy covariance matrix (LAPACK modifies it)
        var a = covariance
        var lda = n
        var w = [Float](repeating: 0, count: dimension) // eigenvalues
        
        // Workspace query
        var work = [Float](repeating: 0, count: 1)
        var lwork = __CLPK_integer(-1)
        var info = __CLPK_integer(0)
        
        // Query optimal workspace size
        ssyev_(&jobz, &uplo, &n, &a, &lda, &w, &work, &lwork, &info)
        
        guard info == 0 else {
            throw DimensionReductionError.eigenDecompositionFailed
        }
        
        // Allocate workspace
        lwork = __CLPK_integer(work[0])
        work = [Float](repeating: 0, count: Int(lwork))
        
        // Compute eigendecomposition
        ssyev_(&jobz, &uplo, &n, &a, &lda, &w, &work, &lwork, &info)
        
        guard info == 0 else {
            throw DimensionReductionError.eigenDecompositionFailed
        }
        
        // Eigenvalues are in ascending order, we want descending
        w.reverse()
        
        // Extract top k eigenvectors (columns of a, but we need to reverse order)
        var topEigenvectors = [Float](repeating: 0, count: numComponents * dimension)
        for i in 0..<numComponents {
            let sourceCol = dimension - 1 - i
            for j in 0..<dimension {
                topEigenvectors[i * dimension + j] = a[j * dimension + sourceCol]
            }
        }
        
        return EigenDecomposition(
            eigenvectors: topEigenvectors,
            eigenvalues: Array(w.prefix(numComponents)),
            allEigenvalues: w
        )
    }
    
    private func projectData(
        centeredData: [Float],
        components: [Float],
        n: Int,
        d: Int,
        k: Int
    ) async throws -> [Float] {
        guard let shaders = pcaShaders else {
            throw DimensionReductionError.shadersNotLoaded
        }
        
        let dataBuffer = try await metalPipeline.allocateBuffer(size: n * d)
        let componentsBuffer = try await metalPipeline.allocateBuffer(size: k * d)
        let projectedBuffer = try await metalPipeline.allocateBuffer(size: n * k)
        
        dataBuffer.buffer.contents().copyMemory(
            from: centeredData,
            byteCount: centeredData.count * MemoryLayout<Float>.stride
        )
        componentsBuffer.buffer.contents().copyMemory(
            from: components,
            byteCount: components.count * MemoryLayout<Float>.stride
        )
        
        try await executeKernel(
            pipeline: shaders.projectData,
            buffers: [dataBuffer.buffer, componentsBuffer.buffer, projectedBuffer.buffer],
            constants: [n, d, k],
            gridSize: MTLSize(width: k, height: n, depth: 1)
        )
        
        let projectedPtr = projectedBuffer.buffer.contents().bindMemory(to: Float.self, capacity: n * k)
        let projected = Array(UnsafeBufferPointer(start: projectedPtr, count: n * k))
        
        await metalPipeline.releaseBuffer(dataBuffer)
        await metalPipeline.releaseBuffer(componentsBuffer)
        await metalPipeline.releaseBuffer(projectedBuffer)
        
        return projected
    }
    
    private func whitenData(
        _ data: [Float],
        eigenvalues: [Float],
        n: Int,
        k: Int
    ) throws -> [Float] {
        var whitened = data
        
        // Whiten by dividing by sqrt(eigenvalue)
        for i in 0..<n {
            for j in 0..<k {
                let idx = i * k + j
                let scale = 1.0 / sqrt(eigenvalues[j] + 1e-8) // Add small epsilon for stability
                whitened[idx] *= scale
            }
        }
        
        return whitened
    }
    
    private func projectWithMatrix(
        vectors: [Vector],
        matrix: [Float],
        targetDim: Int
    ) async throws -> [Vector] {
        let n = vectors.count
        let originalDim = vectors.first!.count
        
        var result = [Vector]()
        
        // Simple matrix multiplication for now
        // TODO: Use Metal for large-scale projection
        for vector in vectors {
            var projected = [Float](repeating: 0, count: targetDim)
            
            for i in 0..<targetDim {
                var sum: Float = 0
                for j in 0..<originalDim {
                    sum += matrix[i * originalDim + j] * vector[j]
                }
                projected[i] = sum
            }
            
            result.append(Vector(projected))
        }
        
        return result
    }
    
    private func generateGaussianRandomMatrix(
        rows: Int,
        cols: Int,
        generator: inout RandomNumberGenerator
    ) -> [Float] {
        var matrix = [Float](repeating: 0, count: rows * cols)
        
        // Box-Muller transform for Gaussian random numbers
        for i in 0..<(rows * cols) {
            let u1 = Float.random(in: 0..<1, using: &generator)
            let u2 = Float.random(in: 0..<1, using: &generator)
            
            let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            matrix[i] = z0
        }
        
        return matrix
    }
    
    private func executeKernel(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        constants: [Int],
        gridSize: MTLSize
    ) async throws {
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            
            // Set buffers
            for (index, buffer) in buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            
            // Set constants
            for (index, constant) in constants.enumerated() {
                var value = UInt32(constant)
                encoder.setBytes(&value, length: MemoryLayout<UInt32>.size, index: buffers.count + index)
            }
            
            // Calculate thread groups
            let threadsPerThreadgroup = MTLSize(
                width: min(16, gridSize.width),
                height: min(16, gridSize.height),
                depth: 1
            )
            
            let threadgroups = MTLSize(
                width: (gridSize.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: (gridSize.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
    }
}

// MARK: - Supporting Types

struct MetalComputePipelines {
    var computeMean: MTLComputePipelineState!
    var centerData: MTLComputePipelineState!
    var computeCovariance: MTLComputePipelineState!
    var projectData: MTLComputePipelineState!
}

struct EigenDecomposition {
    let eigenvectors: [Float]  // [k x d] top k eigenvectors
    let eigenvalues: [Float]   // [k] top k eigenvalues
    let allEigenvalues: [Float] // All eigenvalues for variance calculation
}

struct PCACachedComponents {
    let mean: [Float]
    let components: [Float]
    let eigenvalues: [Float]
    let originalDim: Int
    let reducedDim: Int
    let whiten: Bool
}

/// PCA transformation result
/// 
/// Thread-safe container for PCA results.
/// All properties are immutable value types.
public struct PCAResult: Sendable {
    public let transformedVectors: [Vector]
    public let components: [Float]  // Principal components
    public let explainedVarianceRatio: [Float]
    public let mean: [Float]
}

/// Random projection transformation result
/// 
/// Thread-safe container for random projection results.
/// Contains only immutable value types.
public struct RandomProjectionResult: Sendable {
    public let transformedVectors: [Vector]
    public let projectionMatrix: [Float]
}

struct RandomNumberGenerator: Swift.RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // Linear congruential generator
        state = state &* 1664525 &+ 1013904223
        return state
    }
}

// MARK: - Errors

public enum DimensionReductionError: LocalizedError {
    case emptyInput
    case invalidTargetDimension(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case componentNotTrained
    case shadersNotLoaded
    case eigenDecompositionFailed
    
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input vectors array is empty"
        case .invalidTargetDimension(let message):
            return message
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .componentNotTrained:
            return "PCA components have not been trained yet"
        case .shadersNotLoaded:
            return "Metal shaders not loaded"
        case .eigenDecompositionFailed:
            return "Eigendecomposition failed"
        }
    }
}