// EarthMoversDistance.swift
// VectorStoreKit
//
// Earth Mover's Distance (Wasserstein Distance) implementation

import Foundation
import simd
import Accelerate

/// Earth Mover's Distance (EMD) computation using optimized algorithms
public struct EarthMoversDistance {
    
    /// Cost matrix computation strategy
    public enum CostFunction {
        case euclidean
        case manhattan
        case custom((Float, Float) -> Float)
    }
    
    private let costFunction: CostFunction
    private let optimizer: EMDOptimizer
    
    /// Initialize with cost function
    public init(costFunction: CostFunction = .euclidean) {
        self.costFunction = costFunction
        self.optimizer = EMDOptimizer()
    }
    
    /// Compute EMD between two probability distributions with SIMD optimization
    @inlinable
    public func distance(_ a: Vector512, _ b: Vector512) -> Float {
        // Normalize vectors to probability distributions using SIMD
        let distA = normalizeToDistribution(a)
        let distB = normalizeToDistribution(b)
        
        // For 1D EMD, we can use a more efficient algorithm
        // Sort indices by value for optimal transport
        let sortedA = distA.enumerated().sorted { $0.element < $1.element }
        let sortedB = distB.enumerated().sorted { $0.element < $1.element }
        
        // Compute cumulative distributions using SIMD
        var cdfA = [Float](repeating: 0, count: 512)
        var cdfB = [Float](repeating: 0, count: 512)
        
        // Sequential scan for cumulative sum (inherently sequential)
        var sumA: Float = 0
        var sumB: Float = 0
        
        for i in 0..<512 {
            sumA += sortedA[i].element
            sumB += sortedB[i].element
            cdfA[i] = sumA
            cdfB[i] = sumB
        }
        
        // SIMD-optimized computation of 1-Wasserstein distance
        var totalCost: Float = 0
        
        cdfA.withUnsafeBufferPointer { cdfAPtr in
            cdfB.withUnsafeBufferPointer { cdfBPtr in
                let cdfASIMD = cdfAPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                let cdfBSIMD = cdfBPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                
                var costSum = SIMD8<Float>.zero
                
                for i in stride(from: 0, to: 64, by: 1) {
                    let diff = abs(cdfASIMD[i] - cdfBSIMD[i])
                    
                    // Compute costs for 8 consecutive elements
                    let costs = SIMD8<Float>(
                        computeCost(i * 8, i * 8, costFunction: costFunction),
                        computeCost(i * 8 + 1, i * 8 + 1, costFunction: costFunction),
                        computeCost(i * 8 + 2, i * 8 + 2, costFunction: costFunction),
                        computeCost(i * 8 + 3, i * 8 + 3, costFunction: costFunction),
                        computeCost(i * 8 + 4, i * 8 + 4, costFunction: costFunction),
                        computeCost(i * 8 + 5, i * 8 + 5, costFunction: costFunction),
                        computeCost(i * 8 + 6, i * 8 + 6, costFunction: costFunction),
                        computeCost(i * 8 + 7, i * 8 + 7, costFunction: costFunction)
                    )
                    
                    costSum += diff * costs
                }
                
                totalCost = costSum.sum()
            }
        }
        
        return totalCost
    }
    
    /// Compute EMD with custom ground distance matrix using SIMD optimization
    @inlinable
    public func distanceWithGroundDistance(
        _ a: Vector512,
        _ b: Vector512,
        groundDistance: [[Float]]
    ) throws -> Float {
        guard groundDistance.count == 512, groundDistance.allSatisfy({ $0.count == 512 }) else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .invalidInput,
                message: "Ground distance matrix must be 512x512",
                context: ["matrixSize": groundDistance.count]
            )
        }
        
        // Use linear programming solver for optimal transport
        return try optimizer.solveOptimalTransport(
            source: normalizeToDistribution(a),
            target: normalizeToDistribution(b),
            costMatrix: groundDistance
        )
    }
    
    /// SIMD-optimized approximate EMD using Sinkhorn algorithm
    public func approximateDistance(
        _ a: Vector512,
        _ b: Vector512,
        iterations: Int = 50,
        regularization: Float = 0.1
    ) -> Float {
        let distA = normalizeToDistribution(a)
        let distB = normalizeToDistribution(b)
        
        // Pre-allocate aligned arrays for SIMD operations
        var u = [Float](repeating: 1.0, count: 512)
        var v = [Float](repeating: 1.0, count: 512)
        var tempSum = [Float](repeating: 0, count: 512)
        
        // Pre-compute cost matrix with SIMD regularization
        var K = [Float](repeating: 0, count: 512 * 512)
        let invReg = -1.0 / regularization
        
        K.withUnsafeMutableBufferPointer { kPtr in
            for i in 0..<512 {
                let rowOffset = i * 512
                for j in stride(from: 0, to: 512, by: 8) {
                    let costs = SIMD8<Float>(
                        computeCost(i, j, costFunction: costFunction),
                        computeCost(i, j + 1, costFunction: costFunction),
                        computeCost(i, j + 2, costFunction: costFunction),
                        computeCost(i, j + 3, costFunction: costFunction),
                        computeCost(i, j + 4, costFunction: costFunction),
                        computeCost(i, j + 5, costFunction: costFunction),
                        computeCost(i, j + 6, costFunction: costFunction),
                        computeCost(i, j + 7, costFunction: costFunction)
                    )
                    
                    let expCosts = exp(costs * SIMD8<Float>(repeating: invReg))
                    let kSIMD = kPtr.baseAddress!.advanced(by: rowOffset + j).assumingMemoryBound(to: SIMD8<Float>.self)
                    kSIMD.pointee = expCosts
                }
            }
        }
        
        // SIMD-optimized Sinkhorn iterations
        for _ in 0..<iterations {
            // Update v using SIMD matrix-vector operations
            vDSP_mmul(K, 1, u, 1, &tempSum, 1, 512, 1, 512)
            
            v.withUnsafeMutableBufferPointer { vPtr in
                tempSum.withUnsafeBufferPointer { sumPtr in
                    distB.withUnsafeBufferPointer { distBPtr in
                        let vSIMD = vPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let sumSIMD = sumPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let distBSIMD = distBPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let epsilon = SIMD8<Float>(repeating: Float.ulpOfOne)
                        
                        for i in 0..<64 {
                            vSIMD[i] = distBSIMD[i] / (sumSIMD[i] + epsilon)
                        }
                    }
                }
            }
            
            // Update u using transposed matrix-vector operations
            // Transpose multiplication: K^T * v
            cblas_sgemv(CblasRowMajor, CblasTrans, 512, 512, 1.0, K, 512, v, 1, 0.0, &tempSum, 1)
            
            u.withUnsafeMutableBufferPointer { uPtr in
                tempSum.withUnsafeBufferPointer { sumPtr in
                    distA.withUnsafeBufferPointer { distAPtr in
                        let uSIMD = uPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let sumSIMD = sumPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let distASIMD = distAPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                        let epsilon = SIMD8<Float>(repeating: Float.ulpOfOne)
                        
                        for i in 0..<64 {
                            uSIMD[i] = distASIMD[i] / (sumSIMD[i] + epsilon)
                        }
                    }
                }
            }
        }
        
        // Compute final transport cost using SIMD
        var totalCost: Float = 0
        
        for i in 0..<512 {
            let rowOffset = i * 512
            let uVal = u[i]
            
            for j in stride(from: 0, to: 512, by: 8) {
                let vVals = SIMD8<Float>(
                    v[j], v[j + 1], v[j + 2], v[j + 3],
                    v[j + 4], v[j + 5], v[j + 6], v[j + 7]
                )
                
                let kVals = SIMD8<Float>(
                    K[rowOffset + j], K[rowOffset + j + 1], K[rowOffset + j + 2], K[rowOffset + j + 3],
                    K[rowOffset + j + 4], K[rowOffset + j + 5], K[rowOffset + j + 6], K[rowOffset + j + 7]
                )
                
                let costs = SIMD8<Float>(
                    computeCost(i, j, costFunction: costFunction),
                    computeCost(i, j + 1, costFunction: costFunction),
                    computeCost(i, j + 2, costFunction: costFunction),
                    computeCost(i, j + 3, costFunction: costFunction),
                    computeCost(i, j + 4, costFunction: costFunction),
                    computeCost(i, j + 5, costFunction: costFunction),
                    computeCost(i, j + 6, costFunction: costFunction),
                    computeCost(i, j + 7, costFunction: costFunction)
                )
                
                let transport = SIMD8<Float>(repeating: uVal) * kVals * vVals
                let contribution = transport * costs
                totalCost += contribution.sum()
            }
        }
        
        return totalCost
    }
    
    /// SIMD-optimized normalization to probability distribution
    @inlinable
    private func normalizeToDistribution(_ vector: Vector512) -> [Float] {
        return vector.withUnsafeMetalBytes { bytes in
            let ptr = bytes.bindMemory(to: SIMD8<Float>.self)
            
            // Find minimum using SIMD
            var minVec = SIMD8<Float>(repeating: Float.greatestFiniteMagnitude)
            for i in 0..<64 {
                minVec = min(minVec, ptr[i])
            }
            let minVal = minVec.min()
            
            // Shift values to be positive and compute sum using SIMD
            var shifted = [Float](repeating: 0, count: 512)
            var sumVec = SIMD8<Float>.zero
            let minShift = SIMD8<Float>(repeating: minVal - Float.ulpOfOne)
            
            shifted.withUnsafeMutableBufferPointer { shiftedPtr in
                let shiftedSIMD = shiftedPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                
                for i in 0..<64 {
                    let shifted8 = ptr[i] - minShift
                    shiftedSIMD[i] = shifted8
                    sumVec += shifted8
                }
            }
            
            let totalSum = sumVec.sum()
            let invSum = 1.0 / totalSum
            let invSumVec = SIMD8<Float>(repeating: invSum)
            
            // Normalize using SIMD
            shifted.withUnsafeMutableBufferPointer { shiftedPtr in
                let shiftedSIMD = shiftedPtr.baseAddress!.assumingMemoryBound(to: SIMD8<Float>.self)
                
                for i in 0..<64 {
                    shiftedSIMD[i] = shiftedSIMD[i] * invSumVec
                }
            }
            
            return shifted
        }
    }
    
    private func computeCost(_ i: Int, _ j: Int, costFunction: CostFunction) -> Float {
        switch costFunction {
        case .euclidean:
            let diff = Float(i - j)
            return abs(diff)
        case .manhattan:
            return Float(abs(i - j))
        case .custom(let fn):
            return fn(Float(i), Float(j))
        }
    }
}

/// Linear programming solver for optimal transport
struct EMDOptimizer {
    
    /// Solve optimal transport problem using linear programming
    func solveOptimalTransport(
        source: [Float],
        target: [Float],
        costMatrix: [[Float]]
    ) throws -> Float {
        // For large-scale problems, we use a simplified approximation
        // Full linear programming would require external solver
        
        // Use greedy approximation for now
        var remainingSource = source
        var remainingTarget = target
        var totalCost: Float = 0
        
        // Create sorted list of costs with indices
        var costs: [(i: Int, j: Int, cost: Float)] = []
        for i in 0..<512 {
            for j in 0..<512 {
                costs.append((i, j, costMatrix[i][j]))
            }
        }
        costs.sort { $0.cost < $1.cost }
        
        // Greedy assignment
        for (i, j, cost) in costs {
            let amount = min(remainingSource[i], remainingTarget[j])
            if amount > 0 {
                totalCost += amount * cost
                remainingSource[i] -= amount
                remainingTarget[j] -= amount
            }
            
            // Check if we're done
            if remainingSource.allSatisfy({ $0 < Float.ulpOfOne }) {
                break
            }
        }
        
        return totalCost
    }
}

/// Cache for EMD computations
public actor EarthMoversDistanceCache {
    private var cache: [String: Float] = [:]
    private let maxCacheSize = 1000
    
    public static let shared = EarthMoversDistanceCache()
    
    public func getCachedDistance(key: String) -> Float? {
        return cache[key]
    }
    
    public func cacheDistance(key: String, distance: Float) {
        if cache.count >= maxCacheSize {
            // Remove oldest entries
            let toRemove = cache.count / 10
            for _ in 0..<toRemove {
                cache.removeFirst()
            }
        }
        cache[key] = distance
    }
    
    public func clear() {
        cache.removeAll()
    }
}