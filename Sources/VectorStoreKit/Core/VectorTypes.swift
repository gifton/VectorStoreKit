// VectorStoreKit: Core Vector Types
//
// Advanced type system for vector operations with research-grade capabilities

import Foundation
import simd
import Accelerate

// MARK: - Core Vector Types

/// Vector identifier type optimized for performance
public typealias VectorID = String

/// Distance/similarity score with precision guarantees
public typealias Distance = Float

/// Timestamp with nanosecond precision for research applications
public typealias Timestamp = UInt64

// MARK: - Advanced Vector Entry

/// A sophisticated vector entry with comprehensive metadata and research capabilities
public struct VectorEntry<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Codable, Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Core Properties
    
    /// Unique identifier for this vector
    public let id: VectorID
    
    /// The vector data with SIMD optimization
    public let vector: Vector
    
    /// Associated metadata with type safety
    public let metadata: Metadata
    
    /// Nanosecond-precision timestamp
    public let timestamp: Timestamp
    
    /// Storage tier assignment
    public let tier: StorageTier
    
    /// Vector quality metrics for research
    public let quality: VectorQuality
    
    /// Compression ratio achieved (1.0 = no compression)
    public let compressionRatio: Float
    
    /// Access pattern for optimization
    public private(set) var accessPattern: AccessPattern
    
    // MARK: - Initialization
    
    /// Create a new vector entry with automatic quality assessment
    public init(
        id: VectorID, 
        vector: Vector, 
        metadata: Metadata,
        tier: StorageTier = .auto
    ) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.timestamp = DispatchTime.now().uptimeNanoseconds
        self.tier = tier
        self.quality = VectorQuality.assess(vector)
        self.compressionRatio = 1.0
        self.accessPattern = AccessPattern()
    }
    
    /// Create a new vector entry with cached quality assessment (async)
    public init(
        id: VectorID,
        vector: Vector,
        metadata: Metadata,
        tier: StorageTier = .auto,
        quality: VectorQuality
    ) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.timestamp = DispatchTime.now().uptimeNanoseconds
        self.tier = tier
        self.quality = quality
        self.compressionRatio = 1.0
        self.accessPattern = AccessPattern()
    }
    
    /// Factory method that uses cached quality assessment
    public static func createWithCache(
        id: VectorID,
        vector: Vector,
        metadata: Metadata,
        tier: StorageTier = .auto
    ) async -> VectorEntry {
        let quality = await VectorQuality.assessWithCache(vector)
        return VectorEntry(
            id: id,
            vector: vector,
            metadata: metadata,
            tier: tier,
            quality: quality
        )
    }
    
    /// Update access pattern for ML-driven optimization
    public mutating func recordAccess(at time: Timestamp = DispatchTime.now().uptimeNanoseconds) {
        accessPattern.recordAccess(at: time)
    }
}

// MARK: - Vector Quality Assessment Cache

/// Thread-safe cache for VectorQuality assessments using LRU eviction
actor VectorQualityCache {
    private var cache: [Data: VectorQuality] = [:]
    private var accessOrder: [Data] = []
    private let maxSize: Int
    
    /// Initialize with configurable max size
    init(maxSize: Int = 1000) {
        self.maxSize = maxSize
    }
    
    /// Get cached quality assessment
    func get(for vectorData: Data) -> VectorQuality? {
        if let quality = cache[vectorData] {
            // Move to end for LRU behavior
            if let index = accessOrder.firstIndex(of: vectorData) {
                accessOrder.remove(at: index)
                accessOrder.append(vectorData)
            }
            return quality
        }
        return nil
    }
    
    /// Store quality assessment in cache with LRU eviction
    func set(_ quality: VectorQuality, for vectorData: Data) {
        // Check if we need to evict
        if cache.count >= maxSize && cache[vectorData] == nil {
            // Remove least recently used
            if let oldest = accessOrder.first {
                cache.removeValue(forKey: oldest)
                accessOrder.removeFirst()
            }
        }
        
        // Add to cache
        cache[vectorData] = quality
        if let index = accessOrder.firstIndex(of: vectorData) {
            accessOrder.remove(at: index)
        }
        accessOrder.append(vectorData)
    }
    
    /// Clear the cache
    func clear() {
        cache.removeAll()
        accessOrder.removeAll()
    }
    
    /// Get current cache size
    func size() -> Int {
        cache.count
    }
}

/// Global cache instance
private let qualityCache = VectorQualityCache()

// MARK: - Vector Quality Assessment

/// Advanced vector quality metrics for research and optimization
public struct VectorQuality: Codable, Sendable {
    
    /// Vector magnitude
    public let magnitude: Float
    
    /// Dimensionality utilization (percentage of non-zero dimensions)
    public let sparsity: Float
    
    /// Distribution uniformity measure
    public let entropy: Float
    
    /// Quantization readiness score
    public let quantizability: Float
    
    /// Clustering tendency
    public let clusterability: Float
    
    /// Assess vector quality automatically
    public static func assess<V: SIMD>(_ vector: V) -> VectorQuality 
    where V.Scalar: BinaryFloatingPoint {
        
        let magnitude = Self.computeMagnitude(vector)
        let sparsity = Self.computeSparsity(vector)
        let entropy = Self.computeEntropy(vector)
        let quantizability = Self.computeQuantizability(vector)
        let clusterability = Self.computeClusterability(vector)
        
        return VectorQuality(
            magnitude: magnitude,
            sparsity: sparsity,
            entropy: entropy,
            quantizability: quantizability,
            clusterability: clusterability
        )
    }
    
    /// Assess vector quality with caching support (async version)
    public static func assessWithCache<V: SIMD>(_ vector: V) async -> VectorQuality
    where V.Scalar: BinaryFloatingPoint {
        // Convert vector to Data for cache key
        let vectorData = withUnsafeBytes(of: vector) { Data($0) }
        
        // Check cache first
        if let cached = await qualityCache.get(for: vectorData) {
            return cached
        }
        
        // Compute quality metrics
        let quality = assess(vector)
        
        // Store in cache
        await qualityCache.set(quality, for: vectorData)
        
        return quality
    }
    
    /// Clear the quality assessment cache
    public static func clearCache() async {
        await qualityCache.clear()
    }
    
    private static func computeMagnitude<V: SIMD>(_ vector: V) -> Float 
    where V.Scalar: BinaryFloatingPoint {
        let squares = vector * vector
        return Float(sqrt(squares.sum()))
    }
    
    private static func computeSparsity<V: SIMD>(_ vector: V) -> Float 
    where V.Scalar: BinaryFloatingPoint {
        let nonZeroCount = vector.indices.reduce(0) { count, index in
            vector[index] != 0 ? count + 1 : count
        }
        return Float(nonZeroCount) / Float(vector.scalarCount)
    }
    
    private static func computeEntropy<V: SIMD>(_ vector: V) -> Float 
    where V.Scalar: BinaryFloatingPoint {
        // SIMD-optimized entropy calculation using vDSP
        let count = vector.scalarCount
        
        // Convert SIMD vector to Float array for vDSP
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            values[i] = Float(vector[i])
        }
        
        // Calculate absolute values using vDSP
        var absValues = [Float](repeating: 0, count: count)
        vDSP_vabs(values, 1, &absValues, 1, vDSP_Length(count))
        
        // Calculate sum using vDSP
        var sum: Float = 0
        vDSP_sve(absValues, 1, &sum, vDSP_Length(count))
        guard sum > 0 else { return 0 }
        
        // Calculate probabilities using vDSP (divide by sum)
        var probabilities = [Float](repeating: 0, count: count)
        var reciprocalSum = 1.0 / sum
        vDSP_vsmul(absValues, 1, &reciprocalSum, &probabilities, 1, vDSP_Length(count))
        
        // Calculate -p * log2(p) for entropy using vDSP
        // First, filter out zero probabilities and calculate log2
        var logProbs = [Float](repeating: 0, count: count)
        var entropy: Float = 0
        
        // vDSP doesn't have log2, so we use natural log and convert
        let log2e: Float = 1.0 / log(2.0)
        
        // Calculate entropy: sum of -p * log2(p)
        // We need to handle zeros carefully
        for i in 0..<count {
            if probabilities[i] > 0 {
                logProbs[i] = log(probabilities[i]) * log2e
            }
        }
        
        // Multiply probabilities by their log values
        var entropyTerms = [Float](repeating: 0, count: count)
        vDSP_vmul(probabilities, 1, logProbs, 1, &entropyTerms, 1, vDSP_Length(count))
        
        // Sum the entropy terms
        vDSP_sve(entropyTerms, 1, &entropy, vDSP_Length(count))
        
        return -entropy
    }
    
    private static func computeQuantizability<V: SIMD>(_ vector: V) -> Float 
    where V.Scalar: BinaryFloatingPoint {
        // Measure how well this vector would compress via quantization
        let values = (0..<vector.scalarCount).map { Float(vector[$0]) }
        let _ = values.max()! - values.min()!
        let uniqueValues = Set(values.map { ($0 * 1000).rounded() / 1000 }).count
        return 1.0 - Float(uniqueValues) / Float(values.count)
    }
    
    private static func computeClusterability<V: SIMD>(_ vector: V) -> Float 
    where V.Scalar: BinaryFloatingPoint {
        // Simplified measure of how likely this vector is to form clusters
        let values = (0..<vector.scalarCount).map { Float(vector[$0]) }
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count)
        return min(1.0, variance / (mean * mean + 0.001)) // Coefficient of variation
    }
}

// MARK: - Access Pattern Tracking

/// Sophisticated access pattern tracking for ML optimization
public struct AccessPattern: Codable, Sendable {
    
    /// Total number of accesses
    public private(set) var accessCount: UInt64 = 0
    
    /// Last access timestamp
    public private(set) var lastAccess: Timestamp = 0
    
    /// Access frequency (accesses per second)
    public private(set) var frequency: Float = 0.0
    
    /// Recent access timestamps for pattern analysis
    private var recentAccesses: [Timestamp] = []
    
    /// Maximum number of recent accesses to track
    private var maxRecentAccesses = 100
    
    /// Record a new access for pattern analysis
    public mutating func recordAccess(at timestamp: Timestamp) {
        accessCount += 1
        lastAccess = timestamp
        
        // Add to recent accesses
        recentAccesses.append(timestamp)
        if recentAccesses.count > maxRecentAccesses {
            recentAccesses.removeFirst()
        }
        
        // Update frequency calculation
        updateFrequency()
    }
    
    /// Calculate access patterns for optimization
    public func analyzePattern() -> AccessPatternAnalysis {
        return AccessPatternAnalysis(
            frequency: frequency,
            recency: calculateRecency(),
            regularity: calculateRegularity(),
            burstiness: calculateBurstiness()
        )
    }
    
    private mutating func updateFrequency() {
        guard recentAccesses.count > 1 else { return }
        
        let timeSpan = recentAccesses.last! - recentAccesses.first!
        let timeSpanSeconds = Float(timeSpan) / 1_000_000_000.0 // Convert from nanoseconds
        
        if timeSpanSeconds > 0 {
            frequency = Float(recentAccesses.count - 1) / timeSpanSeconds
        }
    }
    
    private func calculateRecency() -> Float {
        let now = DispatchTime.now().uptimeNanoseconds
        let timeSinceLastAccess = now - lastAccess
        let secondsSinceLastAccess = Float(timeSinceLastAccess) / 1_000_000_000.0
        
        // Recency score (higher = more recent)
        return max(0, 1.0 - secondsSinceLastAccess / 3600.0) // Decay over 1 hour
    }
    
    private func calculateRegularity() -> Float {
        guard recentAccesses.count > 2 else { return 0 }
        
        let intervals = zip(recentAccesses.dropFirst(), recentAccesses).map { $0.0 - $0.1 }
        let meanInterval = intervals.reduce(0, +) / UInt64(intervals.count)
        let variance = intervals.reduce(0) { $0 + pow(Float(Int64($1) - Int64(meanInterval)), 2) } / Float(intervals.count)
        
        // Regularity score (higher = more regular)
        return meanInterval > 0 ? max(0, 1.0 - sqrt(variance) / Float(meanInterval)) : 0
    }
    
    private func calculateBurstiness() -> Float {
        guard recentAccesses.count > 1 else { return 0 }
        
        let intervals = zip(recentAccesses.dropFirst(), recentAccesses).map { Float($0.0 - $0.1) }
        let mean = intervals.reduce(0, +) / Float(intervals.count)
        let variance = intervals.reduce(0) { $0 + pow($1 - mean, 2) } / Float(intervals.count)
        
        // Burstiness coefficient
        return mean > 0 ? (sqrt(variance) - mean) / (sqrt(variance) + mean) : 0
    }
}

/// Analysis results from access pattern tracking
public struct AccessPatternAnalysis: Codable, Sendable {
    public let frequency: Float      // Accesses per second
    public let recency: Float        // How recently accessed (0-1)
    public let regularity: Float     // How regular the access pattern is (0-1)
    public let burstiness: Float     // How bursty the access pattern is (-1 to 1)
}

// MARK: - Storage Tier Enumeration

/// Advanced storage tier system with ML-driven assignment
public enum StorageTier: Int, Codable, Sendable, CaseIterable {
    case hot = 0        // In-memory, fastest access, ML-optimized
    case warm = 1       // Memory-mapped, balanced performance
    case cold = 2       // Disk-based, high capacity, compressed
    case frozen = 3     // Archive storage, maximum compression
    case auto = 99      // ML-driven automatic tier assignment
    
    /// Performance characteristics for each tier
    public var characteristics: TierCharacteristics {
        switch self {
        case .hot:
            return TierCharacteristics(
                latency: .nanoseconds(100),
                throughput: .high,
                capacity: .low,
                compression: .none
            )
        case .warm:
            return TierCharacteristics(
                latency: .microseconds(10),
                throughput: .medium,
                capacity: .medium,
                compression: .light
            )
        case .cold:
            return TierCharacteristics(
                latency: .milliseconds(1),
                throughput: .low,
                capacity: .high,
                compression: .aggressive
            )
        case .frozen:
            return TierCharacteristics(
                latency: .milliseconds(100),
                throughput: .minimal,
                capacity: .unlimited,
                compression: .maximum
            )
        case .auto:
            return TierCharacteristics(
                latency: .variable,
                throughput: .adaptive,
                capacity: .adaptive,
                compression: .adaptive
            )
        }
    }
}

/// Performance characteristics for storage tiers
public struct TierCharacteristics: Sendable {
    public let latency: LatencyTarget
    public let throughput: ThroughputLevel
    public let capacity: CapacityLevel
    public let compression: CompressionLevel
}

public enum LatencyTarget: Sendable {
    case nanoseconds(Int)
    case microseconds(Int)
    case milliseconds(Int)
    case variable
}

public enum ThroughputLevel: Sendable {
    case minimal, low, medium, high, adaptive
}

public enum CapacityLevel: Sendable {
    case low, medium, high, unlimited, adaptive
}

// MARK: - Quantization Types

/// Quantization scheme for vector compression
public enum QuantizationScheme: Sendable, Codable, Equatable {
    case scalar(bits: Int = 8)
    case product(segments: Int, bits: Int = 8)
    case binary
    case vector(codebookSize: Int)
    case learned(modelID: String? = nil)
    case none
    
    // For Metal shader compatibility
    public var metalFunctionName: String {
        switch self {
        case .scalar: return "scalarQuantize"
        case .product: return "productQuantize"
        case .binary: return "binaryQuantize"
        case .vector: return "vectorQuantize"
        case .learned: return "learnedQuantize"
        case .none: return "none"
        }
    }
    
    // For Metal implementation compatibility
    public var rawValue: String {
        switch self {
        case .scalar: return "scalar"
        case .product: return "product"
        case .binary: return "binary"
        case .vector: return "vector"
        case .learned: return "learned"
        case .none: return "none"
        }
    }
}

/// Parameters for quantization operations
public struct QuantizationParameters: Sendable, Codable {
    // API-level parameters
    public let trainingIterations: Int
    public let useAsymmetricDistance: Bool
    public let useOptimizedLayout: Bool
    public let compressionTarget: Float?
    
    // Implementation-level parameters
    public let precision: Int
    public let centroids: Int?
    public let subvectors: Int?
    public let customData: Data?
    
    public init(
        trainingIterations: Int = 25,
        useAsymmetricDistance: Bool = true,
        useOptimizedLayout: Bool = true,
        compressionTarget: Float? = nil,
        precision: Int = 8,
        centroids: Int? = nil,
        subvectors: Int? = nil,
        customData: Data? = nil
    ) {
        self.trainingIterations = trainingIterations
        self.useAsymmetricDistance = useAsymmetricDistance
        self.useOptimizedLayout = useOptimizedLayout
        self.compressionTarget = compressionTarget
        self.precision = precision
        self.centroids = centroids
        self.subvectors = subvectors
        self.customData = customData
    }
}

/// Quantized vector representation
public struct QuantizedVector: Sendable, Codable {
    public let codes: [UInt8]
    public let metadata: QuantizationMetadata
    
    public init(codes: [UInt8], metadata: QuantizationMetadata) {
        self.codes = codes
        self.metadata = metadata
    }
    
    // Convenience properties for Metal implementation
    public var quantizedData: Data { 
        Data(codes) 
    }
    
    public var originalDimensions: Int { 
        metadata.originalDimensions 
    }
    
    public var scheme: QuantizationScheme { 
        metadata.scheme 
    }
    
    public var parameters: QuantizationParameters {
        // Create parameters from metadata for compatibility
        QuantizationParameters(
            compressionTarget: metadata.compressionRatio,
            precision: 8, // Default precision
            centroids: nil,
            subvectors: nil
        )
    }
    
    // Alternative initializer for Metal implementation compatibility
    public init(
        originalDimensions: Int,
        quantizedData: Data,
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) {
        self.codes = Array(quantizedData)
        self.metadata = QuantizationMetadata(
            scheme: scheme,
            originalDimensions: originalDimensions,
            compressionRatio: parameters.compressionTarget ?? 1.0,
            quantizationError: nil
        )
    }
}

/// Metadata for quantized vectors
public struct QuantizationMetadata: Sendable, Codable {
    public let scheme: QuantizationScheme
    public let originalDimensions: Int
    public let compressionRatio: Float
    public let quantizationError: Float?
    
    public init(
        scheme: QuantizationScheme,
        originalDimensions: Int,
        compressionRatio: Float,
        quantizationError: Float? = nil
    ) {
        self.scheme = scheme
        self.originalDimensions = originalDimensions
        self.compressionRatio = compressionRatio
        self.quantizationError = quantizationError
    }
}


