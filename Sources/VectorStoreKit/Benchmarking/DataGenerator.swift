// VectorStoreKit: Data Generator
//
// Generate synthetic vector data for benchmarking

import Foundation
import simd

/// Synthetic data generator for benchmarking
public struct DataGenerator {
    
    // MARK: - Types
    
    public enum Distribution {
        case uniform(min: Float, max: Float)
        case normal(mean: Float, stdDev: Float)
        case exponential(lambda: Float)
        case clustered(centers: Int, spread: Float)
        case zipfian(alpha: Double)
        case powerLaw(alpha: Float)
    }
    
    public enum VectorPattern {
        case random
        case sequential
        case clustered(clusters: Int)
        case grid(dimensions: Int)
        case spiral(turns: Float)
        case wave(frequency: Float)
    }
    
    public struct GeneratorConfig {
        public let seed: UInt64?
        public let distribution: Distribution
        public let pattern: VectorPattern
        public let sparsity: Float // 0.0 = dense, 1.0 = all zeros
        public let noise: Float // Amount of noise to add
        
        public init(
            seed: UInt64? = nil,
            distribution: Distribution = .uniform(min: -1, max: 1),
            pattern: VectorPattern = .random,
            sparsity: Float = 0.0,
            noise: Float = 0.0
        ) {
            self.seed = seed
            self.distribution = distribution
            self.pattern = pattern
            self.sparsity = sparsity
            self.noise = noise
        }
        
        public static let `default` = GeneratorConfig()
        
        public static let clustered = GeneratorConfig(
            distribution: .clustered(centers: 10, spread: 0.1),
            pattern: .clustered(clusters: 10)
        )
        
        public static let sparse = GeneratorConfig(
            sparsity: 0.9
        )
        
        public static let highDimensional = GeneratorConfig(
            distribution: .normal(mean: 0, stdDev: 1),
            noise: 0.01
        )
    }
    
    // MARK: - Properties
    
    private let config: GeneratorConfig
    private var rng: RandomNumberGenerator
    
    // MARK: - Initialization
    
    public init(config: GeneratorConfig = .default) {
        self.config = config
        
        if let seed = config.seed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }
    
    // MARK: - Vector Generation
    
    /// Generate a single vector
    public mutating func generateVector(dimensions: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions)
        
        // Apply pattern
        switch config.pattern {
        case .random:
            for i in 0..<dimensions {
                vector[i] = generateValue()
            }
            
        case .sequential:
            for i in 0..<dimensions {
                vector[i] = Float(i) / Float(dimensions) * 2.0 - 1.0
            }
            
        case .clustered(let clusters):
            let clusterId = Int.random(in: 0..<clusters, using: &rng)
            let center = generateClusterCenter(clusterId: clusterId, dimensions: dimensions)
            for i in 0..<dimensions {
                vector[i] = center[i] + generateNoise()
            }
            
        case .grid(let gridDims):
            let gridSize = Int(pow(Double(dimensions), 1.0 / Double(gridDims)))
            for i in 0..<dimensions {
                let coord = (i / gridSize) % gridSize
                vector[i] = Float(coord) / Float(gridSize) * 2.0 - 1.0
            }
            
        case .spiral(let turns):
            for i in 0..<dimensions {
                let t = Float(i) / Float(dimensions) * turns * 2 * .pi
                vector[i] = cos(t) * (1.0 - Float(i) / Float(dimensions))
                if i + 1 < dimensions {
                    vector[i + 1] = sin(t) * (1.0 - Float(i) / Float(dimensions))
                }
            }
            
        case .wave(let frequency):
            for i in 0..<dimensions {
                let phase = Float(i) / Float(dimensions) * frequency * 2 * .pi
                vector[i] = sin(phase)
            }
        }
        
        // Apply sparsity
        if config.sparsity > 0 {
            for i in 0..<dimensions {
                if Float.random(in: 0...1, using: &rng) < config.sparsity {
                    vector[i] = 0
                }
            }
        }
        
        // Add noise
        if config.noise > 0 {
            for i in 0..<dimensions {
                vector[i] += generateNoise() * config.noise
            }
        }
        
        return vector
    }
    
    /// Generate multiple vectors
    public mutating func generateVectors(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in generateVector(dimensions: dimensions) }
    }
    
    /// Generate SIMD32 vectors
    public mutating func generateSIMD32Vectors(count: Int) -> [SIMD32<Float>] {
        return (0..<count).map { _ in
            var vector = SIMD32<Float>()
            for i in 0..<32 {
                vector[i] = generateValue()
            }
            return vector
        }
    }
    
    /// Generate SIMD64 vectors
    public mutating func generateSIMD64Vectors(count: Int) -> [SIMD64<Float>] {
        return (0..<count).map { _ in
            var vector = SIMD64<Float>()
            for i in 0..<64 {
                vector[i] = generateValue()
            }
            return vector
        }
    }
    
    /// Generate Vector512 instances
    public mutating func generateVector512s(count: Int) -> [Vector512] {
        return (0..<count).map { _ in
            let values = (0..<512).map { _ in generateValue() }
            return Vector512(values)
        }
    }
    
    // MARK: - Metadata Generation
    
    /// Generate test metadata
    public mutating func generateMetadata<M: Codable & Sendable>(
        count: Int,
        generator: (Int) -> M
    ) -> [M] {
        return (0..<count).map { generator($0) }
    }
    
    /// Generate test metadata with common patterns
    public mutating func generateTestMetadata(count: Int) -> [TestMetadata] {
        return (0..<count).map { i in
            TestMetadata(
                id: i,
                category: "category_\(i % 10)",
                timestamp: Date(timeIntervalSinceNow: -Double(i)),
                tags: generateTags(forItem: i)
            )
        }
    }
    
    // MARK: - Query Generation
    
    /// Generate query vectors with specific characteristics
    public mutating func generateQueries(
        count: Int,
        dimensions: Int,
        nearData: [[Float]]? = nil
    ) -> [[Float]] {
        if let nearData = nearData {
            // Generate queries near existing data
            return (0..<count).map { _ in
                let baseVector = nearData.randomElement()!
                return baseVector.map { $0 + Float.random(in: -0.1...0.1, using: &rng) }
            }
        } else {
            // Generate random queries
            return generateVectors(count: count, dimensions: dimensions)
        }
    }
    
    // MARK: - Dataset Generation
    
    /// Generate a complete dataset
    public mutating func generateDataset(
        vectorCount: Int,
        dimensions: Int,
        queryCount: Int
    ) -> (vectors: [[Float]], queries: [[Float]], groundTruth: [[String]]) {
        let vectors = generateVectors(count: vectorCount, dimensions: dimensions)
        let queries = generateQueries(count: queryCount, dimensions: dimensions, nearData: vectors)
        
        // Compute ground truth (brute force)
        let groundTruth = computeGroundTruth(vectors: vectors, queries: queries, k: 100)
        
        return (vectors, queries, groundTruth)
    }
    
    /// Generate dataset with specific data distribution
    public mutating func generateSkewedDataset(
        vectorCount: Int,
        dimensions: Int,
        hotspotRatio: Float = 0.8
    ) -> [[Float]] {
        let hotspotCount = Int(Float(vectorCount) * hotspotRatio)
        let normalCount = vectorCount - hotspotCount
        
        var vectors: [[Float]] = []
        
        // Generate hotspot vectors (clustered)
        let hotspotCenter = generateVector(dimensions: dimensions)
        for _ in 0..<hotspotCount {
            let vector = hotspotCenter.map { $0 + Float.random(in: -0.1...0.1, using: &rng) }
            vectors.append(vector)
        }
        
        // Generate normal vectors
        for _ in 0..<normalCount {
            vectors.append(generateVector(dimensions: dimensions))
        }
        
        return vectors.shuffled()
    }
    
    // MARK: - Time Series Generation
    
    /// Generate time series vector data
    public mutating func generateTimeSeries(
        length: Int,
        dimensions: Int,
        seasonality: Int? = nil
    ) -> [[Float]] {
        var series: [[Float]] = []
        var baseVector = generateVector(dimensions: dimensions)
        
        for t in 0..<length {
            var vector = baseVector
            
            // Add trend
            let trend = Float(t) / Float(length) * 0.1
            for i in 0..<dimensions {
                vector[i] += trend
            }
            
            // Add seasonality
            if let seasonality = seasonality {
                let phase = Float(t % seasonality) / Float(seasonality) * 2 * .pi
                for i in 0..<dimensions {
                    vector[i] += sin(phase + Float(i)) * 0.2
                }
            }
            
            // Add noise
            for i in 0..<dimensions {
                vector[i] += generateNoise() * 0.05
            }
            
            series.append(vector)
            
            // Update base vector (random walk)
            for i in 0..<dimensions {
                baseVector[i] += generateNoise() * 0.01
            }
        }
        
        return series
    }
    
    // MARK: - Private Helpers
    
    private mutating func generateValue() -> Float {
        switch config.distribution {
        case .uniform(let min, let max):
            return Float.random(in: min...max, using: &rng)
            
        case .normal(let mean, let stdDev):
            return generateNormal(mean: mean, stdDev: stdDev)
            
        case .exponential(let lambda):
            let u = Float.random(in: 0..<1, using: &rng)
            return -log(1 - u) / lambda
            
        case .clustered(_, let spread):
            return generateNormal(mean: 0, stdDev: spread)
            
        case .zipfian(let alpha):
            return Float(generateZipfian(alpha: alpha))
            
        case .powerLaw(let alpha):
            let u = Float.random(in: 0..<1, using: &rng)
            return pow(1 - u, -1 / alpha)
        }
    }
    
    private mutating func generateNormal(mean: Float, stdDev: Float) -> Float {
        // Box-Muller transform
        let u1 = Float.random(in: Float.ulpOfOne..<1, using: &rng)
        let u2 = Float.random(in: 0..<1, using: &rng)
        let z0 = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        return mean + stdDev * z0
    }
    
    private mutating func generateZipfian(alpha: Double) -> Int {
        let n = 1000 // Range of values
        var sum = 0.0
        for i in 1...n {
            sum += 1.0 / pow(Double(i), alpha)
        }
        
        let u = Double.random(in: 0..<1, using: &rng) * sum
        var cumSum = 0.0
        
        for i in 1...n {
            cumSum += 1.0 / pow(Double(i), alpha)
            if cumSum >= u {
                return i
            }
        }
        
        return n
    }
    
    private mutating func generateNoise() -> Float {
        return Float.random(in: -1...1, using: &rng)
    }
    
    private mutating func generateClusterCenter(clusterId: Int, dimensions: Int) -> [Float] {
        // Deterministic cluster centers based on ID
        var center = [Float](repeating: 0, count: dimensions)
        var localRng = SeededRandomNumberGenerator(seed: UInt64(clusterId))
        
        for i in 0..<dimensions {
            center[i] = Float.random(in: -1...1, using: &localRng)
        }
        
        return center
    }
    
    private func generateTags(forItem i: Int) -> [String] {
        let tagCount = i % 5 + 1
        return (0..<tagCount).map { "tag_\(i)_\($0)" }
    }
    
    private func computeGroundTruth(
        vectors: [[Float]],
        queries: [[Float]],
        k: Int
    ) -> [[String]] {
        return queries.map { query in
            var distances: [(index: Int, distance: Float)] = []
            
            for (i, vector) in vectors.enumerated() {
                let distance = euclideanDistance(query, vector)
                distances.append((i, distance))
            }
            
            return distances
                .sorted { $0.distance < $1.distance }
                .prefix(k)
                .map { "vec_\($0.index)" }
        }
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}

// MARK: - Seeded Random Number Generator

private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // Linear congruential generator
        state = state &* 2862933555777941757 &+ 3037000493
        return state
    }
}

// MARK: - Convenience Extensions

public extension DataGenerator {
    
    /// Generate a standard benchmark dataset
    static func standardBenchmarkDataset(
        scale: BenchmarkScale = .medium
    ) -> (vectors: [[Float]], queries: [[Float]], groundTruth: [[String]]) {
        var generator = DataGenerator(config: .default)
        
        let (vectorCount, queryCount, dimensions) = scale.parameters
        return generator.generateDataset(
            vectorCount: vectorCount,
            dimensions: dimensions,
            queryCount: queryCount
        )
    }
    
    enum BenchmarkScale {
        case small
        case medium
        case large
        case xlarge
        
        var parameters: (vectors: Int, queries: Int, dimensions: Int) {
            switch self {
            case .small:
                return (1_000, 100, 128)
            case .medium:
                return (10_000, 1_000, 256)
            case .large:
                return (100_000, 10_000, 512)
            case .xlarge:
                return (1_000_000, 100_000, 768)
            }
        }
    }
}

// MARK: - Dataset Formats

public extension DataGenerator {
    
    /// Save dataset to HDF5 format
    func saveHDF5(vectors: [[Float]], queries: [[Float]], path: String) throws {
        // Implementation would use HDF5 library
        fatalError("HDF5 support not implemented")
    }
    
    /// Save dataset to binary format
    func saveBinary(vectors: [[Float]], queries: [[Float]], path: String) throws {
        let data = NSMutableData()
        
        // Write header
        var vectorCount = Int32(vectors.count)
        var queryCount = Int32(queries.count)
        var dimensions = Int32(vectors.first?.count ?? 0)
        
        data.append(&vectorCount, length: 4)
        data.append(&queryCount, length: 4)
        data.append(&dimensions, length: 4)
        
        // Write vectors
        for vector in vectors {
            var floatArray = vector
            data.append(&floatArray, length: vector.count * 4)
        }
        
        // Write queries
        for query in queries {
            var floatArray = query
            data.append(&floatArray, length: query.count * 4)
        }
        
        try data.write(toFile: path)
    }
    
    /// Load dataset from binary format
    static func loadBinary(path: String) throws -> (vectors: [[Float]], queries: [[Float]]) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        
        var offset = 0
        
        // Read header
        let vectorCount = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: offset, as: Int32.self)
        }
        offset += 4
        
        let queryCount = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: offset, as: Int32.self)
        }
        offset += 4
        
        let dimensions = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: offset, as: Int32.self)
        }
        offset += 4
        
        // Read vectors
        var vectors: [[Float]] = []
        for _ in 0..<vectorCount {
            var vector: [Float] = []
            for _ in 0..<dimensions {
                let value = data.withUnsafeBytes { bytes in
                    bytes.load(fromByteOffset: offset, as: Float.self)
                }
                vector.append(value)
                offset += 4
            }
            vectors.append(vector)
        }
        
        // Read queries
        var queries: [[Float]] = []
        for _ in 0..<queryCount {
            var query: [Float] = []
            for _ in 0..<dimensions {
                let value = data.withUnsafeBytes { bytes in
                    bytes.load(fromByteOffset: offset, as: Float.self)
                }
                query.append(value)
                offset += 4
            }
            queries.append(query)
        }
        
        return (vectors, queries)
    }
}