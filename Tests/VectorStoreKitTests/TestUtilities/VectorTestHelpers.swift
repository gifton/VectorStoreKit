import XCTest
import VectorStoreKit
import Accelerate

/// Test utilities for vector operations
enum VectorTestHelpers {
    /// Generate random vectors for testing
    static func randomVectors(
        count: Int,
        dimension: Int,
        seed: UInt64 = 42
    ) -> [[Float]] {
        var generator = SeededRandomGenerator(seed: seed)
        
        return (0..<count).map { _ in
            (0..<dimension).map { _ in
                Float.random(in: -1...1, using: &generator)
            }
        }
    }
    
    /// Generate clustered vectors for testing clustering algorithms
    static func clusteredVectors(
        clusters: Int,
        vectorsPerCluster: Int,
        dimension: Int,
        spread: Float = 0.1
    ) -> (vectors: [[Float]], labels: [Int]) {
        var allVectors: [[Float]] = []
        var labels: [Int] = []
        
        for cluster in 0..<clusters {
            // Generate cluster center
            let center = randomVectors(count: 1, dimension: dimension, seed: UInt64(cluster * 1000))[0]
            
            // Generate vectors around center
            for _ in 0..<vectorsPerCluster {
                let noise = (0..<dimension).map { _ in
                    Float.random(in: -spread...spread)
                }
                
                let vector = zip(center, noise).map { $0 + $1 }
                allVectors.append(vector)
                labels.append(cluster)
            }
        }
        
        return (allVectors, labels)
    }
    
    /// Compute recall for search results
    static func recall(
        retrieved: Set<UUID>,
        relevant: Set<UUID>
    ) -> Float {
        guard !relevant.isEmpty else { return 0.0 }
        let intersection = retrieved.intersection(relevant)
        return Float(intersection.count) / Float(relevant.count)
    }
    
    /// Compute precision for search results
    static func precision(
        retrieved: Set<UUID>,
        relevant: Set<UUID>
    ) -> Float {
        guard !retrieved.isEmpty else { return 0.0 }
        let intersection = retrieved.intersection(relevant)
        return Float(intersection.count) / Float(retrieved.count)
    }
    
    /// Compute F1 score
    static func f1Score(
        retrieved: Set<UUID>,
        relevant: Set<UUID>
    ) -> Float {
        let p = precision(retrieved: retrieved, relevant: relevant)
        let r = recall(retrieved: retrieved, relevant: relevant)
        guard p + r > 0 else { return 0.0 }
        return 2 * p * r / (p + r)
    }
    
    /// Generate synthetic embeddings with specific patterns
    static func syntheticEmbeddings(
        count: Int,
        dimension: Int,
        pattern: EmbeddingPattern = .random
    ) -> [[Float]] {
        switch pattern {
        case .random:
            return randomVectors(count: count, dimension: dimension)
            
        case .sequential:
            // Create embeddings that form a sequence in vector space
            return (0..<count).map { i in
                let base = Float(i) / Float(count)
                return (0..<dimension).map { d in
                    sin(base * Float.pi * Float(d + 1))
                }
            }
            
        case .orthogonal:
            // Create maximally separated embeddings
            var embeddings: [[Float]] = []
            for i in 0..<count {
                var vector = [Float](repeating: 0, count: dimension)
                if i < dimension {
                    vector[i] = 1.0
                } else {
                    // Random orthogonal vectors for remaining
                    vector = randomVectors(count: 1, dimension: dimension)[0]
                    // Orthogonalize against existing
                    for j in 0..<min(i, dimension) {
                        let existing = embeddings[j]
                        let dot = zip(vector, existing).map(*).reduce(0, +)
                        vector = zip(vector, existing).map { $0.0 - dot * $0.1 }
                    }
                    // Normalize
                    let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
                    vector = vector.map { $0 / norm }
                }
                embeddings.append(vector)
            }
            return embeddings
            
        case .similar:
            // Create very similar embeddings (useful for testing precision)
            let base = randomVectors(count: 1, dimension: dimension)[0]
            return (0..<count).map { _ in
                let noise = (0..<dimension).map { _ in Float.random(in: -0.01...0.01) }
                return zip(base, noise).map { $0 + $1 }
            }
        }
    }
    
    /// Embedding patterns for synthetic data
    enum EmbeddingPattern {
        case random
        case sequential
        case orthogonal
        case similar
    }
}

/// Seeded random generator for reproducible tests
struct SeededRandomGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // Linear congruential generator
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

/// Test metadata type
struct TestMetadata: Codable, Sendable, Equatable {
    let id: String
    let category: String
    let timestamp: Date
    let tags: [String]
    
    init(
        id: String = UUID().uuidString,
        category: String = "test",
        timestamp: Date = Date(),
        tags: [String] = []
    ) {
        self.id = id
        self.category = category
        self.timestamp = timestamp
        self.tags = tags
    }
}

/// Performance measurement helpers
extension XCTestCase {
    /// Measure async operation with custom metrics
    func measureAsync<T>(
        metrics: [XCTMetric] = [XCTClockMetric(), XCTMemoryMetric()],
        options: XCTMeasureOptions = .default,
        block: @escaping () async throws -> T
    ) {
        measure(metrics: metrics, options: options) {
            let expectation = expectation(description: "async measurement")
            
            Task {
                _ = try await block()
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 30.0)
        }
    }
    
    /// Assert performance meets requirements
    func assertPerformance<T>(
        _ operation: () async throws -> T,
        maxDuration: TimeInterval,
        maxMemoryDelta: Int? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let memoryBefore = getCurrentMemoryUsage()
        let start = CFAbsoluteTimeGetCurrent()
        
        _ = try await operation()
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let memoryAfter = getCurrentMemoryUsage()
        let memoryDelta = memoryAfter - memoryBefore
        
        XCTAssertLessThan(
            duration,
            maxDuration,
            "Operation took \(duration)s, expected < \(maxDuration)s",
            file: file,
            line: line
        )
        
        if let maxMemory = maxMemoryDelta {
            XCTAssertLessThan(
                memoryDelta,
                maxMemory,
                "Memory increased by \(memoryDelta) bytes, expected < \(maxMemory)",
                file: file,
                line: line
            )
        }
    }
    
    private func getCurrentMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}