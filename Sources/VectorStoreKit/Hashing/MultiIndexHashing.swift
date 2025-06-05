// VectorStoreKit: Multi-Index Hashing
//
// Advanced hash-based retrieval using multiple hash functions and tables

import Foundation
import simd
import CryptoKit


/// Multi-index hash structure
public actor MultiIndexHash {
    
    // MARK: - Properties
    
    private let configuration: MultiIndexHashConfiguration
    private var hashFunctions: [HashFunction]
    private var hashTables: [HashTable]
    private var vectorStore: [String: StoredVector]
    private let hasher = SHA256()
    
    private struct StoredVector {
        let id: String
        let vector: [Float]
        let norm: Float
        let hashes: [UInt64]
    }
    
    private struct HashTable {
        var buckets: [UInt64: Set<String>] = [:]
        
        mutating func insert(hash: UInt64, id: String) {
            buckets[hash, default: []].insert(id)
        }
        
        func retrieve(hash: UInt64) -> Set<String> {
            buckets[hash] ?? []
        }
        
        func multiProbe(hash: UInt64, distance: Int) -> Set<String> {
            var results = Set<String>()
            
            // Get exact match
            results.formUnion(retrieve(hash: hash))
            
            // Probe nearby hashes
            for d in 1...distance {
                // Flip d bits in the hash
                let perturbations = generatePerturbations(hash: hash, flips: d)
                for perturbed in perturbations {
                    results.formUnion(retrieve(hash: perturbed))
                }
            }
            
            return results
        }
        
        private func generatePerturbations(hash: UInt64, flips: Int) -> [UInt64] {
            var perturbations: [UInt64] = []
            
            // Generate all combinations of flipping 'flips' bits
            let positions = Array(0..<64)
            let combinations = generateCombinations(positions, choose: flips)
            
            for combo in combinations.prefix(100) { // Limit for performance
                var perturbed = hash
                for position in combo {
                    perturbed ^= (1 << position)
                }
                perturbations.append(perturbed)
            }
            
            return perturbations
        }
        
        private func generateCombinations<T>(_ array: [T], choose k: Int) -> [[T]] {
            guard k > 0 else { return [[]] }
            guard k <= array.count else { return [] }
            
            if k == 1 {
                return array.map { [$0] }
            }
            
            var result: [[T]] = []
            for (index, element) in array.enumerated() {
                let remaining = Array(array[(index + 1)...])
                let subcombinations = generateCombinations(remaining, choose: k - 1)
                for sub in subcombinations {
                    result.append([element] + sub)
                }
            }
            
            return result
        }
    }
    
    // MARK: - Initialization
    
    public init(configuration: MultiIndexHashConfiguration) async throws {
        try configuration.validate()
        self.configuration = configuration
        self.vectorStore = [:]
        self.hashTables = Array(repeating: HashTable(), count: configuration.numHashTables)
        
        // Initialize hash functions
        self.hashFunctions = try await Self.createHashFunctions(
            count: configuration.numHashTables,
            dimensions: configuration.dimensions,
            hashLength: configuration.hashLength,
            projectionType: configuration.projectionType
        )
    }
    
    // MARK: - Core Operations
    
    /// Insert a vector into the multi-index hash
    public func insert(id: String, vector: [Float]) async throws {
        guard vector.count == configuration.dimensions else {
            throw HashingError.dimensionMismatch
        }
        
        // Compute all hashes
        let hashes = computeHashes(vector)
        
        // Store vector
        let stored = StoredVector(
            id: id,
            vector: vector,
            norm: computeNorm(vector),
            hashes: hashes
        )
        vectorStore[id] = stored
        
        // Insert into hash tables
        for (tableIdx, hash) in hashes.enumerated() {
            hashTables[tableIdx].insert(hash: hash, id: id)
        }
    }
    
    /// Search for nearest neighbors
    public func search(
        query: [Float],
        k: Int,
        multiProbeDistance: Int? = nil
    ) async throws -> [(id: String, distance: Float)] {
        guard query.count == configuration.dimensions else {
            throw HashingError.dimensionMismatch
        }
        
        // Compute query hashes
        let queryHashes = computeHashes(query)
        let queryNorm = computeNorm(query)
        
        // Collect candidates from all hash tables
        var candidateIds = Set<String>()
        let probeDistance = multiProbeDistance ?? (configuration.hashLength / 4)
        
        for (tableIdx, hash) in queryHashes.enumerated() {
            let candidates = hashTables[tableIdx].multiProbe(
                hash: hash,
                distance: probeDistance
            )
            candidateIds.formUnion(candidates)
        }
        
        // Score candidates
        var scores: [(String, Float)] = []
        
        for candidateId in candidateIds {
            guard let stored = vectorStore[candidateId] else { continue }
            
            if configuration.rerank {
                // Exact distance computation
                let distance = computeNormalizedDistance(
                    query: query,
                    queryNorm: queryNorm,
                    candidate: stored.vector,
                    candidateNorm: stored.norm
                )
                scores.append((candidateId, distance))
            } else {
                // Approximate score based on hash collisions
                let collisions = zip(queryHashes, stored.hashes)
                    .filter { $0 == $1 }
                    .count
                let score = Float(configuration.numHashTables - collisions) / Float(configuration.numHashTables)
                scores.append((candidateId, score))
            }
        }
        
        // Sort and return top-k
        scores.sort { $0.1 < $1.1 }
        return Array(scores.prefix(k))
    }
    
    /// Batch insert multiple vectors
    public func batchInsert(_ entries: [(id: String, vector: [Float])]) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            for (id, vector) in entries {
                group.addTask {
                    try await self.insert(id: id, vector: vector)
                }
            }
            
            try await group.waitForAll()
        }
    }
    
    /// Delete a vector
    public func delete(id: String) async -> Bool {
        guard let stored = vectorStore.removeValue(forKey: id) else {
            return false
        }
        
        // Remove from hash tables
        for (tableIdx, hash) in stored.hashes.enumerated() {
            hashTables[tableIdx].buckets[hash]?.remove(id)
        }
        
        return true
    }
    
    /// Get statistics about the hash tables
    public func statistics() async -> HashStatistics {
        var totalBuckets = 0
        var nonEmptyBuckets = 0
        var maxBucketSize = 0
        var totalCollisions = 0
        
        for table in hashTables {
            totalBuckets += table.buckets.count
            nonEmptyBuckets += table.buckets.filter { !$0.value.isEmpty }.count
            
            for bucket in table.buckets.values {
                maxBucketSize = max(maxBucketSize, bucket.count)
                if bucket.count > 1 {
                    totalCollisions += bucket.count - 1
                }
            }
        }
        
        let avgBucketSize = vectorStore.count > 0
            ? Float(vectorStore.count) / Float(nonEmptyBuckets)
            : 0
        
        return HashStatistics(
            vectorCount: vectorStore.count,
            tableCount: configuration.numHashTables,
            totalBuckets: totalBuckets,
            nonEmptyBuckets: nonEmptyBuckets,
            avgBucketSize: avgBucketSize,
            maxBucketSize: maxBucketSize,
            totalCollisions: totalCollisions,
            loadFactor: Float(vectorStore.count) / Float(totalBuckets)
        )
    }
    
    // MARK: - Hash Functions
    
    private static func createHashFunctions(
        count: Int,
        dimensions: Int,
        hashLength: Int,
        projectionType: MultiIndexHashConfiguration.ProjectionType
    ) async throws -> [HashFunction] {
        var functions: [HashFunction] = []
        
        for i in 0..<count {
            let function = try await createHashFunction(
                index: i,
                dimensions: dimensions,
                hashLength: hashLength,
                projectionType: projectionType
            )
            functions.append(function)
        }
        
        return functions
    }
    
    private static func createHashFunction(
        index: Int,
        dimensions: Int,
        hashLength: Int,
        projectionType: MultiIndexHashConfiguration.ProjectionType
    ) async throws -> HashFunction {
        switch projectionType {
        case .random:
            return RandomProjectionHash(
                dimensions: dimensions,
                hashLength: hashLength,
                seed: index
            )
            
        case .learned(let modelPath):
            return try await LearnedHash(
                modelPath: modelPath,
                dimensions: dimensions,
                hashLength: hashLength,
                tableIndex: index
            )
            
        case .spherical:
            return SphericalHash(
                dimensions: dimensions,
                hashLength: hashLength,
                seed: index
            )
            
        case .crossPolytope:
            return CrossPolytopeHash(
                dimensions: dimensions,
                hashLength: hashLength,
                seed: index
            )
            
        case .pStable(let p):
            return PStableHash(
                dimensions: dimensions,
                hashLength: hashLength,
                p: p,
                seed: index
            )
        }
    }
    
    private func computeHashes(_ vector: [Float]) -> [UInt64] {
        hashFunctions.map { $0.hash(vector) }
    }
    
    private func computeNorm(_ vector: [Float]) -> Float {
        sqrt(vector.map { $0 * $0 }.reduce(0, +))
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    private func computeNormalizedDistance(
        query: [Float],
        queryNorm: Float,
        candidate: [Float],
        candidateNorm: Float
    ) -> Float {
        // Compute cosine distance: 1 - cosine_similarity
        // cosine_similarity = dot_product / (norm_a * norm_b)
        
        guard queryNorm > 0 && candidateNorm > 0 else {
            // Handle zero vectors
            return (queryNorm == 0 && candidateNorm == 0) ? 0 : 1
        }
        
        let dotProduct = zip(query, candidate)
            .map { $0 * $1 }
            .reduce(0, +)
        
        let cosineSimilarity = dotProduct / (queryNorm * candidateNorm)
        
        // Convert to distance (0 = identical, 2 = opposite)
        return 1.0 - cosineSimilarity
    }
}

// MARK: - Hash Function Protocol

private protocol HashFunction: Sendable {
    func hash(_ vector: [Float]) -> UInt64
}

// MARK: - Random Projection Hash

private struct RandomProjectionHash: HashFunction, Sendable {
    let projections: [[Float]]
    let thresholds: [Float]
    
    init(dimensions: Int, hashLength: Int, seed: Int) {
        var generator = SeededRandom(seed: seed)
        
        // Generate random projections
        self.projections = (0..<hashLength).map { _ in
            (0..<dimensions).map { _ in generator.nextGaussian() }
        }
        
        // Generate random thresholds
        self.thresholds = (0..<hashLength).map { _ in
            generator.nextUniform() * 2 - 1
        }
    }
    
    func hash(_ vector: [Float]) -> UInt64 {
        var hash: UInt64 = 0
        
        for (i, projection) in projections.enumerated() {
            let dotProduct = zip(vector, projection)
                .map { $0 * $1 }
                .reduce(0, +)
            
            if dotProduct >= thresholds[i] {
                hash |= (1 << i)
            }
        }
        
        return hash
    }
}

// MARK: - Spherical LSH

private struct SphericalHash: HashFunction, Sendable {
    let projections: [[Float]]
    
    init(dimensions: Int, hashLength: Int, seed: Int) {
        var generator = SeededRandom(seed: seed)
        
        // Generate random unit vectors
        self.projections = (0..<hashLength).map { _ in
            let raw = (0..<dimensions).map { _ in generator.nextGaussian() }
            let norm = sqrt(raw.map { $0 * $0 }.reduce(0, +))
            return raw.map { $0 / norm }
        }
    }
    
    func hash(_ vector: [Float]) -> UInt64 {
        var hash: UInt64 = 0
        
        // Normalize input vector
        let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
        guard norm > 0 else { return 0 }
        
        let normalized = vector.map { $0 / norm }
        
        for (i, projection) in projections.enumerated() {
            let dotProduct = zip(normalized, projection)
                .map { $0 * $1 }
                .reduce(0, +)
            
            if dotProduct >= 0 {
                hash |= (1 << i)
            }
        }
        
        return hash
    }
}

// MARK: - Cross-Polytope LSH

private struct CrossPolytopeHash: HashFunction, Sendable {
    let rotationMatrix: [[Float]]
    let hashLength: Int
    
    init(dimensions: Int, hashLength: Int, seed: Int) {
        var generator = SeededRandom(seed: seed)
        self.hashLength = hashLength
        
        // Generate random rotation matrix
        self.rotationMatrix = Self.generateRandomRotation(
            dimensions: dimensions,
            generator: &generator
        )
    }
    
    func hash(_ vector: [Float]) -> UInt64 {
        // Apply rotation
        let rotated = applyRotation(vector)
        
        // Find maximum coordinates
        var maxIndices: [(index: Int, value: Float)] = []
        
        for (i, value) in rotated.enumerated() {
            maxIndices.append((i, abs(value)))
        }
        
        // Sort by absolute value
        maxIndices.sort { $0.value > $1.value }
        
        // Create hash from top coordinates
        var hash: UInt64 = 0
        
        for i in 0..<min(hashLength, maxIndices.count) {
            let idx = maxIndices[i].index
            let bit = UInt64(idx % 64)
            hash |= (1 << bit)
            
            // Include sign
            if rotated[idx] < 0 {
                hash |= (1 << ((bit + 32) % 64))
            }
        }
        
        return hash
    }
    
    private func applyRotation(_ vector: [Float]) -> [Float] {
        var result = Array(repeating: Float(0), count: vector.count)
        
        for i in 0..<rotationMatrix.count {
            for j in 0..<vector.count {
                result[i] += rotationMatrix[i][j] * vector[j]
            }
        }
        
        return result
    }
    
    private static func generateRandomRotation(
        dimensions: Int,
        generator: inout SeededRandom
    ) -> [[Float]] {
        // Generate random orthogonal matrix using QR decomposition
        var matrix = (0..<dimensions).map { _ in
            (0..<dimensions).map { _ in generator.nextGaussian() }
        }
        
        // Simplified orthogonalization
        for i in 0..<dimensions {
            // Normalize row i
            let norm = sqrt(matrix[i].map { $0 * $0 }.reduce(0, +))
            if norm > 0 {
                matrix[i] = matrix[i].map { $0 / norm }
            }
            
            // Orthogonalize subsequent rows
            for j in (i+1)..<dimensions {
                let dot = zip(matrix[i], matrix[j])
                    .map { $0 * $1 }
                    .reduce(0, +)
                
                for k in 0..<dimensions {
                    matrix[j][k] -= dot * matrix[i][k]
                }
            }
        }
        
        return matrix
    }
}

// MARK: - p-Stable LSH

private struct PStableHash: HashFunction, Sendable {
    let projections: [[Float]]
    let biases: [Float]
    let width: Float
    
    init(dimensions: Int, hashLength: Int, p: Float, seed: Int) {
        var generator = SeededRandom(seed: seed)
        
        // Generate p-stable distributions
        if p == 2 {
            // Gaussian for L2
            self.projections = (0..<hashLength).map { _ in
                (0..<dimensions).map { _ in generator.nextGaussian() }
            }
        } else if p == 1 {
            // Cauchy for L1
            self.projections = (0..<hashLength).map { _ in
                (0..<dimensions).map { _ in generator.nextCauchy() }
            }
        } else {
            // Approximate with Gaussian
            self.projections = (0..<hashLength).map { _ in
                (0..<dimensions).map { _ in generator.nextGaussian() }
            }
        }
        
        // Random biases
        self.biases = (0..<hashLength).map { _ in
            generator.nextUniform() * 10
        }
        
        // Hash bucket width
        self.width = 4.0
    }
    
    func hash(_ vector: [Float]) -> UInt64 {
        var hash: UInt64 = 0
        
        for (i, projection) in projections.enumerated() {
            let dotProduct = zip(vector, projection)
                .map { $0 * $1 }
                .reduce(0, +)
            
            let bucketIndex = Int((dotProduct + biases[i]) / width)
            
            // Map bucket index to bit
            if bucketIndex > 0 {
                hash |= (1 << (i % 64))
            }
        }
        
        return hash
    }
}

// MARK: - Learned Hash

private actor LearnedHash: HashFunction {
    private let model: HashModel
    private let dimensions: Int
    private let hashLength: Int
    
    init(modelPath: String, dimensions: Int, hashLength: Int, tableIndex: Int) async throws {
        self.dimensions = dimensions
        self.hashLength = hashLength
        
        // Load or create model
        self.model = try await HashModel.load(
            from: modelPath,
            dimensions: dimensions,
            hashLength: hashLength,
            tableIndex: tableIndex
        )
    }
    
    nonisolated func hash(_ vector: [Float]) -> UInt64 {
        // Use model to compute hash
        // Simplified implementation
        var hash: UInt64 = 0
        
        for i in 0..<min(hashLength, 64) {
            let sum = vector.enumerated()
                .map { Float($0.offset + i) * $0.element }
                .reduce(0, +)
            
            if sum > 0 {
                hash |= (1 << i)
            }
        }
        
        return hash
    }
}

// MARK: - Supporting Types

/// Hash model for learned hashing
private struct HashModel: Codable {
    let dimensions: Int
    let hashLength: Int
    let projectionMatrices: [[[Float]]]  // One projection matrix per hash bit
    let biases: [[Float]]                 // Biases for each hash bit
    let thresholds: [Float]               // Thresholds for binarization
    let modelVersion: String
    let trainingMetadata: TrainingMetadata?
    
    struct TrainingMetadata: Codable {
        let trainingVectorCount: Int
        let trainingEpochs: Int
        let lossValue: Float
        let timestamp: Date
    }
    
    static func load(
        from path: String,
        dimensions: Int,
        hashLength: Int,
        tableIndex: Int
    ) async throws -> HashModel {
        // Construct the full path for this specific table's model
        let modelFileName = "hash_model_table_\(tableIndex).json"
        let modelURL: URL
        
        if path.hasPrefix("/") {
            // Absolute path
            modelURL = URL(fileURLWithPath: path).appendingPathComponent(modelFileName)
        } else {
            // Relative path or just directory name
            let baseURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            modelURL = baseURL.appendingPathComponent(path).appendingPathComponent(modelFileName)
        }
        
        // Check if model file exists
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            // If no trained model exists, create a default one
            return try createDefaultModel(
                dimensions: dimensions,
                hashLength: hashLength,
                tableIndex: tableIndex
            )
        }
        
        // Load the model data
        let modelData = try Data(contentsOf: modelURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        let loadedModel = try decoder.decode(HashModel.self, from: modelData)
        
        // Validate loaded model matches expected dimensions
        guard loadedModel.dimensions == dimensions else {
            throw HashingError.modelDimensionMismatch(
                expected: dimensions,
                actual: loadedModel.dimensions
            )
        }
        
        guard loadedModel.hashLength == hashLength else {
            throw HashingError.modelHashLengthMismatch(
                expected: hashLength,
                actual: loadedModel.hashLength
            )
        }
        
        return loadedModel
    }
    
    private static func createDefaultModel(
        dimensions: Int,
        hashLength: Int,
        tableIndex: Int
    ) throws -> HashModel {
        // Create a default model with random projections
        var generator = SeededRandom(seed: tableIndex * 1000)
        
        // Generate projection matrices for each hash bit
        let projectionMatrices: [[[Float]]] = (0..<hashLength).map { _ in
            // For learned hashing, we typically use a small neural network
            // Here we simulate with random linear projections
            
            // Create a single projection matrix for this hash bit
            // dimensions -> 1 (direct projection for simplicity)
            let projection = (0..<dimensions).map { _ in
                generator.nextGaussian() * sqrt(2.0 / Float(dimensions))
            }
            
            return [projection]
        }
        
        // Generate biases for each hash bit
        let biases = (0..<hashLength).map { _ in
            [generator.nextGaussian() * 0.1]
        }
        
        // Generate thresholds for binarization
        let thresholds = (0..<hashLength).map { _ in
            generator.nextUniform() * 0.2 - 0.1  // Small values around 0
        }
        
        return HashModel(
            dimensions: dimensions,
            hashLength: hashLength,
            projectionMatrices: projectionMatrices,
            biases: biases,
            thresholds: thresholds,
            modelVersion: "1.0",
            trainingMetadata: nil
        )
    }
    
    /// Compute hash for a vector using the learned model
    func computeHash(_ vector: [Float]) -> UInt64 {
        var hash: UInt64 = 0
        
        for (bitIndex, projectionLayer) in projectionMatrices.prefix(64).enumerated() {
            // Simple linear projection for each hash bit
            guard let projection = projectionLayer.first else { continue }
            
            // Compute dot product
            let sum = zip(vector, projection)
                .map { $0 * $1 }
                .reduce(0, +)
            
            // Add bias if available
            let biasValue = bitIndex < biases.count && !biases[bitIndex].isEmpty ?
                biases[bitIndex][0] : 0
            
            let output = sum + biasValue
            
            // Apply threshold to get binary hash bit
            if output > thresholds[bitIndex] {
                hash |= (1 << UInt64(bitIndex))
            }
        }
        
        return hash
    }
}

/// Statistics for hash tables
public struct HashStatistics {
    public let vectorCount: Int
    public let tableCount: Int
    public let totalBuckets: Int
    public let nonEmptyBuckets: Int
    public let avgBucketSize: Float
    public let maxBucketSize: Int
    public let totalCollisions: Int
    public let loadFactor: Float
}

/// Seeded random number generator
private struct SeededRandom {
    private var state: UInt64
    
    init(seed: Int) {
        self.state = UInt64(seed)
    }
    
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
    
    mutating func nextUniform() -> Float {
        Float(next() & 0xFFFFFF) / Float(0xFFFFFF)
    }
    
    mutating func nextGaussian() -> Float {
        // Box-Muller transform
        let u1 = nextUniform()
        let u2 = nextUniform()
        
        return sqrt(-2 * log(max(u1, 1e-7))) * cos(2 * .pi * u2)
    }
    
    mutating func nextCauchy() -> Float {
        let u = nextUniform()
        return tan(.pi * (u - 0.5))
    }
}

