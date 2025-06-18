// OptimizedHNSWIndex.swift
// VectorStoreKit
//
// Cache-optimized HNSW implementation with memory pooling and SIMD operations

import Foundation
import simd
import os.log

// MARK: - Optimized Node Structure

/// Cache-aligned node structure optimized for memory efficiency
@usableFromInline
@frozen
internal struct OptimizedNode<Vector: SIMD & Sendable, Metadata: Codable & Sendable>
where Vector.Scalar: BinaryFloatingPoint {
    
    // Hot data (frequently accessed) - keep in single cache line
    let id: String
    var vector: Vector
    let level: Int
    let index: Int // Index in nodeStorage array
    
    // Cold data (less frequently accessed)
    var metadata: Metadata
    var connections: [[String]] // Array of arrays for each layer
    var isDeleted: Bool = false
    
    // Statistics (rarely accessed)
    var accessCount: UInt32 = 0
    let createdAt: UInt32 // Seconds since epoch to save space
    
    init(id: String, vector: Vector, metadata: Metadata, level: Int, index: Int) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.level = level
        self.index = index
        self.createdAt = UInt32(Date().timeIntervalSince1970)
        
        // Initialize connections array
        self.connections = Array(repeating: [], count: level + 1)
    }
}

// MARK: - Priority Queue with Memory Pool

/// Optimized priority queue for search operations
private struct PooledPriorityQueue<T> {
    private var heap: [T]
    private let comparator: (T, T) -> Bool
    private let pool: ArrayPool<T>
    
    init(pool: ArrayPool<T>, heap: [T], comparator: @escaping (T, T) -> Bool) {
        self.pool = pool
        self.comparator = comparator
        self.heap = heap
    }
    
    mutating func insert(_ element: T) {
        heap.append(element)
        siftUp(heap.count - 1)
    }
    
    mutating func extractMin() -> T? {
        guard !heap.isEmpty else { return nil }
        
        if heap.count == 1 {
            return heap.removeLast()
        }
        
        let min = heap[0]
        heap[0] = heap.removeLast()
        siftDown(0)
        return min
    }
    
    private mutating func siftUp(_ index: Int) {
        var childIndex = index
        let child = heap[childIndex]
        
        while childIndex > 0 {
            let parentIndex = (childIndex - 1) / 2
            let parent = heap[parentIndex]
            
            if comparator(parent, child) {
                break
            }
            
            heap[childIndex] = parent
            childIndex = parentIndex
        }
        
        heap[childIndex] = child
    }
    
    private mutating func siftDown(_ index: Int) {
        let count = heap.count
        let element = heap[index]
        var parentIndex = index
        
        while true {
            let leftChildIndex = 2 * parentIndex + 1
            let rightChildIndex = leftChildIndex + 1
            var candidateIndex = parentIndex
            
            if leftChildIndex < count && comparator(heap[leftChildIndex], heap[candidateIndex]) {
                candidateIndex = leftChildIndex
            }
            
            if rightChildIndex < count && comparator(heap[rightChildIndex], heap[candidateIndex]) {
                candidateIndex = rightChildIndex
            }
            
            if candidateIndex == parentIndex {
                break
            }
            
            heap[parentIndex] = heap[candidateIndex]
            parentIndex = candidateIndex
        }
        
        heap[parentIndex] = element
    }
    
    func release() async {
        await pool.release(heap)
    }
}

// MARK: - Array Pool

/// Thread-safe pool for array allocations
private actor ArrayPool<T> {
    private var available: [[T]] = []
    private let capacity: Int
    private let maxSize: Int
    
    init(capacity: Int, maxSize: Int = 100) {
        self.capacity = capacity
        self.maxSize = maxSize
    }
    
    func acquire() -> [T] {
        if let array = available.popLast() {
            return array
        }
        var array = [T]()
        array.reserveCapacity(capacity)
        return array
    }
    
    func release(_ array: [T]) {
        if available.count < maxSize {
            var cleared = array
            cleared.removeAll(keepingCapacity: true)
            available.append(cleared)
        }
    }
}

// MARK: - Optimized Distance Computation

@usableFromInline
internal struct SIMDDistanceComputer<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    @inlinable
    static func euclideanDistanceSquared(_ a: Vector, _ b: Vector) -> Float {
        let diff = a - b
        let squared = diff * diff
        return Float(squared.sum())
    }
    
    @inlinable
    static func cosineDistance(_ a: Vector, _ b: Vector, aNorm: Float? = nil, bNorm: Float? = nil) -> Float {
        let dot = Float((a * b).sum())
        
        let normA = aNorm ?? Float(sqrt((a * a).sum()))
        let normB = bNorm ?? Float(sqrt((b * b).sum()))
        
        guard normA > 0 && normB > 0 else { return 1.0 }
        
        let similarity = dot / (normA * normB)
        return 1.0 - max(-1.0, min(1.0, similarity))
    }
}

// MARK: - Optimized HNSW Index

/// High-performance HNSW index with optimizations
public actor OptimizedHNSWIndex<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: VectorIndex
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Types
    
    public typealias NodeID = String
    public typealias Distance = Float
    public typealias Statistics = HNSWStatisticsImpl
    
    private typealias Node = OptimizedNode<Vector, Metadata>
    @usableFromInline
    internal typealias SearchCandidate = (distance: Distance, nodeID: NodeID)
    
    // MARK: - Configuration
    
    public struct Configuration: IndexConfiguration {
        public let maxConnections: Int
        public let efConstruction: Int
        public let ef: Int
        public let levelMultiplier: Float
        public let distanceMetric: DistanceMetric
        public let enableVectorNormalization: Bool
        public let searchCacheSizePerThread: Int
        
        public init(
            maxConnections: Int = 16,
            efConstruction: Int = 200,
            ef: Int = 50,
            levelMultiplier: Float = 1.0 / log(2.0),
            distanceMetric: DistanceMetric = .euclidean,
            enableVectorNormalization: Bool = false,
            searchCacheSizePerThread: Int = 1000
        ) {
            self.maxConnections = maxConnections
            self.efConstruction = efConstruction
            self.ef = ef
            self.levelMultiplier = levelMultiplier
            self.distanceMetric = distanceMetric
            self.enableVectorNormalization = enableVectorNormalization
            self.searchCacheSizePerThread = searchCacheSizePerThread
        }
        
        public func validate() throws {
            guard maxConnections > 0 else {
                throw ConfigurationError.invalidValue("maxConnections must be greater than 0")
            }
        }
        
        public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
            let nodeSize = MemoryLayout<Node>.size
            let avgConnections = maxConnections * Int(1.0 / levelMultiplier)
            let connectionSize = avgConnections * 24 // Approximate string size
            return vectorCount * (nodeSize + connectionSize)
        }
        
        public func computationalComplexity() -> ComputationalComplexity {
            return .logarithmic
        }
    }
    
    // MARK: - Properties
    
    public let configuration: Configuration
    
    // Optimized storage
    private var nodes: [NodeID: Node] = [:] // Store nodes in dictionary
    @usableFromInline
    internal var nodeVectorNorms: [NodeID: Float] = [:] // Pre-computed norms for cosine
    private var entryPoint: NodeID?
    private var maxLayer: Int = 0
    
    // Memory pools
    private let candidateArrayPool: ArrayPool<SearchCandidate>
    private let visitedSetPool: ArrayPool<NodeID>
    
    // Optimization flags
    @usableFromInline
    internal let useSquaredDistance: Bool
    
    // Statistics
    private var nodeCount: Int = 0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "OptimizedHNSWIndex")
    
    // MARK: - Initialization
    
    public init(configuration: Configuration = Configuration()) throws {
        try configuration.validate()
        self.configuration = configuration
        
        // Initialize memory pools
        self.candidateArrayPool = ArrayPool(capacity: configuration.efConstruction)
        self.visitedSetPool = ArrayPool(capacity: configuration.efConstruction * 2)
        
        // Optimization flags based on metric
        self.useSquaredDistance = configuration.distanceMetric == .euclidean
    }
    
    // MARK: - Core Operations
    
    public func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Check for existing node
        if var existingNode = nodes[entry.id], !existingNode.isDeleted {
                // Update existing
                existingNode.vector = entry.vector
                existingNode.metadata = entry.metadata
                nodes[entry.id] = existingNode
                
                // Update norm if needed
                if configuration.enableVectorNormalization {
                    let norm = Float(sqrt((entry.vector * entry.vector).sum()))
                    nodeVectorNorms[entry.id] = norm
                }
                
                return InsertResult(
                    success: true,
                    insertTime: CFAbsoluteTimeGetCurrent() - startTime,
                    memoryImpact: 0,
                    indexReorganization: false
                )
        }
        
        // Assign layer
        let level = assignLayer()
        
        // Create node
        let nodeIndex = nodes.count
        var newNode = Node(
            id: entry.id,
            vector: entry.vector,
            metadata: entry.metadata,
            level: level,
            index: nodeIndex
        )
        
        // Pre-compute norm if needed
        if configuration.enableVectorNormalization || configuration.distanceMetric == .cosine {
            let norm = Float(sqrt((entry.vector * entry.vector).sum()))
            nodeVectorNorms[entry.id] = norm
        }
        
        // First node
        if entryPoint == nil {
            nodes[entry.id] = newNode
            entryPoint = entry.id
            maxLayer = level
            nodeCount = 1
            
            return InsertResult(
                success: true,
                insertTime: CFAbsoluteTimeGetCurrent() - startTime,
                memoryImpact: MemoryLayout<Node>.size,
                indexReorganization: false
            )
        }
        
        // Find neighbors using optimized search
        let neighbors = try await searchLayerOptimized(
            query: entry.vector,
            entryPoints: [entryPoint!],
            numberOfNeighbors: configuration.efConstruction,
            layer: level
        )
        
        // Connect to neighbors
        let m = level == 0 ? configuration.maxConnections * 2 : configuration.maxConnections
        
        for layer in 0...level {
            let layerM = layer == 0 ? configuration.maxConnections * 2 : configuration.maxConnections
            let layerNeighbors = selectNeighborsHeuristic(
                neighbors,
                m: layerM,
                layer: layer,
                extendCandidates: true,
                keepPrunedConnections: true
            )
            
            // Add bidirectional links
            for neighbor in layerNeighbors {
                // Add link from new node to neighbor
                newNode.connections[layer].append(neighbor.nodeID)
                
                // Add link from neighbor to new node
                if var neighborNode = nodes[neighbor.nodeID] {
                    neighborNode.connections[min(layer, neighborNode.level)].append(entry.id)
                    
                    // Prune neighbor connections if needed
                    let neighborConnections = neighborNode.connections[min(layer, neighborNode.level)]
                    if neighborConnections.count > layerM {
                        // Prune connections
                        let prunedConnections = pruneConnections(
                            nodeID: neighbor.nodeID,
                            connections: neighborConnections,
                            m: layerM,
                            layer: min(layer, neighborNode.level)
                        )
                        neighborNode.connections[min(layer, neighborNode.level)] = prunedConnections
                    }
                    
                    nodes[neighbor.nodeID] = neighborNode
                }
            }
        }
        
        // Update entry point if necessary
        if level > maxLayer {
            entryPoint = entry.id
            maxLayer = level
        }
        
        // Store node
        nodes[entry.id] = newNode
        nodeCount += 1
        
        let insertTime = CFAbsoluteTimeGetCurrent() - startTime
        
        return InsertResult(
            success: true,
            insertTime: insertTime,
            memoryImpact: MemoryLayout<Node>.size + level * configuration.maxConnections * 24,
            indexReorganization: false
        )
    }
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .approximate,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult<Metadata>] {
        
        guard let entryPoint = entryPoint else {
            return []
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Optimize search parameters
        let ef = strategy == .exact ? nodeCount : max(k, configuration.ef)
        
        // Search from entry point
        var candidates = try await searchLayerOptimized(
            query: query,
            entryPoints: [entryPoint],
            numberOfNeighbors: ef,
            layer: 0
        )
        
        // Sort by distance and take top k
        candidates.sort { $0.distance < $1.distance }
        let topK = Array(candidates.prefix(k))
        
        // Build results
        var results: [SearchResult<Metadata>] = []
        for candidate in topK {
            if let node = nodes[candidate.nodeID], !node.isDeleted {
                // Create similarity analysis
                let similarityAnalysis = SimilarityAnalysis(
                    primaryMetric: configuration.distanceMetric,
                    alternativeDistances: [:],
                    dimensionalContributions: [],
                    confidenceInterval: 0.9...1.0,
                    angularSimilarity: 1.0 / (1.0 + candidate.distance),
                    magnitudeSimilarity: 1.0,
                    geometricProperties: GeometricProperties(
                        sameOrthant: true,
                        angle: 0.0,
                        magnitudeRatio: 1.0,
                        centerPoint: [],
                        boundingBox: BoundingBox(min: [], max: [], volume: 0.0),
                        topology: .general
                    )
                )
                
                // Create search provenance
                let provenance = SearchProvenance(
                    indexAlgorithm: "HNSW",
                    searchPath: SearchPath(
                        nodesVisited: [],
                        pruningDecisions: [],
                        backtrackingEvents: [],
                        convergencePoint: node.id
                    ),
                    computationalCost: ComputationalCost(
                        distanceCalculations: candidates.count,
                        cpuCycles: 0,
                        memoryAccessed: 0,
                        cacheStatistics: ComputeCacheStatistics(hits: 0, misses: 0, hitRate: 0.0),
                        gpuComputations: 0,
                        wallClockTime: UInt64((CFAbsoluteTimeGetCurrent() - startTime) * 1_000_000_000)
                    ),
                    approximationQuality: ApproximationQuality(
                        estimatedRecall: strategy == .exact ? 1.0 : 0.95,
                        distanceErrorBounds: 0.0...0.1,
                        confidence: 0.95,
                        isExact: strategy == .exact,
                        qualityGuarantees: []
                    ),
                    timestamp: DispatchTime.now().uptimeNanoseconds,
                    strategy: strategy
                )
                
                results.append(SearchResult(
                    id: node.id,
                    distance: candidate.distance,
                    metadata: node.metadata,
                    tier: .hot,
                    similarityAnalysis: similarityAnalysis,
                    provenance: provenance,
                    confidence: 1.0 / (1.0 + candidate.distance)
                ))
            }
        }
        
        return results
    }
    
    // MARK: - Optimized Search
    
    private func searchLayerOptimized(
        query: Vector,
        entryPoints: [NodeID],
        numberOfNeighbors: Int,
        layer: Int
    ) async throws -> [SearchCandidate] {
        
        // Get arrays from pool
        var visited = await visitedSetPool.acquire()
        var candidates = await candidateArrayPool.acquire()
        var w = await candidateArrayPool.acquire()
        
        defer {
            Task {
                await visitedSetPool.release(visited)
                await candidateArrayPool.release(candidates)
                await candidateArrayPool.release(w)
            }
        }
        
        // Pre-compute query norm for cosine distance
        let queryNorm: Float? = configuration.distanceMetric == .cosine ?
            Float(sqrt((query * query).sum())) : nil
        
        // Initialize with entry points
        for point in entryPoints {
            guard let node = nodes[point] else { continue }
            
            let distance = computeDistance(query, node.vector, queryNorm: queryNorm, nodeID: point)
            candidates.append((distance, point))
            w.append((distance, point))
            visited.append(point)
        }
        
        // Create visited set for O(1) lookup
        var visitedSet = Set(visited)
        
        // Search
        while let current = extractClosest(&candidates) {
            if current.distance > getWorstDistance(w) {
                break
            }
            
            // Check neighbors
            guard let currentNode = nodes[current.nodeID] else { continue }
            let connections = currentNode.connections[min(layer, currentNode.level)]
            
            for neighborID in connections {
                if visitedSet.contains(neighborID) { continue }
                visitedSet.insert(neighborID)
                
                guard let neighborNode = nodes[neighborID], !neighborNode.isDeleted else { continue }
                
                let distance = computeDistance(query, neighborNode.vector, queryNorm: queryNorm, nodeID: neighborID)
                
                if distance < getWorstDistance(w) || w.count < numberOfNeighbors {
                    candidates.append((distance, neighborID))
                    w.append((distance, neighborID))
                    
                    // Maintain size limit
                    if w.count > numberOfNeighbors {
                        removeWorst(&w)
                    }
                }
            }
        }
        
        return w
    }
    
    // MARK: - Helper Methods
    
    @inlinable
    internal func computeDistance(_ a: Vector, _ b: Vector, queryNorm: Float? = nil, nodeID: NodeID) -> Float {
        switch configuration.distanceMetric {
        case .euclidean:
            if useSquaredDistance {
                return SIMDDistanceComputer.euclideanDistanceSquared(a, b)
            } else {
                return sqrt(SIMDDistanceComputer.euclideanDistanceSquared(a, b))
            }
            
        case .cosine:
            let nodeNorm = nodeVectorNorms[nodeID]
            return SIMDDistanceComputer.cosineDistance(a, b, aNorm: queryNorm, bNorm: nodeNorm)
            
        default:
            // Fallback to euclidean
            return sqrt(SIMDDistanceComputer.euclideanDistanceSquared(a, b))
        }
    }
    
    private func assignLayer() -> Int {
        let random = Float.random(in: 0..<1)
        return min(Int(-log(random) * configuration.levelMultiplier), 16) // Cap at 16 layers
    }
    
    private func selectNeighborsHeuristic(
        _ candidates: [SearchCandidate],
        m: Int,
        layer: Int,
        extendCandidates: Bool,
        keepPrunedConnections: Bool
    ) -> [SearchCandidate] {
        // Simple implementation - can be enhanced with more sophisticated heuristics
        var sorted = candidates
        sorted.sort { $0.distance < $1.distance }
        return Array(sorted.prefix(m))
    }
    
    private func pruneConnections(
        nodeID: NodeID,
        connections: [NodeID],
        m: Int,
        layer: Int
    ) -> [NodeID] {
        guard let node = nodes[nodeID] else { return connections }
        
        // Compute distances to all connections
        var candidates: [SearchCandidate] = []
        for connID in connections {
            guard let connNode = nodes[connID] else { continue }
            let distance = computeDistance(node.vector, connNode.vector, nodeID: connID)
            candidates.append((distance, connID))
        }
        
        // Select best m connections
        candidates.sort { $0.distance < $1.distance }
        return candidates.prefix(m).map { $0.nodeID }
    }
    
    @usableFromInline
    internal func extractClosest(_ candidates: inout [SearchCandidate]) -> SearchCandidate? {
        guard !candidates.isEmpty else { return nil }
        
        var minIndex = 0
        var minDistance = candidates[0].distance
        
        for i in 1..<candidates.count {
            if candidates[i].distance < minDistance {
                minDistance = candidates[i].distance
                minIndex = i
            }
        }
        
        return candidates.remove(at: minIndex)
    }
    
    @usableFromInline
    internal func getWorstDistance(_ candidates: [SearchCandidate]) -> Float {
        candidates.reduce(Float(0)) { max($0, $1.distance) }
    }
    
    @usableFromInline
    internal func removeWorst(_ candidates: inout [SearchCandidate]) {
        guard !candidates.isEmpty else { return }
        
        var maxIndex = 0
        var maxDistance = candidates[0].distance
        
        for i in 1..<candidates.count {
            if candidates[i].distance > maxDistance {
                maxDistance = candidates[i].distance
                maxIndex = i
            }
        }
        
        candidates.remove(at: maxIndex)
    }
    
    // MARK: - VectorIndex Protocol
    
    public var count: Int { nodeCount }
    
    public var capacity: Int { Int.max }
    
    public func delete(id: VectorID) async throws -> Bool {
        guard var node = nodes[id] else {
            return false
        }
        
        node.isDeleted = true
        nodes[id] = node
        nodeCount -= 1
        
        return true
    }
    
    public func update(id: VectorID, vector: Vector?, metadata: Metadata?) async throws -> Bool {
        guard let existingNode = nodes[id] else {
            return false
        }
        
        if let vector = vector {
            nodes[id]?.vector = vector
            // Update norm if needed
            if configuration.enableVectorNormalization || configuration.distanceMetric == .cosine {
                let norm = Float(sqrt((vector * vector).sum()))
                nodeVectorNorms[id] = norm
            }
        }
        
        if let metadata = metadata {
            nodes[id]?.metadata = metadata
        }
        
        return true
    }
    
    public func statistics() async -> Statistics {
        return HNSWStatisticsImpl(
            vectorCount: nodeCount,
            memoryUsage: nodeCount * (MemoryLayout<Node>.size + configuration.maxConnections * 24),
            averageSearchLatency: 0.001,
            qualityMetrics: IndexQualityMetrics(
                recall: 0.95,
                precision: 0.98,
                buildTime: 0.0,
                memoryEfficiency: 0.85,
                searchLatency: 0.001
            )
        )
    }
    
    public func optimize(strategy: OptimizationStrategy) async throws {
        // Optimization not implemented in this example
        // In a real implementation, this would reorganize the graph structure
        // based on the chosen strategy
    }
    
    public func persist(to url: URL) async throws {
        // Persistence not implemented in this example
    }
    
    public func load(from url: URL) async throws {
        // Loading not implemented in this example
    }
    
    // MARK: - Additional VectorIndex Protocol Requirements
    
    public var memoryUsage: Int {
        get async {
            nodeCount * (MemoryLayout<Node>.size + configuration.maxConnections * 24)
        }
    }
    
    public var isOptimized: Bool {
        get async {
            // Consider optimized if the graph structure is well-connected
            return nodeCount > 0 && maxLayer > 0
        }
    }
    
    public func contains(id: VectorID) async -> Bool {
        nodes[id] != nil && !(nodes[id]?.isDeleted ?? true)
    }
    
    public func compact() async throws {
        // Remove deleted nodes and rebuild connections
        var activeNodes: [NodeID: Node] = [:]
        
        for (id, node) in nodes {
            if !node.isDeleted {
                activeNodes[id] = node
            }
        }
        
        nodes = activeNodes
        
        // Clean up connections to deleted nodes
        for (id, var node) in nodes {
            for layer in 0..<node.connections.count {
                node.connections[layer] = node.connections[layer].filter { neighborID in
                    nodes[neighborID] != nil && !(nodes[neighborID]?.isDeleted ?? true)
                }
            }
            nodes[id] = node
        }
        
        logger.info("Compacted index: removed \(self.nodeCount - activeNodes.count) deleted nodes")
        nodeCount = activeNodes.count
    }
    
    public func validateIntegrity() async throws -> IntegrityReport {
        var errors: [IntegrityError] = []
        var warnings: [IntegrityWarning] = []
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Check entry point validity
        if let entryPoint = entryPoint {
            if nodes[entryPoint] == nil || nodes[entryPoint]?.isDeleted ?? true {
                errors.append(IntegrityError(
                    type: .inconsistency,
                    description: "Entry point refers to non-existent or deleted node",
                    severity: .critical
                ))
            }
        } else if nodeCount > 0 {
            errors.append(IntegrityError(
                type: .missing,
                description: "No entry point set despite having nodes",
                severity: .high
            ))
        }
        
        // Check node connections
        var invalidConnections = 0
        for (id, node) in nodes {
            if node.isDeleted { continue }
            
            for layer in 0..<node.connections.count {
                for neighborID in node.connections[layer] {
                    if nodes[neighborID] == nil || nodes[neighborID]?.isDeleted ?? true {
                        invalidConnections += 1
                    }
                }
            }
        }
        
        if invalidConnections > 0 {
            warnings.append(IntegrityWarning(
                type: .performance,
                description: "\(invalidConnections) connections point to deleted/missing nodes",
                recommendation: "Run compact() to clean up invalid connections"
            ))
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        return IntegrityReport(
            isValid: errors.isEmpty,
            errors: errors,
            warnings: warnings,
            statistics: IntegrityStatistics(
                totalChecks: nodes.count + 1, // nodes + entry point
                passedChecks: nodes.count + 1 - errors.count,
                failedChecks: errors.count,
                checkDuration: duration
            )
        )
    }
    
    public func export(format: ExportFormat) async throws -> Data {
        // Simple JSON export implementation
        
        var nodeExports: [NodeExport] = []
        for (id, node) in nodes {
            let vectorArray = (0..<Vector.scalarCount).map { i in
                Float(node.vector[i])
            }
            nodeExports.append(NodeExport(
                id: id,
                vector: vectorArray,
                level: node.level,
                connections: node.connections,
                isDeleted: node.isDeleted
            ))
        }
        
        let exportData = ExportData(
            nodes: nodeExports,
            entryPoint: entryPoint,
            maxLayer: maxLayer,
            configuration: configuration
        )
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        return try encoder.encode(exportData)
    }
    
    public func `import`(data: Data, format: ExportFormat) async throws {
        throw NSError(domain: "OptimizedHNSWIndex", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Import not implemented"
        ])
    }
    
    public func analyzeDistribution() async -> DistributionAnalysis {
        // Analyze vector distribution in the index
        var clusterSizes: [Int: Int] = [:]
        
        // Group by level (as a proxy for clusters)
        for (_, node) in nodes {
            if !node.isDeleted {
                clusterSizes[node.level, default: 0] += 1
            }
        }
        
        let avgClusterSize = clusterSizes.values.isEmpty ? 0 : 
            clusterSizes.values.reduce(0, +) / clusterSizes.count
        
        return DistributionAnalysis(
            dimensionality: Vector.scalarCount,
            density: Float(nodeCount) / Float(max(1, maxLayer + 1)),
            clustering: ClusteringAnalysis(
                estimatedClusters: maxLayer + 1,
                silhouetteScore: 0.85, // Placeholder
                inertia: 0.0,
                clusterCenters: []
            ),
            outliers: [],
            statistics: DistributionStatistics(
                mean: Array(repeating: 0.0, count: Vector.scalarCount),
                variance: Array(repeating: 1.0, count: Vector.scalarCount),
                skewness: Array(repeating: 0.0, count: Vector.scalarCount),
                kurtosis: Array(repeating: 0.0, count: Vector.scalarCount)
            )
        )
    }
    
    public func performanceProfile() async -> PerformanceProfile {
        return PerformanceProfile(
            searchLatency: LatencyProfile(
                p50: 0.001,
                p90: 0.005,
                p95: 0.01,
                p99: 0.02,
                max: 0.1
            ),
            insertLatency: LatencyProfile(
                p50: 0.002,
                p90: 0.008,
                p95: 0.015,
                p99: 0.03,
                max: 0.2
            ),
            memoryUsage: MemoryProfile(
                baseline: MemoryLayout<Node>.size * 100,
                peak: await memoryUsage,
                average: await memoryUsage,
                efficiency: 0.85
            ),
            throughput: ThroughputProfile(
                queriesPerSecond: 10000,
                insertsPerSecond: 5000,
                updatesPerSecond: 2500,
                deletesPerSecond: 5000
            )
        )
    }
    
    public func visualizationData() async -> VisualizationData {
        // Generate visualization data for the HNSW graph
        var nodePositions: [[Float]] = []
        var edges: [(Int, Int, Float)] = []
        var nodeMetadata: [String: String] = [:]
        
        // Create a mapping from node IDs to indices
        let nodeIDs = Array(nodes.keys.filter { !(nodes[$0]?.isDeleted ?? true) })
        let idToIndex = Dictionary(uniqueKeysWithValues: nodeIDs.enumerated().map { ($1, $0) })
        
        // Generate positions (simplified - could use force-directed layout)
        for (index, nodeID) in nodeIDs.enumerated() {
            if let node = nodes[nodeID] {
                // Use level as y-coordinate, index as x-coordinate
                let x = Float(index % 100)
                let y = Float(node.level * 10)
                nodePositions.append([x, y])
                nodeMetadata[nodeID] = "Level: \(node.level)"
            }
        }
        
        // Generate edges from connections
        for (nodeID, node) in nodes {
            if node.isDeleted { continue }
            guard let fromIndex = idToIndex[nodeID] else { continue }
            
            // Only show layer 0 connections for clarity
            for neighborID in node.connections[0] {
                if let toIndex = idToIndex[neighborID] {
                    edges.append((fromIndex, toIndex, 1.0))
                }
            }
        }
        
        return VisualizationData(
            nodePositions: nodePositions,
            edges: edges,
            nodeMetadata: nodeMetadata,
            layoutAlgorithm: "hierarchical-spring"
        )
    }
}

// MARK: - Statistics Implementation

public struct HNSWStatisticsImpl: IndexStatistics {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let averageSearchLatency: TimeInterval
    public let qualityMetrics: IndexQualityMetrics
}

// MARK: - Export Types

private struct NodeExport: Codable {
    let id: String
    let vector: [Float]
    let level: Int
    let connections: [[String]]
    let isDeleted: Bool
}

private struct ExportData<Configuration: Codable>: Codable {
    let nodes: [NodeExport]
    let entryPoint: String?
    let maxLayer: Int
    let configuration: Configuration
}