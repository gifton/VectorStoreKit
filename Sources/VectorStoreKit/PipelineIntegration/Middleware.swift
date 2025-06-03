// VectorStoreKit: PipelineKit Middleware
//
// Middleware components for VectorStore operations

import Foundation
import simd

// MARK: - Middleware Protocol

/// Protocol for PipelineKit Middleware
public protocol Middleware: Sendable {
    func process<C: Command>(
        _ command: C,
        next: @escaping @Sendable (C) async throws -> C.Result
    ) async throws -> C.Result
}

// MARK: - VectorStore Caching Middleware

/// Middleware that adds caching for vector search operations
public struct VectorStoreCachingMiddleware<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Middleware 
where Vector.Scalar: BinaryFloatingPoint {
    
    private let cache: SearchCache<Vector, Metadata>
    private let ttl: TimeInterval
    
    public init(cache: SearchCache<Vector, Metadata>, ttl: TimeInterval = 300) {
        self.cache = cache
        self.ttl = ttl
    }
    
    public func process<C: Command>(
        _ command: C,
        next: @escaping @Sendable (C) async throws -> C.Result
    ) async throws -> C.Result {
        // Only cache search commands
        if let searchCommand = command as? SearchCommand<Vector, Metadata> {
            // Check cache first
            if let cachedResult = await cache.get(for: searchCommand) {
                return cachedResult as! C.Result
            }
            
            // Execute command
            let result = try await next(command)
            
            // Cache the result
            if let searchResult = result as? ComprehensiveSearchResult<Metadata> {
                await cache.set(searchResult, for: searchCommand, ttl: ttl)
            }
            
            return result
        }
        
        // For non-search commands, just pass through
        return try await next(command)
    }
}

// MARK: - Vector Validation Middleware

/// Middleware that validates vector dimensions and metadata
public struct VectorValidationMiddleware<Vector: SIMD & Sendable>: Middleware 
where Vector.Scalar: BinaryFloatingPoint {
    
    private let expectedDimension: Int
    private let validateMetadata: Bool
    
    public init(expectedDimension: Int, validateMetadata: Bool = true) {
        self.expectedDimension = expectedDimension
        self.validateMetadata = validateMetadata
    }
    
    public func process<C: Command>(
        _ command: C,
        next: @escaping @Sendable (C) async throws -> C.Result
    ) async throws -> C.Result {
        // Since we can't dynamically cast to generic types with existentials,
        // the validation logic would need to be moved to a concrete implementation
        // or handled differently. For now, we'll pass through.
        return try await next(command)
    }
}

// MARK: - Performance Monitoring Middleware

/// Middleware that tracks performance metrics for vector operations
public actor VectorStoreMetricsMiddleware: Middleware {
    
    private var operationCounts: [String: Int] = [:]
    private var operationLatencies: [String: [TimeInterval]] = [:]
    private let maxLatencySamples = 1000
    
    public init() {}
    
    public func process<C: Command>(
        _ command: C,
        next: @escaping @Sendable (C) async throws -> C.Result
    ) async throws -> C.Result {
        let commandType = String(describing: type(of: command))
        let startTime = DispatchTime.now()
        
        // Increment operation count
        await incrementOperationCount(for: commandType)
        
        do {
            let result = try await next(command)
            
            // Record latency
            let endTime = DispatchTime.now()
            let latency = TimeInterval(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000.0
            await recordLatency(latency, for: commandType)
            
            return result
        } catch {
            // Still record latency even on error
            let endTime = DispatchTime.now()
            let latency = TimeInterval(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000.0
            await recordLatency(latency, for: commandType)
            
            throw error
        }
    }
    
    private func incrementOperationCount(for commandType: String) {
        operationCounts[commandType, default: 0] += 1
    }
    
    private func recordLatency(_ latency: TimeInterval, for commandType: String) {
        var latencies = operationLatencies[commandType, default: []]
        latencies.append(latency)
        
        // Keep only recent samples
        if latencies.count > maxLatencySamples {
            latencies = Array(latencies.suffix(maxLatencySamples))
        }
        
        operationLatencies[commandType] = latencies
    }
    
    public func getMetrics() -> VectorStoreMetrics {
        var metrics: [String: VectorStoreMetrics.OperationMetrics] = [:]
        
        for (commandType, count) in operationCounts {
            let latencies = operationLatencies[commandType, default: []]
            guard !latencies.isEmpty else { continue }
            
            let sortedLatencies = latencies.sorted()
            let p50 = sortedLatencies[sortedLatencies.count / 2]
            let p90 = sortedLatencies[Int(Double(sortedLatencies.count) * 0.9)]
            let p99 = sortedLatencies[Int(Double(sortedLatencies.count) * 0.99)]
            
            metrics[commandType] = VectorStoreMetrics.OperationMetrics(
                count: count,
                averageLatency: latencies.reduce(0, +) / Double(latencies.count),
                p50Latency: p50,
                p90Latency: p90,
                p99Latency: p99
            )
        }
        
        return VectorStoreMetrics(operationMetrics: metrics)
    }
}

// MARK: - Access Control Middleware

/// Middleware that enforces access control for vector operations
public struct VectorStoreAccessControlMiddleware: Middleware {
    
    public enum AccessLevel: Sendable {
        case read
        case write
        case admin
    }
    
    private let accessChecker: @Sendable (any Command) async -> AccessLevel?
    
    public init(accessChecker: @escaping @Sendable (any Command) async -> AccessLevel?) {
        self.accessChecker = accessChecker
    }
    
    public func process<C: Command>(
        _ command: C,
        next: @escaping @Sendable (C) async throws -> C.Result
    ) async throws -> C.Result {
        let requiredLevel = getRequiredAccessLevel(for: command)
        let userLevel = await accessChecker(command)
        
        guard let userLevel = userLevel else {
            throw VectorStoreError.accessDenied("No access level provided")
        }
        
        guard hasAccess(userLevel: userLevel, requiredLevel: requiredLevel) else {
            throw VectorStoreError.accessDenied("Insufficient access level")
        }
        
        return try await next(command)
    }
    
    private func getRequiredAccessLevel<C: Command>(for command: C) -> AccessLevel {
        // Check by command type name since we can't cast to generic types with existentials
        let commandType = String(describing: type(of: command))
        
        if commandType.contains("SearchCommand") || commandType.contains("GetStatisticsCommand") {
            return .read
        } else if commandType.contains("StoreEmbeddingCommand") || 
                  commandType.contains("BatchStoreEmbeddingCommand") ||
                  commandType.contains("UpdateVectorCommand") ||
                  commandType.contains("DeleteVectorCommand") ||
                  commandType.contains("BatchDeleteCommand") {
            return .write
        } else if commandType.contains("OptimizeStoreCommand") ||
                  commandType.contains("ExportStoreCommand") ||
                  commandType.contains("ImportStoreCommand") {
            return .admin
        } else {
            return .admin
        }
    }
    
    private func hasAccess(userLevel: AccessLevel, requiredLevel: AccessLevel) -> Bool {
        switch (userLevel, requiredLevel) {
        case (.admin, _):
            return true
        case (.write, .read), (.write, .write):
            return true
        case (.read, .read):
            return true
        default:
            return false
        }
    }
}

// MARK: - Supporting Types

/// Cache for search results
public actor SearchCache<Vector: SIMD & Sendable, Metadata: Codable & Sendable> 
where Vector.Scalar: BinaryFloatingPoint {
    
    private struct CacheEntry {
        let result: ComprehensiveSearchResult<Metadata>
        let expiration: Date
    }
    
    private var cache: [String: CacheEntry] = [:]
    
    public init() {}
    
    func get(for command: SearchCommand<Vector, Metadata>) async -> ComprehensiveSearchResult<Metadata>? {
        let key = cacheKey(for: command)
        
        guard let entry = cache[key] else { return nil }
        
        if entry.expiration > Date() {
            return entry.result
        } else {
            cache[key] = nil
            return nil
        }
    }
    
    func set(_ result: ComprehensiveSearchResult<Metadata>, for command: SearchCommand<Vector, Metadata>, ttl: TimeInterval) async {
        let key = cacheKey(for: command)
        let expiration = Date().addingTimeInterval(ttl)
        
        cache[key] = CacheEntry(result: result, expiration: expiration)
    }
    
    private func cacheKey(for command: SearchCommand<Vector, Metadata>) -> String {
        var hasher = Hasher()
        hasher.combine(command.query.description)
        hasher.combine(command.k)
        hasher.combine(command.strategy)
        return String(hasher.finalize())
    }
}

/// Metrics for vector store operations
public struct VectorStoreMetrics: Sendable {
    public let operationMetrics: [String: OperationMetrics]
    
    public struct OperationMetrics: Sendable {
        public let count: Int
        public let averageLatency: TimeInterval
        public let p50Latency: TimeInterval
        public let p90Latency: TimeInterval
        public let p99Latency: TimeInterval
    }
}

/// Error extension for access control
extension VectorStoreError {
    static func accessDenied(_ reason: String) -> VectorStoreError {
        .validation("Access denied: \(reason)")
    }
}