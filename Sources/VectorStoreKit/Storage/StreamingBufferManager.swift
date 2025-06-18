// StreamingBufferManager.swift
// VectorStoreKit
//
// Efficient streaming buffer management for large vector datasets

import Foundation
import Metal
import os.log

/// Memory-mapped file wrapper for large datasets
public struct MemoryMappedFile: @unchecked Sendable {
    private let fileHandle: FileHandle
    private let fileSize: Int
    private let baseAddress: UnsafeRawPointer
    
    public init(url: URL, mode: Mode = .readOnly) throws {
        // Open file
        let fileHandle = try FileHandle(forReadingFrom: url)
        self.fileHandle = fileHandle
        
        // Get file size
        let fileSize = try fileHandle.seekToEnd()
        self.fileSize = Int(fileSize)
        try fileHandle.seek(toOffset: 0)
        
        // Memory map the file
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped),
              !data.isEmpty else {
            throw StreamingError.memoryMappingFailed
        }
        
        // Store the data to keep it alive
        // Note: This is a simplified implementation. In production, you'd want to
        // properly manage the memory-mapped region
        self.baseAddress = data.withUnsafeBytes { $0.baseAddress! }
    }
    
    public enum Mode {
        case readOnly
        case readWrite
    }
    
    /// Map a specific region of the file
    public func mapRegion(offset: Int, length: Int) throws -> UnsafeRawBufferPointer {
        guard offset >= 0 && offset + length <= fileSize else {
            throw StreamingError.invalidRange(offset: offset, length: length, fileSize: fileSize)
        }
        
        return UnsafeRawBufferPointer(
            start: baseAddress.advanced(by: offset),
            count: length
        )
    }
}

/// Manages streaming buffers for large vector datasets
public actor StreamingBufferManager {
    // MARK: - Properties
    
    private let targetDatasetSize: Int
    private let bufferPool: MetalBufferPool
    private let pageSize: Int
    private let device: MTLDevice
    
    // Memory-mapped files for different tiers
    private var hotTierFile: MemoryMappedFile?
    private var warmTierFile: MemoryMappedFile?
    private var coldTierFile: MemoryMappedFile?
    
    // Buffer management
    private var activeBuffers: [BufferID: StreamingBuffer] = [:]
    private var bufferAccessCounts: [BufferID: Int] = [:]
    
    // Metrics
    private var totalBytesStreamed: Int = 0
    private var streamingOperations: Int = 0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "StreamingBufferManager")
    
    // MARK: - Types
    
    public typealias BufferID = String
    
    private struct StreamingBuffer {
        let id: BufferID
        let metalBuffer: MTLBuffer
        let offset: Int
        let length: Int
        let tier: StorageTier
        var lastAccess: Date
    }
    
    // MARK: - Initialization
    
    public init(
        targetSize: Int,
        device: MTLDevice,
        pageSize: Int = 65536 // 64KB pages
    ) async throws {
        self.targetDatasetSize = targetSize
        self.device = device
        self.pageSize = pageSize
        
        // Configure buffer pool for streaming with appropriate sizes
        let bufferSizes = [pageSize, pageSize * 4, pageSize * 16]
        let config = MetalBufferPoolConfiguration(
            maxBuffersPerSize: 10,
            preallocationSizes: bufferSizes
        )
        
        self.bufferPool = MetalBufferPool(
            device: device,
            configuration: config
        )
    }
    
    // MARK: - File Management
    
    /// Initialize memory-mapped files for each storage tier
    public func initializeStorageTiers(
        hotTierURL: URL?,
        warmTierURL: URL?,
        coldTierURL: URL?
    ) async throws {
        if let url = hotTierURL {
            hotTierFile = try MemoryMappedFile(url: url)
        }
        if let url = warmTierURL {
            warmTierFile = try MemoryMappedFile(url: url)
        }
        if let url = coldTierURL {
            coldTierFile = try MemoryMappedFile(url: url)
        }
    }
    
    // MARK: - Streaming Operations
    
    /// Stream a batch of vectors from disk to GPU
    public func streamBatch(
        range: Range<Int>,
        vectorDimension: Int,
        tier: StorageTier = .warm
    ) async throws -> MTLBuffer {
        let vectorSize = vectorDimension * MemoryLayout<Float>.size
        let offset = range.lowerBound * vectorSize
        let length = range.count * vectorSize
        
        let vectorCount = range.count
        logger.debug("Streaming \(vectorCount) vectors from \(tier.rawValue) tier")
        
        // Select appropriate file based on tier
        let file = selectFile(for: tier)
        guard let file = file else {
            throw StreamingError.tierNotAvailable(tier)
        }
        
        // Map file region to memory
        let bufferPointer = try file.mapRegion(offset: offset, length: length)
        
        // Create Metal buffer without copying
        let metalBuffer = try await createMetalBuffer(from: bufferPointer, tier: tier)
        
        // Track buffer
        let bufferID = UUID().uuidString
        activeBuffers[bufferID] = StreamingBuffer(
            id: bufferID,
            metalBuffer: metalBuffer,
            offset: offset,
            length: length,
            tier: tier,
            lastAccess: Date()
        )
        bufferAccessCounts[bufferID] = 1
        
        // Update metrics
        totalBytesStreamed += length
        streamingOperations += 1
        
        return metalBuffer
    }
    
    /// Stream vectors asynchronously for processing
    public func streamVectors<T>(
        vectorDimension: Int,
        batchSize: Int = 1024,
        tier: StorageTier = .warm,
        transform: @escaping (MTLBuffer, Int) async throws -> T
    ) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let file = selectFile(for: tier)
                    guard let file = file else {
                        throw StreamingError.tierNotAvailable(tier)
                    }
                    
                    let vectorSize = vectorDimension * MemoryLayout<Float>.size
                    let totalVectors = targetDatasetSize / vectorSize
                    
                    for offset in stride(from: 0, to: totalVectors, by: batchSize) {
                        let count = min(batchSize, totalVectors - offset)
                        let range = offset..<(offset + count)
                        
                        do {
                            let buffer = try await streamBatch(
                                range: range,
                                vectorDimension: vectorDimension,
                                tier: tier
                            )
                            
                            let result = try await transform(buffer, count)
                            continuation.yield(result)
                            
                            // Return buffer to pool
                            await returnStreamingBuffer(buffer)
                        } catch {
                            continuation.finish(throwing: error)
                            return
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Buffer Management
    
    /// Create Metal buffer from memory-mapped data
    private func createMetalBuffer(
        from data: UnsafeRawBufferPointer,
        tier: StorageTier
    ) async throws -> MTLBuffer {
        // Determine storage mode based on tier
        let storageMode: MTLResourceOptions
        switch tier {
        case .hot:
            storageMode = .storageModeShared // CPU+GPU access
        case .warm:
            storageMode = .storageModePrivate // GPU only
        case .cold, .frozen:
            storageMode = .storageModeManaged // Managed by system
        default:
            storageMode = .storageModeShared
        }
        
        // Create buffer without copying data
        guard let buffer = device.makeBuffer(
            bytes: data.baseAddress!,
            length: data.count,
            options: storageMode
        ) else {
            throw StreamingError.bufferCreationFailed
        }
        
        return buffer
    }
    
    /// Return a streaming buffer to the pool
    public func returnStreamingBuffer(_ buffer: MTLBuffer) async {
        // Find and remove from active buffers
        if let entry = activeBuffers.first(where: { $0.value.metalBuffer === buffer }) {
            activeBuffers.removeValue(forKey: entry.key)
            bufferAccessCounts.removeValue(forKey: entry.key)
        }
        
        // Return to pool for reuse
        await bufferPool.returnBuffer(buffer)
    }
    
    /// Prefetch data for upcoming operations
    public func prefetchBatch(
        range: Range<Int>,
        vectorDimension: Int,
        tier: StorageTier = .warm
    ) async throws {
        // This could trigger OS-level prefetching
        let vectorSize = vectorDimension * MemoryLayout<Float>.size
        let offset = range.lowerBound * vectorSize
        let length = range.count * vectorSize
        
        let file = selectFile(for: tier)
        guard let file = file else { return }
        
        // Touch pages to trigger prefetch
        let _ = try file.mapRegion(offset: offset, length: length)
    }
    
    // MARK: - Memory Pressure Management
    
    /// Evict least recently used buffers under memory pressure
    public func handleMemoryPressure() async {
        logger.warning("Handling memory pressure")
        
        // Sort buffers by access count and last access time
        let sortedBuffers = activeBuffers.values.sorted { buffer1, buffer2 in
            let count1 = bufferAccessCounts[buffer1.id] ?? 0
            let count2 = bufferAccessCounts[buffer2.id] ?? 0
            
            if count1 != count2 {
                return count1 < count2
            }
            return buffer1.lastAccess < buffer2.lastAccess
        }
        
        // Evict bottom 25% of buffers
        let evictCount = max(1, sortedBuffers.count / 4)
        for i in 0..<evictCount {
            let buffer = sortedBuffers[i]
            await returnStreamingBuffer(buffer.metalBuffer)
        }
    }
    
    // MARK: - Utilities
    
    private func selectFile(for tier: StorageTier) -> MemoryMappedFile? {
        switch tier {
        case .hot:
            return hotTierFile
        case .warm:
            return warmTierFile
        case .cold, .frozen:
            return coldTierFile
        case .auto:
            // Default to warm tier for auto
            return warmTierFile
        }
    }
    
    /// Get streaming statistics
    public func getStatistics() -> StreamingStatistics {
        StreamingStatistics(
            totalBytesStreamed: totalBytesStreamed,
            streamingOperations: streamingOperations,
            activeBufferCount: activeBuffers.count,
            averageBufferSize: activeBuffers.isEmpty ? 0 : 
                activeBuffers.values.map(\.length).reduce(0, +) / activeBuffers.count
        )
    }
}

// MARK: - Supporting Types

public struct StreamingStatistics: Sendable {
    public let totalBytesStreamed: Int
    public let streamingOperations: Int
    public let activeBufferCount: Int
    public let averageBufferSize: Int
    
    public var formattedTotalSize: String {
        ByteCountFormatter.string(fromByteCount: Int64(totalBytesStreamed), countStyle: .binary)
    }
}

public enum StreamingError: LocalizedError {
    case memoryMappingFailed
    case invalidRange(offset: Int, length: Int, fileSize: Int)
    case tierNotAvailable(StorageTier)
    case bufferCreationFailed
    
    public var errorDescription: String? {
        switch self {
        case .memoryMappingFailed:
            return "Failed to memory map file"
        case .invalidRange(let offset, let length, let fileSize):
            return "Invalid range: offset=\(offset), length=\(length), fileSize=\(fileSize)"
        case .tierNotAvailable(let tier):
            return "Storage tier not available: \(tier)"
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        }
    }
}

// MARK: - Extensions

extension StreamingBufferManager {
    /// Convenience method for streaming Vector512 types
    public func streamVector512Batch(
        range: Range<Int>,
        tier: StorageTier = .warm
    ) async throws -> MTLBuffer {
        return try await streamBatch(
            range: range,
            vectorDimension: 512,
            tier: tier
        )
    }
    
    /// Stream vectors with automatic dimension detection
    public func streamVectorsAutoDimension(
        dimension: VectorDimension,
        batchSize: Int? = nil,
        tier: StorageTier = .warm
    ) -> AsyncThrowingStream<MTLBuffer, Error> {
        let actualBatchSize = batchSize ?? dimension.optimalBatchSize
        
        return streamVectors(
            vectorDimension: dimension.dimension,
            batchSize: actualBatchSize,
            tier: tier
        ) { buffer, count in
            return buffer
        }
    }
}