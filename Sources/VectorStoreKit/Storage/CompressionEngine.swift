import Foundation
import Compression

/// High-performance compression engine supporting LZ4 and ZSTD
public actor CompressionEngine {
    /// Compression statistics for monitoring
    public struct Statistics: Sendable {
        public var totalCompressed: Int = 0
        public var totalDecompressed: Int = 0
        public var totalBytesIn: Int = 0
        public var totalBytesOut: Int = 0
        public var avgCompressionRatio: Double = 0
        public var compressionErrors: Int = 0
        
        mutating func updateCompression(bytesIn: Int, bytesOut: Int) {
            totalCompressed += 1
            totalBytesIn += bytesIn
            totalBytesOut += bytesOut
            avgCompressionRatio = Double(totalBytesIn - totalBytesOut) / Double(max(totalBytesIn, 1))
        }
    }
    
    private var statistics = Statistics()
    
    /// Get current compression statistics
    public func getStatistics() -> Statistics {
        statistics
    }
    
    /// Compress data using the specified algorithm
    public func compress(
        _ data: Data,
        using type: CompressionType,
        level: Int = 1
    ) async throws -> Data {
        switch type {
        case .none:
            return data
        case .lz4:
            return try compressLZ4(data, level: level)
        case .zstd:
            return try compressZSTD(data, level: level)
        }
    }
    
    /// Decompress data using the specified algorithm
    public func decompress(
        _ data: Data,
        using type: CompressionType
    ) async throws -> Data {
        switch type {
        case .none:
            return data
        case .lz4:
            return try decompressLZ4(data)
        case .zstd:
            return try decompressZSTD(data)
        }
    }
    
    // MARK: - LZ4 Implementation
    
    private func compressLZ4(_ data: Data, level: Int) throws -> Data {
        let algorithm = COMPRESSION_LZ4
        // LZ4 typically compresses to at most the original size + some overhead
        let bufferSize = data.count + (data.count / 255) + 16
        
        var compressedData = Data(count: bufferSize)
        let compressedSize = compressedData.withUnsafeMutableBytes { compressedBuffer in
            data.withUnsafeBytes { sourceBuffer in
                compression_encode_buffer(
                    compressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    bufferSize,
                    sourceBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        
        guard compressedSize > 0 else {
            statistics.compressionErrors += 1
            throw CompressionError.compressionFailed
        }
        
        compressedData.count = compressedSize
        statistics.updateCompression(bytesIn: data.count, bytesOut: compressedSize)
        
        return compressedData
    }
    
    private func decompressLZ4(_ data: Data) throws -> Data {
        let algorithm = COMPRESSION_LZ4
        // For LZ4, we need to estimate the decompressed size
        // In production, we'd store this as metadata
        let estimatedSize = data.count * 10 // Conservative estimate
        
        var decompressedData = Data(count: estimatedSize)
        let decompressedSize = decompressedData.withUnsafeMutableBytes { decompressedBuffer in
            data.withUnsafeBytes { compressedBuffer in
                compression_decode_buffer(
                    decompressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    estimatedSize,
                    compressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        
        guard decompressedSize > 0 else {
            throw CompressionError.decompressionFailed
        }
        
        decompressedData.count = decompressedSize
        statistics.totalDecompressed += 1
        
        return decompressedData
    }
    
    // MARK: - ZSTD Implementation
    
    private func compressZSTD(_ data: Data, level: Int) throws -> Data {
        // For now, we'll use ZLIB as a placeholder for ZSTD
        // In production, we'd integrate the actual ZSTD library
        let algorithm = COMPRESSION_ZLIB
        // ZLIB typically compresses to at most the original size + some overhead
        let bufferSize = data.count + 64
        
        var compressedData = Data(count: bufferSize)
        let compressedSize = compressedData.withUnsafeMutableBytes { compressedBuffer in
            data.withUnsafeBytes { sourceBuffer in
                compression_encode_buffer(
                    compressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    bufferSize,
                    sourceBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        
        guard compressedSize > 0 else {
            statistics.compressionErrors += 1
            throw CompressionError.compressionFailed
        }
        
        compressedData.count = compressedSize
        statistics.updateCompression(bytesIn: data.count, bytesOut: compressedSize)
        
        return compressedData
    }
    
    private func decompressZSTD(_ data: Data) throws -> Data {
        // Using ZLIB as placeholder for ZSTD
        let algorithm = COMPRESSION_ZLIB
        let estimatedSize = data.count * 10
        
        var decompressedData = Data(count: estimatedSize)
        let decompressedSize = decompressedData.withUnsafeMutableBytes { decompressedBuffer in
            data.withUnsafeBytes { compressedBuffer in
                compression_decode_buffer(
                    decompressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    estimatedSize,
                    compressedBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        
        guard decompressedSize > 0 else {
            throw CompressionError.decompressionFailed
        }
        
        decompressedData.count = decompressedSize
        statistics.totalDecompressed += 1
        
        return decompressedData
    }
}

/// Compression-related errors
public enum CompressionError: LocalizedError {
    case compressionFailed
    case decompressionFailed
    case unsupportedAlgorithm
    case dataTooLarge
    
    public var errorDescription: String? {
        switch self {
        case .compressionFailed:
            return "Failed to compress data"
        case .decompressionFailed:
            return "Failed to decompress data"
        case .unsupportedAlgorithm:
            return "Unsupported compression algorithm"
        case .dataTooLarge:
            return "Data exceeds maximum compression size"
        }
    }
}

/// Compressed data wrapper to store metadata
public struct CompressedData: Codable, Sendable {
    public let compressedBytes: Data
    public let originalSize: Int
    public let compressionType: CompressionType
    public let timestamp: Date
    
    public init(
        compressedBytes: Data,
        originalSize: Int,
        compressionType: CompressionType,
        timestamp: Date = Date()
    ) {
        self.compressedBytes = compressedBytes
        self.originalSize = originalSize
        self.compressionType = compressionType
        self.timestamp = timestamp
    }
    
    /// Compression ratio (0-1, higher is better)
    public var compressionRatio: Double {
        Double(originalSize - compressedBytes.count) / Double(max(originalSize, 1))
    }
}