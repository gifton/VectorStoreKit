// VectorStoreKit: Model Serialization
//
// High-performance model checkpoint system with Metal buffer serialization
//
// Key features:
// - Efficient Metal buffer serialization with compression
// - Versioned checkpoint format for backward compatibility  
// - Integrity validation with checksums
// - Support for incremental/differential checkpoints
// - Async/await for non-blocking I/O

import Foundation
import Compression
import CryptoKit
@preconcurrency import Metal

// MARK: - Core Protocol

/// Protocol for serializable ML models
public protocol ModelSerializable: Actor {
    /// Save model checkpoint to URL
    /// - Parameters:
    ///   - url: Destination URL for checkpoint
    ///   - options: Serialization options
    func saveCheckpoint(to url: URL, options: CheckpointOptions) async throws
    
    /// Load model checkpoint from URL
    /// - Parameters:
    ///   - url: Source URL for checkpoint
    ///   - options: Deserialization options
    func loadCheckpoint(from url: URL, options: CheckpointOptions) async throws
    
    /// Get checkpoint metadata without loading full model
    static func checkpointMetadata(from url: URL) async throws -> CheckpointMetadata
}

// Default implementations and convenience methods
public extension ModelSerializable {
    /// Save checkpoint with default options
    func saveCheckpoint(to url: URL) async throws {
        try await saveCheckpoint(to: url, options: .default)
    }
    
    /// Load checkpoint with default options
    func loadCheckpoint(from url: URL) async throws {
        try await loadCheckpoint(from: url, options: .default)
    }
    
    static func checkpointMetadata(from url: URL) async throws -> CheckpointMetadata {
        // Create a temporary device for loading metadata
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CheckpointError.bufferCreationFailed
        }
        
        let checkpointManager = CheckpointManager(device: device)
        return try await checkpointManager.loadMetadata(from: url)
    }
}

// MARK: - Checkpoint Format

/// Checkpoint container format
/// 
/// Thread-safe checkpoint structure for serializing ML model state.
/// All properties are immutable and contain only Sendable types.
public struct Checkpoint: Codable, Sendable {
    /// Format version for compatibility
    public let version: CheckpointVersion
    
    /// Model architecture configuration
    public let architecture: ArchitectureConfig
    
    /// Serialized parameters
    public let parameters: [ParameterData]
    
    /// Training state
    public let trainingState: TrainingState?
    
    /// Metadata
    public let metadata: CheckpointMetadata
    
    /// Integrity checksum
    public let checksum: String
}

/// Checkpoint format version
public struct CheckpointVersion: Codable, Comparable, Sendable {
    public let major: Int
    public let minor: Int
    public let patch: Int
    
    public static let current = CheckpointVersion(major: 1, minor: 0, patch: 0)
    
    public static func < (lhs: CheckpointVersion, rhs: CheckpointVersion) -> Bool {
        if lhs.major != rhs.major { return lhs.major < rhs.major }
        if lhs.minor != rhs.minor { return lhs.minor < rhs.minor }
        return lhs.patch < rhs.patch
    }
}

/// Model architecture configuration
/// 
/// Immutable configuration describing model architecture.
/// Safe to share across actor boundaries.
public struct ArchitectureConfig: Codable, Sendable {
    /// Model type identifier
    public let modelType: String
    
    /// Layer configurations
    public let layers: [LayerConfig]
    
    /// Global model settings
    public let settings: [String: String]
}

/// Layer configuration
/// 
/// Immutable layer configuration with thread-safe properties.
public struct LayerConfig: Codable, Sendable {
    /// Layer type identifier
    public let type: String
    
    /// Layer name
    public let name: String
    
    /// Layer-specific configuration
    public let config: [String: String]
    
    /// Parameter shapes
    public let parameterShapes: [String: [Int]]
}

/// Serialized parameter data
/// 
/// Thread-safe container for serialized model parameters.
/// Contains only value types and immutable data.
public struct ParameterData: Codable, Sendable {
    /// Parameter name
    public let name: String
    
    /// Parameter shape
    public let shape: [Int]
    
    /// Data type
    public let dtype: DataType
    
    /// Compressed data
    public let data: Data
    
    /// Compression algorithm used
    public let compression: CompressionAlgorithm
    
    /// Original size before compression
    public let originalSize: Int
}

/// Supported data types
/// 
/// Thread-safe enumeration of supported parameter data types.
public enum DataType: String, Codable, Sendable {
    case float32 = "float32"
    case float16 = "float16"
    case int32 = "int32"
    case int16 = "int16"
    case int8 = "int8"
    case uint8 = "uint8"
    
    var stride: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .int16: return 2
        case .int8, .uint8: return 1
        }
    }
}

/// Training state for resumable training
/// 
/// Immutable snapshot of training state, safe for concurrent access.
public struct TrainingState: Codable, Sendable {
    /// Current epoch
    public let epoch: Int
    
    /// Current step within epoch
    public let step: Int
    
    /// Optimizer state
    public let optimizerState: OptimizerState?
    
    /// Training history
    public let history: NetworkTrainingHistory
    
    /// Random seed for reproducibility
    public let randomSeed: UInt64
}

/// Optimizer state
/// 
/// Thread-safe optimizer state snapshot.
public struct OptimizerState: Codable, Sendable {
    /// Optimizer type
    public let type: String
    
    /// Learning rate
    public let learningRate: Float
    
    /// Momentum buffers
    public let momentumBuffers: [ParameterData]?
    
    /// Velocity buffers (for Adam)
    public let velocityBuffers: [ParameterData]?
    
    /// Additional optimizer-specific state
    public let additionalState: [String: String]
}

/// Checkpoint metadata
public struct CheckpointMetadata: Codable, Sendable {
    /// Creation timestamp
    public let timestamp: Date
    
    /// Model description
    public let description: String?
    
    /// Training metrics at checkpoint
    public let metrics: [String: Float]
    
    /// Hardware info
    public let hardware: HardwareInfo
    
    /// Custom metadata
    public let custom: [String: String]
}

/// Hardware information
/// 
/// Immutable hardware configuration snapshot.
/// Safe to share across concurrency domains.
public struct HardwareInfo: Codable, Sendable {
    /// Device name
    public let deviceName: String
    
    /// Device type (GPU, Neural Engine, etc.)
    public let deviceType: String
    
    /// Available memory at checkpoint time
    public let availableMemory: Int
    
    /// Metal features used
    public let metalFeatures: [String]
}

/// Checkpoint options
public struct CheckpointOptions: Sendable {
    /// Compression algorithm
    public let compression: CompressionAlgorithm
    
    /// Include training state
    public let includeTrainingState: Bool
    
    /// Include optimizer state
    public let includeOptimizerState: Bool
    
    /// Validate on save
    public let validateOnSave: Bool
    
    /// Description
    public let description: String?
    
    /// Custom metadata
    public let customMetadata: [String: String]
    
    public init(
        compression: CompressionAlgorithm = .zstd,
        includeTrainingState: Bool = true,
        includeOptimizerState: Bool = true,
        validateOnSave: Bool = true,
        description: String? = nil,
        customMetadata: [String: String] = [:]
    ) {
        self.compression = compression
        self.includeTrainingState = includeTrainingState
        self.includeOptimizerState = includeOptimizerState
        self.validateOnSave = validateOnSave
        self.description = description
        self.customMetadata = customMetadata
    }
    
    public static let `default` = CheckpointOptions()
}

// MARK: - Serialization Engine

/// High-performance checkpoint serialization engine
public actor CheckpointSerializer {
    private let device: MTLDevice
    private let compressionLevel: Int
    
    public init(device: MTLDevice, compressionLevel: Int = 6) {
        self.device = device
        self.compressionLevel = compressionLevel
    }
    
    /// Serialize Metal buffer to compressed data
    public func serializeBuffer(
        _ buffer: MetalBuffer,
        name: String,
        compression: CompressionAlgorithm
    ) async throws -> ParameterData {
        // Get buffer data
        let contents = buffer.buffer.contents()
        let byteLength = buffer.count * MemoryLayout<Float>.stride
        
        // Create data from buffer
        let data = Data(bytes: contents, count: byteLength)
        
        // Compress data
        let compressedData = try await compressData(data, algorithm: compression)
        
        return ParameterData(
            name: name,
            shape: buffer.shape.dimensions,
            dtype: .float32,
            data: compressedData,
            compression: compression,
            originalSize: byteLength
        )
    }
    
    /// Deserialize compressed data to Metal buffer
    public func deserializeBuffer(
        _ parameterData: ParameterData
    ) async throws -> MetalBuffer {
        // Decompress data
        let decompressedData = try await decompressData(
            parameterData.data,
            algorithm: parameterData.compression,
            originalSize: parameterData.originalSize
        )
        
        // Create Metal buffer
        guard let buffer = device.makeBuffer(
            bytes: decompressedData.withUnsafeBytes { $0.baseAddress! },
            length: decompressedData.count,
            options: .storageModeShared
        ) else {
            throw CheckpointError.bufferCreationFailed
        }
        
        let shape = TensorShape(dimensions: parameterData.shape)
        return MetalBuffer(buffer: buffer, shape: shape)
    }
    
    /// Compress data using specified algorithm
    private func compressData(
        _ data: Data,
        algorithm: CompressionAlgorithm
    ) async throws -> Data {
        switch algorithm {
        case .none:
            return data
            
        case .lz4:
            return try compressLZ4(data)
            
        case .zstd:
            return try compressZSTD(data)
            
        case .quantization:
            // Use quantization compressor
            let compressor = try await QuantizationCompressor(device: device)
            let shape = [data.count / MemoryLayout<Float>.stride]
            let compressed = try await compressor.compress(
                data: data,
                shape: shape,
                quantizationType: .uint8 // Use 8-bit quantization for maximum compression
            )
            // Encode the compressed parameter
            let encoder = JSONEncoder()
            let quantizedData = try encoder.encode(compressed)
            
            // Apply zlib compression on top of quantization for even better compression
            return try compressZSTD(quantizedData)
            
        default:
            throw CheckpointError.unsupportedCompression(algorithm)
        }
    }
    
    /// Decompress data using specified algorithm
    private func decompressData(
        _ data: Data,
        algorithm: CompressionAlgorithm,
        originalSize: Int
    ) async throws -> Data {
        switch algorithm {
        case .none:
            return data
            
        case .lz4:
            return try decompressLZ4(data, originalSize: originalSize)
            
        case .zstd:
            return try decompressZSTD(data, originalSize: originalSize)
            
        case .quantization:
            // Try to decompress with different buffer sizes until successful
            var quantizedData: Data?
            for multiplier in [2, 4, 8, 16, 32] {
                do {
                    quantizedData = try decompressZSTD(data, originalSize: data.count * multiplier)
                    break
                } catch {
                    // Try next size
                    continue
                }
            }
            
            guard let decompressedData = quantizedData else {
                throw CheckpointError.decompressionFailed
            }
            
            // Use quantization decompressor
            let compressor = try await QuantizationCompressor(device: device)
            // Decode the compressed parameter
            let decoder = JSONDecoder()
            let compressed = try decoder.decode(CompressedParameter.self, from: decompressedData)
            return try await compressor.decompress(compressed)
            
        default:
            throw CheckpointError.unsupportedCompression(algorithm)
        }
    }
    
    /// LZ4 compression
    private func compressLZ4(_ data: Data) throws -> Data {
        return try data.withUnsafeBytes { sourceBuffer in
            let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress!
            let destSize = compression_encode_scratch_buffer_size(COMPRESSION_LZ4_RAW)
            let destBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destSize)
            defer { destBuffer.deallocate() }
            
            let compressedSize = compression_encode_buffer(
                destBuffer, destSize,
                sourcePtr, data.count,
                nil, COMPRESSION_LZ4_RAW
            )
            
            guard compressedSize > 0 else {
                throw CheckpointError.compressionFailed
            }
            
            return Data(bytes: destBuffer, count: compressedSize)
        }
    }
    
    /// LZ4 decompression
    private func decompressLZ4(_ data: Data, originalSize: Int) throws -> Data {
        return try data.withUnsafeBytes { sourceBuffer in
            let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress!
            let destBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: originalSize)
            defer { destBuffer.deallocate() }
            
            let decompressedSize = compression_decode_buffer(
                destBuffer, originalSize,
                sourcePtr, data.count,
                nil, COMPRESSION_LZ4_RAW
            )
            
            guard decompressedSize == originalSize else {
                throw CheckpointError.decompressionFailed
            }
            
            return Data(bytes: destBuffer, count: decompressedSize)
        }
    }
    
    /// ZSTD compression
    private func compressZSTD(_ data: Data) throws -> Data {
        return try data.withUnsafeBytes { sourceBuffer in
            let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress!
            let destSize = compression_encode_scratch_buffer_size(COMPRESSION_ZLIB)
            let destBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destSize)
            defer { destBuffer.deallocate() }
            
            let compressedSize = compression_encode_buffer(
                destBuffer, destSize,
                sourcePtr, data.count,
                nil, COMPRESSION_ZLIB
            )
            
            guard compressedSize > 0 else {
                throw CheckpointError.compressionFailed
            }
            
            return Data(bytes: destBuffer, count: compressedSize)
        }
    }
    
    /// ZSTD decompression
    private func decompressZSTD(_ data: Data, originalSize: Int) throws -> Data {
        return try data.withUnsafeBytes { sourceBuffer in
            let sourcePtr = sourceBuffer.bindMemory(to: UInt8.self).baseAddress!
            let destBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: originalSize)
            defer { destBuffer.deallocate() }
            
            let decompressedSize = compression_decode_buffer(
                destBuffer, originalSize,
                sourcePtr, data.count,
                nil, COMPRESSION_ZLIB
            )
            
            guard decompressedSize == originalSize else {
                throw CheckpointError.decompressionFailed
            }
            
            return Data(bytes: destBuffer, count: decompressedSize)
        }
    }
}

// MARK: - Checkpoint Manager

/// Manages checkpoint save/load operations
public actor CheckpointManager {
    private let serializer: CheckpointSerializer
    private let fileManager = FileManager.default
    
    public init(device: MTLDevice) {
        self.serializer = CheckpointSerializer(device: device)
    }
    
    /// Save checkpoint to file
    public func save(
        _ checkpoint: Checkpoint,
        to url: URL
    ) async throws {
        // Ensure directory exists
        let directory = url.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        
        // Encode checkpoint
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        
        let data = try encoder.encode(checkpoint)
        
        // Write to file
        try data.write(to: url)
    }
    
    /// Load checkpoint from file
    public func load(from url: URL) async throws -> Checkpoint {
        // Read file
        let data = try Data(contentsOf: url)
        
        // Decode checkpoint
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        let checkpoint = try decoder.decode(Checkpoint.self, from: data)
        
        // Validate version compatibility
        guard checkpoint.version <= CheckpointVersion.current else {
            throw CheckpointError.incompatibleVersion(
                checkpoint.version,
                current: CheckpointVersion.current
            )
        }
        
        // Validate checksum
        let computedChecksum = try computeChecksum(for: checkpoint)
        guard computedChecksum == checkpoint.checksum else {
            throw CheckpointError.checksumMismatch
        }
        
        return checkpoint
    }
    
    /// Load checkpoint metadata without full deserialization
    public func loadMetadata(from url: URL) async throws -> CheckpointMetadata {
        // Read only the metadata portion
        // This is a simplified implementation - in production, we'd parse JSON partially
        let checkpoint = try await load(from: url)
        return checkpoint.metadata
    }
    
    /// Compute SHA256 checksum for checkpoint
    private func computeChecksum(for checkpoint: Checkpoint) throws -> String {
        // Encode architecture and parameters for checksum
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        encoder.dateEncodingStrategy = .iso8601
        
        // Create data for checksum by combining architecture and parameters
        var dataToHash = Data()
        
        // Add architecture
        let architectureData = try encoder.encode(checkpoint.architecture)
        dataToHash.append(architectureData)
        
        // Add parameters data
        for param in checkpoint.parameters {
            dataToHash.append(param.data)
        }
        
        // Add metadata
        let metadataData = try encoder.encode(checkpoint.metadata)
        dataToHash.append(metadataData)
        
        // Compute SHA256 using CryptoKit
        let hash = SHA256.hash(data: dataToHash)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    /// Get serializer for buffer operations
    public func getSerializer() -> CheckpointSerializer {
        serializer
    }
}

// MARK: - Errors

/// Checkpoint serialization errors
public enum CheckpointError: LocalizedError {
    case bufferCreationFailed
    case compressionFailed
    case decompressionFailed
    case unsupportedCompression(CompressionAlgorithm)
    case incompatibleVersion(CheckpointVersion, current: CheckpointVersion)
    case checksumMismatch
    case invalidFormat
    case missingParameter(String)
    
    public var errorDescription: String? {
        switch self {
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        case .compressionFailed:
            return "Failed to compress data"
        case .decompressionFailed:
            return "Failed to decompress data"
        case .unsupportedCompression(let algorithm):
            return "Unsupported compression algorithm: \(algorithm)"
        case .incompatibleVersion(let version, let current):
            return "Incompatible checkpoint version \(version), current version is \(current)"
        case .checksumMismatch:
            return "Checkpoint integrity validation failed"
        case .invalidFormat:
            return "Invalid checkpoint format"
        case .missingParameter(let name):
            return "Missing parameter: \(name)"
        }
    }
}

// MARK: - Helper Extensions

extension Data {
    /// Compute SHA256 hash using CryptoKit
    func sha256Hash() -> String {
        let hash = SHA256.hash(data: self)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Backward Compatibility

/// Version migration support
public protocol CheckpointMigrator {
    /// Source version this migrator handles
    var sourceVersion: CheckpointVersion { get }
    
    /// Target version after migration
    var targetVersion: CheckpointVersion { get }
    
    /// Migrate checkpoint data
    func migrate(_ checkpoint: Checkpoint) async throws -> Checkpoint
}

/// Registry for checkpoint migrators
public actor CheckpointMigrationRegistry {
    private var migrators: [String: any CheckpointMigrator] = [:]
    
    /// Register a migrator
    public func register(_ migrator: any CheckpointMigrator) {
        let key = "\(migrator.sourceVersion)->\(migrator.targetVersion)"
        migrators[key] = migrator
    }
    
    /// Find migration path from source to target version
    public func migrationPath(
        from source: CheckpointVersion,
        to target: CheckpointVersion
    ) -> [any CheckpointMigrator]? {
        // Simple implementation - in production, use graph traversal
        let key = "\(source)->\(target)"
        if let migrator = migrators[key] {
            return [migrator]
        }
        return nil
    }
}