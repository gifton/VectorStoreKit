# VectorStoreKit API Reference

## Vector512

### Declaration
```swift
public struct Vector512: SIMD, Sendable, Codable, Collection
```

### Properties
```swift
public static var scalarCount: Int { 512 }
public var scalarCount: Int { 512 }
public typealias MaskStorage = SIMD512<Float.SIMDMaskScalar>
```

### Initializers
```swift
// Default initializer (zeros)
public init()

// Repeating value
public init(repeating scalar: Float)

// From array
public init(_ values: [Float])

// From pointer
public init(_ pointer: UnsafePointer<Float>)
```

### Methods
```swift
// Element access
public subscript(index: Int) -> Float { get set }

// Memory operations
public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R

// Collection conformance
public var startIndex: Int { 0 }
public var endIndex: Int { 512 }
public func index(after i: Int) -> Int
```

---

## BatchProcessor

### Declaration
```swift
public actor BatchProcessor
```

### Configuration
```swift
public struct BatchProcessingConfiguration: Sendable {
    public let optimalBatchSize: Int
    public let maxConcurrentBatches: Int
    public let memoryLimit: Int
    public let useMetalAcceleration: Bool
    
    public init(
        optimalBatchSize: Int = 1000,
        maxConcurrentBatches: Int = ProcessInfo.processInfo.activeProcessorCount,
        memoryLimit: Int = 1_073_741_824, // 1GB
        useMetalAcceleration: Bool = true
    )
    
    public func optimalBatchSize(forVectorSize: Int) -> Int
}
```

### Operations
```swift
public enum BatchOperation<T: Sendable>: Sendable {
    case indexing(index: T)
    case quantization(type: ScalarQuantizationType)
    case distanceComputation(query: T, metric: DistanceMetric)
    case transformation(transform: @Sendable (T) async throws -> T)
    case filtering(predicate: @Sendable (T) -> Bool)
    case aggregation(reducer: @Sendable ([T]) async -> T)
}
```

### Methods
```swift
public init(
    configuration: BatchProcessingConfiguration = BatchProcessingConfiguration(),
    metalCompute: MetalCompute? = nil,
    bufferPool: MetalBufferPool? = nil
)

public func processBatches<T, R>(
    dataset: any LargeVectorDataset<T>,
    operation: BatchOperation<T>,
    progressHandler: ((BatchProgress) -> Void)? = nil
) async throws -> [R] where T: Sendable, R: Sendable

public func streamProcessBatches<T, R>(
    stream: AsyncStream<T>,
    batchSize: Int? = nil,
    operation: BatchOperation<T>
) -> AsyncThrowingStream<[R], Error> where T: Sendable, R: Sendable

public func processVector512Batches(
    vectors: [Vector512],
    operation: BatchOperation<Vector512>,
    progressHandler: ((BatchProgress) -> Void)? = nil
) async throws -> [Any]
```

### Progress Tracking
```swift
public struct BatchProgress: Sendable {
    public let totalItems: Int
    public let processedItems: Int
    public let currentBatch: Int
    public let totalBatches: Int
    public let elapsedTime: TimeInterval
    public let estimatedTimeRemaining: TimeInterval?
    
    public var percentComplete: Float { get }
    public var itemsPerSecond: Double { get }
}
```

---

## HierarchicalIndex

### Declaration
```swift
public actor HierarchicalIndex<Vector: SIMD> where Vector: Sendable, Vector.Scalar: BinaryFloatingPoint
```

### Configuration
```swift
public struct HierarchicalConfiguration: Sendable {
    public let topLevelClusters: Int
    public let leafIndexSize: Int
    public let probesPerQuery: Int
    public let enableDynamicProbing: Bool
    public let rebalanceThreshold: Float
    
    public init(
        topLevelClusters: Int = 1024,
        leafIndexSize: Int = 1000,
        probesPerQuery: Int = 10,
        enableDynamicProbing: Bool = true,
        rebalanceThreshold: Float = 2.0
    )
    
    public static func forDatasetSize(_ size: Int) -> HierarchicalConfiguration
}
```

### Methods
```swift
public init(
    dimension: Int,
    configuration: HierarchicalConfiguration = HierarchicalConfiguration(),
    distanceMetric: DistanceMetric = .euclidean
) async throws

// Insertion
public func insert(
    _ vector: Vector,
    id: VectorID,
    metadata: Data? = nil
) async throws

public func insertBatch(
    _ entries: [(Vector, VectorID, Data?)]
) async throws

// Search
public func search(
    query: Vector,
    k: Int,
    filter: ((Data?) -> Bool)? = nil
) async throws -> [SearchResult<Data>]

public func batchSearch(
    queries: [Vector],
    k: Int,
    filter: ((Data?) -> Bool)? = nil
) async throws -> [[SearchResult<Data>]]

// Maintenance
public func rebalance() async throws
public func getStatistics() async -> HierarchicalStatistics
```

### Statistics
```swift
public struct HierarchicalStatistics: Sendable {
    public let totalVectors: Int
    public let leafIndices: [Int: IndexStatistics]
    public let clusterSizes: [Int: Int]
    public let clusterImbalance: Float
    public let averageSearchLatency: TimeInterval
}
```

---

## ScalarQuantizer

### Declaration
```swift
public actor ScalarQuantizer
```

### Types
```swift
public enum ScalarQuantizationType: Sendable {
    case int8
    case uint8
    case float16
    case dynamic
    
    public var compressionRatio: Float { get }
    public var bytesPerValue: Int { get }
}

public struct QuantizationStatistics: Sendable {
    public let min: Float
    public let max: Float
    public let mean: Float
    public let stdDev: Float
    public let sparsity: Float
    
    public var recommendedType: ScalarQuantizationType { get }
}

public struct QuantizedVectorStore: Sendable {
    public let quantizedData: Data
    public let quantizationType: ScalarQuantizationType
    public let scale: Float
    public let offset: Float
    public let originalDimension: Int
    public let statistics: QuantizationStatistics
    
    public var compressionRatio: Float { get }
}
```

### Methods
```swift
public init(
    metalDevice: MetalDevice? = nil,
    bufferPool: MetalBufferPool? = nil
)

// Statistics
public func computeStatistics(
    for vectors: [[Float]]
) async throws -> QuantizationStatistics

// Quantization
public func quantize(
    _ vector: [Float],
    type: ScalarQuantizationType,
    statistics: QuantizationStatistics? = nil
) async throws -> QuantizedVectorStore

public func quantizeBatch(
    vectors: [[Float]],
    type: ScalarQuantizationType,
    statistics: QuantizationStatistics? = nil
) async throws -> [QuantizedVectorStore]

// Dequantization
public func dequantize(
    _ quantized: QuantizedVectorStore
) async throws -> [Float]

public func dequantizeBatch(
    _ quantizedBatch: [QuantizedVectorStore]
) async throws -> [[Float]]

// Analysis
public func analyzeQuantization(
    vectors: [[Float]],
    type: ScalarQuantizationType
) async throws -> QuantizationAnalysis
```

### Analysis
```swift
public struct QuantizationAnalysis: Sendable {
    public let averageError: Float
    public let maxError: Float
    public let standardDeviation: Float
    public let compressionRatio: Float
    public let signalToNoiseRatio: Float
    public let errorDistribution: [Float]
}
```

---

## StreamingBufferManager

### Declaration
```swift
public actor StreamingBufferManager
```

### Types
```swift
public struct FileMetadata: Sendable {
    public let vectorCount: Int
    public let dimension: Int
    public let vectorSize: Int
    
    public init(
        vectorCount: Int,
        dimension: Int,
        vectorSize: Int
    )
}

public struct StreamingStatistics: Sendable {
    public let totalFiles: Int
    public let totalBytesManaged: Int
    public let totalBytesLoaded: Int
    public let cacheHitRate: Float
    public let averageLoadTime: TimeInterval
}
```

### Methods
```swift
public init(
    device: MTLDevice,
    targetSize: Int,
    pageSize: Int = 4096
) async throws

// File Management
public func addFile(
    url: URL,
    tier: StorageTier,
    metadata: FileMetadata
) async throws

public func removeFile(
    url: URL
) async throws

// Vector Loading
public func loadVectors(
    tier: StorageTier,
    offset: Int,
    count: Int
) async throws -> Data

public func prefetchVectors(
    tier: StorageTier,
    ranges: [Range<Int>]
) async throws

// Streaming
public func streamVectors(
    tier: StorageTier,
    batchSize: Int = 100
) async -> AsyncThrowingStream<Data, Error>

// Metal Integration
public func createMetalBuffer(
    from data: Data,
    options: MTLResourceOptions
) async throws -> MTLBuffer

// Statistics
public func getStatistics() async -> StreamingStatistics
```

### Memory-Mapped File
```swift
public struct MemoryMappedFile: @unchecked Sendable {
    public init(url: URL, mode: Mode = .readOnly) throws
    
    public enum Mode {
        case readOnly
        case readWrite
    }
    
    public func mapRegion(
        offset: Int,
        length: Int
    ) throws -> UnsafeRawBufferPointer
}
```

---

## Error Types

### ScalarQuantizationError
```swift
public enum ScalarQuantizationError: LocalizedError {
    case inconsistentDimensions
    case unsupportedType
    case metalExecutionFailed
    
    public var errorDescription: String? { get }
}
```

### StreamingError
```swift
public enum StreamingError: Error {
    case fileNotFound(URL)
    case memoryMappingFailed
    case invalidRange(offset: Int, length: Int, fileSize: Int)
    case tierNotFound(StorageTier)
    case bufferCreationFailed
}
```

---

## Protocol Conformances

### LargeVectorDataset
```swift
public protocol LargeVectorDataset<Element>: Sendable {
    associatedtype Element: Sendable
    
    var count: Int { get async }
    
    func loadBatch(range: Range<Int>) async throws -> [Element]
    func asyncIterator() -> AsyncStream<Element>
}
```

### Vector512Dataset
```swift
public struct Vector512Dataset: LargeVectorDataset {
    public typealias Element = (Vector512, String, Data?)
    
    public init(vectors: [Vector512])
    
    public var count: Int { get async }
    public func loadBatch(range: Range<Int>) async throws -> [Element]
    public func asyncIterator() -> AsyncStream<Element>
}
```