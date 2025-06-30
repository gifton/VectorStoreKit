// VectorStoreKit: Metal Shader Compiler
//
// Standardized shader compilation and caching system
// Provides efficient compilation, caching, and variant generation
//
// Design principles:
// - Lazy compilation with persistent caching
// - Function constant specialization for runtime optimization
// - Automatic variant generation for different data types and dimensions
// - Performance profiling and shader selection

import Foundation
import Metal
import MetalPerformanceShaders
import os.log
import CryptoKit

/// Centralized shader compiler with advanced caching and optimization
public actor MetalShaderCompiler {
    
    // MARK: - Properties
    
    private let device: MTLDevice
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalShaderCompiler")
    
    // Multi-level cache: memory -> disk
    private var memoryCache: [ShaderCacheKey: MTLComputePipelineState] = [:]
    private let diskCacheURL: URL
    
    // Shader library management
    private var libraries: [String: MTLLibrary] = [:]
    private let defaultLibrary: MTLLibrary?
    
    // Compilation options
    private let options: ShaderCompilationOptions
    
    // Performance metrics
    private var compilationMetrics: [String: ShaderCompilationMetric] = [:]
    
    // MARK: - Initialization
    
    public init(
        device: MTLDevice,
        cacheDirectory: URL? = nil,
        options: ShaderCompilationOptions = .default
    ) throws {
        self.device = device
        self.options = options
        
        // Set up disk cache
        if let cacheDir = cacheDirectory {
            self.diskCacheURL = cacheDir
        } else {
            let cacheDir = FileManager.default.urls(
                for: .cachesDirectory,
                in: .userDomainMask
            ).first!.appendingPathComponent("VectorStoreKit/ShaderCache")
            self.diskCacheURL = cacheDir
        }
        
        // Create cache directory
        try FileManager.default.createDirectory(
            at: diskCacheURL,
            withIntermediateDirectories: true
        )
        
        // Load default library
        self.defaultLibrary = device.makeDefaultLibrary()
        
        // Load cached pipelines if warm start enabled
        if options.warmStart {
            Task {
                await loadCachedPipelines()
            }
        }
        
        logger.info("Initialized shader compiler with cache at: \(self.diskCacheURL.path)")
    }
    
    // MARK: - Public Interface
    
    /// Compile and cache a compute pipeline with optional specialization
    public func compilePipeline(
        functionName: String,
        constants: ShaderFunctionConstants? = nil,
        library: String? = nil
    ) async throws -> MTLComputePipelineState {
        
        // Generate cache key
        let cacheKey = ShaderCacheKey(
            functionName: functionName,
            constants: constants,
            libraryName: library
        )
        
        // Check memory cache
        if let cached = memoryCache[cacheKey] {
            logger.debug("Using memory cached pipeline: \(functionName)")
            return cached
        }
        
        // Check disk cache
        if options.diskCaching,
           let diskCached = try? await loadFromDiskCache(key: cacheKey) {
            memoryCache[cacheKey] = diskCached
            logger.debug("Loaded pipeline from disk cache: \(functionName)")
            return diskCached
        }
        
        // Compile new pipeline
        logger.info("Compiling new pipeline: \(functionName)")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let pipeline = try await compileNewPipeline(
            functionName: functionName,
            constants: constants,
            libraryName: library
        )
        
        let compilationTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Cache the compiled pipeline
        memoryCache[cacheKey] = pipeline
        
        if options.diskCaching {
            try? await saveToDiskCache(key: cacheKey, pipeline: pipeline)
        }
        
        // Record metrics
        compilationMetrics[functionName] = ShaderCompilationMetric(
            functionName: functionName,
            compilationTime: compilationTime,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup
        )
        
        logger.info("Compiled \(functionName) in \(compilationTime)s, threads: \(pipeline.threadExecutionWidth)")
        
        return pipeline
    }
    
    /// Compile multiple pipeline variants for different configurations
    public func compilePipelineVariants(
        baseFunctionName: String,
        variants: [ShaderPipelineVariant]
    ) async throws -> [ShaderPipelineVariant: MTLComputePipelineState] {
        
        var results: [ShaderPipelineVariant: MTLComputePipelineState] = [:]
        
        // Compile variants in parallel
        typealias VariantResult = (variant: ShaderPipelineVariant, result: Result<MTLComputePipelineState, Error>)
        
        await withTaskGroup(of: VariantResult.self) { (group: inout TaskGroup<VariantResult>) in
            for variant in variants {
                group.addTask {
                    do {
                        let functionName = self.generateVariantFunctionName(
                            base: baseFunctionName,
                            variant: variant
                        )
                        
                        let constants = self.generateFunctionConstants(for: variant)
                        
                        let pipeline = try await self.compilePipeline(
                            functionName: functionName,
                            constants: constants
                        )
                        
                        return (variant, .success(pipeline))
                    } catch {
                        return (variant, .failure(error))
                    }
                }
            }
            
            for await variantResult in group {
                switch variantResult.result {
                case .success(let pipeline):
                    results[variantResult.variant] = pipeline
                case .failure(let error):
                    logger.error("Failed to compile variant \(variantResult.variant): \(error)")
                }
            }
        }
        
        return results
    }
    
    /// Get optimal pipeline variant for given parameters
    public func getOptimalPipeline(
        baseFunctionName: String,
        dimensions: Int,
        dataType: ShaderDataType,
        precision: ShaderComputePrecision = .full
    ) async throws -> MTLComputePipelineState {
        
        // Determine optimal variant based on parameters
        let variant = ShaderPipelineVariant(
            dimensions: dimensions,
            dataType: dataType,
            precision: precision,
            simdWidth: determineSIMDWidth(for: dimensions),
            useSharedMemory: dimensions > 128
        )
        
        let functionName = generateVariantFunctionName(
            base: baseFunctionName,
            variant: variant
        )
        
        let constants = generateFunctionConstants(for: variant)
        
        return try await compilePipeline(
            functionName: functionName,
            constants: constants
        )
    }
    
    /// Precompile common shader variants
    public func precompileCommonVariants() async {
        logger.info("Precompiling common shader variants")
        
        let commonFunctions = [
            "euclideanDistance",
            "cosineDistance",
            "dotProduct",
            "matrixMultiply",
            "vectorQuantization"
        ]
        
        let commonDimensions = [128, 256, 512, 768, 1024, 1536]
        let dataTypes: [ShaderDataType] = [.float32, .float16]
        
        var compiledCount = 0
        
        for function in commonFunctions {
            for dimension in commonDimensions {
                for dataType in dataTypes {
                    do {
                        _ = try await getOptimalPipeline(
                            baseFunctionName: function,
                            dimensions: dimension,
                            dataType: dataType
                        )
                        compiledCount += 1
                    } catch {
                        logger.warning("Failed to precompile \(function) for \(dimension)d \(dataType): \(error)")
                    }
                }
            }
        }
        
        logger.info("Precompiled \(compiledCount) shader variants")
    }
    
    // MARK: - Shader Library Management
    
    /// Load a Metal shader library from source or file
    public func loadLibrary(name: String, source: String? = nil, url: URL? = nil) async throws {
        if let source = source {
            // Compile from source
            let options = MTLCompileOptions()
            options.fastMathEnabled = self.options.fastMath
            
            if #available(macOS 13.0, iOS 16.0, *) {
                options.optimizationLevel = self.options.optimizationLevel
            }
            
            let library = try device.makeLibrary(source: source, options: options)
            libraries[name] = library
            
        } else if let url = url {
            // Load from metallib file
            let library = try device.makeLibrary(URL: url)
            libraries[name] = library
            
        } else {
            throw ShaderCompilerError.invalidLibrarySpecification
        }
        
        logger.info("Loaded shader library: \(name)")
    }
    
    /// Get all available functions across loaded libraries
    public func availableFunctions() -> [String] {
        var functions: Set<String> = []
        
        // Add functions from default library
        if let defaultLib = defaultLibrary {
            functions.formUnion(defaultLib.functionNames)
        }
        
        // Add functions from loaded libraries
        for library in libraries.values {
            functions.formUnion(library.functionNames)
        }
        
        return Array(functions).sorted()
    }
    
    // MARK: - Cache Management
    
    /// Clear all cached pipelines
    public func clearCache() async {
        memoryCache.removeAll()
        
        if options.diskCaching {
            try? FileManager.default.removeItem(at: diskCacheURL)
            try? FileManager.default.createDirectory(
                at: diskCacheURL,
                withIntermediateDirectories: true
            )
        }
        
        logger.info("Cleared shader cache")
    }
    
    /// Get cache statistics
    public func getCacheStatistics() -> ShaderCacheStatistics {
        let diskCacheSize = calculateDiskCacheSize()
        
        return ShaderCacheStatistics(
            memoryCachedPipelines: memoryCache.count,
            diskCacheSize: diskCacheSize,
            compilationMetrics: Array(compilationMetrics.values)
        )
    }
    
    // MARK: - Private Implementation
    
    private func compileNewPipeline(
        functionName: String,
        constants: ShaderFunctionConstants?,
        libraryName: String?
    ) async throws -> MTLComputePipelineState {
        
        // Find the function in appropriate library
        let library: MTLLibrary
        if let libName = libraryName, let customLib = libraries[libName] {
            library = customLib
        } else if let defaultLib = defaultLibrary {
            library = defaultLib
        } else {
            throw ShaderCompilerError.noLibraryAvailable
        }
        
        // Get base function
        guard var function = library.makeFunction(name: functionName) else {
            // Try to find function with different suffixes
            let possibleNames = [
                functionName,
                "\(functionName)_kernel",
                "\(functionName)_compute"
            ]
            
            var foundFunction: MTLFunction?
            for name in possibleNames {
                if let f = library.makeFunction(name: name) {
                    foundFunction = f
                    break
                }
            }
            
            guard let function = foundFunction else {
                throw ShaderCompilerError.functionNotFound(functionName)
            }
            
            self.function = function
        }
        
        // Apply function constants if provided
        if let constants = constants {
            let constantValues = MTLFunctionConstantValues()
            
            for (index, value) in constants.values {
                switch value {
                case .uint32(let v):
                    var val = v
                    constantValues.setConstantValue(&val, type: .uint, index: index)
                case .float32(let v):
                    var val = v
                    constantValues.setConstantValue(&val, type: .float, index: index)
                case .bool(let v):
                    var val = v
                    constantValues.setConstantValue(&val, type: .bool, index: index)
                }
            }
            
            function = try library.makeFunction(
                name: functionName,
                constantValues: constantValues
            )
        }
        
        // Create compute pipeline descriptor for advanced options
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = functionName
        
        if #available(macOS 13.0, iOS 16.0, *) {
            descriptor.maxTotalThreadsPerThreadgroup = options.maxThreadsPerThreadgroup
        }
        
        // Compile with descriptor
        let pipeline = try await device.makeComputePipelineState(descriptor: descriptor, options: [])
        
        return pipeline.0  // Return pipeline state, ignore reflection
    }
    
    private func generateVariantFunctionName(base: String, variant: ShaderPipelineVariant) -> String {
        var name = base
        
        // Add dimension suffix for optimized variants
        if [128, 256, 512, 768, 1024].contains(variant.dimensions) {
            name += "\(variant.dimensions)"
        }
        
        // Add data type suffix
        switch variant.dataType {
        case .float16:
            name += "_half"
        case .int8:
            name += "_int8"
        default:
            break
        }
        
        // Add SIMD suffix
        if variant.simdWidth > 1 {
            name += "_simd\(variant.simdWidth)"
        }
        
        return name
    }
    
    private func generateFunctionConstants(for variant: ShaderPipelineVariant) -> ShaderFunctionConstants {
        var constants = ShaderFunctionConstants()
        
        // Standard constants
        constants.setValue(.uint32(UInt32(variant.dimensions)), at: 0)
        constants.setValue(.bool(variant.precision == .fast), at: 1)
        constants.setValue(.uint32(UInt32(variant.simdWidth)), at: 2)
        constants.setValue(.bool(variant.useSharedMemory), at: 3)
        
        return constants
    }
    
    private func determineSIMDWidth(for dimensions: Int) -> Int {
        // Optimize SIMD width based on dimensions
        if dimensions % 4 == 0 {
            return 4
        } else if dimensions % 2 == 0 {
            return 2
        } else {
            return 1
        }
    }
    
    // MARK: - Disk Cache Implementation
    
    private func loadFromDiskCache(key: ShaderCacheKey) async throws -> MTLComputePipelineState? {
        let cacheFile = diskCacheURL.appendingPathComponent(key.cacheFileName)
        
        guard FileManager.default.fileExists(atPath: cacheFile.path) else {
            return nil
        }
        
        // Load binary pipeline data
        let data = try Data(contentsOf: cacheFile)
        
        // Note: Metal doesn't support direct pipeline serialization
        // This would need custom implementation or use of Metal Binary Archives
        // For now, return nil to trigger recompilation
        return nil
    }
    
    private func saveToDiskCache(key: ShaderCacheKey, pipeline: MTLComputePipelineState) async throws {
        // Note: Actual implementation would use Metal Binary Archives
        // or custom serialization format
        let cacheFile = diskCacheURL.appendingPathComponent(key.cacheFileName)
        
        // Save metadata for now
        let metadata = PipelineMetadata(
            functionName: key.functionName,
            compilationDate: Date(),
            deviceName: device.name
        )
        
        let encoder = JSONEncoder()
        let data = try encoder.encode(metadata)
        try data.write(to: cacheFile)
    }
    
    private func loadCachedPipelines() async {
        // Load pipeline metadata from disk cache
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: diskCacheURL,
            includingPropertiesForKeys: nil
        ) else { return }
        
        logger.info("Loading \(files.count) cached pipeline entries")
        
        // In production, would deserialize actual pipeline states
    }
    
    private func calculateDiskCacheSize() -> Int {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: diskCacheURL,
            includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        
        return files.reduce(0) { total, file in
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + size
        }
    }
}

// MARK: - Supporting Types

/// Shader compilation options
public enum OptimizationLevel: Int, Sendable {
    case none = 0
    case `default` = 1
    case performance = 2
}

public struct ShaderCompilationOptions: Sendable {
    public let optimizationLevel: OptimizationLevel
    public let fastMath: Bool
    public let diskCaching: Bool
    public let warmStart: Bool
    public let maxThreadsPerThreadgroup: Int
    
    public static let `default` = ShaderCompilationOptions(
        optimizationLevel: .default,
        fastMath: true,
        diskCaching: true,
        warmStart: true,
        maxThreadsPerThreadgroup: 1024
    )
    
    public static let performance = ShaderCompilationOptions(
        optimizationLevel: .performance,
        fastMath: true,
        diskCaching: true,
        warmStart: true,
        maxThreadsPerThreadgroup: 1024
    )
    
    public static let debug = ShaderCompilationOptions(
        optimizationLevel: .none,
        fastMath: false,
        diskCaching: false,
        warmStart: false,
        maxThreadsPerThreadgroup: 512
    )
}

/// Function constants for shader specialization
public struct ShaderFunctionConstants: Hashable, Equatable, Sendable {
    public enum Value: Hashable, Sendable {
        case uint32(UInt32)
        case float32(Float)
        case bool(Bool)
    }
    
    private struct ConstantEntry: Hashable {
        let index: Int
        let value: Value
    }
    
    private var entries: [ConstantEntry] = []
    
    public var values: [(index: Int, value: Value)] {
        entries.map { ($0.index, $0.value) }
    }
    
    public mutating func setValue(_ value: Value, at index: Int) {
        entries.append(ConstantEntry(index: index, value: value))
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(entries)
    }
    
    public static func == (lhs: ShaderFunctionConstants, rhs: ShaderFunctionConstants) -> Bool {
        lhs.entries == rhs.entries
    }
}

/// Pipeline variant specification
public struct ShaderPipelineVariant: Hashable, Sendable {
    public let dimensions: Int
    public let dataType: ShaderDataType
    public let precision: ShaderComputePrecision
    public let simdWidth: Int
    public let useSharedMemory: Bool
}

/// Supported data types
public enum ShaderDataType: String, Sendable {
    case float32 = "float"
    case float16 = "half"
    case int8 = "char"
    case uint8 = "uchar"
}

/// Compute precision modes
public enum ShaderComputePrecision: String, Sendable {
    case full = "full"
    case fast = "fast"
    case mixed = "mixed"
}

/// Cache key for pipeline lookup
private struct ShaderCacheKey: Hashable {
    let functionName: String
    let constants: ShaderFunctionConstants?
    let libraryName: String?
    
    var cacheFileName: String {
        var hasher = SHA256()
        hasher.update(functionName)
        
        if let constants = constants {
            for (index, value) in constants.values {
                hasher.update("\(index):\(value)")
            }
        }
        
        if let lib = libraryName {
            hasher.update(lib)
        }
        
        let hash = hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
        return "\(functionName)_\(hash).pipelineCache"
    }
}

/// Compilation metrics
public struct ShaderCompilationMetric: Sendable {
    public let functionName: String
    public let compilationTime: TimeInterval
    public let threadExecutionWidth: Int
    public let maxTotalThreadsPerThreadgroup: Int
}

/// Cache statistics
public struct ShaderCacheStatistics: Sendable {
    public let memoryCachedPipelines: Int
    public let diskCacheSize: Int
    public let compilationMetrics: [ShaderCompilationMetric]
}

/// Pipeline metadata for caching
private struct PipelineMetadata: Codable {
    let functionName: String
    let compilationDate: Date
    let deviceName: String
}

/// Shader compiler errors
public enum ShaderCompilerError: Error, LocalizedError {
    case noLibraryAvailable
    case functionNotFound(String)
    case invalidLibrarySpecification
    case compilationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .noLibraryAvailable:
            return "No Metal library available"
        case .functionNotFound(let name):
            return "Function not found: \(name)"
        case .invalidLibrarySpecification:
            return "Invalid library specification - provide either source or URL"
        case .compilationFailed(let reason):
            return "Shader compilation failed: \(reason)"
        }
    }
}