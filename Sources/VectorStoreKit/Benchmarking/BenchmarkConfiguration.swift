// VectorStoreKit: Benchmark Configuration
//
// Configurable benchmark parameters

import Foundation

/// Master configuration for all benchmarks
public struct BenchmarkConfiguration: Codable, Sendable {
    
    // MARK: - General Settings
    
    public struct GeneralSettings: Codable, Sendable {
        public let name: String
        public let description: String
        public let outputDirectory: String
        public let saveResults: Bool
        public let compareBaseline: Bool
        public let baselinePath: String?
        public let randomSeed: UInt64?
        public let timeout: TimeInterval
        
        public init(
            name: String = "VectorStoreKit Benchmarks",
            description: String = "Comprehensive performance benchmarks",
            outputDirectory: String = "./benchmark-results",
            saveResults: Bool = true,
            compareBaseline: Bool = false,
            baselinePath: String? = nil,
            randomSeed: UInt64? = nil,
            timeout: TimeInterval = 3600 // 1 hour
        ) {
            self.name = name
            self.description = description
            self.outputDirectory = outputDirectory
            self.saveResults = saveResults
            self.compareBaseline = compareBaseline
            self.baselinePath = baselinePath
            self.randomSeed = randomSeed
            self.timeout = timeout
        }
    }
    
    // MARK: - Execution Settings
    
    public struct ExecutionSettings: Codable, Sendable {
        public let iterations: Int
        public let warmupIterations: Int
        public let cooldownTime: TimeInterval
        public let parallelExecution: Bool
        public let maxConcurrency: Int
        public let collectSystemMetrics: Bool
        public let profileMemory: Bool
        public let profileCPU: Bool
        public let profileGPU: Bool
        
        public init(
            iterations: Int = 10,
            warmupIterations: Int = 2,
            cooldownTime: TimeInterval = 1.0,
            parallelExecution: Bool = false,
            maxConcurrency: Int = 4,
            collectSystemMetrics: Bool = true,
            profileMemory: Bool = true,
            profileCPU: Bool = true,
            profileGPU: Bool = true
        ) {
            self.iterations = iterations
            self.warmupIterations = warmupIterations
            self.cooldownTime = cooldownTime
            self.parallelExecution = parallelExecution
            self.maxConcurrency = maxConcurrency
            self.collectSystemMetrics = collectSystemMetrics
            self.profileMemory = profileMemory
            self.profileCPU = profileCPU
            self.profileGPU = profileGPU
        }
        
        public static let quick = ExecutionSettings(
            iterations: 3,
            warmupIterations: 1,
            cooldownTime: 0.5
        )
        
        public static let standard = ExecutionSettings()
        
        public static let thorough = ExecutionSettings(
            iterations: 100,
            warmupIterations: 10,
            cooldownTime: 2.0
        )
    }
    
    // MARK: - Data Settings
    
    public struct DataSettings: Codable, Sendable {
        public let vectorCounts: [Int]
        public let dimensions: [Int]
        public let queryCounts: [Int]
        public let kValues: [Int]
        public let dataDistribution: String
        public let sparsityLevels: [Float]
        public let noiseLevel: Float
        
        public init(
            vectorCounts: [Int] = [1_000, 10_000, 100_000],
            dimensions: [Int] = [128, 256, 512],
            queryCounts: [Int] = [100, 1_000],
            kValues: [Int] = [10, 50, 100],
            dataDistribution: String = "uniform",
            sparsityLevels: [Float] = [0.0],
            noiseLevel: Float = 0.0
        ) {
            self.vectorCounts = vectorCounts
            self.dimensions = dimensions
            self.queryCounts = queryCounts
            self.kValues = kValues
            self.dataDistribution = dataDistribution
            self.sparsityLevels = sparsityLevels
            self.noiseLevel = noiseLevel
        }
        
        public static let small = DataSettings(
            vectorCounts: [100, 1_000],
            dimensions: [64, 128],
            queryCounts: [10, 100],
            kValues: [10]
        )
        
        public static let standard = DataSettings()
        
        public static let large = DataSettings(
            vectorCounts: [100_000, 1_000_000],
            dimensions: [256, 512, 768],
            queryCounts: [1_000, 10_000],
            kValues: [10, 50, 100, 500]
        )
    }
    
    // MARK: - Index Settings
    
    public struct IndexSettings: Codable, Sendable {
        public let enableHNSW: Bool
        public let enableIVF: Bool
        public let enableLearned: Bool
        public let enableHybrid: Bool
        public let hnswSettings: HNSWSettings?
        public let ivfSettings: IVFSettings?
        public let learnedSettings: LearnedSettings?
        public let hybridSettings: HybridSettings?
        
        public struct HNSWSettings: Codable, Sendable {
            public let mValues: [Int]
            public let efConstructionValues: [Int]
            public let efSearchValues: [Int]
            
            public init(
                mValues: [Int] = [16, 32],
                efConstructionValues: [Int] = [200, 400],
                efSearchValues: [Int] = [50, 100, 200]
            ) {
                self.mValues = mValues
                self.efConstructionValues = efConstructionValues
                self.efSearchValues = efSearchValues
            }
        }
        
        public struct IVFSettings: Codable, Sendable {
            public let centroidCounts: [Int]
            public let nprobeValues: [Int]
            
            public init(
                centroidCounts: [Int] = [256, 1024],
                nprobeValues: [Int] = [1, 10, 50]
            ) {
                self.centroidCounts = centroidCounts
                self.nprobeValues = nprobeValues
            }
        }
        
        public struct LearnedSettings: Codable, Sendable {
            public let modelArchitectures: [String]
            public let bucketSizes: [Int]
            
            public init(
                modelArchitectures: [String] = ["mlp", "transformer"],
                bucketSizes: [Int] = [100, 1000]
            ) {
                self.modelArchitectures = modelArchitectures
                self.bucketSizes = bucketSizes
            }
        }
        
        public struct HybridSettings: Codable, Sendable {
            public let routingStrategies: [String]
            public let adaptiveThresholds: [Float]
            
            public init(
                routingStrategies: [String] = ["static", "adaptive", "learned"],
                adaptiveThresholds: [Float] = [0.5, 0.7, 0.9]
            ) {
                self.routingStrategies = routingStrategies
                self.adaptiveThresholds = adaptiveThresholds
            }
        }
        
        public init(
            enableHNSW: Bool = true,
            enableIVF: Bool = true,
            enableLearned: Bool = true,
            enableHybrid: Bool = true,
            hnswSettings: HNSWSettings? = HNSWSettings(),
            ivfSettings: IVFSettings? = IVFSettings(),
            learnedSettings: LearnedSettings? = LearnedSettings(),
            hybridSettings: HybridSettings? = HybridSettings()
        ) {
            self.enableHNSW = enableHNSW
            self.enableIVF = enableIVF
            self.enableLearned = enableLearned
            self.enableHybrid = enableHybrid
            self.hnswSettings = hnswSettings
            self.ivfSettings = ivfSettings
            self.learnedSettings = learnedSettings
            self.hybridSettings = hybridSettings
        }
    }
    
    // MARK: - Benchmark Suites
    
    public struct BenchmarkSuites: Codable, Sendable {
        public let runVectorOperations: Bool
        public let runIndexBenchmarks: Bool
        public let runMetalAcceleration: Bool
        public let runDistributed: Bool
        public let runCache: Bool
        public let runML: Bool
        public let runScalability: Bool
        public let runConcurrency: Bool
        public let runMemory: Bool
        public let runEndToEnd: Bool
        
        public init(
            runVectorOperations: Bool = true,
            runIndexBenchmarks: Bool = true,
            runMetalAcceleration: Bool = true,
            runDistributed: Bool = true,
            runCache: Bool = true,
            runML: Bool = true,
            runScalability: Bool = true,
            runConcurrency: Bool = true,
            runMemory: Bool = true,
            runEndToEnd: Bool = true
        ) {
            self.runVectorOperations = runVectorOperations
            self.runIndexBenchmarks = runIndexBenchmarks
            self.runMetalAcceleration = runMetalAcceleration
            self.runDistributed = runDistributed
            self.runCache = runCache
            self.runML = runML
            self.runScalability = runScalability
            self.runConcurrency = runConcurrency
            self.runMemory = runMemory
            self.runEndToEnd = runEndToEnd
        }
        
        public static let minimal = BenchmarkSuites(
            runVectorOperations: true,
            runIndexBenchmarks: true,
            runMetalAcceleration: false,
            runDistributed: false,
            runCache: false,
            runML: false,
            runScalability: false,
            runConcurrency: false,
            runMemory: false,
            runEndToEnd: false
        )
        
        public static let standard = BenchmarkSuites()
        
        public static let comprehensive = BenchmarkSuites()
    }
    
    // MARK: - Properties
    
    public let general: GeneralSettings
    public let execution: ExecutionSettings
    public let data: DataSettings
    public let index: IndexSettings
    public let suites: BenchmarkSuites
    
    // MARK: - Initialization
    
    public init(
        general: GeneralSettings = GeneralSettings(),
        execution: ExecutionSettings = ExecutionSettings(),
        data: DataSettings = DataSettings(),
        index: IndexSettings = IndexSettings(),
        suites: BenchmarkSuites = BenchmarkSuites()
    ) {
        self.general = general
        self.execution = execution
        self.data = data
        self.index = index
        self.suites = suites
    }
    
    // MARK: - Presets
    
    public static let quick = BenchmarkConfiguration(
        general: GeneralSettings(
            name: "Quick Benchmarks",
            saveResults: false
        ),
        execution: ExecutionSettings.quick,
        data: DataSettings.small,
        suites: BenchmarkSuites.minimal
    )
    
    public static let standard = BenchmarkConfiguration()
    
    public static let comprehensive = BenchmarkConfiguration(
        general: GeneralSettings(
            name: "Comprehensive Benchmarks",
            compareBaseline: true
        ),
        execution: ExecutionSettings.thorough,
        data: DataSettings.large,
        suites: BenchmarkSuites.comprehensive
    )
    
    public static let ci = BenchmarkConfiguration(
        general: GeneralSettings(
            name: "CI Benchmarks",
            outputDirectory: "./ci-benchmark-results",
            timeout: 1800 // 30 minutes
        ),
        execution: ExecutionSettings(
            iterations: 5,
            warmupIterations: 1
        ),
        data: DataSettings(
            vectorCounts: [1_000, 10_000],
            dimensions: [128],
            queryCounts: [100]
        ),
        suites: BenchmarkSuites(
            runVectorOperations: true,
            runIndexBenchmarks: true,
            runMetalAcceleration: false,
            runDistributed: false,
            runCache: true,
            runML: false,
            runScalability: true,
            runConcurrency: true,
            runMemory: true,
            runEndToEnd: false
        )
    )
}

// MARK: - Configuration Loading

public extension BenchmarkConfiguration {
    
    /// Load configuration from JSON file
    static func load(from path: String) throws -> BenchmarkConfiguration {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        return try decoder.decode(BenchmarkConfiguration.self, from: data)
    }
    
    /// Save configuration to JSON file
    func save(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: URL(fileURLWithPath: path))
    }
    
    /// Load configuration from environment variables
    static func fromEnvironment() -> BenchmarkConfiguration {
        var config = BenchmarkConfiguration.standard
        
        // Override with environment variables
        if let iterations = ProcessInfo.processInfo.environment["BENCH_ITERATIONS"],
           let iterationCount = Int(iterations) {
            config = BenchmarkConfiguration(
                general: config.general,
                execution: ExecutionSettings(
                    iterations: iterationCount,
                    warmupIterations: config.execution.warmupIterations,
                    cooldownTime: config.execution.cooldownTime,
                    parallelExecution: config.execution.parallelExecution,
                    maxConcurrency: config.execution.maxConcurrency,
                    collectSystemMetrics: config.execution.collectSystemMetrics,
                    profileMemory: config.execution.profileMemory,
                    profileCPU: config.execution.profileCPU,
                    profileGPU: config.execution.profileGPU
                ),
                data: config.data,
                index: config.index,
                suites: config.suites
            )
        }
        
        return config
    }
}

// MARK: - Configuration Validation

public extension BenchmarkConfiguration {
    
    /// Validate configuration
    func validate() throws {
        // Check vector counts
        guard !data.vectorCounts.isEmpty else {
            throw ConfigurationError.invalid("Vector counts cannot be empty")
        }
        
        // Check dimensions
        guard !data.dimensions.isEmpty else {
            throw ConfigurationError.invalid("Dimensions cannot be empty")
        }
        
        guard data.dimensions.allSatisfy({ $0 > 0 && $0 <= 4096 }) else {
            throw ConfigurationError.invalid("Dimensions must be between 1 and 4096")
        }
        
        // Check iterations
        guard execution.iterations > 0 else {
            throw ConfigurationError.invalid("Iterations must be positive")
        }
        
        // Check timeout
        guard general.timeout > 0 else {
            throw ConfigurationError.invalid("Timeout must be positive")
        }
        
        // Check output directory
        if general.saveResults {
            let url = URL(fileURLWithPath: general.outputDirectory)
            if !FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.createDirectory(
                    at: url,
                    withIntermediateDirectories: true
                )
            }
        }
    }
    
    enum ConfigurationError: LocalizedError {
        case invalid(String)
        
        public var errorDescription: String? {
            switch self {
            case .invalid(let message):
                return "Invalid configuration: \(message)"
            }
        }
    }
}

// MARK: - Configuration Builder

public class BenchmarkConfigurationBuilder {
    private var general = BenchmarkConfiguration.GeneralSettings()
    private var execution = BenchmarkConfiguration.ExecutionSettings()
    private var data = BenchmarkConfiguration.DataSettings()
    private var index = BenchmarkConfiguration.IndexSettings()
    private var suites = BenchmarkConfiguration.BenchmarkSuites()
    
    public init() {}
    
    @discardableResult
    public func withName(_ name: String) -> Self {
        general = BenchmarkConfiguration.GeneralSettings(
            name: name,
            description: general.description,
            outputDirectory: general.outputDirectory,
            saveResults: general.saveResults,
            compareBaseline: general.compareBaseline,
            baselinePath: general.baselinePath,
            randomSeed: general.randomSeed,
            timeout: general.timeout
        )
        return self
    }
    
    @discardableResult
    public func withIterations(_ iterations: Int) -> Self {
        execution = BenchmarkConfiguration.ExecutionSettings(
            iterations: iterations,
            warmupIterations: execution.warmupIterations,
            cooldownTime: execution.cooldownTime,
            parallelExecution: execution.parallelExecution,
            maxConcurrency: execution.maxConcurrency,
            collectSystemMetrics: execution.collectSystemMetrics,
            profileMemory: execution.profileMemory,
            profileCPU: execution.profileCPU,
            profileGPU: execution.profileGPU
        )
        return self
    }
    
    @discardableResult
    public func withVectorCounts(_ counts: [Int]) -> Self {
        data = BenchmarkConfiguration.DataSettings(
            vectorCounts: counts,
            dimensions: data.dimensions,
            queryCounts: data.queryCounts,
            kValues: data.kValues,
            dataDistribution: data.dataDistribution,
            sparsityLevels: data.sparsityLevels,
            noiseLevel: data.noiseLevel
        )
        return self
    }
    
    @discardableResult
    public func withDimensions(_ dimensions: [Int]) -> Self {
        data = BenchmarkConfiguration.DataSettings(
            vectorCounts: data.vectorCounts,
            dimensions: dimensions,
            queryCounts: data.queryCounts,
            kValues: data.kValues,
            dataDistribution: data.dataDistribution,
            sparsityLevels: data.sparsityLevels,
            noiseLevel: data.noiseLevel
        )
        return self
    }
    
    @discardableResult
    public func withSuites(_ suites: BenchmarkConfiguration.BenchmarkSuites) -> Self {
        self.suites = suites
        return self
    }
    
    public func build() -> BenchmarkConfiguration {
        return BenchmarkConfiguration(
            general: general,
            execution: execution,
            data: data,
            index: index,
            suites: suites
        )
    }
}