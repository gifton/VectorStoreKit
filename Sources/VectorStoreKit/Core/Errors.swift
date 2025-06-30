import Foundation

/// Comprehensive error type system for VectorStoreKit
public struct VectorStoreError: Error, LocalizedError, CustomStringConvertible {
    /// The category of error that occurred
    public let category: ErrorCategory
    
    /// Detailed error code for programmatic handling
    public let code: ErrorCode
    
    /// Human-readable description of the error
    public let message: String
    
    /// Additional context about the error
    public let context: ErrorContext
    
    /// The underlying error if this error wraps another
    public let underlyingError: Error?
    
    /// File where the error occurred
    public let file: String
    
    /// Line number where the error occurred
    public let line: Int
    
    /// Function where the error occurred
    public let function: String
    
    /// Timestamp when the error occurred
    public let timestamp: Date
    
    /// Unique identifier for this error instance
    public let errorID: UUID
    
    public init(
        category: ErrorCategory,
        code: ErrorCode,
        message: String,
        context: ErrorContext = [:],
        underlyingError: Error? = nil,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) {
        self.category = category
        self.code = code
        self.message = message
        self.context = context
        self.underlyingError = underlyingError
        self.file = URL(fileURLWithPath: file).lastPathComponent
        self.line = line
        self.function = function
        self.timestamp = Date()
        self.errorID = UUID()
    }
    
    // MARK: - LocalizedError
    
    public var errorDescription: String? {
        var components = ["\(category) Error [\(code)]"]
        components.append(message)
        
        if !context.isEmpty {
            let contextString = context.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
            components.append("Context: {\(contextString)}")
        }
        
        if let underlying = underlyingError {
            components.append("Underlying: \(underlying)")
        }
        
        components.append("Location: \(file):\(line) in \(function)")
        
        return components.joined(separator: "\n")
    }
    
    public var failureReason: String? {
        message
    }
    
    public var recoverySuggestion: String? {
        code.recoverySuggestion
    }
    
    // MARK: - CustomStringConvertible
    
    public var description: String {
        errorDescription ?? "Unknown VectorStoreError"
    }
}

// MARK: - Error Categories

public enum ErrorCategory: String, CaseIterable {
    case memoryAllocation = "Memory Allocation"
    case distanceComputation = "Distance Computation"
    case indexOperation = "Index Operation"
    case batchProcessing = "Batch Processing"
    case configuration = "Configuration"
    case storage = "Storage"
    case concurrency = "Concurrency"
    case metalCompute = "Metal Compute"
    case validation = "Validation"
    case serialization = "Serialization"
    case network = "Network"
    case initialization = "Initialization"
}

// MARK: - Error Codes

public enum ErrorCode: String, CaseIterable {
    // Memory Allocation Errors
    case insufficientMemory = "MEMORY_001"
    case allocationFailed = "MEMORY_002"
    case bufferOverflow = "MEMORY_003"
    case memoryLeak = "MEMORY_004"
    case alignmentViolation = "MEMORY_005"
    
    // Distance Computation Errors
    case dimensionMismatch = "DISTANCE_001"
    case invalidMetric = "DISTANCE_002"
    case numericalOverflow = "DISTANCE_003"
    case nanOrInfinite = "DISTANCE_004"
    case unsupportedVectorType = "DISTANCE_005"
    
    // Index Operation Errors
    case indexFull = "INDEX_001"
    case indexCorrupted = "INDEX_002"
    case vectorNotFound = "INDEX_003"
    case duplicateVector = "INDEX_004"
    case indexNotInitialized = "INDEX_005"
    case invalidIndexParameters = "INDEX_006"
    case rebuildRequired = "INDEX_007"
    
    // Batch Processing Errors
    case batchSizeExceeded = "BATCH_001"
    case batchValidationFailed = "BATCH_002"
    case partialBatchFailure = "BATCH_003"
    case batchTimeout = "BATCH_004"
    case batchCancelled = "BATCH_005"
    
    // Configuration Errors
    case invalidConfiguration = "CONFIG_001"
    case missingRequiredParameter = "CONFIG_002"
    case parameterOutOfRange = "CONFIG_003"
    case incompatibleSettings = "CONFIG_004"
    case configurationNotFound = "CONFIG_005"
    
    // Storage Errors
    case storageUnavailable = "STORAGE_001"
    case readFailure = "STORAGE_002"
    case writeFailure = "STORAGE_003"
    case corruptedData = "STORAGE_004"
    case insufficientSpace = "STORAGE_005"
    case migrationRequired = "STORAGE_006"
    
    // Concurrency Errors
    case deadlock = "CONCUR_001"
    case raceCondition = "CONCUR_002"
    case actorIsolationViolation = "CONCUR_003"
    case taskCancelled = "CONCUR_004"
    case timeout = "CONCUR_005"
    
    // Metal Compute Errors
    case deviceNotAvailable = "METAL_001"
    case shaderCompilationFailed = "METAL_002"
    case bufferCreationFailed = "METAL_003"
    case kernelExecutionFailed = "METAL_004"
    case textureCreationFailed = "METAL_005"
    case commandBufferFailed = "METAL_006"
    
    // Validation Errors
    case invalidInput = "VALID_001"
    case preconditionFailed = "VALID_002"
    case postconditionFailed = "VALID_003"
    case invariantViolation = "VALID_004"
    case typeValidationFailed = "VALID_005"
    
    // Serialization Errors
    case encodingFailed = "SERIAL_001"
    case decodingFailed = "SERIAL_002"
    case versionMismatch = "SERIAL_003"
    case formatNotSupported = "SERIAL_004"
    case dataCorrupted = "SERIAL_005"
    
    // Network Errors
    case connectionFailed = "NET_001"
    case requestTimeout = "NET_002"
    case responseInvalid = "NET_003"
    case networkUnavailable = "NET_004"
    case authenticationFailed = "NET_005"
    
    // Initialization Errors
    case initializationFailed = "INIT_001"
    case dependencyMissing = "INIT_002"
    case environmentInvalid = "INIT_003"
    case resourcesUnavailable = "INIT_004"
    
    // Store Operation Errors
    case insertionFailed = "STORE_001"
    case storeNotReady = "STORE_002" 
    case validationFailed = "STORE_003"
    case storageOperationFailed = "STORE_004"
    case exportOperationFailed = "STORE_005"
    
    var recoverySuggestion: String? {
        switch self {
        // Memory Allocation
        case .insufficientMemory:
            return "Free up memory by reducing batch sizes or closing other applications"
        case .allocationFailed:
            return "Check system memory availability and reduce allocation size"
        case .bufferOverflow:
            return "Verify buffer sizes and input data dimensions"
        case .memoryLeak:
            return "Review resource management and ensure proper cleanup"
        case .alignmentViolation:
            return "Ensure memory alignment requirements are met for SIMD operations"
            
        // Distance Computation
        case .dimensionMismatch:
            return "Ensure all vectors have the same dimension"
        case .invalidMetric:
            return "Use a supported distance metric (euclidean, cosine, dot)"
        case .numericalOverflow:
            return "Normalize vectors or use a different precision level"
        case .nanOrInfinite:
            return "Check input vectors for NaN or infinite values"
        case .unsupportedVectorType:
            return "Convert vectors to a supported type (Float32, Float16)"
            
        // Index Operations
        case .indexFull:
            return "Increase index capacity or remove old vectors"
        case .indexCorrupted:
            return "Rebuild the index from scratch"
        case .vectorNotFound:
            return "Verify the vector ID exists in the index"
        case .duplicateVector:
            return "Remove duplicate before insertion or use update operation"
        case .indexNotInitialized:
            return "Initialize the index before performing operations"
        case .invalidIndexParameters:
            return "Check index parameters against documentation"
        case .rebuildRequired:
            return "Rebuild the index to incorporate recent changes"
            
        // Batch Processing
        case .batchSizeExceeded:
            return "Reduce batch size or increase system limits"
        case .batchValidationFailed:
            return "Verify all items in the batch meet requirements"
        case .partialBatchFailure:
            return "Check individual items for errors and retry failed ones"
        case .batchTimeout:
            return "Increase timeout or reduce batch size"
        case .batchCancelled:
            return "Check for cancellation reasons and retry if needed"
            
        // Configuration
        case .invalidConfiguration:
            return "Review configuration against schema requirements"
        case .missingRequiredParameter:
            return "Provide all required configuration parameters"
        case .parameterOutOfRange:
            return "Adjust parameter to be within valid range"
        case .incompatibleSettings:
            return "Review setting combinations for compatibility"
        case .configurationNotFound:
            return "Create configuration file or use defaults"
            
        // Storage
        case .storageUnavailable:
            return "Check storage permissions and availability"
        case .readFailure:
            return "Verify file exists and has read permissions"
        case .writeFailure:
            return "Check write permissions and available disk space"
        case .corruptedData:
            return "Restore from backup or rebuild data"
        case .insufficientSpace:
            return "Free up disk space or use external storage"
        case .migrationRequired:
            return "Run migration tool to update data format"
            
        // Concurrency
        case .deadlock:
            return "Review actor isolation and async patterns"
        case .raceCondition:
            return "Add proper synchronization mechanisms"
        case .actorIsolationViolation:
            return "Ensure actor-isolated properties are accessed correctly"
        case .taskCancelled:
            return "Check cancellation reason and retry if appropriate"
        case .timeout:
            return "Increase timeout or optimize operation performance"
            
        // Metal Compute
        case .deviceNotAvailable:
            return "Ensure Metal-capable device is available"
        case .shaderCompilationFailed:
            return "Check shader syntax and Metal version compatibility"
        case .bufferCreationFailed:
            return "Reduce buffer size or check device limits"
        case .kernelExecutionFailed:
            return "Verify kernel parameters and input data"
        case .textureCreationFailed:
            return "Check texture format and dimensions"
        case .commandBufferFailed:
            return "Review Metal command sequence for errors"
            
        // Validation
        case .invalidInput:
            return "Verify input meets validation requirements"
        case .preconditionFailed:
            return "Ensure preconditions are met before operation"
        case .postconditionFailed:
            return "Operation completed but results are invalid"
        case .invariantViolation:
            return "Internal consistency check failed - report bug"
        case .typeValidationFailed:
            return "Ensure data types match expected schema"
            
        // Serialization
        case .encodingFailed:
            return "Check data is encodable to target format"
        case .decodingFailed:
            return "Verify data format matches decoder expectations"
        case .versionMismatch:
            return "Update to compatible version or migrate data"
        case .formatNotSupported:
            return "Use a supported serialization format"
        case .dataCorrupted:
            return "Data integrity check failed - use backup"
            
        // Network
        case .connectionFailed:
            return "Check network connectivity and server status"
        case .requestTimeout:
            return "Increase timeout or check server response time"
        case .responseInvalid:
            return "Server returned unexpected response format"
        case .networkUnavailable:
            return "Wait for network connectivity to be restored"
        case .authenticationFailed:
            return "Verify credentials and authentication method"
            
        // Initialization
        case .initializationFailed:
            return "Check initialization parameters and system state"
        case .dependencyMissing:
            return "Install required dependencies"
        case .environmentInvalid:
            return "Verify environment meets system requirements"
        case .resourcesUnavailable:
            return "Ensure required resources are accessible"
            
        // Store Operations
        case .insertionFailed:
            return "Check vector format and index capacity"
        case .storeNotReady:
            return "Wait for initialization to complete or check store status"
        case .validationFailed:
            return "Verify input data meets schema requirements"
        case .storageOperationFailed:
            return "Check storage permissions and available space"
        case .exportOperationFailed:
            return "Verify export path and format compatibility"
        }
    }
}

// MARK: - Error Context

public typealias ErrorContext = [String: Any]

// MARK: - Error Creation Helpers

public extension VectorStoreError {
    // Memory Allocation Helpers
    static func insufficientMemory(
        required: Int,
        available: Int,
        operation: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .memoryAllocation,
            code: .insufficientMemory,
            message: "Insufficient memory for \(operation)",
            context: [
                "requiredBytes": required,
                "availableBytes": available,
                "operation": operation
            ],
            file: file,
            line: line,
            function: function
        )
    }
    
    // Distance Computation Helpers
    static func dimensionMismatch(
        expected: Int,
        actual: Int,
        vectorID: String? = nil,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        var context: ErrorContext = [
            "expectedDimension": expected,
            "actualDimension": actual
        ]
        if let id = vectorID {
            context["vectorID"] = id
        }
        
        return VectorStoreError(
            category: .distanceComputation,
            code: .dimensionMismatch,
            message: "Vector dimension mismatch: expected \(expected), got \(actual)",
            context: context,
            file: file,
            line: line,
            function: function
        )
    }
    
    // Index Operation Helpers
    static func indexFull(
        capacity: Int,
        attempted: Int,
        indexType: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .indexOperation,
            code: .indexFull,
            message: "Index is at full capacity",
            context: [
                "maxCapacity": capacity,
                "attemptedSize": attempted,
                "indexType": indexType
            ],
            file: file,
            line: line,
            function: function
        )
    }
    
    // Batch Processing Helpers
    static func batchSizeExceeded(
        maxSize: Int,
        actualSize: Int,
        operation: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .batchProcessing,
            code: .batchSizeExceeded,
            message: "Batch size exceeds maximum allowed",
            context: [
                "maxBatchSize": maxSize,
                "actualBatchSize": actualSize,
                "operation": operation
            ],
            file: file,
            line: line,
            function: function
        )
    }
    
    // Configuration Helpers
    static func missingParameter(
        parameter: String,
        component: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .configuration,
            code: .missingRequiredParameter,
            message: "Missing required parameter '\(parameter)' for \(component)",
            context: [
                "parameter": parameter,
                "component": component
            ],
            file: file,
            line: line,
            function: function
        )
    }
    
    // Metal Compute Helpers
    static func metalDeviceUnavailable(
        requiredFeatures: [String] = [],
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .metalCompute,
            code: .deviceNotAvailable,
            message: "Metal device not available or doesn't support required features",
            context: [
                "requiredFeatures": requiredFeatures
            ],
            file: file,
            line: line,
            function: function
        )
    }
    
    // Validation Helpers
    static func invalidInput<T>(
        value: T,
        reason: String,
        validRange: String? = nil,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        var context: ErrorContext = [
            "invalidValue": String(describing: value),
            "reason": reason
        ]
        if let range = validRange {
            context["validRange"] = range
        }
        
        return VectorStoreError(
            category: .validation,
            code: .invalidInput,
            message: "Invalid input: \(reason)",
            context: context,
            file: file,
            line: line,
            function: function
        )
    }
    
    // Configuration Helpers - Additional
    static func configurationInvalid(
        _ reason: String,
        component: String? = nil,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        var context: ErrorContext = [
            "reason": reason
        ]
        if let comp = component {
            context["component"] = comp
        }
        
        return VectorStoreError(
            category: .configuration,
            code: .invalidConfiguration,
            message: reason,
            context: context,
            file: file,
            line: line,
            function: function
        )
    }
    
    // Store Operation Helpers
    static func insertion(
        reason: String,
        vectorId: String? = nil,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        var context: ErrorContext = ["reason": reason]
        if let id = vectorId {
            context["vectorId"] = id
        }
        
        return VectorStoreError(
            category: .indexOperation,
            code: .insertionFailed,
            message: "Vector insertion failed: \(reason)",
            context: context,
            file: file,
            line: line,
            function: function
        )
    }

    static func notReady(
        component: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .initialization,
            code: .storeNotReady,
            message: "Store component not ready: \(component)",
            context: ["component": component],
            file: file,
            line: line,
            function: function
        )
    }

    static func validation(
        field: String,
        value: Any,
        reason: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .validation,
            code: .validationFailed,
            message: "Validation failed for \(field): \(reason)",
            context: [
                "field": field,
                "value": String(describing: value),
                "reason": reason
            ],
            file: file,
            line: line,
            function: function
        )
    }

    static func storageError(
        operation: String,
        reason: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .storage,
            code: .storageOperationFailed,
            message: "Storage operation failed: \(operation)",
            context: [
                "operation": operation,
                "reason": reason
            ],
            file: file,
            line: line,
            function: function
        )
    }

    static func exportError(
        format: String,
        reason: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .serialization,
            code: .exportOperationFailed,
            message: "Export operation failed for format \(format): \(reason)",
            context: [
                "format": format,
                "reason": reason
            ],
            file: file,
            line: line,
            function: function
        )
    }
}

// MARK: - Error Analysis

public extension VectorStoreError {
    /// Severity level of the error
    enum Severity {
        case critical   // System cannot continue
        case high      // Operation failed, needs immediate attention
        case medium    // Operation failed but recoverable
        case low       // Warning or informational
    }
    
    var severity: Severity {
        switch code {
        case .indexCorrupted, .deadlock, .memoryLeak:
            return .critical
        case .allocationFailed, .deviceNotAvailable, .initializationFailed:
            return .high
        case .batchTimeout, .vectorNotFound, .configurationNotFound:
            return .medium
        case .duplicateVector, .batchCancelled:
            return .low
        default:
            return .medium
        }
    }
    
    /// Whether this error is recoverable
    var isRecoverable: Bool {
        switch code {
        case .indexCorrupted, .deadlock, .memoryLeak, .invariantViolation:
            return false
        default:
            return true
        }
    }
    
    /// Whether this error should be retried
    var shouldRetry: Bool {
        switch code {
        case .timeout, .networkUnavailable, .connectionFailed, .taskCancelled:
            return true
        default:
            return false
        }
    }
    
    /// Suggested retry delay in seconds
    var retryDelay: TimeInterval? {
        guard shouldRetry else { return nil }
        
        switch code {
        case .timeout:
            return 5.0
        case .networkUnavailable:
            return 30.0
        case .connectionFailed:
            return 10.0
        case .taskCancelled:
            return 1.0
        default:
            return nil
        }
    }
}

// MARK: - Error Logging

public extension VectorStoreError {
    /// Format error for structured logging
    var logFormat: [String: Any] {
        var log: [String: Any] = [
            "errorID": errorID.uuidString,
            "timestamp": ISO8601DateFormatter().string(from: timestamp),
            "category": category.rawValue,
            "code": code.rawValue,
            "message": message,
            "severity": String(describing: severity),
            "file": file,
            "line": line,
            "function": function,
            "isRecoverable": isRecoverable,
            "shouldRetry": shouldRetry
        ]
        
        if !context.isEmpty {
            log["context"] = context
        }
        
        if let retryDelay = retryDelay {
            log["retryDelay"] = retryDelay
        }
        
        if let underlying = underlyingError {
            log["underlyingError"] = String(describing: underlying)
        }
        
        return log
    }
}

// MARK: - Result Extensions

public extension Result where Failure == VectorStoreError {
    /// Create a failure result with a VectorStoreError
    static func failure(
        _ error: VectorStoreError
    ) -> Result<Success, VectorStoreError> {
        Result<Success, VectorStoreError>.failure(error)
    }
    
    /// Map any error to VectorStoreError
    static func catching(
        category: ErrorCategory,
        code: ErrorCode,
        message: String,
        context: ErrorContext = [:],
        file: String = #file,
        line: Int = #line,
        function: String = #function,
        body: () throws -> Success
    ) -> Result<Success, VectorStoreError> {
        do {
            return .success(try body())
        } catch let error as VectorStoreError {
            return .failure(error)
        } catch {
            return .failure(VectorStoreError(
                category: category,
                code: code,
                message: message,
                context: context,
                underlyingError: error,
                file: file,
                line: line,
                function: function
            ))
        }
    }
}