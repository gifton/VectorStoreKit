// VectorStoreKit: ML Benchmarks
//
// Neural network and encoding performance benchmarks

import Foundation
import simd
import Metal
import MetalPerformanceShaders

/// Benchmarks for ML components
public struct MLBenchmarks {
    
    private let framework: BenchmarkFramework
    private let metrics: PerformanceMetrics
    private let device: MTLDevice
    
    public init(
        framework: BenchmarkFramework = BenchmarkFramework(),
        metrics: PerformanceMetrics = PerformanceMetrics()
    ) throws {
        self.framework = framework
        self.metrics = metrics
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.metalNotAvailable
        }
        self.device = device
    }
    
    enum BenchmarkError: Error {
        case metalNotAvailable
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all ML benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Neural network benchmarks
        results.merge(try await runNeuralNetworkBenchmarks()) { _, new in new }
        
        // Autoencoder benchmarks
        results.merge(try await runAutoencoderBenchmarks()) { _, new in new }
        
        // Neural clustering benchmarks
        results.merge(try await runNeuralClusteringBenchmarks()) { _, new in new }
        
        // Layer operation benchmarks
        results.merge(try await runLayerOperationBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Neural Network Benchmarks
    
    private func runNeuralNetworkBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Neural Network Operations",
            description: "Benchmarks for neural network inference and training"
        ) {
            let batchSizes = [1, 32, 128, 512]
            let dimensions = [128, 256, 512]
            
            for batchSize in batchSizes {
                for dim in dimensions {
                    // Forward pass benchmark
                    benchmark(name: "nn_forward_batch\(batchSize)_dim\(dim)") {
                        let network = try await createTestNetwork(inputDim: dim)
                        let input = generateBatch(batchSize: batchSize, dimensions: dim)
                        
                        // Process batch samples individually
                        for sample in input {
                            let output = try await network.forward(sample)
                            blackHole(output)
                        }
                    }
                    
                    // Backward pass benchmark
                    benchmark(name: "nn_backward_batch\(batchSize)_dim\(dim)") {
                        let network = try await createTestNetwork(inputDim: dim)
                        let input = generateBatch(batchSize: batchSize, dimensions: dim)
                        
                        // Process each sample in the batch
                        for sample in input {
                            let output = try await network.forward(sample)
                            // Generate gradient of same size as output
                            let gradient = (0..<output.count).map { _ in Float.random(in: -0.1...0.1) }
                            let gradients = await network.backward(gradient, learningRate: 0.01)
                            blackHole(gradients)
                        }
                    }
                }
            }
            
            // Optimization benchmarks
            benchmark(name: "optimizer_adam") {
                try await benchmarkOptimizer(type: .adam)
            }
            
            benchmark(name: "optimizer_sgd") {
                try await benchmarkOptimizer(type: .sgd)
            }
            
            benchmark(name: "optimizer_rmsprop") {
                try await benchmarkOptimizer(type: .rmsprop)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Autoencoder Benchmarks
    
    private func runAutoencoderBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Autoencoder Operations",
            description: "Benchmarks for different autoencoder variants"
        ) {
            let dimensions = [128, 256, 512]
            let batchSize = 32
            
            for dim in dimensions {
                // Standard autoencoder
                benchmark(name: "autoencoder_standard_\(dim)d") {
                    let autoencoder = try Autoencoder(
                        inputDimension: dim,
                        hiddenDimensions: [dim/2, dim/4],
                        outputDimension: dim
                    )
                    
                    let input = generateBatch(batchSize: batchSize, dimensions: dim)
                    let encoded = try await autoencoder.encode(input)
                    let decoded = try await autoencoder.decode(encoded)
                    blackHole(decoded)
                }
                
                // Variational autoencoder
                benchmark(name: "autoencoder_variational_\(dim)d") {
                    let vae = try VariationalAutoencoder(
                        inputDimension: dim,
                        latentDimension: dim/4
                    )
                    
                    let input = generateBatch(batchSize: batchSize, dimensions: dim)
                    let result = try await vae.forward(input)
                    blackHole(result)
                }
                
                // Sparse autoencoder
                benchmark(name: "autoencoder_sparse_\(dim)d") {
                    let sparse = try SparseAutoencoder(
                        inputDimension: dim,
                        hiddenDimension: dim/2,
                        sparsityParam: 0.05
                    )
                    
                    let input = generateBatch(batchSize: batchSize, dimensions: dim)
                    let result = try await sparse.forward(input)
                    blackHole(result)
                }
                
                // Denoising autoencoder
                benchmark(name: "autoencoder_denoising_\(dim)d") {
                    let denoising = try DenoisingAutoencoder(
                        inputDimension: dim,
                        noiseLevel: 0.1
                    )
                    
                    let input = generateBatch(batchSize: batchSize, dimensions: dim)
                    let result = try await denoising.forward(input)
                    blackHole(result)
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Neural Clustering Benchmarks
    
    private func runNeuralClusteringBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Neural Clustering",
            description: "Benchmarks for neural clustering algorithms"
        ) {
            let vectorCounts = [1_000, 10_000, 100_000]
            let dimensions = 128
            let clusterCounts = [10, 50, 100]
            
            for count in vectorCounts {
                for clusters in clusterCounts {
                    benchmark(name: "neural_clustering_\(count)vectors_\(clusters)clusters") {
                        let config = NeuralClusteringConfiguration(
                            epochs: 10,
                            batchSize: 32
                        )
                        
                        let clustering = try await NeuralClustering(configuration: config)
                        let vectors = generateVectors(count: count, dimensions: dimensions)
                        
                        let assignments = try await clustering.fit(vectors)
                        blackHole(assignments)
                    }
                }
            }
            
            // Incremental clustering
            benchmark(name: "incremental_neural_clustering") {
                let config = NeuralClusteringConfiguration(
                    dimensions: dimensions,
                    numberOfClusters: 50,
                    enableIncremental: true
                )
                
                let clustering = try await NeuralClustering(configuration: config)
                
                // Initial training
                let initialVectors = generateVectors(count: 10_000, dimensions: dimensions)
                _ = try await clustering.fit(initialVectors)
                
                // Incremental updates
                for _ in 0..<10 {
                    let newVectors = generateVectors(count: 1_000, dimensions: dimensions)
                    try await clustering.incrementalUpdate(newVectors)
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Layer Operation Benchmarks
    
    private func runLayerOperationBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Layer Operations",
            description: "Benchmarks for individual layer types"
        ) {
            let batchSize = 32
            
            // Dense layer
            benchmark(name: "layer_dense_512x256") {
                let device = try MetalDevice()
                let metalPipeline = try await MetalMLPipeline(device: device)
                let layer = try await DenseLayer(
                    inputSize: 512,
                    outputSize: 256,
                    activation: .relu,
                    metalPipeline: metalPipeline
                )
                
                let input = generateBatch(batchSize: batchSize, dimensions: 512)
                // Process each sample
                for sample in input {
                    let output = try await layer.forward(sample)
                    blackHole(output)
                }
            }
            
            
            // Batch normalization
            benchmark(name: "layer_batchnorm") {
                let layer = try BatchNormLayer(
                    numFeatures: 256
                )
                
                let input = generateBatch(batchSize: batchSize, dimensions: 256)
                let output = try await layer.forward(input, training: true)
                blackHole(output)
            }
            
            // Dropout
            benchmark(name: "layer_dropout") {
                let layer = try DropoutLayer(
                    dropoutRate: 0.5
                )
                
                let input = generateBatch(batchSize: batchSize, dimensions: 512)
                let output = try await layer.forward(input, training: true)
                blackHole(output)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func createTestNetwork(inputDim: Int) async throws -> NeuralNetwork {
        let config = NeuralNetwork.Configuration(
            inputDimension: inputDim,
            layers: [
                .dense(outputSize: inputDim / 2, activation: .relu),
                .batchNorm(numFeatures: inputDim / 2),
                .dropout(rate: 0.5),
                .dense(outputSize: inputDim / 4, activation: .relu),
                .dense(outputSize: 64, activation: .none)
            ],
            optimizer: .adam(learningRate: 0.001),
            device: device
        )
        
        return try NeuralNetwork(configuration: config)
    }
    
    private func benchmarkOptimizer(type: OptimizerType) async throws {
        let dimensions = 10_000
        let parameters = generateParameters(count: dimensions)
        let gradients = generateParameters(count: dimensions)
        
        let optimizer = createOptimizer(type: type)
        
        for _ in 0..<100 {
            var updatedParameters: [Float] = []
            for (i, (param, grad)) in zip(parameters, gradients).enumerated() {
                let updated = await optimizer.update(
                    parameter: param,
                    gradient: grad,
                    name: "param_\(i)"
                )
                updatedParameters.append(updated)
            }
            blackHole(updatedParameters)
        }
    }
    
    private func createOptimizer(type: OptimizerType) -> Optimizer {
        switch type {
        case .adam:
            return AdamOptimizer(learningRate: 0.001)
        case .sgd:
            return SGDOptimizer(learningRate: 0.01, momentum: 0.9)
        case .rmsprop:
            return RMSPropOptimizer(learningRate: 0.001)
        }
    }
    
    private func generateBatch(batchSize: Int, dimensions: Int) -> [[Float]] {
        return (0..<batchSize).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    private func generateVectors(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    private func generateImageBatch(batchSize: Int, channels: Int, height: Int, width: Int) -> [[[[Float]]]] {
        return (0..<batchSize).map { _ in
            (0..<channels).map { _ in
                (0..<height).map { _ in
                    (0..<width).map { _ in Float.random(in: 0...1) }
                }
            }
        }
    }
    
    private func generateSequenceBatch(batchSize: Int, sequenceLength: Int, dimensions: Int) -> [[[Float]]] {
        return (0..<batchSize).map { _ in
            (0..<sequenceLength).map { _ in
                (0..<dimensions).map { _ in Float.random(in: -1...1) }
            }
        }
    }
    
    private func generateParameters(count: Int) -> [Float] {
        return (0..<count).map { _ in Float.random(in: -1...1) }
    }
}

// MARK: - Optimizer Types

private enum OptimizerType {
    case adam
    case sgd
    case rmsprop
}

// MARK: - ML Performance Report

public extension MLBenchmarks {
    
    struct MLPerformanceReport {
        public let component: String
        public let configuration: String
        public let forwardPassTime: TimeInterval
        public let backwardPassTime: TimeInterval?
        public let throughput: Double // samples/second
        public let memoryUsage: Int
        public let gpuUtilization: Double
        
        public var summary: String {
            var report = """
                Component: \(component)
                Configuration: \(configuration)
                  Forward Pass: \(formatTime(forwardPassTime))
                  Throughput: \(formatNumber(throughput)) samples/s
                  Memory: \(formatBytes(memoryUsage))
                  GPU Usage: \(String(format: "%.1f%%", gpuUtilization * 100))
                """
            
            if let backward = backwardPassTime {
                report += "\n  Backward Pass: \(formatTime(backward))"
            }
            
            return report
        }
        
        private func formatTime(_ seconds: Double) -> String {
            if seconds < 0.001 {
                return String(format: "%.2f μs", seconds * 1_000_000)
            } else if seconds < 1.0 {
                return String(format: "%.2f ms", seconds * 1_000)
            } else {
                return String(format: "%.2f s", seconds)
            }
        }
        
        private func formatNumber(_ number: Double) -> String {
            if number >= 1_000_000 {
                return String(format: "%.2fM", number / 1_000_000)
            } else if number >= 1_000 {
                return String(format: "%.2fK", number / 1_000)
            } else {
                return String(format: "%.2f", number)
            }
        }
        
        private func formatBytes(_ bytes: Int) -> String {
            let units = ["B", "KB", "MB", "GB"]
            var size = Double(bytes)
            var unitIndex = 0
            
            while size >= 1024 && unitIndex < units.count - 1 {
                size /= 1024
                unitIndex += 1
            }
            
            return String(format: "%.1f %@", size, units[unitIndex])
        }
    }
}