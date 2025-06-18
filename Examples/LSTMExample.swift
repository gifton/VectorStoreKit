// VectorStoreKit: LSTM Example
//
// Demonstrates using LSTM layers for sequence processing in vector databases

import Foundation
import Metal
import VectorStoreKit

@main
struct LSTMExample {
    static func main() async throws {
        print("=== VectorStoreKit LSTM Example ===\n")
        
        // Initialize Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        // Create ML pipeline
        let metalPipeline = try MetalMLPipeline(device: device)
        
        // Example 1: Basic LSTM for sequence classification
        try await sequenceClassificationExample(metalPipeline: metalPipeline)
        
        // Example 2: LSTM for vector sequence encoding
        try await vectorSequenceEncodingExample(metalPipeline: metalPipeline)
        
        // Example 3: Bidirectional LSTM simulation
        try await bidirectionalLSTMExample(metalPipeline: metalPipeline)
    }
    
    // MARK: - Example 1: Sequence Classification
    
    static func sequenceClassificationExample(metalPipeline: MetalMLPipeline) async throws {
        print("Example 1: Sequence Classification with LSTM")
        print("=" * 40)
        
        // Configuration
        let inputSize = 128  // Vector dimension
        let hiddenSize = 64
        let sequenceLength = 10
        let batchSize = 1
        
        // Create LSTM layer
        let lstmConfig = LSTMConfig(
            hiddenSize: hiddenSize,
            returnSequences: false,  // Only return last hidden state
            dropout: 0.0
        )
        
        let lstm = try await LSTMLayer(
            inputSize: inputSize,
            config: lstmConfig,
            name: "sequence_classifier",
            metalPipeline: metalPipeline
        )
        
        // Create input sequence (10 timesteps, each with 128 features)
        var inputData: [Float] = []
        for t in 0..<sequenceLength {
            for i in 0..<inputSize {
                // Generate some pattern that changes over time
                inputData.append(Float(sin(Double(t) * 0.1 + Double(i) * 0.01)))
            }
        }
        
        // Convert to Metal buffer
        let inputBuffer = try MetalBuffer(device: device, array: inputData)
        
        // Forward pass
        print("Processing sequence of length \(sequenceLength)...")
        let startTime = Date()
        
        let output = try await lstm.forward(inputBuffer)
        
        let processingTime = Date().timeIntervalSince(startTime)
        print("Processing time: \(String(format: "%.3f", processingTime * 1000))ms")
        
        // Output is the final hidden state
        print("Output shape: [\(batchSize), \(hiddenSize)]")
        
        // Extract and display first few values
        let outputArray = output.toArray()
        print("First 10 output values: \(outputArray.prefix(10).map { String(format: "%.4f", $0) })")
        print()
    }
    
    // MARK: - Example 2: Vector Sequence Encoding
    
    static func vectorSequenceEncodingExample(metalPipeline: MetalMLPipeline) async throws {
        print("\nExample 2: Vector Sequence Encoding")
        print("=" * 40)
        
        // Configuration for encoding variable-length sequences
        let inputSize = 256
        let hiddenSize = 128
        let maxSequenceLength = 20
        
        // Create LSTM with sequence output
        let lstmConfig = LSTMConfig(
            hiddenSize: hiddenSize,
            returnSequences: true,  // Return all hidden states
            dropout: 0.0
        )
        
        let lstm = try await LSTMLayer(
            inputSize: inputSize,
            config: lstmConfig,
            name: "sequence_encoder",
            metalPipeline: metalPipeline
        )
        
        // Simulate encoding multiple sequences
        let sequences = [5, 10, 15, 20]  // Different sequence lengths
        
        for seqLen in sequences {
            // Generate random sequence
            var inputData: [Float] = []
            for _ in 0..<seqLen {
                for _ in 0..<inputSize {
                    inputData.append(Float.random(in: -1...1))
                }
            }
            
            let inputBuffer = try MetalBuffer(device: device, array: inputData)
            
            // Process sequence
            let startTime = Date()
            let output = try await lstm.forward(inputBuffer)
            let processingTime = Date().timeIntervalSince(startTime)
            
            print("Sequence length: \(seqLen)")
            print("  Input shape: [\(seqLen), 1, \(inputSize)]")
            print("  Output shape: [\(seqLen), 1, \(hiddenSize)]")
            print("  Processing time: \(String(format: "%.3f", processingTime * 1000))ms")
            print("  Throughput: \(String(format: "%.0f", Double(seqLen) / processingTime)) sequences/sec")
        }
        print()
    }
    
    // MARK: - Example 3: Bidirectional LSTM Simulation
    
    static func bidirectionalLSTMExample(metalPipeline: MetalMLPipeline) async throws {
        print("\nExample 3: Bidirectional LSTM (Simulated)")
        print("=" * 40)
        
        // Create forward and backward LSTMs
        let inputSize = 128
        let hiddenSize = 64
        let sequenceLength = 15
        
        let config = LSTMConfig(
            hiddenSize: hiddenSize,
            returnSequences: true
        )
        
        let forwardLSTM = try await LSTMLayer(
            inputSize: inputSize,
            config: config,
            name: "forward_lstm",
            metalPipeline: metalPipeline
        )
        
        let backwardLSTM = try await LSTMLayer(
            inputSize: inputSize,
            config: config,
            name: "backward_lstm",
            metalPipeline: metalPipeline
        )
        
        // Generate input sequence
        var inputData: [Float] = []
        for t in 0..<sequenceLength {
            for i in 0..<inputSize {
                inputData.append(Float(cos(Double(t) * 0.2) * sin(Double(i) * 0.1)))
            }
        }
        
        let inputBuffer = try MetalBuffer(device: device, array: inputData)
        
        // Process forward
        print("Processing forward direction...")
        let forwardOutput = try await forwardLSTM.forward(inputBuffer)
        
        // For backward, we'd need to reverse the sequence
        // This is a simplified simulation
        print("Processing backward direction...")
        let backwardOutput = try await backwardLSTM.forward(inputBuffer)
        
        print("Forward output shape: [1, \(sequenceLength), \(hiddenSize)]")
        print("Backward output shape: [1, \(sequenceLength), \(hiddenSize)]")
        print("Combined output would be: [1, \(sequenceLength), \(hiddenSize * 2)]")
        
        // Performance metrics
        let totalParams = await forwardLSTM.getParameterCount() + await backwardLSTM.getParameterCount()
        print("\nTotal parameters: \(totalParams)")
        print("Memory usage: ~\(totalParams * 4 / 1024 / 1024) MB")
    }
}

// MARK: - Helper Extensions

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// MARK: - Usage in Vector Store Context

/*
 LSTM layers can be used in vector databases for:
 
 1. **Temporal Vector Sequences**: When vectors represent time series data
    - Financial market embeddings over time
    - User behavior sequences
    - Sensor data embeddings
 
 2. **Document Sequence Processing**: For hierarchical document understanding
    - Sentence embeddings → paragraph embeddings
    - Section embeddings → document embedding
 
 3. **Query Understanding**: Processing multi-turn conversations
    - Chat history encoding
    - Context-aware query reformulation
 
 4. **Similarity Learning**: Learning temporal patterns in similarity
    - How vector similarities change over time
    - Detecting drift in embedding spaces
 
 Example integration with VectorStore:
 
 ```swift
 // Create a temporal-aware vector store
 let store = VectorStore(
     collection: "temporal_vectors",
     indexType: .hierarchical(
         config: HierarchicalIndexConfig(
             clusterCount: 100,
             subIndexType: .hnsw
         )
     )
 )
 
 // Use LSTM to encode sequences before storage
 let lstm = try await LSTMLayer(
     inputSize: 512,
     config: LSTMConfig(hiddenSize: 256, returnSequences: false),
     metalPipeline: metalPipeline
 )
 
 // Process sequence of vectors
 let sequence = getVectorSequence()  // [Vector]
 let encoded = try await lstm.forward(sequenceToBuffer(sequence))
 
 // Store the encoded representation
 try await store.addVector(bufferToVector(encoded), id: "seq_001")
 ```
 */