import Foundation
import CoreML
import NaturalLanguage

/// Local embedding provider using on-device models
public actor LocalEmbeddingProvider: EmbeddingProvider {
    public let configuration: EmbeddingConfiguration
    private let encoder: OptimizedVectorEncoder
    private let tokenizer: NLTokenizer
    private let accelerationEngine: MetalAccelerationEngine
    
    /// Initialize with a pre-trained encoder
    public init(
        encoder: OptimizedVectorEncoder,
        dimensions: Int = 768,
        model: String = "local-bert-base"
    ) {
        self.encoder = encoder
        self.configuration = EmbeddingConfiguration(
            model: model,
            dimensions: dimensions,
            maxTokens: 512,  // BERT-style limit
            normalize: true,
            batchSize: 32    // Optimized for on-device
        )
        
        self.tokenizer = NLTokenizer(unit: .word)
        self.accelerationEngine = MetalAccelerationEngine.shared
    }
    
    /// Initialize with default sentence transformer
    public static func sentenceTransformer(
        dimensions: Int = 384
    ) async throws -> LocalEmbeddingProvider {
        // Create a pre-configured encoder for sentence embeddings
        let encoder = try await OptimizedVectorEncoder()
        
        // Configure the encoder for sentence embeddings
        // Note: The encoder's configure method takes inputDimension, encodedDimension and optional config
        try await encoder.configure(
            inputDimension: 768,    // BERT hidden size
            encodedDimension: dimensions
            // Using default configuration
        )
        
        return LocalEmbeddingProvider(
            encoder: encoder,
            dimensions: dimensions,
            model: "sentence-transformer-local"
        )
    }
    
    public func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        // Validate batch size
        guard texts.count <= configuration.batchSize else {
            throw EmbeddingError.batchTooLarge(
                size: texts.count,
                maxSize: configuration.batchSize
            )
        }
        
        // Tokenize and validate lengths
        let tokenizedTexts = try texts.map { text -> [String] in
            let tokens = tokenize(text)
            if tokens.count > configuration.maxTokens {
                throw EmbeddingError.textTooLong(
                    tokens: tokens.count,
                    maxTokens: configuration.maxTokens
                )
            }
            return tokens
        }
        
        // Convert to input vectors (simplified - would use actual token embeddings)
        let inputVectors = try await convertToVectors(tokenizedTexts)
        
        // Encode using the neural encoder
        let embeddings = try await encoder.encode(inputVectors)
        
        // Convert [[Float]] to [EmbeddingVector] and normalize if requested
        return embeddings.map { floatArray in
            var vector = EmbeddingVector(data: floatArray)
            if configuration.normalize {
                vector.normalize()
            }
            return vector
        }
    }
    
    public func estimateTokens(for text: String) -> Int {
        tokenize(text).count
    }
    
    public func isAvailable() async -> Bool {
        // Check if encoder is properly initialized
        // For now, assume it's available if we have an encoder instance
        return true
    }
    
    // MARK: - Private Methods
    
    private func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let token = String(text[tokenRange])
            tokens.append(token.lowercased())
            return true
        }
        
        return tokens
    }
    
    private func convertToVectors(_ tokenizedTexts: [[String]]) async throws -> [[Float]] {
        // In a real implementation, this would:
        // 1. Convert tokens to IDs using a vocabulary
        // 2. Look up token embeddings
        // 3. Apply positional encodings
        // 4. Pool tokens into sentence embeddings
        
        // For now, create random vectors as placeholder
        return tokenizedTexts.map { tokens in
            // Simple bag-of-words style representation
            var vector = Array(repeating: Float(0.0), count: configuration.dimensions)
            
            // Hash tokens into vector space (simplified)
            for (index, token) in tokens.enumerated() {
                let hash = token.hash
                let position = abs(hash) % configuration.dimensions
                vector[position] += Float(1.0 / Float(tokens.count))
                
                // Add some position information
                if index < configuration.dimensions {
                    vector[index] += Float(0.1)
                }
            }
            
            return vector
        }
    }
}

/// Core ML based embedding provider
@available(macOS 14.0, iOS 17.0, *)
public actor CoreMLEmbeddingProvider: EmbeddingProvider {
    public let configuration: EmbeddingConfiguration
    private let model: MLModel
    private let accelerationEngine: MetalAccelerationEngine
    
    /// Initialize with a Core ML model
    public init(
        modelURL: URL,
        dimensions: Int,
        modelName: String = "coreml-embeddings"
    ) async throws {
        // Load Core ML model
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        
        self.model = try await MLModel.load(
            contentsOf: modelURL,
            configuration: config
        )
        
        self.configuration = EmbeddingConfiguration(
            model: modelName,
            dimensions: dimensions,
            maxTokens: 512,
            normalize: true,
            batchSize: 16
        )
        
        self.accelerationEngine = MetalAccelerationEngine.shared
    }
    
    public func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        var embeddings: [EmbeddingVector] = []
        
        for text in texts {
            // Prepare input for Core ML model
            let input = try prepareInput(text)
            
            // Run inference
            let output = try await model.prediction(from: input)
            
            // Extract embeddings from output
            let embedding = try extractEmbedding(from: output)
            
            // Normalize if needed
            var vector = embedding
            if configuration.normalize {
                vector.normalize()
            }
            
            embeddings.append(vector)
        }
        
        return embeddings
    }
    
    public func estimateTokens(for text: String) -> Int {
        // Simple estimation
        let words = text.split(separator: " ").count
        return words + 2  // Account for special tokens
    }
    
    public func isAvailable() async -> Bool {
        // Check if model is loaded
        return true
    }
    
    private func prepareInput(_ text: String) throws -> MLFeatureProvider {
        // This would prepare the input according to the model's requirements
        // Placeholder implementation
        throw EmbeddingError.processingFailed("Core ML input preparation not implemented")
    }
    
    private func extractEmbedding(from output: MLFeatureProvider) throws -> EmbeddingVector {
        // Extract embedding vector from Core ML output
        // Placeholder implementation
        throw EmbeddingError.processingFailed("Core ML output extraction not implemented")
    }
}

/// Hybrid embedding provider that can fall back between providers
public actor HybridEmbeddingProvider: EmbeddingProvider {
    public let configuration: EmbeddingConfiguration
    private let primaryProvider: any EmbeddingProvider
    private let fallbackProvider: any EmbeddingProvider
    private var useFallback: Bool = false
    
    public init(
        primary: any EmbeddingProvider,
        fallback: any EmbeddingProvider
    ) async {
        self.primaryProvider = primary
        self.fallbackProvider = fallback
        
        // Use primary provider's configuration
        self.configuration = await primary.configuration
    }
    
    public func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        // Try primary provider first
        if !useFallback {
            do {
                return try await primaryProvider.embed(texts)
            } catch {
                print("Primary provider failed, switching to fallback: \(error)")
                useFallback = true
            }
        }
        
        // Use fallback provider
        return try await fallbackProvider.embed(texts)
    }
    
    public func estimateTokens(for text: String) -> Int {
        // Use primary provider's estimation
        Task {
            await primaryProvider.estimateTokens(for: text)
        }
        return (text.count + 3) / 4  // Default estimation
    }
    
    public func isAvailable() async -> Bool {
        if await primaryProvider.isAvailable() {
            useFallback = false
            return true
        }
        return await fallbackProvider.isAvailable()
    }
    
    /// Reset to try primary provider again
    public func reset() {
        useFallback = false
    }
}

// MARK: - Embedding Cache

/// Cache for embeddings to avoid recomputation
public actor EmbeddingCache {
    private var cache: [String: EmbeddingVector] = [:]
    private var accessOrder: [String] = []
    private let maxSize: Int
    private let provider: any EmbeddingProvider
    
    public init(
        provider: any EmbeddingProvider,
        maxSize: Int = 10000
    ) {
        self.provider = provider
        self.maxSize = maxSize
    }
    
    /// Get embedding with caching
    public func embed(_ text: String) async throws -> EmbeddingVector {
        // Check cache
        if let cached = cache[text] {
            // Move to end (LRU)
            accessOrder.removeAll { $0 == text }
            accessOrder.append(text)
            return cached
        }
        
        // Compute embedding
        let embedding = try await provider.embed(text)
        
        // Update cache
        cache[text] = embedding
        accessOrder.append(text)
        
        // Evict if needed
        if cache.count > maxSize {
            if let oldest = accessOrder.first {
                cache.removeValue(forKey: oldest)
                accessOrder.removeFirst()
            }
        }
        
        return embedding
    }
    
    /// Batch embed with caching
    public func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        var results: [EmbeddingVector] = []
        var uncached: [(index: Int, text: String)] = []
        
        // Check cache for each text
        for (index, text) in texts.enumerated() {
            if let cached = cache[text] {
                results.append(cached)
                // Update LRU
                accessOrder.removeAll { $0 == text }
                accessOrder.append(text)
            } else {
                results.append(EmbeddingVector(dimensions: 0))  // Placeholder
                uncached.append((index, text))
            }
        }
        
        // Compute uncached embeddings
        if !uncached.isEmpty {
            let uncachedTexts = uncached.map { $0.text }
            let embeddings = try await provider.embed(uncachedTexts)
            
            // Update results and cache
            for (embedding, (index, text)) in zip(embeddings, uncached) {
                results[index] = embedding
                
                // Update cache
                cache[text] = embedding
                accessOrder.append(text)
                
                // Evict if needed
                if cache.count > maxSize {
                    if let oldest = accessOrder.first {
                        cache.removeValue(forKey: oldest)
                        accessOrder.removeFirst()
                    }
                }
            }
        }
        
        return results
    }
    
    /// Clear the cache
    public func clear() {
        cache.removeAll()
        accessOrder.removeAll()
    }
    
    /// Get cache statistics
    public func statistics() -> (size: Int, hitRate: Double) {
        (size: cache.count, hitRate: 0.0)  // Would track hits/misses in production
    }
}