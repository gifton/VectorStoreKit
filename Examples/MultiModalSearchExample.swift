import Foundation
import VectorStoreKit

/// Demonstrates multi-modal search capabilities combining text and image embeddings
/// in a unified vector space for cross-modal retrieval
@main
struct MultiModalSearchExample {
    static func main() async throws {
        print("=== VectorStoreKit Multi-Modal Search Example ===\n")
        
        let example = MultiModalSearchExample()
        
        // Demonstrate different multi-modal search strategies
        try await example.demonstrateEarlyFusion()
        try await example.demonstrateLateFusion()
        try await example.demonstrateHybridFusion()
        try await example.demonstrateCrossModalSearch()
        try await example.demonstrateWeightedSearch()
        try await example.demonstrateECommerceUseCase()
    }
    
    // MARK: - Multi-Modal Data Structures
    
    /// Represents a multi-modal item with both text and image embeddings
    struct MultiModalItem {
        let id: String
        let textContent: String
        let imageDescription: String
        let textEmbedding: [Float]
        let imageEmbedding: [Float]
        let metadata: [String: Any]
        
        /// Combined embedding using specified fusion strategy
        func fusedEmbedding(strategy: FusionStrategy) -> [Float] {
            switch strategy {
            case .early(let weights):
                return earlyFusion(textWeight: weights.text, imageWeight: weights.image)
            case .late:
                return lateFusion()
            case .hybrid(let alpha):
                return hybridFusion(alpha: alpha)
            }
        }
        
        private func earlyFusion(textWeight: Float, imageWeight: Float) -> [Float] {
            // Concatenate and weight embeddings
            let weightedText = textEmbedding.map { $0 * textWeight }
            let weightedImage = imageEmbedding.map { $0 * imageWeight }
            return weightedText + weightedImage
        }
        
        private func lateFusion() -> [Float] {
            // Keep embeddings separate, combine at search time
            return textEmbedding + imageEmbedding
        }
        
        private func hybridFusion(alpha: Float) -> [Float] {
            // Weighted average of embeddings
            return zip(textEmbedding, imageEmbedding).map { text, image in
                alpha * text + (1 - alpha) * image
            }
        }
    }
    
    /// Fusion strategies for combining multi-modal embeddings
    enum FusionStrategy {
        case early(weights: (text: Float, image: Float))
        case late
        case hybrid(alpha: Float)
    }
    
    /// Multi-modal query supporting different modality combinations
    struct MultiModalQuery {
        let text: String?
        let imageEmbedding: [Float]?
        let modalityWeights: (text: Float, image: Float)
        
        func toEmbedding(embeddingDim: Int) -> [Float] {
            var embedding = [Float](repeating: 0, count: embeddingDim)
            
            if let text = text {
                let textEmb = generateMockTextEmbedding(text: text, dimension: embeddingDim / 2)
                for i in 0..<textEmb.count {
                    embedding[i] = textEmb[i] * modalityWeights.text
                }
            }
            
            if let imageEmb = imageEmbedding {
                let offset = embeddingDim / 2
                for i in 0..<min(imageEmb.count, embeddingDim / 2) {
                    embedding[offset + i] = imageEmb[i] * modalityWeights.image
                }
            }
            
            return embedding
        }
    }
    
    // MARK: - Embedding Generation
    
    /// Generate mock text embedding using character-based features
    static func generateMockTextEmbedding(text: String, dimension: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: dimension)
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        
        // Simple feature extraction
        for (i, word) in words.enumerated() {
            let hash = word.hashValue
            let index = abs(hash) % dimension
            embedding[index] += 1.0 / Float(words.count)
            
            // Add some semantic features
            if i < dimension / 4 {
                embedding[i] = Float(word.count) / 10.0
            }
        }
        
        // Normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            embedding = embedding.map { $0 / norm }
        }
        
        return embedding
    }
    
    /// Generate mock image embedding based on description
    static func generateMockImageEmbedding(description: String, dimension: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: dimension)
        
        // Simulate visual features based on description
        let visualFeatures = description.lowercased()
        
        // Color features
        if visualFeatures.contains("red") { embedding[0] = 0.8 }
        if visualFeatures.contains("blue") { embedding[1] = 0.8 }
        if visualFeatures.contains("green") { embedding[2] = 0.8 }
        
        // Shape features
        if visualFeatures.contains("round") { embedding[10] = 0.7 }
        if visualFeatures.contains("square") { embedding[11] = 0.7 }
        
        // Texture features
        if visualFeatures.contains("smooth") { embedding[20] = 0.6 }
        if visualFeatures.contains("rough") { embedding[21] = 0.6 }
        
        // Add some randomness for realism
        for i in 30..<dimension {
            embedding[i] = Float.random(in: -0.1...0.1)
        }
        
        // Normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            embedding = embedding.map { $0 / norm }
        }
        
        return embedding
    }
    
    // MARK: - Early Fusion Demonstration
    
    func demonstrateEarlyFusion() async throws {
        print("=== Early Fusion Strategy ===")
        print("Concatenating weighted embeddings before indexing\n")
        
        let universe = VectorUniverse()
        let config = StoreConfiguration(
            indexType: .hierarchical,
            dimensions: 256, // 128 text + 128 image
            distanceMetric: .cosine
        )
        
        let store = try await universe.createStore(named: "early_fusion", config: config)
        
        // Create multi-modal items
        let items = createSampleMultiModalItems()
        
        // Index with early fusion
        let strategy = FusionStrategy.early(weights: (text: 0.6, image: 0.4))
        
        for item in items {
            let vector = Vector(
                id: item.id,
                values: item.fusedEmbedding(strategy: strategy),
                metadata: item.metadata
            )
            try await store.add([vector])
        }
        
        // Search with text query
        let textQuery = MultiModalQuery(
            text: "comfortable running shoes",
            imageEmbedding: nil,
            modalityWeights: (text: 1.0, image: 0.0)
        )
        
        let results = try await store.search(
            query: textQuery.toEmbedding(embeddingDim: 256),
            k: 3
        )
        
        print("Text search results:")
        for result in results {
            print("  - \(result.id): score = \(String(format: "%.3f", result.score))")
        }
        print()
    }
    
    // MARK: - Late Fusion Demonstration
    
    func demonstrateLateFusion() async throws {
        print("=== Late Fusion Strategy ===")
        print("Combining scores from separate modality searches\n")
        
        let universe = VectorUniverse()
        
        // Create separate stores for each modality
        let textConfig = StoreConfiguration(
            indexType: .hnsw,
            dimensions: 128,
            distanceMetric: .cosine
        )
        let imageConfig = StoreConfiguration(
            indexType: .hnsw,
            dimensions: 128,
            distanceMetric: .cosine
        )
        
        let textStore = try await universe.createStore(named: "text_store", config: textConfig)
        let imageStore = try await universe.createStore(named: "image_store", config: imageConfig)
        
        // Index items in both stores
        let items = createSampleMultiModalItems()
        
        for item in items {
            let textVector = Vector(
                id: item.id,
                values: item.textEmbedding,
                metadata: ["type": "text"]
            )
            let imageVector = Vector(
                id: item.id,
                values: item.imageEmbedding,
                metadata: ["type": "image"]
            )
            
            try await textStore.add([textVector])
            try await imageStore.add([imageVector])
        }
        
        // Perform late fusion search
        let query = "athletic footwear with good cushioning"
        let textResults = try await textStore.search(
            query: Self.generateMockTextEmbedding(text: query, dimension: 128),
            k: 5
        )
        
        let imageQuery = Self.generateMockImageEmbedding(
            description: "sporty shoe with thick sole",
            dimension: 128
        )
        let imageResults = try await imageStore.search(query: imageQuery, k: 5)
        
        // Combine scores
        let fusedResults = lateFusionCombine(
            textResults: textResults,
            imageResults: imageResults,
            textWeight: 0.7,
            imageWeight: 0.3
        )
        
        print("Late fusion results:")
        for (id, score) in fusedResults.prefix(3) {
            print("  - \(id): combined score = \(String(format: "%.3f", score))")
        }
        print()
    }
    
    // MARK: - Hybrid Fusion Demonstration
    
    func demonstrateHybridFusion() async throws {
        print("=== Hybrid Fusion Strategy ===")
        print("Weighted average of embeddings with learned alpha\n")
        
        let universe = VectorUniverse()
        let config = StoreConfiguration(
            indexType: .ivf(nCentroids: 10),
            dimensions: 128,
            distanceMetric: .euclidean
        )
        
        let store = try await universe.createStore(named: "hybrid_fusion", config: config)
        
        // Create items with hybrid fusion
        let items = createSampleMultiModalItems()
        let alpha: Float = 0.65 // Learned parameter favoring text
        
        for item in items {
            let fusedEmbedding = item.fusedEmbedding(strategy: .hybrid(alpha: alpha))
            let vector = Vector(
                id: item.id,
                values: fusedEmbedding,
                metadata: item.metadata
            )
            try await store.add([vector])
        }
        
        // Multi-modal query
        let textPart = "lightweight breathable material"
        let imagePart = "mesh fabric texture"
        
        let textEmb = Self.generateMockTextEmbedding(text: textPart, dimension: 128)
        let imageEmb = Self.generateMockImageEmbedding(description: imagePart, dimension: 128)
        
        // Fuse query embeddings
        let queryEmbedding = zip(textEmb, imageEmb).map { text, image in
            alpha * text + (1 - alpha) * image
        }
        
        let results = try await store.search(query: queryEmbedding, k: 3)
        
        print("Hybrid fusion results:")
        for result in results {
            print("  - \(result.id): score = \(String(format: "%.3f", result.score))")
        }
        print()
    }
    
    // MARK: - Cross-Modal Search
    
    func demonstrateCrossModalSearch() async throws {
        print("=== Cross-Modal Search ===")
        print("Search images with text and text with images\n")
        
        let universe = VectorUniverse()
        
        // Use a shared embedding space
        let config = StoreConfiguration(
            indexType: .hnsw,
            dimensions: 128,
            distanceMetric: .cosine
        )
        
        let store = try await universe.createStore(named: "cross_modal", config: config)
        
        // Create items with aligned embeddings
        let items = createAlignedMultiModalItems()
        
        // Index all embeddings in the same space
        for item in items {
            // Add text representation
            let textVector = Vector(
                id: "\(item.id)_text",
                values: item.textEmbedding,
                metadata: [
                    "modality": "text",
                    "content": item.textContent,
                    "item_id": item.id
                ]
            )
            
            // Add image representation
            let imageVector = Vector(
                id: "\(item.id)_image",
                values: item.imageEmbedding,
                metadata: [
                    "modality": "image",
                    "description": item.imageDescription,
                    "item_id": item.id
                ]
            )
            
            try await store.add([textVector, imageVector])
        }
        
        // Search images with text
        print("1. Searching images with text query:")
        let textQuery = "professional business attire"
        let textQueryEmb = Self.generateMockTextEmbedding(text: textQuery, dimension: 128)
        
        let imageResults = try await store.search(
            query: textQueryEmb,
            k: 5,
            filter: { metadata in
                metadata["modality"] as? String == "image"
            }
        )
        
        for result in imageResults.prefix(3) {
            if let desc = result.metadata?["description"] as? String {
                print("  - Found image: \(desc) (score: \(String(format: "%.3f", result.score)))")
            }
        }
        
        // Search text with images
        print("\n2. Searching text with image query:")
        let imageQuery = Self.generateMockImageEmbedding(
            description: "casual denim jacket",
            dimension: 128
        )
        
        let textResults = try await store.search(
            query: imageQuery,
            k: 5,
            filter: { metadata in
                metadata["modality"] as? String == "text"
            }
        )
        
        for result in textResults.prefix(3) {
            if let content = result.metadata?["content"] as? String {
                print("  - Found text: \(content) (score: \(String(format: "%.3f", result.score)))")
            }
        }
        print()
    }
    
    // MARK: - Weighted Search
    
    func demonstrateWeightedSearch() async throws {
        print("=== Weighted Multi-Modal Search ===")
        print("Dynamic weighting based on query confidence\n")
        
        let universe = VectorUniverse()
        let config = StoreConfiguration(
            indexType: .hierarchical,
            dimensions: 256,
            distanceMetric: .cosine
        )
        
        let store = try await universe.createStore(named: "weighted_search", config: config)
        
        // Index items
        let items = createSampleMultiModalItems()
        for item in items {
            let vector = Vector(
                id: item.id,
                values: item.fusedEmbedding(strategy: .early(weights: (text: 0.5, image: 0.5))),
                metadata: item.metadata
            )
            try await store.add([vector])
        }
        
        // Different query scenarios with adaptive weights
        let scenarios = [
            (
                name: "Text-heavy query",
                text: "ergonomic design for long walks",
                imageDesc: "generic shoe",
                weights: (text: 0.8, image: 0.2)
            ),
            (
                name: "Image-heavy query",
                text: "shoe",
                imageDesc: "bright neon colors with reflective strips",
                weights: (text: 0.2, image: 0.8)
            ),
            (
                name: "Balanced query",
                text: "vintage style sneaker",
                imageDesc: "retro design with classic colors",
                weights: (text: 0.5, image: 0.5)
            )
        ]
        
        for scenario in scenarios {
            print("\(scenario.name) (weights: text=\(scenario.weights.text), image=\(scenario.weights.image)):")
            
            let query = createWeightedQuery(
                text: scenario.text,
                imageDescription: scenario.imageDesc,
                weights: scenario.weights
            )
            
            let results = try await store.search(query: query, k: 2)
            
            for result in results {
                if let name = result.metadata?["name"] as? String {
                    print("  - \(name): score = \(String(format: "%.3f", result.score))")
                }
            }
            print()
        }
    }
    
    // MARK: - E-Commerce Use Case
    
    func demonstrateECommerceUseCase() async throws {
        print("=== E-Commerce Product Search ===")
        print("Real-world multi-modal product discovery\n")
        
        let universe = VectorUniverse()
        
        // Configure for e-commerce scale
        let config = StoreConfiguration(
            indexType: .hybrid(
                primary: .hnsw,
                secondary: .ivf(nCentroids: 100)
            ),
            dimensions: 384, // Larger embeddings for richer features
            distanceMetric: .cosine,
            quantization: .product(numSubvectors: 8, bitsPerSubvector: 8)
        )
        
        let store = try await universe.createStore(named: "ecommerce", config: config)
        
        // Create product catalog
        let products = createECommerceProducts()
        
        // Index products with rich embeddings
        for product in products {
            let vector = Vector(
                id: product.id,
                values: product.multiModalEmbedding,
                metadata: [
                    "name": product.name,
                    "category": product.category,
                    "price": product.price,
                    "brand": product.brand,
                    "colors": product.colors,
                    "materials": product.materials,
                    "tags": product.tags
                ]
            )
            try await store.add([vector])
        }
        
        // Simulate different customer queries
        print("1. Visual similarity search:")
        let visualQuery = createProductQuery(
            text: nil,
            imageFeatures: ["blue", "leather", "formal", "oxford style"],
            intent: .visualSimilarity
        )
        
        let visualResults = try await store.search(query: visualQuery, k: 3)
        printProductResults(visualResults, prefix: "  ")
        
        print("\n2. Text + filter search:")
        let textFilterQuery = createProductQuery(
            text: "comfortable walking shoes under $100",
            imageFeatures: nil,
            intent: .budgetConscious
        )
        
        let budgetResults = try await store.search(
            query: textFilterQuery,
            k: 5,
            filter: { metadata in
                if let price = metadata["price"] as? Double {
                    return price <= 100
                }
                return false
            }
        )
        printProductResults(budgetResults, prefix: "  ")
        
        print("\n3. Hybrid preference search:")
        let hybridQuery = createProductQuery(
            text: "sustainable materials eco-friendly",
            imageFeatures: ["minimalist", "neutral colors"],
            intent: .sustainability
        )
        
        let sustainableResults = try await store.search(
            query: hybridQuery,
            k: 3,
            filter: { metadata in
                if let tags = metadata["tags"] as? [String] {
                    return tags.contains("sustainable") || tags.contains("eco-friendly")
                }
                return false
            }
        )
        printProductResults(sustainableResults, prefix: "  ")
        
        // Performance metrics
        print("\n4. Search performance:")
        let start = CFAbsoluteTimeGetCurrent()
        
        let _ = try await store.search(
            query: createProductQuery(
                text: "sports shoes",
                imageFeatures: ["athletic", "breathable"],
                intent: .performance
            ),
            k: 10
        )
        
        let searchTime = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("  - Search latency: \(String(format: "%.2f", searchTime))ms")
        print("  - Index size: \(products.count) products")
        print("  - Embedding dimensions: 384")
        print("  - Quantization: 8x8 PQ")
    }
    
    // MARK: - Helper Functions
    
    func createSampleMultiModalItems() -> [MultiModalItem] {
        [
            MultiModalItem(
                id: "item1",
                textContent: "Professional running shoes with advanced cushioning technology",
                imageDescription: "White and blue athletic shoe with thick sole",
                textEmbedding: Self.generateMockTextEmbedding(
                    text: "Professional running shoes with advanced cushioning technology",
                    dimension: 128
                ),
                imageEmbedding: Self.generateMockImageEmbedding(
                    description: "White and blue athletic shoe with thick sole",
                    dimension: 128
                ),
                metadata: ["name": "ProRunner 3000", "category": "athletic"]
            ),
            MultiModalItem(
                id: "item2",
                textContent: "Elegant leather dress shoes for formal occasions",
                imageDescription: "Black leather oxford shoe with polished finish",
                textEmbedding: Self.generateMockTextEmbedding(
                    text: "Elegant leather dress shoes for formal occasions",
                    dimension: 128
                ),
                imageEmbedding: Self.generateMockImageEmbedding(
                    description: "Black leather oxford shoe with polished finish",
                    dimension: 128
                ),
                metadata: ["name": "Executive Oxford", "category": "formal"]
            ),
            MultiModalItem(
                id: "item3",
                textContent: "Comfortable walking shoes with memory foam insoles",
                imageDescription: "Brown casual shoe with soft cushioning",
                textEmbedding: Self.generateMockTextEmbedding(
                    text: "Comfortable walking shoes with memory foam insoles",
                    dimension: 128
                ),
                imageEmbedding: Self.generateMockImageEmbedding(
                    description: "Brown casual shoe with soft cushioning",
                    dimension: 128
                ),
                metadata: ["name": "ComfortWalk Plus", "category": "casual"]
            ),
            MultiModalItem(
                id: "item4",
                textContent: "Lightweight hiking boots with waterproof protection",
                imageDescription: "Green and gray boot with rugged sole",
                textEmbedding: Self.generateMockTextEmbedding(
                    text: "Lightweight hiking boots with waterproof protection",
                    dimension: 128
                ),
                imageEmbedding: Self.generateMockImageEmbedding(
                    description: "Green and gray boot with rugged sole",
                    dimension: 128
                ),
                metadata: ["name": "TrailBlazer Pro", "category": "outdoor"]
            ),
            MultiModalItem(
                id: "item5",
                textContent: "Classic canvas sneakers with vintage appeal",
                imageDescription: "Red canvas shoe with white rubber sole",
                textEmbedding: Self.generateMockTextEmbedding(
                    text: "Classic canvas sneakers with vintage appeal",
                    dimension: 128
                ),
                imageEmbedding: Self.generateMockImageEmbedding(
                    description: "Red canvas shoe with white rubber sole",
                    dimension: 128
                ),
                metadata: ["name": "RetroKicks", "category": "casual"]
            )
        ]
    }
    
    func createAlignedMultiModalItems() -> [MultiModalItem] {
        // Create items where text and image embeddings are in aligned space
        let items = createSampleMultiModalItems()
        
        // Simulate alignment through shared features
        return items.map { item in
            var alignedTextEmb = item.textEmbedding
            var alignedImageEmb = item.imageEmbedding
            
            // Add shared semantic features
            for i in 0..<min(32, alignedTextEmb.count) {
                let sharedFeature = (alignedTextEmb[i] + alignedImageEmb[i]) / 2
                alignedTextEmb[i] = alignedTextEmb[i] * 0.7 + sharedFeature * 0.3
                alignedImageEmb[i] = alignedImageEmb[i] * 0.7 + sharedFeature * 0.3
            }
            
            return MultiModalItem(
                id: item.id,
                textContent: item.textContent,
                imageDescription: item.imageDescription,
                textEmbedding: alignedTextEmb,
                imageEmbedding: alignedImageEmb,
                metadata: item.metadata
            )
        }
    }
    
    func lateFusionCombine(
        textResults: [SearchResult],
        imageResults: [SearchResult],
        textWeight: Float,
        imageWeight: Float
    ) -> [(String, Float)] {
        var combinedScores: [String: Float] = [:]
        
        // Add text scores
        for result in textResults {
            combinedScores[result.id] = result.score * textWeight
        }
        
        // Add image scores
        for result in imageResults {
            if let existing = combinedScores[result.id] {
                combinedScores[result.id] = existing + result.score * imageWeight
            } else {
                combinedScores[result.id] = result.score * imageWeight
            }
        }
        
        // Sort by combined score
        return combinedScores.sorted { $0.value > $1.value }
    }
    
    func createWeightedQuery(
        text: String,
        imageDescription: String,
        weights: (text: Float, image: Float)
    ) -> [Float] {
        let textEmb = Self.generateMockTextEmbedding(text: text, dimension: 128)
        let imageEmb = Self.generateMockImageEmbedding(description: imageDescription, dimension: 128)
        
        // Weighted concatenation
        let weightedText = textEmb.map { $0 * weights.text }
        let weightedImage = imageEmb.map { $0 * weights.image }
        
        return weightedText + weightedImage
    }
    
    // MARK: - E-Commerce Helpers
    
    struct Product {
        let id: String
        let name: String
        let category: String
        let price: Double
        let brand: String
        let colors: [String]
        let materials: [String]
        let tags: [String]
        let textDescription: String
        let visualFeatures: String
        
        var multiModalEmbedding: [Float] {
            // Rich embedding combining all features
            let textEmb = MultiModalSearchExample.generateMockTextEmbedding(
                text: "\(name) \(textDescription) \(brand) \(tags.joined(separator: " "))",
                dimension: 192
            )
            
            let visualEmb = MultiModalSearchExample.generateMockImageEmbedding(
                description: "\(colors.joined(separator: " ")) \(materials.joined(separator: " ")) \(visualFeatures)",
                dimension: 192
            )
            
            return textEmb + visualEmb
        }
    }
    
    enum SearchIntent {
        case visualSimilarity
        case budgetConscious
        case sustainability
        case performance
    }
    
    func createECommerceProducts() -> [Product] {
        [
            Product(
                id: "prod001",
                name: "EcoRunner Pro",
                category: "athletic",
                price: 129.99,
                brand: "GreenStride",
                colors: ["white", "green"],
                materials: ["recycled polyester", "natural rubber"],
                tags: ["sustainable", "running", "eco-friendly", "performance"],
                textDescription: "High-performance running shoe made from recycled materials",
                visualFeatures: "sleek design with green accents and breathable mesh"
            ),
            Product(
                id: "prod002",
                name: "Executive Elite",
                category: "formal",
                price: 249.99,
                brand: "Prestige",
                colors: ["black", "brown"],
                materials: ["genuine leather", "leather sole"],
                tags: ["business", "formal", "luxury", "handcrafted"],
                textDescription: "Handcrafted Italian leather dress shoes for the modern executive",
                visualFeatures: "polished leather with classic oxford styling"
            ),
            Product(
                id: "prod003",
                name: "CloudWalk Comfort",
                category: "casual",
                price: 89.99,
                brand: "ComfortZone",
                colors: ["gray", "navy"],
                materials: ["memory foam", "knit fabric"],
                tags: ["comfort", "walking", "everyday", "orthopedic"],
                textDescription: "All-day comfort walking shoes with orthopedic support",
                visualFeatures: "soft knit upper with cushioned sole"
            ),
            Product(
                id: "prod004",
                name: "TrailMaster X",
                category: "outdoor",
                price: 159.99,
                brand: "AdventureGear",
                colors: ["brown", "orange"],
                materials: ["waterproof leather", "vibram sole"],
                tags: ["hiking", "waterproof", "durable", "outdoor"],
                textDescription: "Rugged hiking boots built for challenging terrain",
                visualFeatures: "sturdy construction with aggressive tread pattern"
            ),
            Product(
                id: "prod005",
                name: "UrbanFlex",
                category: "lifestyle",
                price: 79.99,
                brand: "StreetStyle",
                colors: ["black", "white", "red"],
                materials: ["canvas", "rubber"],
                tags: ["casual", "streetwear", "versatile", "affordable"],
                textDescription: "Versatile sneakers perfect for urban lifestyle",
                visualFeatures: "minimalist design with bold color options"
            ),
            Product(
                id: "prod006",
                name: "AquaShield Pro",
                category: "athletic",
                price: 139.99,
                brand: "WaterTech",
                colors: ["blue", "silver"],
                materials: ["waterproof mesh", "synthetic"],
                tags: ["water-sports", "quick-dry", "athletic", "marine"],
                textDescription: "Water-resistant athletic shoes for aquatic activities",
                visualFeatures: "mesh design with drainage ports and grip sole"
            ),
            Product(
                id: "prod007",
                name: "ZeroGravity",
                category: "performance",
                price: 199.99,
                brand: "TechRunner",
                colors: ["neon yellow", "black"],
                materials: ["carbon fiber", "ultralight foam"],
                tags: ["racing", "lightweight", "performance", "marathon"],
                textDescription: "Ultra-lightweight racing shoes with carbon fiber plate",
                visualFeatures: "aerodynamic profile with striking neon accents"
            ),
            Product(
                id: "prod008",
                name: "Heritage Classic",
                category: "casual",
                price: 69.99,
                brand: "Vintage",
                colors: ["brown", "beige"],
                materials: ["suede", "gum rubber"],
                tags: ["retro", "classic", "sustainable", "timeless"],
                textDescription: "Timeless design inspired by 1970s court shoes",
                visualFeatures: "vintage suede with classic gum sole"
            )
        ]
    }
    
    func createProductQuery(
        text: String?,
        imageFeatures: [String]?,
        intent: SearchIntent
    ) -> [Float] {
        var embedding = [Float](repeating: 0, count: 384)
        
        // Text features (first 192 dimensions)
        if let text = text {
            let textEmb = Self.generateMockTextEmbedding(text: text, dimension: 192)
            for i in 0..<textEmb.count {
                embedding[i] = textEmb[i]
            }
        }
        
        // Visual features (last 192 dimensions)
        if let features = imageFeatures {
            let visualEmb = Self.generateMockImageEmbedding(
                description: features.joined(separator: " "),
                dimension: 192
            )
            for i in 0..<visualEmb.count {
                embedding[192 + i] = visualEmb[i]
            }
        }
        
        // Adjust weights based on intent
        switch intent {
        case .visualSimilarity:
            // Boost visual features
            for i in 192..<384 {
                embedding[i] *= 1.5
            }
        case .budgetConscious:
            // Add price sensitivity signal
            embedding[50] = 0.8  // Arbitrary dimension for price
        case .sustainability:
            // Boost eco-related features
            embedding[100] = 0.9  // Arbitrary dimension for sustainability
        case .performance:
            // Boost performance features
            embedding[150] = 0.9  // Arbitrary dimension for performance
        }
        
        // Normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            embedding = embedding.map { $0 / norm }
        }
        
        return embedding
    }
    
    func printProductResults(_ results: [SearchResult], prefix: String) {
        for result in results {
            if let name = result.metadata?["name"] as? String,
               let price = result.metadata?["price"] as? Double,
               let brand = result.metadata?["brand"] as? String {
                print("\(prefix)- \(name) by \(brand) ($\(String(format: "%.2f", price))) - score: \(String(format: "%.3f", result.score))")
            }
        }
    }
}
        <function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Design multi-modal data structures for text and image embeddings", "status": "completed", "priority": "high"}, {"id": "2", "content": "Implement embedding generators for mock text and image data", "status": "completed", "priority": "high"}, {"id": "3", "content": "Create multi-modal fusion strategies (early, late, hybrid fusion)", "status": "in_progress", "priority": "high"}, {"id": "4", "content": "Implement cross-modal search functionality", "status": "pending", "priority": "high"}, {"id": "5", "content": "Add weighted search across modalities", "status": "pending", "priority": "medium"}, {"id": "6", "content": "Create e-commerce product search use case", "status": "pending", "priority": "medium"}, {"id": "7", "content": "Add relevance scoring and ranking logic", "status": "pending", "priority": "medium"}, {"id": "8", "content": "Implement performance benchmarking for multi-modal search", "status": "pending", "priority": "low"}]