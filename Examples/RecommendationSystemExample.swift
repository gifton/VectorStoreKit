// VectorStoreKit: Recommendation System Example
//
// Demonstrates building a sophisticated recommendation system using vector embeddings
// Shows collaborative filtering, content-based filtering, and hybrid approaches

import Foundation
import VectorStoreKit

@main
struct RecommendationSystemExample {
    
    static func main() async throws {
        print("🎯 VectorStoreKit Recommendation System Example")
        print("=" * 60)
        print()
        
        // Configuration
        let config = RecommendationSystemConfiguration(
            dimensions: 256,
            userEmbeddingSize: 128,
            itemEmbeddingSize: 128,
            interactionWeight: 0.7,
            contentWeight: 0.3,
            enableRealTimeUpdates: true,
            minInteractionsForRecommendation: 3
        )
        
        // Create recommendation engine
        let recommendationEngine = try await RecommendationEngine(configuration: config)
        
        // Example 1: User and Item Profiling
        print("👤 Example 1: Building User and Item Profiles")
        print("-" * 40)
        try await buildUserAndItemProfiles(engine: recommendationEngine)
        
        // Example 2: Collaborative Filtering
        print("\n🤝 Example 2: Collaborative Filtering Recommendations")
        print("-" * 40)
        try await collaborativeFilteringDemo(engine: recommendationEngine)
        
        // Example 3: Content-Based Filtering
        print("\n📊 Example 3: Content-Based Recommendations")
        print("-" * 40)
        try await contentBasedFilteringDemo(engine: recommendationEngine)
        
        // Example 4: Hybrid Recommendations
        print("\n🔄 Example 4: Hybrid Recommendation System")
        print("-" * 40)
        try await hybridRecommendationsDemo(engine: recommendationEngine)
        
        // Example 5: Real-time Personalization
        print("\n⚡ Example 5: Real-time Personalization")
        print("-" * 40)
        try await realtimePersonalizationDemo(engine: recommendationEngine)
        
        // Example 6: Cold Start Problem
        print("\n❄️ Example 6: Handling Cold Start")
        print("-" * 40)
        try await coldStartDemo(engine: recommendationEngine)
        
        // Example 7: Recommendation Explanations
        print("\n💡 Example 7: Explainable Recommendations")
        print("-" * 40)
        try await explainableRecommendationsDemo(engine: recommendationEngine)
        
        print("\n✅ Recommendation system example completed!")
    }
    
    // MARK: - Example 1: User and Item Profiling
    
    static func buildUserAndItemProfiles(engine: RecommendationEngine) async throws {
        // Create sample users
        let users = [
            User(
                id: "user1",
                name: "Alice",
                demographics: Demographics(age: 28, gender: "F", location: "New York"),
                preferences: UserPreferences(
                    categories: ["Technology", "Books", "Fitness"],
                    priceRange: .medium,
                    qualityPreference: .high
                )
            ),
            User(
                id: "user2",
                name: "Bob",
                demographics: Demographics(age: 35, gender: "M", location: "San Francisco"),
                preferences: UserPreferences(
                    categories: ["Gaming", "Technology", "Movies"],
                    priceRange: .high,
                    qualityPreference: .premium
                )
            ),
            User(
                id: "user3",
                name: "Carol",
                demographics: Demographics(age: 42, gender: "F", location: "Chicago"),
                preferences: UserPreferences(
                    categories: ["Home & Garden", "Cooking", "Books"],
                    priceRange: .medium,
                    qualityPreference: .medium
                )
            ),
            User(
                id: "user4",
                name: "David",
                demographics: Demographics(age: 24, gender: "M", location: "Austin"),
                preferences: UserPreferences(
                    categories: ["Music", "Fashion", "Technology"],
                    priceRange: .low,
                    qualityPreference: .medium
                )
            )
        ]
        
        // Create sample items
        let items = [
            // Technology products
            Item(
                id: "item1",
                name: "Wireless Headphones Pro",
                category: "Technology",
                subcategory: "Audio",
                price: 299.99,
                features: ItemFeatures(
                    brand: "TechBrand",
                    quality: .premium,
                    attributes: ["noise-canceling", "wireless", "long-battery"],
                    popularity: 0.85
                )
            ),
            Item(
                id: "item2",
                name: "Smart Home Hub",
                category: "Technology",
                subcategory: "Smart Home",
                price: 149.99,
                features: ItemFeatures(
                    brand: "HomeTech",
                    quality: .high,
                    attributes: ["voice-control", "automation", "energy-saving"],
                    popularity: 0.72
                )
            ),
            
            // Books
            Item(
                id: "item3",
                name: "The AI Revolution",
                category: "Books",
                subcategory: "Technology",
                price: 24.99,
                features: ItemFeatures(
                    brand: "TechPress",
                    quality: .high,
                    attributes: ["educational", "bestseller", "tech-focused"],
                    popularity: 0.78
                )
            ),
            Item(
                id: "item4",
                name: "Mastering the Kitchen",
                category: "Books",
                subcategory: "Cooking",
                price: 34.99,
                features: ItemFeatures(
                    brand: "CulinaryPress",
                    quality: .high,
                    attributes: ["cookbook", "recipes", "beginner-friendly"],
                    popularity: 0.65
                )
            ),
            
            // Gaming
            Item(
                id: "item5",
                name: "Gaming Keyboard RGB",
                category: "Gaming",
                subcategory: "Peripherals",
                price: 179.99,
                features: ItemFeatures(
                    brand: "GameGear",
                    quality: .premium,
                    attributes: ["mechanical", "rgb-lighting", "programmable"],
                    popularity: 0.88
                )
            ),
            
            // Fitness
            Item(
                id: "item6",
                name: "Smart Fitness Tracker",
                category: "Fitness",
                subcategory: "Wearables",
                price: 199.99,
                features: ItemFeatures(
                    brand: "FitTech",
                    quality: .high,
                    attributes: ["heart-rate", "gps", "water-resistant"],
                    popularity: 0.82
                )
            )
        ]
        
        print("Building user and item profiles...")
        
        // Index users
        for user in users {
            let userEmbedding = try await engine.generateUserEmbedding(user)
            try await engine.indexUser(user, embedding: userEmbedding)
            
            print("  ✓ Indexed user: \(user.name)")
            print("    Interests: \(user.preferences.categories.joined(separator: ", "))")
        }
        
        // Index items
        for item in items {
            let itemEmbedding = try await engine.generateItemEmbedding(item)
            try await engine.indexItem(item, embedding: itemEmbedding)
            
            print("\n  ✓ Indexed item: \(item.name)")
            print("    Category: \(item.category) - \(item.subcategory)")
            print("    Price: $\(String(format: "%.2f", item.price))")
            print("    Features: \(item.features.attributes.joined(separator: ", "))")
        }
        
        // Create interaction history
        let interactions = [
            // Alice's interactions
            Interaction(userId: "user1", itemId: "item1", type: .purchase, rating: 5, timestamp: Date()),
            Interaction(userId: "user1", itemId: "item3", type: .purchase, rating: 4, timestamp: Date()),
            Interaction(userId: "user1", itemId: "item6", type: .view, rating: nil, timestamp: Date()),
            
            // Bob's interactions
            Interaction(userId: "user2", itemId: "item5", type: .purchase, rating: 5, timestamp: Date()),
            Interaction(userId: "user2", itemId: "item1", type: .purchase, rating: 5, timestamp: Date()),
            Interaction(userId: "user2", itemId: "item2", type: .view, rating: nil, timestamp: Date()),
            
            // Carol's interactions
            Interaction(userId: "user3", itemId: "item4", type: .purchase, rating: 5, timestamp: Date()),
            Interaction(userId: "user3", itemId: "item2", type: .purchase, rating: 4, timestamp: Date()),
            Interaction(userId: "user3", itemId: "item3", type: .view, rating: nil, timestamp: Date()),
            
            // David's interactions
            Interaction(userId: "user4", itemId: "item1", type: .wishlist, rating: nil, timestamp: Date()),
            Interaction(userId: "user4", itemId: "item6", type: .view, rating: nil, timestamp: Date())
        ]
        
        print("\n\nRecording user interactions...")
        for interaction in interactions {
            try await engine.recordInteraction(interaction)
        }
        
        print("  ✓ Recorded \(interactions.count) interactions")
        
        // Show statistics
        let stats = await engine.statistics()
        print("\nSystem Statistics:")
        print("  Total users: \(stats.userCount)")
        print("  Total items: \(stats.itemCount)")
        print("  Total interactions: \(stats.interactionCount)")
        print("  Average interactions per user: \(String(format: "%.1f", stats.avgInteractionsPerUser))")
    }
    
    // MARK: - Example 2: Collaborative Filtering
    
    static func collaborativeFilteringDemo(engine: RecommendationEngine) async throws {
        print("Generating collaborative filtering recommendations...")
        
        let users = ["user1", "user2", "user3", "user4"]
        
        for userId in users {
            print("\n\nRecommendations for \(userId):")
            
            let recommendations = try await engine.getCollaborativeRecommendations(
                userId: userId,
                count: 5,
                strategy: .userBased
            )
            
            for (index, rec) in recommendations.enumerated() {
                print("\n  \(index + 1). \(rec.item.name)")
                print("     Score: \(String(format: "%.2f", rec.score))")
                print("     Reason: \(rec.explanation)")
                
                // Show which similar users also liked this
                if let similarUsers = rec.metadata["similar_users"] as? [String] {
                    print("     Liked by similar users: \(similarUsers.joined(separator: ", "))")
                }
            }
        }
        
        // Show item-based collaborative filtering
        print("\n\nItem-based collaborative filtering:")
        
        let itemRecommendations = try await engine.getCollaborativeRecommendations(
            userId: "user1",
            count: 3,
            strategy: .itemBased
        )
        
        print("\nFor user1 (based on item similarities):")
        for rec in itemRecommendations {
            print("  - \(rec.item.name): \(String(format: "%.2f", rec.score))")
            if let basedOn = rec.metadata["based_on_items"] as? [String] {
                print("    Based on your interest in: \(basedOn.joined(separator: ", "))")
            }
        }
    }
    
    // MARK: - Example 3: Content-Based Filtering
    
    static func contentBasedFilteringDemo(engine: RecommendationEngine) async throws {
        print("Generating content-based recommendations...")
        
        // Content-based recommendations for each user
        for userId in ["user1", "user2", "user3"] {
            print("\n\nContent-based recommendations for \(userId):")
            
            let recommendations = try await engine.getContentBasedRecommendations(
                userId: userId,
                count: 4,
                diversityFactor: 0.3
            )
            
            for (index, rec) in recommendations.enumerated() {
                print("\n  \(index + 1). \(rec.item.name)")
                print("     Category: \(rec.item.category)")
                print("     Match score: \(String(format: "%.2f", rec.score))")
                print("     Matched attributes: \(rec.matchedAttributes.joined(separator: ", "))")
            }
        }
        
        // Show recommendations based on specific item
        print("\n\nSimilar items to 'Wireless Headphones Pro':")
        
        let similarItems = try await engine.getSimilarItems(
            itemId: "item1",
            count: 3,
            similarity: .feature
        )
        
        for (index, rec) in similarItems.enumerated() {
            print("\n  \(index + 1). \(rec.item.name)")
            print("     Similarity: \(String(format: "%.2f%%", rec.similarity * 100))")
            print("     Common features: \(rec.commonFeatures.joined(separator: ", "))")
        }
    }
    
    // MARK: - Example 4: Hybrid Recommendations
    
    static func hybridRecommendationsDemo(engine: RecommendationEngine) async throws {
        print("Generating hybrid recommendations (collaborative + content)...")
        
        // Configure hybrid strategy
        let hybridConfig = HybridRecommendationConfig(
            collaborativeWeight: 0.6,
            contentWeight: 0.4,
            popularityBoost: 0.1,
            diversityPenalty: 0.2,
            enableExplanation: true
        )
        
        for userId in ["user1", "user2"] {
            print("\n\nHybrid recommendations for \(userId):")
            
            let recommendations = try await engine.getHybridRecommendations(
                userId: userId,
                count: 5,
                configuration: hybridConfig
            )
            
            for (index, rec) in recommendations.enumerated() {
                print("\n  \(index + 1). \(rec.item.name) - $\(String(format: "%.2f", rec.item.price))")
                print("     Overall score: \(String(format: "%.2f", rec.hybridScore))")
                print("     Breakdown:")
                print("       - Collaborative: \(String(format: "%.2f", rec.collaborativeScore))")
                print("       - Content-based: \(String(format: "%.2f", rec.contentScore))")
                print("       - Popularity: \(String(format: "%.2f", rec.popularityScore))")
                
                if let explanation = rec.explanation {
                    print("     Why recommended: \(explanation)")
                }
            }
        }
        
        // Show how diversity affects recommendations
        print("\n\nEffect of diversity on recommendations:")
        
        let highDiversityConfig = HybridRecommendationConfig(
            collaborativeWeight: 0.5,
            contentWeight: 0.5,
            diversityPenalty: 0.8,  // High diversity
            enableExplanation: false
        )
        
        let diverseRecs = try await engine.getHybridRecommendations(
            userId: "user1",
            count: 5,
            configuration: highDiversityConfig
        )
        
        print("\nWith high diversity:")
        let categories = Set(diverseRecs.map { $0.item.category })
        print("  Categories covered: \(categories.joined(separator: ", "))")
        print("  Items:")
        for rec in diverseRecs {
            print("    - \(rec.item.name) (\(rec.item.category))")
        }
    }
    
    // MARK: - Example 5: Real-time Personalization
    
    static func realtimePersonalizationDemo(engine: RecommendationEngine) async throws {
        print("Demonstrating real-time personalization...")
        
        let userId = "user_realtime"
        let sessionUser = User(
            id: userId,
            name: "Real-time User",
            demographics: Demographics(age: 30, gender: "M", location: "Boston"),
            preferences: UserPreferences(
                categories: ["Technology"],
                priceRange: .medium,
                qualityPreference: .high
            )
        )
        
        // Index the new user
        let userEmbedding = try await engine.generateUserEmbedding(sessionUser)
        try await engine.indexUser(sessionUser, embedding: userEmbedding)
        
        print("\nSimulating user session with real-time updates...")
        
        // Initial recommendations
        print("\n1. Initial recommendations (no history):")
        var recommendations = try await engine.getRealtimeRecommendations(
            userId: userId,
            sessionContext: SessionContext(
                currentTime: Date(),
                deviceType: "mobile",
                location: "home"
            )
        )
        
        for rec in recommendations.prefix(3) {
            print("   - \(rec.item.name): \(String(format: "%.2f", rec.score))")
        }
        
        // User views an item
        let viewInteraction = Interaction(
            userId: userId,
            itemId: "item1",
            type: .view,
            timestamp: Date()
        )
        try await engine.recordInteraction(viewInteraction)
        
        print("\n2. After viewing 'Wireless Headphones Pro':")
        recommendations = try await engine.getRealtimeRecommendations(
            userId: userId,
            sessionContext: SessionContext(
                currentTime: Date(),
                deviceType: "mobile",
                location: "home",
                recentViews: ["item1"]
            )
        )
        
        for rec in recommendations.prefix(3) {
            print("   - \(rec.item.name): \(String(format: "%.2f", rec.score))")
        }
        
        // User adds to cart
        let cartInteraction = Interaction(
            userId: userId,
            itemId: "item1",
            type: .addToCart,
            timestamp: Date()
        )
        try await engine.recordInteraction(cartInteraction)
        
        print("\n3. After adding to cart:")
        recommendations = try await engine.getRealtimeRecommendations(
            userId: userId,
            sessionContext: SessionContext(
                currentTime: Date(),
                deviceType: "mobile",
                location: "home",
                cartItems: ["item1"]
            )
        )
        
        print("   Complementary items:")
        for rec in recommendations.prefix(3) {
            print("   - \(rec.item.name): \(rec.complementaryReason ?? "")")
        }
        
        // Time-based recommendations
        print("\n4. Time-sensitive recommendations:")
        
        let morningContext = SessionContext(
            currentTime: Calendar.current.date(bySettingHour: 8, minute: 0, second: 0, of: Date())!,
            deviceType: "mobile",
            location: "commute"
        )
        
        let morningRecs = try await engine.getRealtimeRecommendations(
            userId: userId,
            sessionContext: morningContext
        )
        
        print("   Morning commute recommendations:")
        for rec in morningRecs.prefix(3) {
            print("   - \(rec.item.name)")
        }
    }
    
    // MARK: - Example 6: Cold Start Problem
    
    static func coldStartDemo(engine: RecommendationEngine) async throws {
        print("Handling cold start scenarios...")
        
        // New user with no history
        let newUser = User(
            id: "cold_user",
            name: "New User",
            demographics: Demographics(age: 26, gender: "F", location: "Seattle"),
            preferences: UserPreferences(
                categories: ["Books", "Fitness"],
                priceRange: .medium,
                qualityPreference: .high
            )
        )
        
        print("\n1. New user with preferences but no interactions:")
        
        let newUserEmbedding = try await engine.generateUserEmbedding(newUser)
        try await engine.indexUser(newUser, embedding: newUserEmbedding)
        
        let coldStartRecs = try await engine.getColdStartRecommendations(
            userId: newUser.id,
            strategy: .preferencesBased
        )
        
        print("   Recommendations based on stated preferences:")
        for rec in coldStartRecs {
            print("   - \(rec.item.name) (\(rec.item.category))")
            print("     Match reason: \(rec.explanation)")
        }
        
        // New item with no interactions
        let newItem = Item(
            id: "cold_item",
            name: "Revolutionary Gadget X",
            category: "Technology",
            subcategory: "Innovation",
            price: 399.99,
            features: ItemFeatures(
                brand: "FutureTech",
                quality: .premium,
                attributes: ["innovative", "smart", "eco-friendly"],
                popularity: 0.0  // No popularity yet
            )
        )
        
        print("\n2. New item with no interaction history:")
        
        let itemEmbedding = try await engine.generateItemEmbedding(newItem)
        try await engine.indexItem(newItem, embedding: itemEmbedding)
        
        // Find users who might be interested
        let potentialUsers = try await engine.findPotentialUsersForItem(
            itemId: newItem.id,
            count: 3
        )
        
        print("   Potential early adopters:")
        for user in potentialUsers {
            print("   - \(user.name): \(user.matchReason)")
        }
        
        // Popular items for new users
        print("\n3. Popular items strategy for new users:")
        
        let popularRecs = try await engine.getColdStartRecommendations(
            userId: "brand_new_user",
            strategy: .popularItems
        )
        
        print("   Top trending items:")
        for rec in popularRecs.prefix(5) {
            print("   - \(rec.item.name): Popularity \(String(format: "%.2f", rec.item.features.popularity))")
        }
    }
    
    // MARK: - Example 7: Explainable Recommendations
    
    static func explainableRecommendationsDemo(engine: RecommendationEngine) async throws {
        print("Generating explainable recommendations...")
        
        let userId = "user1"
        
        // Get recommendations with detailed explanations
        let explainableRecs = try await engine.getExplainableRecommendations(
            userId: userId,
            count: 3,
            explanationLevel: .detailed
        )
        
        for (index, rec) in explainableRecs.enumerated() {
            print("\n\(index + 1). Recommendation: \(rec.item.name)")
            print("   Overall Score: \(String(format: "%.2f", rec.score))")
            
            print("\n   Detailed Explanation:")
            print("   " + "-" * 40)
            
            // User preference match
            print("   User Preference Match:")
            for (preference, score) in rec.explanationDetails.preferenceMatches {
                print("     - \(preference): \(String(format: "%.2f", score))")
            }
            
            // Similar users
            if !rec.explanationDetails.similarUsers.isEmpty {
                print("\n   Similar Users Who Liked This:")
                for similarUser in rec.explanationDetails.similarUsers.prefix(3) {
                    print("     - \(similarUser.name) (similarity: \(String(format: "%.2f", similarUser.similarity)))")
                }
            }
            
            // Item features
            print("\n   Matching Item Features:")
            for feature in rec.explanationDetails.matchingFeatures {
                print("     - \(feature)")
            }
            
            // Past behavior
            if !rec.explanationDetails.relatedPastItems.isEmpty {
                print("\n   Based on Your Interest In:")
                for pastItem in rec.explanationDetails.relatedPastItems {
                    print("     - \(pastItem)")
                }
            }
            
            // Confidence and reasoning
            print("\n   Recommendation Confidence: \(String(format: "%.2f%%", rec.explanationDetails.confidence * 100))")
            print("   Primary Reason: \(rec.explanationDetails.primaryReason)")
            
            // Visual explanation
            print("\n   Score Breakdown Visualization:")
            print("   " + generateScoreVisualization(rec.explanationDetails.scoreBreakdown))
        }
        
        // A/B test different explanation styles
        print("\n\nA/B Testing Explanation Styles:")
        
        let explanationStyles: [ExplanationStyle] = [.simple, .social, .feature]
        
        for style in explanationStyles {
            print("\n\(style) explanation:")
            
            let styledRec = try await engine.getExplainableRecommendations(
                userId: userId,
                count: 1,
                explanationLevel: .simple,
                style: style
            ).first!
            
            print("Item: \(styledRec.item.name)")
            print("Explanation: \(styledRec.styledExplanation)")
        }
    }
    
    // MARK: - Helper Functions
    
    static func generateScoreVisualization(_ breakdown: [String: Float]) -> String {
        var visualization = ""
        
        for (component, score) in breakdown.sorted(by: { $0.value > $1.value }) {
            let barLength = Int(score * 20)
            let bar = String(repeating: "█", count: barLength)
            let padding = String(repeating: " ", count: 20 - component.count)
            visualization += "\n   \(component):\(padding) \(bar) \(String(format: "%.2f", score))"
        }
        
        return visualization
    }
    
    static func elapsedTime(since start: DispatchTime) -> TimeInterval {
        let end = DispatchTime.now()
        return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }
}

// MARK: - Supporting Types

struct RecommendationSystemConfiguration {
    let dimensions: Int
    let userEmbeddingSize: Int
    let itemEmbeddingSize: Int
    let interactionWeight: Float
    let contentWeight: Float
    let enableRealTimeUpdates: Bool
    let minInteractionsForRecommendation: Int
}

actor RecommendationEngine {
    private let configuration: RecommendationSystemConfiguration
    private let userIndex: any VectorIndex
    private let itemIndex: any VectorIndex
    private let interactionStore: InteractionStore
    private let userStore: UserStore
    private let itemStore: ItemStore
    private let embeddingModel: EmbeddingModel
    
    init(configuration: RecommendationSystemConfiguration) async throws {
        self.configuration = configuration
        
        // Initialize user index
        let userIndexConfig = HNSWConfiguration(
            dimensions: configuration.userEmbeddingSize,
            m: 16,
            efConstruction: 200
        )
        self.userIndex = try await HNSWIndex<SIMD128<Float>, UserMetadata>(
            configuration: userIndexConfig
        )
        
        // Initialize item index
        let itemIndexConfig = HNSWConfiguration(
            dimensions: configuration.itemEmbeddingSize,
            m: 32,
            efConstruction: 200
        )
        self.itemIndex = try await HNSWIndex<SIMD128<Float>, ItemMetadata>(
            configuration: itemIndexConfig
        )
        
        // Initialize stores
        self.interactionStore = InteractionStore()
        self.userStore = UserStore()
        self.itemStore = ItemStore()
        
        // Initialize embedding model
        self.embeddingModel = EmbeddingModel(
            userDimensions: configuration.userEmbeddingSize,
            itemDimensions: configuration.itemEmbeddingSize
        )
    }
    
    func generateUserEmbedding(_ user: User) async throws -> [Float] {
        embeddingModel.encodeUser(user)
    }
    
    func generateItemEmbedding(_ item: Item) async throws -> [Float] {
        embeddingModel.encodeItem(item)
    }
    
    func indexUser(_ user: User, embedding: [Float]) async throws {
        try await userStore.store(user)
        
        let metadata = UserMetadata(
            userId: user.id,
            categories: user.preferences.categories,
            location: user.demographics.location
        )
        
        let vector = SIMD128<Float>(embedding)
        let entry = VectorEntry(
            id: user.id,
            vector: vector,
            metadata: metadata
        )
        
        _ = try await userIndex.insert(entry)
    }
    
    func indexItem(_ item: Item, embedding: [Float]) async throws {
        try await itemStore.store(item)
        
        let metadata = ItemMetadata(
            itemId: item.id,
            category: item.category,
            price: item.price,
            popularity: item.features.popularity
        )
        
        let vector = SIMD128<Float>(embedding)
        let entry = VectorEntry(
            id: item.id,
            vector: vector,
            metadata: metadata
        )
        
        _ = try await itemIndex.insert(entry)
    }
    
    func recordInteraction(_ interaction: Interaction) async throws {
        try await interactionStore.record(interaction)
        
        // Update embeddings if real-time updates are enabled
        if configuration.enableRealTimeUpdates {
            await updateEmbeddingsForInteraction(interaction)
        }
    }
    
    func getCollaborativeRecommendations(
        userId: String,
        count: Int,
        strategy: CollaborativeStrategy
    ) async throws -> [Recommendation] {
        switch strategy {
        case .userBased:
            return try await getUserBasedRecommendations(userId: userId, count: count)
        case .itemBased:
            return try await getItemBasedRecommendations(userId: userId, count: count)
        }
    }
    
    func getContentBasedRecommendations(
        userId: String,
        count: Int,
        diversityFactor: Float
    ) async throws -> [ContentRecommendation] {
        guard let user = try await userStore.get(userId) else {
            throw RecommendationError.userNotFound
        }
        
        let userInteractions = await interactionStore.getInteractions(for: userId)
        let likedItems = userInteractions
            .filter { $0.rating ?? 0 >= 4 || $0.type == .purchase }
            .compactMap { interaction in
                try? await itemStore.get(interaction.itemId)
            }
        
        // Generate content profile
        let contentProfile = generateContentProfile(user: user, likedItems: await likedItems)
        
        // Find similar items
        var recommendations: [ContentRecommendation] = []
        
        for item in try await itemStore.getAllItems() {
            if !userInteractions.contains(where: { $0.itemId == item.id }) {
                let similarity = calculateContentSimilarity(
                    profile: contentProfile,
                    item: item
                )
                
                if similarity > 0.5 {
                    let matchedAttributes = findMatchedAttributes(
                        profile: contentProfile,
                        item: item
                    )
                    
                    recommendations.append(ContentRecommendation(
                        item: item,
                        score: similarity,
                        matchedAttributes: matchedAttributes,
                        explanation: "Based on your interest in \(matchedAttributes.joined(separator: ", "))"
                    ))
                }
            }
        }
        
        // Apply diversity
        if diversityFactor > 0 {
            recommendations = applyDiversity(
                recommendations: recommendations,
                factor: diversityFactor
            )
        }
        
        return Array(recommendations.sorted { $0.score > $1.score }.prefix(count))
    }
    
    func getHybridRecommendations(
        userId: String,
        count: Int,
        configuration: HybridRecommendationConfig
    ) async throws -> [HybridRecommendation] {
        // Get collaborative recommendations
        let collaborativeRecs = try await getCollaborativeRecommendations(
            userId: userId,
            count: count * 2,
            strategy: .userBased
        )
        
        // Get content-based recommendations
        let contentRecs = try await getContentBasedRecommendations(
            userId: userId,
            count: count * 2,
            diversityFactor: 0
        )
        
        // Combine and score
        var hybridScores: [String: HybridRecommendation] = [:]
        
        // Process collaborative recommendations
        for rec in collaborativeRecs {
            let hybridRec = HybridRecommendation(
                item: rec.item,
                hybridScore: rec.score * configuration.collaborativeWeight,
                collaborativeScore: rec.score,
                contentScore: 0,
                popularityScore: rec.item.features.popularity * configuration.popularityBoost,
                explanation: configuration.enableExplanation ? 
                    "Recommended based on similar users' preferences" : nil
            )
            hybridScores[rec.item.id] = hybridRec
        }
        
        // Process content recommendations
        for rec in contentRecs {
            if var existing = hybridScores[rec.item.id] {
                existing.contentScore = rec.score
                existing.hybridScore += rec.score * configuration.contentWeight
                hybridScores[rec.item.id] = existing
            } else {
                let hybridRec = HybridRecommendation(
                    item: rec.item,
                    hybridScore: rec.score * configuration.contentWeight + 
                                rec.item.features.popularity * configuration.popularityBoost,
                    collaborativeScore: 0,
                    contentScore: rec.score,
                    popularityScore: rec.item.features.popularity * configuration.popularityBoost,
                    explanation: configuration.enableExplanation ?
                        "Matches your preferences for \(rec.matchedAttributes.joined(separator: ", "))" : nil
                )
                hybridScores[rec.item.id] = hybridRec
            }
        }
        
        // Apply diversity penalty if needed
        var results = Array(hybridScores.values)
        if configuration.diversityPenalty > 0 {
            results = applyDiversityPenalty(
                recommendations: results,
                penalty: configuration.diversityPenalty
            )
        }
        
        return Array(results.sorted { $0.hybridScore > $1.hybridScore }.prefix(count))
    }
    
    func getRealtimeRecommendations(
        userId: String,
        sessionContext: SessionContext
    ) async throws -> [RealtimeRecommendation] {
        // Get base recommendations
        var recommendations = try await getHybridRecommendations(
            userId: userId,
            count: 10,
            configuration: HybridRecommendationConfig()
        ).map { RealtimeRecommendation(base: $0) }
        
        // Adjust based on session context
        
        // Time-based adjustments
        let hour = Calendar.current.component(.hour, from: sessionContext.currentTime)
        recommendations = adjustRecommendationsForTime(
            recommendations: recommendations,
            hour: hour
        )
        
        // Device-based adjustments
        if sessionContext.deviceType == "mobile" {
            // Prioritize mobile-friendly items
            recommendations = recommendations.map { rec in
                var adjusted = rec
                if rec.item.features.attributes.contains("mobile-friendly") {
                    adjusted.score *= 1.2
                }
                return adjusted
            }
        }
        
        // Recent views boost
        if let recentViews = sessionContext.recentViews {
            for viewedId in recentViews {
                if let viewedItem = try await itemStore.get(viewedId) {
                    // Boost similar items
                    recommendations = boostSimilarItems(
                        recommendations: recommendations,
                        referenceItem: viewedItem,
                        boost: 1.3
                    )
                }
            }
        }
        
        // Cart complementary items
        if let cartItems = sessionContext.cartItems {
            for cartId in cartItems {
                if let cartItem = try await itemStore.get(cartId) {
                    // Find complementary items
                    recommendations = findComplementaryItems(
                        recommendations: recommendations,
                        cartItem: cartItem
                    )
                }
            }
        }
        
        return Array(recommendations.sorted { $0.score > $1.score }.prefix(5))
    }
    
    func getColdStartRecommendations(
        userId: String,
        strategy: ColdStartStrategy
    ) async throws -> [Recommendation] {
        switch strategy {
        case .popularItems:
            return try await getPopularItemRecommendations(count: 10)
            
        case .preferencesBased:
            guard let user = try await userStore.get(userId) else {
                throw RecommendationError.userNotFound
            }
            
            // Find items matching user preferences
            var recommendations: [Recommendation] = []
            
            for item in try await itemStore.getAllItems() {
                if user.preferences.categories.contains(item.category) {
                    let score = calculatePreferenceMatch(user: user, item: item)
                    
                    recommendations.append(Recommendation(
                        item: item,
                        score: score,
                        explanation: "Matches your interest in \(item.category)",
                        metadata: ["cold_start": true]
                    ))
                }
            }
            
            return Array(recommendations.sorted { $0.score > $1.score }.prefix(10))
        }
    }
    
    func getSimilarItems(
        itemId: String,
        count: Int,
        similarity: SimilarityType
    ) async throws -> [ItemSimilarity] {
        guard let item = try await itemStore.get(itemId) else {
            throw RecommendationError.itemNotFound
        }
        
        let embedding = generateItemEmbedding(item)
        let vector = SIMD128<Float>(embedding)
        
        let results = try await itemIndex.search(
            query: vector,
            k: count + 1 // +1 because the item itself will be in results
        )
        
        var similarities: [ItemSimilarity] = []
        
        for result in results {
            if result.metadata.itemId != itemId {
                if let similarItem = try await itemStore.get(result.metadata.itemId) {
                    let commonFeatures = findCommonFeatures(item1: item, item2: similarItem)
                    
                    similarities.append(ItemSimilarity(
                        item: similarItem,
                        similarity: 1.0 - result.distance,
                        commonFeatures: commonFeatures
                    ))
                }
            }
        }
        
        return similarities
    }
    
    func findPotentialUsersForItem(
        itemId: String,
        count: Int
    ) async throws -> [PotentialUser] {
        guard let item = try await itemStore.get(itemId) else {
            throw RecommendationError.itemNotFound
        }
        
        var potentialUsers: [PotentialUser] = []
        
        // Find users with matching preferences
        for user in try await userStore.getAllUsers() {
            if user.preferences.categories.contains(item.category) {
                let matchScore = calculateUserItemMatch(user: user, item: item)
                
                if matchScore > 0.6 {
                    potentialUsers.append(PotentialUser(
                        id: user.id,
                        name: user.name,
                        matchScore: matchScore,
                        matchReason: "Interested in \(item.category) products"
                    ))
                }
            }
        }
        
        return Array(potentialUsers.sorted { $0.matchScore > $1.matchScore }.prefix(count))
    }
    
    func getExplainableRecommendations(
        userId: String,
        count: Int,
        explanationLevel: ExplanationLevel,
        style: ExplanationStyle = .balanced
    ) async throws -> [ExplainableRecommendation] {
        let recommendations = try await getHybridRecommendations(
            userId: userId,
            count: count,
            configuration: HybridRecommendationConfig(enableExplanation: true)
        )
        
        var explainableRecs: [ExplainableRecommendation] = []
        
        for rec in recommendations {
            let details = try await generateExplanationDetails(
                userId: userId,
                recommendation: rec,
                level: explanationLevel
            )
            
            let styledExplanation = generateStyledExplanation(
                item: rec.item,
                details: details,
                style: style
            )
            
            explainableRecs.append(ExplainableRecommendation(
                item: rec.item,
                score: rec.hybridScore,
                explanation: rec.explanation ?? "",
                explanationDetails: details,
                styledExplanation: styledExplanation
            ))
        }
        
        return explainableRecs
    }
    
    func statistics() async -> RecommendationStatistics {
        RecommendationStatistics(
            userCount: await userStore.count(),
            itemCount: await itemStore.count(),
            interactionCount: await interactionStore.count(),
            avgInteractionsPerUser: await calculateAvgInteractionsPerUser()
        )
    }
    
    // MARK: - Private Methods
    
    private func getUserBasedRecommendations(
        userId: String,
        count: Int
    ) async throws -> [Recommendation] {
        guard let user = try await userStore.get(userId) else {
            throw RecommendationError.userNotFound
        }
        
        // Find similar users
        let userEmbedding = generateUserEmbedding(user)
        let vector = SIMD128<Float>(userEmbedding)
        
        let similarUsers = try await userIndex.search(
            query: vector,
            k: 20
        )
        
        // Get items liked by similar users
        var itemScores: [String: Float] = [:]
        var itemSimilarUsers: [String: [String]] = [:]
        
        for similarUser in similarUsers {
            if similarUser.metadata.userId != userId {
                let interactions = await interactionStore.getInteractions(
                    for: similarUser.metadata.userId
                )
                
                for interaction in interactions {
                    if interaction.type == .purchase || (interaction.rating ?? 0) >= 4 {
                        let similarity = 1.0 - similarUser.distance
                        itemScores[interaction.itemId, default: 0] += similarity
                        itemSimilarUsers[interaction.itemId, default: []].append(
                            similarUser.metadata.userId
                        )
                    }
                }
            }
        }
        
        // Filter out items user already interacted with
        let userInteractions = await interactionStore.getInteractions(for: userId)
        let interactedItems = Set(userInteractions.map { $0.itemId })
        
        // Create recommendations
        var recommendations: [Recommendation] = []
        
        for (itemId, score) in itemScores {
            if !interactedItems.contains(itemId) {
                if let item = try await itemStore.get(itemId) {
                    recommendations.append(Recommendation(
                        item: item,
                        score: score,
                        explanation: "Users with similar tastes also liked this",
                        metadata: [
                            "similar_users": itemSimilarUsers[itemId] ?? []
                        ]
                    ))
                }
            }
        }
        
        return Array(recommendations.sorted { $0.score > $1.score }.prefix(count))
    }
    
    private func getItemBasedRecommendations(
        userId: String,
        count: Int
    ) async throws -> [Recommendation] {
        let userInteractions = await interactionStore.getInteractions(for: userId)
        let likedItems = userInteractions
            .filter { $0.type == .purchase || (interaction.rating ?? 0) >= 4 }
            .compactMap { $0.itemId }
        
        var recommendations: [String: Float] = [:]
        var basedOnItems: [String: [String]] = [:]
        
        // For each liked item, find similar items
        for itemId in likedItems {
            let similarItems = try await getSimilarItems(
                itemId: itemId,
                count: 10,
                similarity: .feature
            )
            
            for similar in similarItems {
                if !likedItems.contains(similar.item.id) {
                    recommendations[similar.item.id, default: 0] += similar.similarity
                    basedOnItems[similar.item.id, default: []].append(itemId)
                }
            }
        }
        
        // Create recommendation objects
        var results: [Recommendation] = []
        
        for (itemId, score) in recommendations {
            if let item = try await itemStore.get(itemId) {
                results.append(Recommendation(
                    item: item,
                    score: score,
                    explanation: "Similar to items you've liked",
                    metadata: [
                        "based_on_items": basedOnItems[itemId] ?? []
                    ]
                ))
            }
        }
        
        return Array(results.sorted { $0.score > $1.score }.prefix(count))
    }
    
    private func updateEmbeddingsForInteraction(_ interaction: Interaction) async {
        // Update user embedding based on interaction
        // This would involve retraining or fine-tuning the embedding model
        // For demo purposes, we'll skip the actual implementation
    }
    
    private func generateContentProfile(user: User, likedItems: [Item]) -> ContentProfile {
        var categoryWeights: [String: Float] = [:]
        var attributeWeights: [String: Float] = [:]
        
        // User preferences
        for category in user.preferences.categories {
            categoryWeights[category] = 1.0
        }
        
        // Learn from liked items
        for item in likedItems {
            categoryWeights[item.category, default: 0] += 0.5
            
            for attribute in item.features.attributes {
                attributeWeights[attribute, default: 0] += 0.3
            }
        }
        
        return ContentProfile(
            categoryWeights: categoryWeights,
            attributeWeights: attributeWeights,
            priceRange: user.preferences.priceRange,
            qualityPreference: user.preferences.qualityPreference
        )
    }
    
    private func calculateContentSimilarity(profile: ContentProfile, item: Item) -> Float {
        var score: Float = 0
        
        // Category match
        if let categoryWeight = profile.categoryWeights[item.category] {
            score += categoryWeight * 0.4
        }
        
        // Attribute match
        for attribute in item.features.attributes {
            if let attrWeight = profile.attributeWeights[attribute] {
                score += attrWeight * 0.2
            }
        }
        
        // Price range match
        if isPriceInRange(price: item.price, range: profile.priceRange) {
            score += 0.2
        }
        
        // Quality match
        if item.features.quality == profile.qualityPreference {
            score += 0.2
        }
        
        return min(score, 1.0)
    }
    
    private func findMatchedAttributes(profile: ContentProfile, item: Item) -> [String] {
        item.features.attributes.filter { attribute in
            profile.attributeWeights[attribute] != nil
        }
    }
    
    private func applyDiversity<T: ScoredRecommendation>(
        recommendations: [T],
        factor: Float
    ) -> [T] {
        var diverse: [T] = []
        var categories: Set<String> = []
        
        for rec in recommendations.sorted(by: { $0.score > $1.score }) {
            if !categories.contains(rec.item.category) || diverse.count < 3 {
                diverse.append(rec)
                categories.insert(rec.item.category)
            } else {
                // Apply diversity penalty
                var penalized = rec
                penalized.score *= (1.0 - factor)
                diverse.append(penalized)
            }
        }
        
        return diverse
    }
    
    private func applyDiversityPenalty(
        recommendations: [HybridRecommendation],
        penalty: Float
    ) -> [HybridRecommendation] {
        var results = recommendations
        var categories: [String: Int] = [:]
        
        for i in 0..<results.count {
            let category = results[i].item.category
            let count = categories[category, default: 0]
            
            if count > 0 {
                results[i].hybridScore *= (1.0 - penalty * Float(count) * 0.1)
            }
            
            categories[category] = count + 1
        }
        
        return results
    }
    
    private func adjustRecommendationsForTime(
        recommendations: [RealtimeRecommendation],
        hour: Int
    ) -> [RealtimeRecommendation] {
        recommendations.map { rec in
            var adjusted = rec
            
            // Morning boost for certain categories
            if hour >= 6 && hour <= 10 {
                if rec.item.category == "Books" || rec.item.category == "Fitness" {
                    adjusted.score *= 1.2
                }
            }
            
            // Evening boost for entertainment
            else if hour >= 18 && hour <= 23 {
                if rec.item.category == "Gaming" || rec.item.category == "Movies" {
                    adjusted.score *= 1.2
                }
            }
            
            return adjusted
        }
    }
    
    private func boostSimilarItems(
        recommendations: [RealtimeRecommendation],
        referenceItem: Item,
        boost: Float
    ) -> [RealtimeRecommendation] {
        recommendations.map { rec in
            var adjusted = rec
            
            if rec.item.category == referenceItem.category {
                adjusted.score *= boost
            }
            
            // Check for common attributes
            let commonAttrs = Set(rec.item.features.attributes)
                .intersection(referenceItem.features.attributes)
            
            if !commonAttrs.isEmpty {
                adjusted.score *= (1.0 + Float(commonAttrs.count) * 0.1)
            }
            
            return adjusted
        }
    }
    
    private func findComplementaryItems(
        recommendations: [RealtimeRecommendation],
        cartItem: Item
    ) -> [RealtimeRecommendation] {
        recommendations.map { rec in
            var adjusted = rec
            
            // Define complementary relationships
            let complementaryPairs: [(String, String)] = [
                ("Technology", "Technology"), // Accessories
                ("Books", "Books"), // Related books
                ("Fitness", "Fitness") // Fitness accessories
            ]
            
            for (cat1, cat2) in complementaryPairs {
                if cartItem.category == cat1 && rec.item.category == cat2 {
                    adjusted.score *= 1.5
                    adjusted.complementaryReason = "Goes well with \(cartItem.name)"
                }
            }
            
            return adjusted
        }
    }
    
    private func getPopularItemRecommendations(count: Int) async throws -> [Recommendation] {
        let allItems = try await itemStore.getAllItems()
        
        let popularItems = allItems
            .sorted { $0.features.popularity > $1.features.popularity }
            .prefix(count)
        
        return popularItems.map { item in
            Recommendation(
                item: item,
                score: item.features.popularity,
                explanation: "Trending item",
                metadata: ["popularity_rank": item.features.popularity]
            )
        }
    }
    
    private func calculatePreferenceMatch(user: User, item: Item) -> Float {
        var score: Float = 0
        
        // Category match
        if user.preferences.categories.contains(item.category) {
            score += 0.5
        }
        
        // Price range match
        if isPriceInRange(price: item.price, range: user.preferences.priceRange) {
            score += 0.3
        }
        
        // Quality preference match
        if item.features.quality == user.preferences.qualityPreference {
            score += 0.2
        }
        
        return score
    }
    
    private func isPriceInRange(price: Double, range: PriceRange) -> Bool {
        switch range {
        case .low:
            return price < 50
        case .medium:
            return price >= 50 && price < 200
        case .high:
            return price >= 200
        }
    }
    
    private func findCommonFeatures(item1: Item, item2: Item) -> [String] {
        var common: [String] = []
        
        if item1.category == item2.category {
            common.append(item1.category)
        }
        
        let commonAttrs = Set(item1.features.attributes)
            .intersection(item2.features.attributes)
        
        common.append(contentsOf: commonAttrs)
        
        return common
    }
    
    private func calculateUserItemMatch(user: User, item: Item) -> Float {
        calculatePreferenceMatch(user: user, item: item)
    }
    
    private func generateExplanationDetails(
        userId: String,
        recommendation: HybridRecommendation,
        level: ExplanationLevel
    ) async throws -> ExplanationDetails {
        var details = ExplanationDetails()
        
        // Preference matches
        if let user = try await userStore.get(userId) {
            for category in user.preferences.categories {
                if recommendation.item.category == category {
                    details.preferenceMatches["Category"] = 1.0
                }
            }
        }
        
        // Similar users
        if level == .detailed {
            let similarUsers = try await findSimilarUsersWhoLiked(
                itemId: recommendation.item.id,
                excludeUserId: userId
            )
            details.similarUsers = similarUsers
        }
        
        // Matching features
        details.matchingFeatures = recommendation.item.features.attributes
        
        // Related past items
        let userInteractions = await interactionStore.getInteractions(for: userId)
        let relatedItems = userInteractions
            .filter { $0.type == .purchase }
            .compactMap { interaction in
                try? await itemStore.get(interaction.itemId)
            }
            .filter { $0.category == recommendation.item.category }
            .map { $0.name }
        
        details.relatedPastItems = await relatedItems
        
        // Score breakdown
        details.scoreBreakdown = [
            "Collaborative": recommendation.collaborativeScore,
            "Content": recommendation.contentScore,
            "Popularity": recommendation.popularityScore
        ]
        
        // Confidence and primary reason
        details.confidence = min(recommendation.hybridScore, 1.0)
        details.primaryReason = determineS(recommendation: recommendation)
        
        return details
    }
    
    private func findSimilarUsersWhoLiked(
        itemId: String,
        excludeUserId: String
    ) async throws -> [SimilarUser] {
        let interactions = await interactionStore.getAllInteractions()
        
        let usersWhoLiked = interactions
            .filter { $0.itemId == itemId && $0.userId != excludeUserId }
            .filter { $0.type == .purchase || ($0.rating ?? 0) >= 4 }
            .map { $0.userId }
        
        var similarUsers: [SimilarUser] = []
        
        for userId in Set(usersWhoLiked).prefix(3) {
            if let user = try await userStore.get(userId) {
                similarUsers.append(SimilarUser(
                    id: userId,
                    name: user.name,
                    similarity: Float.random(in: 0.7...0.95) // Mock similarity
                ))
            }
        }
        
        return similarUsers
    }
    
    private func determinePrimaryReason(recommendation: HybridRecommendation) -> String {
        if recommendation.collaborativeScore > recommendation.contentScore {
            return "Popular with similar users"
        } else if recommendation.contentScore > recommendation.collaborativeScore {
            return "Matches your preferences"
        } else {
            return "Balanced recommendation"
        }
    }
    
    private func generateStyledExplanation(
        item: Item,
        details: ExplanationDetails,
        style: ExplanationStyle
    ) -> String {
        switch style {
        case .simple:
            return "We think you'll like this based on your interests"
            
        case .social:
            if !details.similarUsers.isEmpty {
                return "\(details.similarUsers.count) people with similar tastes loved this"
            } else {
                return "Trending in your interest areas"
            }
            
        case .feature:
            let features = details.matchingFeatures.prefix(2).joined(separator: " and ")
            return "Perfect for someone interested in \(features)"
            
        case .balanced:
            return details.primaryReason
        }
    }
    
    private func calculateAvgInteractionsPerUser() async -> Double {
        let userCount = await userStore.count()
        let interactionCount = await interactionStore.count()
        
        return userCount > 0 ? Double(interactionCount) / Double(userCount) : 0
    }
}

// Data models
struct User: Sendable {
    let id: String
    let name: String
    let demographics: Demographics
    let preferences: UserPreferences
}

struct Demographics: Sendable {
    let age: Int
    let gender: String
    let location: String
}

struct UserPreferences: Sendable {
    let categories: [String]
    let priceRange: PriceRange
    let qualityPreference: QualityLevel
}

struct Item: Sendable {
    let id: String
    let name: String
    let category: String
    let subcategory: String
    let price: Double
    let features: ItemFeatures
}

struct ItemFeatures: Sendable {
    let brand: String
    let quality: QualityLevel
    let attributes: [String]
    let popularity: Float
}

struct Interaction: Sendable {
    let userId: String
    let itemId: String
    let type: InteractionType
    let rating: Int?
    let timestamp: Date
}

// Enums
enum PriceRange: Sendable {
    case low
    case medium
    case high
}

enum QualityLevel: Sendable {
    case basic
    case medium
    case high
    case premium
}

enum InteractionType: Sendable {
    case view
    case purchase
    case rating
    case wishlist
    case addToCart
}

enum CollaborativeStrategy {
    case userBased
    case itemBased
}

enum ColdStartStrategy {
    case popularItems
    case preferencesBased
}

enum SimilarityType {
    case feature
    case interaction
    case hybrid
}

enum ExplanationLevel {
    case simple
    case detailed
}

enum ExplanationStyle {
    case simple
    case social
    case feature
    case balanced
}

// Recommendation types
protocol ScoredRecommendation {
    var item: Item { get }
    var score: Float { get set }
}

struct Recommendation: ScoredRecommendation {
    let item: Item
    var score: Float
    let explanation: String
    let metadata: [String: Any]
}

struct ContentRecommendation: ScoredRecommendation {
    let item: Item
    var score: Float
    let matchedAttributes: [String]
    let explanation: String
}

struct HybridRecommendation {
    let item: Item
    var hybridScore: Float
    var collaborativeScore: Float
    var contentScore: Float
    var popularityScore: Float
    let explanation: String?
}

struct RealtimeRecommendation: ScoredRecommendation {
    let item: Item
    var score: Float
    var complementaryReason: String?
    
    init(base: HybridRecommendation) {
        self.item = base.item
        self.score = base.hybridScore
        self.complementaryReason = nil
    }
}

struct ItemSimilarity {
    let item: Item
    let similarity: Float
    let commonFeatures: [String]
}

struct PotentialUser {
    let id: String
    let name: String
    let matchScore: Float
    let matchReason: String
}

struct ExplainableRecommendation {
    let item: Item
    let score: Float
    let explanation: String
    let explanationDetails: ExplanationDetails
    let styledExplanation: String
}

struct ExplanationDetails {
    var preferenceMatches: [String: Float] = [:]
    var similarUsers: [SimilarUser] = []
    var matchingFeatures: [String] = []
    var relatedPastItems: [String] = []
    var scoreBreakdown: [String: Float] = [:]
    var confidence: Float = 0
    var primaryReason: String = ""
}

struct SimilarUser {
    let id: String
    let name: String
    let similarity: Float
}

// Context types
struct SessionContext {
    let currentTime: Date
    let deviceType: String
    let location: String
    let recentViews: [String]? = nil
    let cartItems: [String]? = nil
}

struct ContentProfile {
    let categoryWeights: [String: Float]
    let attributeWeights: [String: Float]
    let priceRange: PriceRange
    let qualityPreference: QualityLevel
}

// Configuration types
struct HybridRecommendationConfig {
    let collaborativeWeight: Float = 0.5
    let contentWeight: Float = 0.5
    let popularityBoost: Float = 0.1
    let diversityPenalty: Float = 0.0
    let enableExplanation: Bool = false
}

// Statistics
struct RecommendationStatistics {
    let userCount: Int
    let itemCount: Int
    let interactionCount: Int
    let avgInteractionsPerUser: Double
}

// Metadata for vector storage
struct UserMetadata: Codable, Sendable {
    let userId: String
    let categories: [String]
    let location: String
}

struct ItemMetadata: Codable, Sendable {
    let itemId: String
    let category: String
    let price: Double
    let popularity: Float
}

// Mock stores
actor UserStore {
    private var users: [String: User] = [:]
    
    func store(_ user: User) async throws {
        users[user.id] = user
    }
    
    func get(_ id: String) async throws -> User? {
        users[id]
    }
    
    func getAllUsers() async throws -> [User] {
        Array(users.values)
    }
    
    func count() async -> Int {
        users.count
    }
}

actor ItemStore {
    private var items: [String: Item] = [:]
    
    func store(_ item: Item) async throws {
        items[item.id] = item
    }
    
    func get(_ id: String) async throws -> Item? {
        items[id]
    }
    
    func getAllItems() async throws -> [Item] {
        Array(items.values)
    }
    
    func count() async -> Int {
        items.count
    }
}

actor InteractionStore {
    private var interactions: [Interaction] = []
    
    func record(_ interaction: Interaction) async throws {
        interactions.append(interaction)
    }
    
    func getInteractions(for userId: String) async -> [Interaction] {
        interactions.filter { $0.userId == userId }
    }
    
    func getAllInteractions() async -> [Interaction] {
        interactions
    }
    
    func count() async -> Int {
        interactions.count
    }
}

// Embedding model
struct EmbeddingModel {
    let userDimensions: Int
    let itemDimensions: Int
    
    func encodeUser(_ user: User) -> [Float] {
        var embedding = [Float](repeating: 0, count: userDimensions)
        
        // Encode demographics
        embedding[0] = Float(user.demographics.age) / 100.0
        embedding[1] = user.demographics.gender == "M" ? 1.0 : 0.0
        
        // Encode preferences
        let categories = ["Technology", "Books", "Gaming", "Fitness", "Home & Garden", "Cooking", "Movies", "Music", "Fashion"]
        for (i, category) in categories.enumerated() where i + 2 < userDimensions {
            embedding[i + 2] = user.preferences.categories.contains(category) ? 1.0 : 0.0
        }
        
        // Price range encoding
        switch user.preferences.priceRange {
        case .low: embedding[20] = 0.33
        case .medium: embedding[20] = 0.66
        case .high: embedding[20] = 1.0
        }
        
        // Add some noise for variety
        for i in 30..<userDimensions {
            embedding[i] = Float.random(in: -0.1...0.1)
        }
        
        // Normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            embedding = embedding.map { $0 / norm }
        }
        
        return embedding
    }
    
    func encodeItem(_ item: Item) -> [Float] {
        var embedding = [Float](repeating: 0, count: itemDimensions)
        
        // Encode category
        let categories = ["Technology", "Books", "Gaming", "Fitness", "Home & Garden", "Cooking"]
        for (i, category) in categories.enumerated() where i < itemDimensions {
            embedding[i] = item.category == category ? 1.0 : 0.0
        }
        
        // Encode price (normalized)
        embedding[10] = Float(min(item.price / 1000.0, 1.0))
        
        // Encode quality
        switch item.features.quality {
        case .basic: embedding[11] = 0.25
        case .medium: embedding[11] = 0.5
        case .high: embedding[11] = 0.75
        case .premium: embedding[11] = 1.0
        }
        
        // Encode attributes
        let allAttributes = ["wireless", "smart", "eco-friendly", "bestseller", "mechanical", "voice-control"]
        for (i, attr) in allAttributes.enumerated() where i + 20 < itemDimensions {
            embedding[i + 20] = item.features.attributes.contains(attr) ? 1.0 : 0.0
        }
        
        // Popularity
        embedding[30] = item.features.popularity
        
        // Add some noise
        for i in 40..<itemDimensions {
            embedding[i] = Float.random(in: -0.1...0.1)
        }
        
        // Normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            embedding = embedding.map { $0 / norm }
        }
        
        return embedding
    }
}

// Error types
enum RecommendationError: Error {
    case userNotFound
    case itemNotFound
    case insufficientData
}

// String multiplication helper
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// SIMD initialization helper
extension SIMD128<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(128, array.count) {
            self[i] = array[i]
        }
    }
}