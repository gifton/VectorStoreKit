import Foundation
import VectorStoreKit

// MARK: - Geospatial Data Structures

/// Represents a geographic location with coordinates and attributes
struct Location {
    let id: String
    let name: String
    let latitude: Double
    let longitude: Double
    let category: LocationCategory
    let amenities: Set<String>
    let popularityScore: Float
    let priceLevel: Int // 1-4 scale
    
    /// Calculate Haversine distance to another location in kilometers
    func distance(to other: Location) -> Double {
        let R = 6371.0 // Earth's radius in kilometers
        let lat1Rad = latitude * .pi / 180
        let lat2Rad = other.latitude * .pi / 180
        let deltaLat = (other.latitude - latitude) * .pi / 180
        let deltaLon = (other.longitude - longitude) * .pi / 180
        
        let a = sin(deltaLat/2) * sin(deltaLat/2) +
                cos(lat1Rad) * cos(lat2Rad) *
                sin(deltaLon/2) * sin(deltaLon/2)
        let c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    }
}

enum LocationCategory: String, CaseIterable {
    case restaurant = "restaurant"
    case hotel = "hotel"
    case attraction = "attraction"
    case shopping = "shopping"
    case transportation = "transportation"
    case park = "park"
    case museum = "museum"
    case cafe = "cafe"
}

/// Represents a route or trajectory
struct Route {
    let id: String
    let waypoints: [Location]
    let totalDistance: Double
    let estimatedDuration: TimeInterval
    let transportMode: TransportMode
    
    enum TransportMode: String {
        case walking, driving, cycling, transit
    }
}

/// Represents a geographic region for geo-fencing
struct GeoRegion {
    let center: (latitude: Double, longitude: Double)
    let radiusKm: Double
    let name: String
    
    func contains(_ location: Location) -> Bool {
        let centerLocation = Location(
            id: "center",
            name: "",
            latitude: center.latitude,
            longitude: center.longitude,
            category: .attraction,
            amenities: [],
            popularityScore: 0,
            priceLevel: 1
        )
        return centerLocation.distance(to: location) <= radiusKm
    }
}

// MARK: - Geospatial Embedding Strategies

/// Converts geospatial data into vector embeddings
struct GeoSpatialEmbedder {
    
    /// Strategy 1: Coordinate-based embedding with normalization
    static func coordinateEmbedding(location: Location) -> [Float] {
        // Normalize coordinates to [-1, 1] range
        let normalizedLat = Float(location.latitude / 90.0)
        let normalizedLon = Float(location.longitude / 180.0)
        
        // Add sine/cosine transformations for circular continuity
        let latRad = Float(location.latitude * .pi / 180)
        let lonRad = Float(location.longitude * .pi / 180)
        
        return [
            normalizedLat,
            normalizedLon,
            sin(latRad),
            cos(latRad),
            sin(lonRad),
            cos(lonRad)
        ]
    }
    
    /// Strategy 2: Feature-based embedding for POI characteristics
    static func featureEmbedding(location: Location) -> [Float] {
        var features = [Float](repeating: 0, count: 32)
        
        // Category one-hot encoding
        if let categoryIndex = LocationCategory.allCases.firstIndex(of: location.category) {
            features[categoryIndex] = 1.0
        }
        
        // Amenity features (simplified)
        let amenityMapping = [
            "wifi": 8, "parking": 9, "outdoor_seating": 10,
            "delivery": 11, "takeout": 12, "reservations": 13,
            "wheelchair_accessible": 14, "pet_friendly": 15
        ]
        
        for (amenity, index) in amenityMapping {
            if location.amenities.contains(amenity) {
                features[index] = 1.0
            }
        }
        
        // Normalized numeric features
        features[16] = location.popularityScore
        features[17] = Float(location.priceLevel) / 4.0
        
        return features
    }
    
    /// Strategy 3: Hybrid embedding combining coordinates and features
    static func hybridEmbedding(location: Location, weights: (coordinate: Float, feature: Float) = (0.5, 0.5)) -> [Float] {
        let coordEmbed = coordinateEmbedding(location: location).map { $0 * weights.coordinate }
        let featureEmbed = featureEmbedding(location: location).map { $0 * weights.feature }
        return coordEmbed + featureEmbed
    }
    
    /// Strategy 4: Grid-based embedding for spatial indexing
    static func gridEmbedding(location: Location, gridSize: Double = 0.01) -> [Float] {
        let gridX = Int(location.longitude / gridSize)
        let gridY = Int(location.latitude / gridSize)
        
        // Create a sparse embedding based on grid cell
        var embedding = [Float](repeating: 0, count: 64)
        let hashIndex = abs((gridX &* 73856093) ^ (gridY &* 19349663)) % 64
        embedding[hashIndex] = 1.0
        
        // Add local coordinates within grid cell
        let localX = Float(location.longitude.truncatingRemainder(dividingBy: gridSize) / gridSize)
        let localY = Float(location.latitude.truncatingRemainder(dividingBy: gridSize) / gridSize)
        
        return embedding + [localX, localY]
    }
    
    /// Route embedding for trajectory similarity
    static func routeEmbedding(route: Route) -> [Float] {
        guard !route.waypoints.isEmpty else { return [] }
        
        // Start and end point features
        let start = route.waypoints.first!
        let end = route.waypoints.last!
        
        var features: [Float] = []
        
        // Geographic features
        features += coordinateEmbedding(location: start)
        features += coordinateEmbedding(location: end)
        
        // Route characteristics
        features.append(Float(route.totalDistance / 100.0)) // Normalized by 100km
        features.append(Float(route.estimatedDuration / 3600.0)) // Hours
        features.append(Float(route.waypoints.count) / 20.0) // Normalized waypoint count
        
        // Transport mode one-hot
        let modes: [Route.TransportMode] = [.walking, .driving, .cycling, .transit]
        for mode in modes {
            features.append(mode == route.transportMode ? 1.0 : 0.0)
        }
        
        // Statistical features of the route
        let latitudes = route.waypoints.map { $0.latitude }
        let longitudes = route.waypoints.map { $0.longitude }
        
        features.append(Float(latitudes.min() ?? 0) / 90.0)
        features.append(Float(latitudes.max() ?? 0) / 90.0)
        features.append(Float(longitudes.min() ?? 0) / 180.0)
        features.append(Float(longitudes.max() ?? 0) / 180.0)
        
        return features
    }
}

// MARK: - Mock Data Generator

struct MockDataGenerator {
    
    static func generateLocations() -> [Location] {
        return [
            // San Francisco landmarks and POIs
            Location(id: "sf_1", name: "Golden Gate Bridge", latitude: 37.8199, longitude: -122.4783,
                    category: .attraction, amenities: ["parking", "wheelchair_accessible"],
                    popularityScore: 0.95, priceLevel: 1),
            Location(id: "sf_2", name: "Alcatraz Island", latitude: 37.8267, longitude: -122.4230,
                    category: .attraction, amenities: ["wheelchair_accessible"],
                    popularityScore: 0.90, priceLevel: 3),
            Location(id: "sf_3", name: "Ferry Building Marketplace", latitude: 37.7956, longitude: -122.3933,
                    category: .shopping, amenities: ["wifi", "wheelchair_accessible", "pet_friendly"],
                    popularityScore: 0.85, priceLevel: 2),
            
            // Restaurants
            Location(id: "rest_1", name: "The French Laundry", latitude: 38.4045, longitude: -122.3650,
                    category: .restaurant, amenities: ["reservations", "parking"],
                    popularityScore: 0.98, priceLevel: 4),
            Location(id: "rest_2", name: "Tartine Bakery", latitude: 37.7614, longitude: -122.4241,
                    category: .cafe, amenities: ["takeout", "outdoor_seating"],
                    popularityScore: 0.88, priceLevel: 2),
            Location(id: "rest_3", name: "House of Prime Rib", latitude: 37.7934, longitude: -122.4226,
                    category: .restaurant, amenities: ["reservations", "parking"],
                    popularityScore: 0.83, priceLevel: 3),
            
            // Hotels
            Location(id: "hotel_1", name: "St. Regis San Francisco", latitude: 37.7863, longitude: -122.4020,
                    category: .hotel, amenities: ["wifi", "parking", "wheelchair_accessible", "pet_friendly"],
                    popularityScore: 0.92, priceLevel: 4),
            Location(id: "hotel_2", name: "Hotel Zephyr", latitude: 37.8075, longitude: -122.4208,
                    category: .hotel, amenities: ["wifi", "parking", "wheelchair_accessible"],
                    popularityScore: 0.78, priceLevel: 3),
            
            // Parks
            Location(id: "park_1", name: "Golden Gate Park", latitude: 37.7694, longitude: -122.4862,
                    category: .park, amenities: ["parking", "wheelchair_accessible", "pet_friendly"],
                    popularityScore: 0.93, priceLevel: 1),
            Location(id: "park_2", name: "Dolores Park", latitude: 37.7596, longitude: -122.4269,
                    category: .park, amenities: ["pet_friendly"],
                    popularityScore: 0.87, priceLevel: 1),
            
            // Museums
            Location(id: "museum_1", name: "SFMOMA", latitude: 37.7857, longitude: -122.4011,
                    category: .museum, amenities: ["wifi", "wheelchair_accessible"],
                    popularityScore: 0.89, priceLevel: 3),
            Location(id: "museum_2", name: "California Academy of Sciences", latitude: 37.7699, longitude: -122.4661,
                    category: .museum, amenities: ["wifi", "parking", "wheelchair_accessible"],
                    popularityScore: 0.91, priceLevel: 3),
            
            // Transportation hubs
            Location(id: "trans_1", name: "San Francisco International Airport", latitude: 37.6213, longitude: -122.3790,
                    category: .transportation, amenities: ["wifi", "parking", "wheelchair_accessible"],
                    popularityScore: 0.85, priceLevel: 2),
            Location(id: "trans_2", name: "Caltrain Station", latitude: 37.7764, longitude: -122.3947,
                    category: .transportation, amenities: ["wifi", "wheelchair_accessible"],
                    popularityScore: 0.75, priceLevel: 1)
        ]
    }
    
    static func generateRoutes(locations: [Location]) -> [Route] {
        return [
            // Tourist route
            Route(id: "route_1",
                  waypoints: [locations[0], locations[2], locations[10], locations[8]],
                  totalDistance: 12.5,
                  estimatedDuration: 3600,
                  transportMode: .walking),
            
            // Restaurant tour
            Route(id: "route_2",
                  waypoints: [locations[3], locations[4], locations[5]],
                  totalDistance: 45.2,
                  estimatedDuration: 5400,
                  transportMode: .driving),
            
            // Airport to hotel
            Route(id: "route_3",
                  waypoints: [locations[12], locations[6]],
                  totalDistance: 22.8,
                  estimatedDuration: 1800,
                  transportMode: .driving),
            
            // Park hopping
            Route(id: "route_4",
                  waypoints: [locations[8], locations[9]],
                  totalDistance: 3.2,
                  estimatedDuration: 2400,
                  transportMode: .cycling)
        ]
    }
}

// MARK: - Main Example Application

@main
struct GeoSpatialExample {
    
    static func main() async throws {
        print("🌍 VectorStoreKit Geospatial Example")
        print("=" * 50)
        
        // Initialize vector store with geospatial configuration
        let config = StoreConfiguration(
            dimension: 38, // Hybrid embedding dimension
            cacheConfig: .lru(maxSize: 1000),
            distanceMetric: .euclidean
        )
        
        let vectorStore = try await VectorStore(configuration: config)
        
        // Generate mock data
        let locations = MockDataGenerator.generateLocations()
        let routes = MockDataGenerator.generateRoutes(locations: locations)
        
        // Example 1: Basic location similarity search
        print("\n📍 Example 1: Location-Based Similarity Search")
        print("-" * 40)
        try await demonstrateLocationSearch(vectorStore: vectorStore, locations: locations)
        
        // Example 2: Geo-fencing and region queries
        print("\n🗺️ Example 2: Geo-Fencing and Region Queries")
        print("-" * 40)
        try await demonstrateGeoFencing(locations: locations)
        
        // Example 3: Category-based POI recommendations
        print("\n🏪 Example 3: POI Recommendations")
        print("-" * 40)
        try await demonstratePOIRecommendations(vectorStore: vectorStore, locations: locations)
        
        // Example 4: Route similarity analysis
        print("\n🛣️ Example 4: Route Similarity Analysis")
        print("-" * 40)
        try await demonstrateRouteSimilarity(routes: routes)
        
        // Example 5: Multi-criteria location search
        print("\n🔍 Example 5: Multi-Criteria Location Search")
        print("-" * 40)
        try await demonstrateMultiCriteriaSearch(vectorStore: vectorStore, locations: locations)
        
        // Example 6: Performance benchmarks
        print("\n⚡ Example 6: Geospatial Performance Benchmarks")
        print("-" * 40)
        try await benchmarkGeospatialOperations(vectorStore: vectorStore, locations: locations)
    }
    
    // MARK: - Example Implementations
    
    static func demonstrateLocationSearch(vectorStore: VectorStore, locations: [Location]) async throws {
        // Index locations using hybrid embeddings
        for location in locations {
            let embedding = GeoSpatialEmbedder.hybridEmbedding(location: location)
            let vector = Vector(id: location.id, values: embedding)
            try await vectorStore.add(vector: vector)
        }
        
        // Search for locations similar to Golden Gate Bridge
        let queryLocation = locations[0] // Golden Gate Bridge
        let queryEmbedding = GeoSpatialEmbedder.hybridEmbedding(location: queryLocation)
        let queryVector = Vector(id: "query", values: queryEmbedding)
        
        let results = try await vectorStore.search(
            query: queryVector,
            k: 5,
            threshold: 0.8
        )
        
        print("Searching for locations similar to \(queryLocation.name):")
        for (index, result) in results.enumerated() {
            if let location = locations.first(where: { $0.id == result.vector.id }) {
                let distance = queryLocation.distance(to: location)
                print("\(index + 1). \(location.name) (Category: \(location.category.rawValue))")
                print("   Vector similarity: \(result.score), Geographic distance: \(String(format: "%.2f", distance)) km")
            }
        }
    }
    
    static func demonstrateGeoFencing(locations: [Location]) async throws {
        // Define geo-fence regions
        let regions = [
            GeoRegion(center: (37.7749, -122.4194), radiusKm: 5.0, name: "Downtown SF"),
            GeoRegion(center: (37.8044, -122.2712), radiusKm: 3.0, name: "Oakland"),
            GeoRegion(center: (37.7699, -122.4661), radiusKm: 2.0, name: "Golden Gate Park Area")
        ]
        
        for region in regions {
            print("\nLocations within \(region.name) (\(region.radiusKm)km radius):")
            let locationsInRegion = locations.filter { region.contains($0) }
            
            for location in locationsInRegion {
                let centerLoc = Location(
                    id: "center",
                    name: "Center",
                    latitude: region.center.latitude,
                    longitude: region.center.longitude,
                    category: .attraction,
                    amenities: [],
                    popularityScore: 0,
                    priceLevel: 1
                )
                let distance = centerLoc.distance(to: location)
                print("  - \(location.name): \(String(format: "%.2f", distance)) km from center")
            }
        }
    }
    
    static func demonstratePOIRecommendations(vectorStore: VectorStore, locations: [Location]) async throws {
        // Create a new vector store for feature-based search
        let featureConfig = StoreConfiguration(
            dimension: 32,
            cacheConfig: .lru(maxSize: 500),
            distanceMetric: .cosine
        )
        let featureStore = try await VectorStore(configuration: featureConfig)
        
        // Index locations by features
        for location in locations {
            let embedding = GeoSpatialEmbedder.featureEmbedding(location: location)
            let vector = Vector(id: location.id, values: embedding)
            try await featureStore.add(vector: vector)
        }
        
        // Find restaurants similar to The French Laundry
        let queryLocation = locations.first { $0.id == "rest_1" }!
        let queryEmbedding = GeoSpatialEmbedder.featureEmbedding(location: queryLocation)
        let queryVector = Vector(id: "query", values: queryEmbedding)
        
        let results = try await featureStore.search(
            query: queryVector,
            k: 5,
            threshold: 0.7
        )
        
        print("\nRecommendations similar to \(queryLocation.name):")
        for (index, result) in results.enumerated() {
            if let location = locations.first(where: { $0.id == result.vector.id }),
               location.id != queryLocation.id {
                print("\(index + 1). \(location.name)")
                print("   Category: \(location.category.rawValue), Price: \(String(repeating: "$", count: location.priceLevel))")
                print("   Similarity score: \(result.score)")
            }
        }
    }
    
    static func demonstrateRouteSimilarity(routes: [Route]) async throws {
        // Create embeddings for all routes
        let routeEmbeddings = routes.map { route in
            (route: route, embedding: GeoSpatialEmbedder.routeEmbedding(route: route))
        }
        
        // Compare tourist route with others
        let queryRoute = routes[0] // Tourist route
        let queryEmbedding = routeEmbeddings[0].embedding
        
        print("\nRoutes similar to \(queryRoute.id) (Tourist route):")
        
        for (route, embedding) in routeEmbeddings where route.id != queryRoute.id {
            // Calculate cosine similarity
            let dotProduct = zip(queryEmbedding, embedding).map { $0 * $1 }.reduce(0, +)
            let queryNorm = sqrt(queryEmbedding.map { $0 * $0 }.reduce(0, +))
            let embeddingNorm = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            let similarity = dotProduct / (queryNorm * embeddingNorm)
            
            print("\n  Route: \(route.id)")
            print("  Transport: \(route.transportMode.rawValue)")
            print("  Distance: \(String(format: "%.1f", route.totalDistance)) km")
            print("  Similarity: \(String(format: "%.3f", similarity))")
        }
    }
    
    static func demonstrateMultiCriteriaSearch(vectorStore: VectorStore, locations: [Location]) async throws {
        // Create a custom scoring function that combines vector similarity with constraints
        print("\nSearching for: High-rated restaurants with outdoor seating near downtown")
        
        let downtownCenter = Location(
            id: "downtown",
            name: "Downtown Center",
            latitude: 37.7749,
            longitude: -122.4194,
            category: .restaurant,
            amenities: ["outdoor_seating"],
            popularityScore: 0.8,
            priceLevel: 2
        )
        
        let queryEmbedding = GeoSpatialEmbedder.hybridEmbedding(location: downtownCenter)
        let queryVector = Vector(id: "query", values: queryEmbedding)
        
        // Search with filtering
        let allResults = try await vectorStore.search(query: queryVector, k: 20)
        
        // Apply multi-criteria filtering
        let filteredResults = allResults.compactMap { result -> (location: Location, score: Float)? in
            guard let location = locations.first(where: { $0.id == result.vector.id }) else { return nil }
            
            // Check criteria
            let isRestaurant = location.category == .restaurant || location.category == .cafe
            let hasOutdoorSeating = location.amenities.contains("outdoor_seating")
            let isNearDowntown = location.distance(to: downtownCenter) <= 10.0
            let isHighRated = location.popularityScore >= 0.8
            
            if isRestaurant && hasOutdoorSeating && isNearDowntown && isHighRated {
                // Combine vector similarity with geographic proximity
                let geoScore = Float(1.0 / (1.0 + location.distance(to: downtownCenter)))
                let combinedScore = result.score * 0.7 + geoScore * 0.3
                return (location, combinedScore)
            }
            
            return nil
        }.sorted { $0.score > $1.score }
        
        print("\nResults:")
        for (index, (location, score)) in filteredResults.enumerated() {
            let distance = location.distance(to: downtownCenter)
            print("\(index + 1). \(location.name)")
            print("   Distance from downtown: \(String(format: "%.2f", distance)) km")
            print("   Popularity: \(location.popularityScore), Combined score: \(String(format: "%.3f", score))")
        }
    }
    
    static func benchmarkGeospatialOperations(vectorStore: VectorStore, locations: [Location]) async throws {
        let iterations = 1000
        
        // Benchmark 1: Coordinate embedding performance
        let coordStart = Date()
        for _ in 0..<iterations {
            _ = GeoSpatialEmbedder.coordinateEmbedding(location: locations[0])
        }
        let coordDuration = Date().timeIntervalSince(coordStart)
        print("Coordinate embedding: \(String(format: "%.2f", coordDuration * 1000 / Double(iterations))) ms/op")
        
        // Benchmark 2: Feature embedding performance
        let featureStart = Date()
        for _ in 0..<iterations {
            _ = GeoSpatialEmbedder.featureEmbedding(location: locations[0])
        }
        let featureDuration = Date().timeIntervalSince(featureStart)
        print("Feature embedding: \(String(format: "%.2f", featureDuration * 1000 / Double(iterations))) ms/op")
        
        // Benchmark 3: Haversine distance calculation
        let distStart = Date()
        for _ in 0..<iterations {
            _ = locations[0].distance(to: locations[1])
        }
        let distDuration = Date().timeIntervalSince(distStart)
        print("Haversine distance: \(String(format: "%.2f", distDuration * 1000 / Double(iterations))) ms/op")
        
        // Benchmark 4: Vector search performance
        let searchStart = Date()
        let queryVector = Vector(id: "bench", values: [Float](repeating: 0.5, count: 38))
        for _ in 0..<100 {
            _ = try await vectorStore.search(query: queryVector, k: 10)
        }
        let searchDuration = Date().timeIntervalSince(searchStart)
        print("Vector search (k=10): \(String(format: "%.2f", searchDuration * 10)) ms/op")
        
        // Benchmark 5: Geo-fence containment check
        let region = GeoRegion(center: (37.7749, -122.4194), radiusKm: 5.0, name: "Test")
        let geoStart = Date()
        for _ in 0..<iterations {
            _ = region.contains(locations[0])
        }
        let geoDuration = Date().timeIntervalSince(geoStart)
        print("Geo-fence check: \(String(format: "%.2f", geoDuration * 1000 / Double(iterations))) ms/op")
        
        print("\nTotal vectors indexed: \(locations.count)")
        print("Embedding strategies demonstrated: 4")
        print("Example scenarios covered: 6")
    }
}

// MARK: - Helper Extensions

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}