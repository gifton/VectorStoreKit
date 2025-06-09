// VectorStoreKit: FilterEvaluator Tests
//
// Tests for the shared filter evaluation utilities

import XCTest
@testable import VectorStoreKit

final class FilterEvaluatorTests: XCTestCase {
    
    // MARK: - Test Data
    
    private func createTestVector(
        id: String = UUID().uuidString,
        vector: [Float] = [1.0, 2.0, 3.0, 4.0],
        metadata: [String: String]? = nil
    ) -> StoredVector {
        let metadataData: Data? = if let metadata = metadata {
            try? JSONSerialization.data(withJSONObject: metadata)
        } else {
            nil
        }
        
        return StoredVector(
            id: id,
            vector: vector,
            metadata: metadataData,
            timestamp: Date()
        )
    }
    
    // MARK: - Metadata Filtering Tests
    
    func testMetadataFilterEquals() async throws {
        let vector = createTestVector(metadata: ["category": "electronics", "brand": "apple"])
        
        // Test equals
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "electronics"
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
        
        // Test not equals
        let filter2 = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "clothing"
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertFalse(result2)
    }
    
    func testMetadataFilterContains() async throws {
        let vector = createTestVector(metadata: ["description": "high-quality electronic device"])
        
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "description",
            operation: .contains,
            value: "electronic"
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
    }
    
    func testMetadataFilterIn() async throws {
        let vector = createTestVector(metadata: ["category": "electronics"])
        
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .in,
            value: "electronics, clothing, books"
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
    }
    
    func testMetadataFilterMissingKey() async throws {
        let vector = createTestVector(metadata: ["category": "electronics"])
        
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "nonexistent",
            operation: .equals,
            value: "value"
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertFalse(result)
    }
    
    // MARK: - Vector Filtering Tests
    
    func testVectorFilterMagnitude() async throws {
        let vector = createTestVector(vector: [3.0, 4.0]) // magnitude = 5.0
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .magnitude(4.0...6.0)
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
        
        // Test out of range
        let filter2 = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .magnitude(6.0...10.0)
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertFalse(result2)
    }
    
    func testVectorFilterSparsity() async throws {
        let vector = createTestVector(vector: [1.0, 0.0, 2.0, 0.0, 3.0]) // sparsity = 3/5 = 0.6
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .sparsity(0.5...0.7)
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
    }
    
    func testVectorFilterCustomPredicate() async throws {
        // Test SIMD4 vector
        let vector = createTestVector(vector: [1.0, 2.0, 3.0, 4.0])
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .custom { (simdVector: any SIMD) -> Bool in
                // Check if it's a SIMD4 and sum is greater than 8
                if let vec = simdVector as? SIMD4<Float> {
                    return vec.sum() > 8.0
                }
                return false
            }
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result) // 1+2+3+4 = 10 > 8
        
        // Test with a different predicate
        let filter2 = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .custom { (simdVector: any SIMD) -> Bool in
                // Check if all components are positive
                if let vec = simdVector as? SIMD4<Float> {
                    return vec.min() > 0
                }
                return false
            }
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertTrue(result2)
        
        // Test with vector containing negative values
        let vectorWithNegative = createTestVector(vector: [1.0, -2.0, 3.0, 4.0])
        let result3 = try await FilterEvaluator.evaluateFilter(filter2, vector: vectorWithNegative)
        XCTAssertFalse(result3)
    }
    
    func testVectorFilterDimension() async throws {
        let vector = createTestVector(vector: [1.0, 2.0, 3.0, 4.0])
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: 2,
            range: 2.5...3.5,
            constraint: .magnitude(0...100)
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
        
        // Test out of range
        let filter2 = SearchFilter.vector(VectorFilter(
            dimension: 2,
            range: 4.0...5.0,
            constraint: .magnitude(0...100)
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertFalse(result2)
    }
    
    func testVectorFilterCustomPredicateNonStandardSize() async throws {
        // Test with non-standard SIMD size (e.g., 5 elements)
        let vector = createTestVector(vector: [1.0, 2.0, 3.0, 4.0, 5.0])
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .custom { _ in
                // This should not be called for non-standard sizes
                XCTFail("Custom predicate should not be called for non-standard SIMD sizes")
                return true
            }
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertFalse(result) // Should return false for non-standard sizes
    }
    
    // MARK: - Composite Filtering Tests
    
    func testCompositeFilterAnd() async throws {
        let vector = createTestVector(
            vector: [3.0, 4.0],
            metadata: ["category": "electronics", "brand": "apple"]
        )
        
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .and,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "electronics")),
                .metadata(MetadataFilter(key: "brand", operation: .equals, value: "apple")),
                .vector(VectorFilter(dimension: nil, range: nil, constraint: .magnitude(4.0...6.0)))
            ]
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result)
        
        // Test with one failing condition
        let filter2 = SearchFilter.composite(CompositeFilter(
            operation: .and,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "electronics")),
                .metadata(MetadataFilter(key: "brand", operation: .equals, value: "samsung")), // This will fail
                .vector(VectorFilter(dimension: nil, range: nil, constraint: .magnitude(4.0...6.0)))
            ]
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertFalse(result2)
    }
    
    func testCompositeFilterOr() async throws {
        let vector = createTestVector(
            vector: [3.0, 4.0],
            metadata: ["category": "electronics"]
        )
        
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .or,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "clothing")), // False
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "electronics")), // True
                .vector(VectorFilter(dimension: nil, range: nil, constraint: .magnitude(10.0...20.0))) // False
            ]
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result) // Should be true because one condition matches
    }
    
    func testCompositeFilterNot() async throws {
        let vector = createTestVector(metadata: ["category": "electronics"])
        
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .not,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "clothing"))
            ]
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result) // Should be true because the vector is NOT clothing
        
        // Test negative case
        let filter2 = SearchFilter.composite(CompositeFilter(
            operation: .not,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "electronics"))
            ]
        ))
        
        let result2 = try await FilterEvaluator.evaluateFilter(filter2, vector: vector)
        XCTAssertFalse(result2) // Should be false because the vector IS electronics
    }
    
    // MARK: - Batch Filtering Tests
    
    func testBatchFiltering() async throws {
        let vectors = [
            createTestVector(id: "1", vector: [1.0, 2.0], metadata: ["category": "electronics"]),
            createTestVector(id: "2", vector: [3.0, 4.0], metadata: ["category": "clothing"]),
            createTestVector(id: "3", vector: [5.0, 6.0], metadata: ["category": "electronics"]),
            createTestVector(id: "4", vector: [7.0, 8.0], metadata: ["category": "books"])
        ]
        
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "electronics"
        ))
        
        let filtered = try await FilterEvaluator.filterVectors(vectors, filter: filter)
        
        XCTAssertEqual(filtered.count, 2)
        XCTAssertEqual(Set(filtered.map { $0.id }), Set(["1", "3"]))
    }
    
    // MARK: - Edge Cases
    
    func testNilMetadata() async throws {
        let vector = createTestVector(metadata: nil)
        
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "electronics"
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertFalse(result) // Should be false because there's no metadata
    }
    
    func testEmptyVector() async throws {
        let vector = createTestVector(vector: [])
        
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .magnitude(0...1.0)
        ))
        
        let result = try await FilterEvaluator.evaluateFilter(filter, vector: vector)
        XCTAssertTrue(result) // Empty vector has magnitude 0
    }
    
    // MARK: - Performance Tests
    
    func testFilteringPerformance() async throws {
        // Create a large dataset
        let vectors = (0..<10000).map { i in
            createTestVector(
                id: "vector-\(i)",
                vector: Array(repeating: Float(i), count: 128),
                metadata: ["index": "\(i)", "category": i % 2 == 0 ? "even" : "odd"]
            )
        }
        
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .and,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "even")),
                .vector(VectorFilter(dimension: nil, range: nil, constraint: .magnitude(0...1000)))
            ]
        ))
        
        measure {
            Task {
                let _ = try await FilterEvaluator.filterVectors(vectors, filter: filter)
            }
        }
    }
}