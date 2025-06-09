// VectorStoreKit: Shared Filter Evaluation Utilities
//
// Provides common filtering logic for vector search operations

import Foundation
import simd

/// Shared utility for evaluating search filters across different index implementations
public struct FilterEvaluator {
    
    // MARK: - Main Filter Evaluation
    
    /// Evaluates a search filter against a stored vector
    public static func evaluateFilter(
        _ filter: SearchFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        switch filter {
        case .metadata(let metadataFilter):
            return evaluateMetadataFilter(metadataFilter, vector: vector, decoder: decoder, encoder: encoder)
            
        case .vector(let vectorFilter):
            return evaluateVectorFilter(vectorFilter, vector: vector.vector)
            
        case .composite(let compositeFilter):
            return try await evaluateCompositeFilter(compositeFilter, vector: vector, decoder: decoder, encoder: encoder)
            
        case .learned(let learnedFilter):
            // Simple implementation - can be enhanced with actual ML model
            return evaluateLearnedFilter(learnedFilter)
        }
    }
    
    // MARK: - Metadata Filtering
    
    /// Evaluates a metadata filter against a stored vector
    public static func evaluateMetadataFilter(
        _ filter: MetadataFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) -> Bool {
        // Try to decode as dictionary
        guard let metadataDict = try? JSONSerialization.jsonObject(with: vector.metadata) as? [String: Any],
              let value = metadataDict[filter.key] else {
            return false
        }
        
        // Convert value to string for comparison
        let valueString = String(describing: value)
        
        return evaluateFilterOperation(
            value: valueString,
            operation: filter.operation,
            filterValue: filter.value
        )
    }
    
    // MARK: - Vector Filtering
    
    /// Evaluates a vector filter against a vector array
    public static func evaluateVectorFilter(
        _ filter: VectorFilter,
        vector: [Float]
    ) -> Bool {
        // Check dimension filter if specified
        if let dimension = filter.dimension,
           dimension < vector.count,
           let range = filter.range {
            guard range.contains(vector[dimension]) else {
                return false
            }
        }
        
        // Apply vector constraint
        switch filter.constraint {
        case .magnitude(let range):
            let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            return range.contains(magnitude)
            
        case .sparsity(let range):
            let nonZeroCount = vector.filter { $0 != 0 }.count
            let sparsity = Float(nonZeroCount) / Float(vector.count)
            return range.contains(sparsity)
            
        case .custom(let predicate):
            // Convert array to appropriate SIMD vector based on length
            switch vector.count {
            case 2:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD2<Float>.self)
            case 3:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD3<Float>.self)
            case 4:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD4<Float>.self)
            case 8:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD8<Float>.self)
            case 16:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD16<Float>.self)
            case 32:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD32<Float>.self)
            case 64:
                return evaluateCustomPredicate(vector, predicate: predicate, type: SIMD64<Float>.self)
            default:
                // For non-standard SIMD sizes, we can't evaluate the predicate
                // In production, you might want to handle this differently
                return false
            }
        }
    }
    
    // MARK: - SIMD Conversion Helpers
    
    /// Evaluates a custom predicate with a specific SIMD type
    private static func evaluateCustomPredicate<T: SIMD>(
        _ vector: [Float],
        predicate: @Sendable (any SIMD) -> Bool,
        type: T.Type
    ) -> Bool where T.Scalar == Float {
        guard let simdVector = arrayToSIMD(vector, type: type) else {
            return false
        }
        return predicate(simdVector)
    }
    
    /// Converts a Float array to a specific SIMD type
    private static func arrayToSIMD<T: SIMD>(_ array: [Float], type: T.Type) -> T? where T.Scalar == Float {
        guard array.count == T.scalarCount else {
            return nil
        }
        
        var result = T()
        for i in 0..<T.scalarCount {
            result[i] = array[i]
        }
        return result
    }
    
    // MARK: - Composite Filtering
    
    /// Evaluates a composite filter against a stored vector
    public static func evaluateCompositeFilter(
        _ filter: CompositeFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        switch filter.operation {
        case .and:
            // All filters must match
            for subFilter in filter.filters {
                if !(try await evaluateFilter(subFilter, vector: vector, decoder: decoder, encoder: encoder)) {
                    return false
                }
            }
            return true
            
        case .or:
            // At least one filter must match
            for subFilter in filter.filters {
                if try await evaluateFilter(subFilter, vector: vector, decoder: decoder, encoder: encoder) {
                    return true
                }
            }
            return false
            
        case .not:
            // First filter must not match
            guard let firstFilter = filter.filters.first else {
                return true
            }
            return !(try await evaluateFilter(firstFilter, vector: vector, decoder: decoder, encoder: encoder))
        }
    }
    
    // MARK: - Learned Filtering
    
    /// Evaluates a learned filter (placeholder implementation)
    public static func evaluateLearnedFilter(_ filter: LearnedFilter) -> Bool {
        // Simple confidence-based filtering
        // In a real implementation, this would use the learned model
        return filter.confidence > 0.5
    }
    
    // MARK: - Filter Operation Evaluation
    
    /// Evaluates a filter operation between two string values
    public static func evaluateFilterOperation(
        value: String,
        operation: FilterOperation,
        filterValue: String
    ) -> Bool {
        switch operation {
        case .equals:
            return value == filterValue
        case .notEquals:
            return value != filterValue
        case .lessThan:
            return value < filterValue
        case .lessThanOrEqual:
            return value <= filterValue
        case .greaterThan:
            return value > filterValue
        case .greaterThanOrEqual:
            return value >= filterValue
        case .contains:
            return value.contains(filterValue)
        case .notContains:
            return !value.contains(filterValue)
        case .startsWith:
            return value.hasPrefix(filterValue)
        case .endsWith:
            return value.hasSuffix(filterValue)
        case .in:
            let values = filterValue.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            return values.contains(value)
        case .notIn:
            let values = filterValue.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            return !values.contains(value)
        case .regex:
            return (try? NSRegularExpression(pattern: filterValue).firstMatch(
                in: value,
                range: NSRange(location: 0, length: value.utf16.count)
            )) != nil
        }
    }
    
    // MARK: - Batch Filtering
    
    /// Filters an array of stored vectors based on a search filter
    public static func filterVectors(
        _ vectors: [StoredVector],
        filter: SearchFilter,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> [StoredVector] {
        var filtered: [StoredVector] = []
        
        for vector in vectors {
            if try await evaluateFilter(filter, vector: vector, decoder: decoder, encoder: encoder) {
                filtered.append(vector)
            }
        }
        
        return filtered
    }
}

// MARK: - StoredVector Extension

extension StoredVector {
    /// Convenience method to check if this vector matches a filter
    public func matchesFilter(
        _ filter: SearchFilter,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        return try await FilterEvaluator.evaluateFilter(filter, vector: self, decoder: decoder, encoder: encoder)
    }
}
