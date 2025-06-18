# Core Folder Refactoring Plan

This document outlines the comprehensive work required to create a lean, robust, and high-performance Core folder for VectorStoreKit.

## Executive Summary

The Core folder requires significant refactoring to achieve a lean architecture that leverages Apple Silicon's capabilities while maintaining code clarity and performance. Key areas include removing redundant implementations, completing stub methods, and optimizing for Metal acceleration.

## 1. Code Removal (Lean Architecture)

### Immediate Removals
- [ ] **VectorTypes.swift:343-358** - Remove all legacy naming support methods
  ```swift
  // Remove: legacyNamingSupport(), oldAPICompatibility(), etc.
  ```
- [ ] **DistanceComputation512.swift:256** - Remove redundant SIMD sum() extension
- [ ] **Vector512.swift** - Remove duplicate SIMD4 sum() extension
- [ ] **BatchProcessor.swift:448-475** - Remove AsyncSemaphore, use Swift's built-in actors/tasks

### Consolidation Tasks
- [ ] **Merge distance computation files** into a single `DistanceComputation.swift`
  - Combine DistanceComputation512.swift and OptimizedDistanceComputation.swift
  - Remove overlapping implementations
  - Keep only the most performant versions

## 2. Missing Implementation Fixes

### Critical Implementations

#### BatchProcessor.swift
- [ ] Implement `processSingleBatch` for `.indexing` operation
  ```swift
  case .indexing(let indexType):
      // Implement actual indexing logic using Metal acceleration
      // Support HNSW, IVF, and Learned index types
  ```
- [ ] Implement `processSingleBatch` for `.quantization` operation
  ```swift
  case .quantization(let params):
      // Implement product quantization using Metal kernels
      // Support scalar and vector quantization
  ```
- [ ] Implement `processSingleBatch` for `.distanceComputation` operation
  ```swift
  case .distanceComputation(let metric):
      // Route to unified distance computation with Metal acceleration
  ```

#### FilterEvaluator.swift
- [ ] Complete `evaluateLearnedFilter` implementation
  ```swift
  func evaluateLearnedFilter(_ filter: LearnedFilter, vector: Vector) -> Bool {
      // Implement neural network-based filtering
      // Use Metal Performance Shaders for inference
  }
  ```

#### DistanceComputation.swift (consolidated)
- [ ] Complete `distanceMatrix` implementation with all distance metrics
- [ ] Add proper error handling for unsupported metrics

## 3. Performance Optimizations

### Memory Management

#### MemoryPoolManager.swift
- [ ] Fix fatal error in `release` method
  ```swift
  func release(_ buffer: BufferType) {
      guard isManaged(buffer) else {
          // Log warning and handle gracefully
          return
      }
      // Proper release logic
  }
  ```
- [ ] Implement system memory pressure integration
  ```swift
  private func setupMemoryPressureHandler() {
      let source = DispatchSource.makeMemoryPressureSource(
          eventMask: [.warning, .critical],
          queue: .main
      )
      // Handle memory pressure events
  }
  ```

#### Vector512.swift
- [ ] Optimize `toArray()` to avoid copies
  ```swift
  func toArray() -> [Float] {
      return withUnsafeBytes(of: self) { buffer in
          Array(buffer.bindMemory(to: Float.self))
      }
  }
  ```

### Computation Optimizations

#### VectorTypes.swift
- [ ] Cache `VectorQuality.assess` results
  ```swift
  actor QualityCache {
      private var cache: [Vector.ID: VectorQuality] = [:]
      
      func assess(_ vector: Vector) async -> VectorQuality {
          if let cached = cache[vector.id] { return cached }
          // Compute and cache
      }
  }
  ```
- [ ] Optimize entropy calculation using SIMD
  ```swift
  private func computeEntropy(_ data: [Float]) -> Float {
      // Use vDSP for histogram computation
      // Vectorize probability calculations
  }
  ```

#### FilterEvaluator.swift
- [ ] Replace switch statement with lookup table
  ```swift
  private let predicateEvaluators: [CustomPredicate.Type: (Vector) -> Bool] = [
      // Pre-computed evaluator functions
  ]
  ```

### Metal Acceleration

#### OptimizedDistanceComputation.swift
- [ ] Implement SIMD16/SIMD32 variants for newer hardware
  ```swift
  @available(macOS 13.0, *)
  func dotProductSIMD32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
      // Utilize wider SIMD types when available
  }
  ```

## 4. Architecture Improvements

### Protocol Refinements

#### Protocols.swift
- [ ] Add default implementations for common patterns
  ```swift
  extension VectorIndexProtocol {
      func batchAdd(_ vectors: [Vector]) async throws {
          // Default implementation using single add
      }
  }
  ```

### Type Safety

#### OptimizedDistanceComputation.swift
- [ ] Implement proper move semantics for `AlignedVector`
  ```swift
  struct AlignedVector: ~Copyable {
      consuming func move() -> AlignedVector {
          // Proper move implementation
      }
  }
  ```

## 5. New Core Components

### Unified Distance Engine
- [ ] Create `DistanceEngine.swift` consolidating all distance computations
  ```swift
  actor DistanceEngine {
      private let metalCompute: MetalCompute
      private let cpuFallback: CPUDistanceCompute
      
      func compute(/* params */) async -> [Float] {
          // Intelligent routing between Metal and CPU
      }
  }
  ```

### Memory Pressure Manager
- [ ] Create `MemoryPressureManager.swift` for system-wide memory coordination
  ```swift
  actor MemoryPressureManager {
      func registerComponent(_ component: MemoryManaged) 
      func handlePressure(_ level: MemoryPressureLevel)
  }
  ```

## 6. Testing Requirements

### Unit Tests
- [ ] Complete test coverage for all distance computation methods
- [ ] Test memory pool behavior under pressure
- [ ] Verify filter evaluation correctness
- [ ] Benchmark all optimization improvements

### Performance Tests
- [ ] Create benchmarks for before/after optimization comparison
- [ ] Metal vs CPU fallback performance tests
- [ ] Memory allocation/deallocation benchmarks
- [ ] Large-scale batch processing tests

## 7. Documentation Updates

### Code Documentation
- [ ] Document all Metal acceleration paths
- [ ] Add complexity annotations to algorithms
- [ ] Include hardware requirements for optimizations

### Architecture Documentation
- [ ] Create Core/README.md explaining the architecture
- [ ] Document the unified distance computation strategy
- [ ] Explain memory management approach

## Implementation Priority

### Phase 1: Critical Fixes (Week 1)
1. Fix MemoryPoolManager fatal error
2. Remove all redundant code
3. Implement BatchProcessor stub methods

### Phase 2: Performance (Week 2)
1. Consolidate distance computation
2. Implement Metal acceleration paths
3. Optimize memory operations

### Phase 3: Architecture (Week 3)
1. Create unified components
2. Implement system integration
3. Complete testing suite

## Success Metrics

- **Code Reduction**: 30% fewer lines of code
- **Performance**: 2x improvement in distance computation
- **Memory**: 50% reduction in allocation overhead
- **Reliability**: Zero fatal errors, proper error handling throughout

## Risk Mitigation

- **Compatibility**: Ensure CPU fallbacks for all Metal operations
- **Testing**: Comprehensive test suite before removing old code
- **Performance**: Benchmark each change to prevent regressions
- **Memory**: Monitor for leaks and excessive allocation

---

This plan ensures the Core folder becomes a lean, robust foundation for VectorStoreKit while fully leveraging Apple Silicon's capabilities.