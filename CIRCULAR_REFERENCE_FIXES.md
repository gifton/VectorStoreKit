# Circular Reference Memory Leak Fixes

## Summary
Fixed circular reference memory leaks in VectorStoreKit's training system by implementing weak self pattern in Task closures.

## Changes Made

### 1. NeuralClustering.swift (Lines 495-502)
Fixed a circular reference in the `updateMetrics` method where a Task closure was capturing `self` strongly.

**Before:**
```swift
let probTask = Task {
    try await getClusterProbabilities(for: query)
}
```

**After:**
```swift
let probTask = Task { [weak self] in
    guard let self else { return nil }
    return try await self.getClusterProbabilities(for: query)
}
```

The conditional check was also updated to handle the optional return value:
```swift
if let probabilities = try? await probTask.value,
   let probs = probabilities {
    performanceMetrics.updateClusterUtilization(probs)
}
```

## Analysis Performed

1. **Searched for Task closures without [weak self]** - Found one instance in NeuralClustering.swift
2. **Checked Metal completion handlers** - Found they already use weak self where appropriate or use withCheckedContinuation (which doesn't capture self)
3. **Looked for stored closures, delegates, and observers** - No circular reference issues found
4. **Verified async closures** - No problematic patterns found

## Best Practices Applied

1. **Use [weak self] in Task closures** when the closure captures self and could outlive the enclosing scope
2. **Guard against nil self** with `guard let self else { return }` pattern
3. **Return appropriate values** when self is nil (nil for optional returns, empty arrays/default values for non-optionals)

## Testing

The fix was verified to build successfully. The change is minimal and focused only on fixing the memory leak without altering functionality.

## Recommendations

1. Continue to use `[weak self]` pattern consistently in Task closures throughout the codebase
2. Consider adding a linting rule to catch Task closures that capture self without weak reference
3. Monitor for similar patterns in future code additions