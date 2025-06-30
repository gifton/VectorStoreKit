#!/bin/bash

echo "🧹 Removing duplicate implementations..."

# Remove duplicate distance computation files
echo "Removing duplicate distance computation files..."
rm -f Sources/VectorStoreKit/Core/DistanceComputation512.swift
rm -f Sources/VectorStoreKit/Core/DistanceComputation512+Advanced.swift
rm -f Sources/VectorStoreKit/Core/OptimizedDistanceComputation.swift
rm -f Sources/VectorStoreKit/Acceleration/Metal/MetalDistanceComputeOptimized.swift

# Keep only the unified implementation and specialized metrics
echo "Keeping unified distance computation and specialized metrics"

# Remove duplicate shader files if they exist
echo "Checking for duplicate Metal shaders..."
shaders_dir="Sources/VectorStoreKit/Acceleration/Metal/Shaders"
if [ -d "$shaders_dir" ]; then
    # List all shader files
    find "$shaders_dir" -name "*.metal" | while read shader; do
        basename=$(basename "$shader")
        echo "  Found shader: $basename"
    done
    
    # Remove old/duplicate versions
    rm -f "$shaders_dir/distance_old.metal"
    rm -f "$shaders_dir/distance_v2.metal"
    rm -f "$shaders_dir/distance_legacy.metal"
fi

# Remove duplicate benchmark directories
echo "Consolidating benchmark directories..."
if [ -d "Benchmarking" ] && [ -d "Benchmarks" ]; then
    echo "Found both Benchmarking/ and Benchmarks/ - keeping only Benchmarks/"
    rm -rf Benchmarking/
fi

# Count remaining source files
echo ""
echo "📊 Summary:"
echo "Distance computation files remaining:"
find Sources/VectorStoreKit -name "*Distance*.swift" -type f | wc -l

echo "Metal shader files:"
find Sources/VectorStoreKit -name "*.metal" -type f 2>/dev/null | wc -l

echo "Total Swift source files:"
find Sources/VectorStoreKit -name "*.swift" -type f | wc -l

echo ""
echo "✅ Duplicate removal complete!"