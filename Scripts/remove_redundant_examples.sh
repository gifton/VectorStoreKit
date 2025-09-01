#!/bin/bash

# Remove redundant/duplicate functionality examples
rm -f Examples/CoreMLIntegrationExample.swift
rm -f Examples/VectorMLPipelineExample.swift
rm -f Examples/ModelSerializationExample.swift

# Remove too specific/niche examples
rm -f Examples/BioinformaticsExample.swift
rm -f Examples/FinancialAnalysisExample.swift
rm -f Examples/GeoSpatialExample.swift
rm -f Examples/DistributedSystemExample.swift

# Remove examples that should be documentation
rm -f Examples/DebuggingExample.swift
rm -f Examples/ValidationExample.swift
rm -f Examples/ProductionFilterExample.swift

# Remove redundant migration examples
rm -f Examples/Migration/GradualMigrationExample.swift
rm -f Examples/Migration/ZeroDowntimeMigrationExample.swift

# Remove other redundant examples
rm -f Examples/ImageSimilarityExample.swift
rm -f Examples/RealTimeAnalyticsExample.swift

echo "Removed redundant examples"

# Count remaining examples
remaining=$(find Examples -name "*.swift" -type f 2>/dev/null | wc -l)
echo "Remaining examples: $remaining"