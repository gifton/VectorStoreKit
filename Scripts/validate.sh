#!/bin/bash

# VectorStoreKit Architecture Validation Script
# Usage: ./Scripts/validate.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CATEGORY=""
OUTPUT_FORMAT="markdown"
OUTPUT_FILE=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --category <name>    Run validation for specific category"
            echo "                          Categories: api, performance, memory, threadSafety, documentation, naming"
            echo "  -f, --format <format>   Output format: markdown (default) or junit"
            echo "  -o, --output <file>     Save report to file"
            echo "  -v, --verbose           Enable verbose output"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run all validations"
            echo "  ./Scripts/validate.sh"
            echo ""
            echo "  # Run only API validation"
            echo "  ./Scripts/validate.sh -c api"
            echo ""
            echo "  # Generate JUnit report"
            echo "  ./Scripts/validate.sh -f junit -o report.xml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Header
print_color $BLUE "╔══════════════════════════════════════════════╗"
print_color $BLUE "║     VectorStoreKit Architecture Validation    ║"
print_color $BLUE "╚══════════════════════════════════════════════╝"
echo ""

# Build the project first
print_color $YELLOW "Building VectorStoreKit..."
if [ "$VERBOSE" = true ]; then
    swift build
else
    swift build 2>&1 | grep -E "(error:|warning:)" || true
fi

# Prepare validation command
VALIDATION_CMD="swift run VectorStoreValidation"

if [ -n "$CATEGORY" ]; then
    VALIDATION_CMD="$VALIDATION_CMD --category $CATEGORY"
    print_color $YELLOW "Running validation for category: $CATEGORY"
else
    print_color $YELLOW "Running all validation checks..."
fi

if [ "$OUTPUT_FORMAT" = "junit" ]; then
    VALIDATION_CMD="$VALIDATION_CMD --junit"
fi

if [ -n "$OUTPUT_FILE" ]; then
    VALIDATION_CMD="$VALIDATION_CMD --output $OUTPUT_FILE"
fi

# Run validation
print_color $YELLOW "Executing validation..."
echo ""

if [ "$VERBOSE" = true ]; then
    $VALIDATION_CMD
else
    # Capture output for parsing
    OUTPUT=$($VALIDATION_CMD 2>&1)
    EXIT_CODE=$?
    
    # Extract summary information
    PASS_RATE=$(echo "$OUTPUT" | grep -oE "Pass Rate: [0-9.]+%" | head -1)
    PASSED=$(echo "$OUTPUT" | grep -oE "✅ Passed: [0-9]+" | head -1)
    WARNINGS=$(echo "$OUTPUT" | grep -oE "⚠️ Warnings: [0-9]+" | head -1)
    ERRORS=$(echo "$OUTPUT" | grep -oE "❌ Errors: [0-9]+" | head -1)
    CRITICAL=$(echo "$OUTPUT" | grep -oE "🚨 Critical: [0-9]+" | head -1)
    
    # Display summary
    echo "$OUTPUT" | grep -A 20 "## Summary" || echo "$OUTPUT"
    echo ""
    
    # Display colored status based on exit code
    case $EXIT_CODE in
        0)
            print_color $GREEN "✅ Validation PASSED!"
            ;;
        1)
            print_color $YELLOW "⚠️  Validation completed with warnings"
            ;;
        2)
            print_color $RED "❌ Validation FAILED with errors"
            ;;
        3)
            print_color $RED "🚨 Validation FAILED with critical errors"
            ;;
    esac
    
    # Exit with the same code as validation
    exit $EXIT_CODE
fi

# Check if output file was created
if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
    print_color $GREEN "Report saved to: $OUTPUT_FILE"
fi