import Foundation
import VectorStoreKit

// MARK: - Validation CLI

@main
struct VectorStoreValidation {
    static func main() async {
        print("🔍 VectorStoreKit Architecture Validation")
        print("========================================\n")
        
        // Parse command line arguments
        let arguments = CommandLine.arguments
        let outputFormat = arguments.contains("--junit") ? "junit" : "markdown"
        let category = parseCategory(from: arguments)
        
        // Create validator
        let validator = ArchitectureValidator()
        
        // Run validation
        let startTime = Date()
        print("Running validation checks...")
        
        let report: ValidationReport
        if let category = category {
            print("Category: \(category.rawValue)")
            report = await validator.validate(category: category)
        } else {
            print("Running all validation checks...")
            report = await validator.validate()
        }
        
        // Output results
        switch outputFormat {
        case "junit":
            let xml = ValidationCI.generateJUnitXML(from: report)
            if let outputPath = parseOutputPath(from: arguments) {
                try? xml.write(toFile: outputPath, atomically: true, encoding: .utf8)
                print("JUnit XML report written to: \(outputPath)")
            } else {
                print(xml)
            }
            
        default:
            print(report.markdown())
            
            // Save to file if specified
            if let outputPath = parseOutputPath(from: arguments) {
                try? report.markdown().write(toFile: outputPath, atomically: true, encoding: .utf8)
                print("\nReport saved to: \(outputPath)")
            }
        }
        
        // Exit with appropriate code
        if report.summary.critical > 0 {
            print("\n❌ Validation failed with critical errors")
            exit(3)
        } else if report.summary.errors > 0 {
            print("\n❌ Validation failed with errors")
            exit(2)
        } else if report.summary.warnings > 0 {
            print("\n⚠️  Validation completed with warnings")
            exit(1)
        } else {
            print("\n✅ Validation passed!")
            exit(0)
        }
    }
    
    private static func parseCategory(from arguments: [String]) -> ValidationCategory? {
        guard let categoryIndex = arguments.firstIndex(of: "--category"),
              categoryIndex + 1 < arguments.count else {
            return nil
        }
        
        let categoryString = arguments[categoryIndex + 1]
        return ValidationCategory(rawValue: categoryString)
    }
    
    private static func parseOutputPath(from arguments: [String]) -> String? {
        guard let outputIndex = arguments.firstIndex(of: "--output"),
              outputIndex + 1 < arguments.count else {
            return nil
        }
        
        return arguments[outputIndex + 1]
    }
}

// MARK: - Usage Instructions

/*
 Usage: swift run VectorStoreValidation [options]
 
 Options:
   --category <name>    Run validation for specific category
                       Categories: api, performance, memory, threadSafety, documentation, naming
   --junit             Output in JUnit XML format
   --output <path>     Save report to file
 
 Examples:
   # Run all validations
   swift run VectorStoreValidation
   
   # Run only performance validation
   swift run VectorStoreValidation --category performance
   
   # Generate JUnit report for CI
   swift run VectorStoreValidation --junit --output report.xml
   
   # Save markdown report
   swift run VectorStoreValidation --output validation-report.md
 */