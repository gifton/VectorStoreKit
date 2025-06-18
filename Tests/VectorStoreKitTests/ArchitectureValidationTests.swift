import XCTest
@testable import VectorStoreKit

final class ArchitectureValidationTests: XCTestCase {
    
    // MARK: - Validation Result Tests
    
    func testValidationResultCreation() {
        let result = ValidationResult(
            category: .api,
            severity: .warning,
            message: "Test warning",
            details: ["key": "value"],
            location: ValidationLocation(file: "test.swift", line: 42)
        )
        
        XCTAssertEqual(result.category, .api)
        XCTAssertEqual(result.severity, .warning)
        XCTAssertEqual(result.message, "Test warning")
        XCTAssertEqual(result.details["key"] as? String, "value")
        XCTAssertNotNil(result.location)
        XCTAssertEqual(result.location?.line, 42)
    }
    
    func testValidationSeverityComparison() {
        XCTAssertTrue(ValidationSeverity.info < ValidationSeverity.warning)
        XCTAssertTrue(ValidationSeverity.warning < ValidationSeverity.error)
        XCTAssertTrue(ValidationSeverity.error < ValidationSeverity.critical)
    }
    
    // MARK: - Validation Report Tests
    
    func testValidationReportSummary() {
        let results = [
            ValidationResult(category: .api, severity: .info, message: "Info"),
            ValidationResult(category: .api, severity: .warning, message: "Warning"),
            ValidationResult(category: .performance, severity: .error, message: "Error"),
            ValidationResult(category: .memory, severity: .critical, message: "Critical")
        ]
        
        let report = ValidationReport(
            results: results,
            summary: generateSummary(from: results),
            timestamp: Date(),
            duration: 1.5
        )
        
        XCTAssertEqual(report.summary.totalChecks, 4)
        XCTAssertEqual(report.summary.passed, 1)
        XCTAssertEqual(report.summary.warnings, 1)
        XCTAssertEqual(report.summary.errors, 1)
        XCTAssertEqual(report.summary.critical, 1)
        XCTAssertTrue(report.summary.hasErrors)
        XCTAssertEqual(report.summary.passRate, 0.25)
    }
    
    func testValidationReportMarkdown() {
        let results = [
            ValidationResult(
                category: .api,
                severity: .error,
                message: "API consistency error",
                details: ["api": "testAPI", "issue": "missing async"],
                location: ValidationLocation(file: "API.swift", line: 100, symbol: "fetchData")
            )
        ]
        
        let report = ValidationReport(
            results: results,
            summary: generateSummary(from: results),
            timestamp: Date(),
            duration: 0.5
        )
        
        let markdown = report.markdown()
        
        XCTAssertTrue(markdown.contains("# VectorStoreKit Validation Report"))
        XCTAssertTrue(markdown.contains("API consistency error"))
        XCTAssertTrue(markdown.contains("API.swift"))
        XCTAssertTrue(markdown.contains("fetchData"))
    }
    
    // MARK: - Architecture Validator Tests
    
    func testArchitectureValidatorInitialization() async {
        let validator = ArchitectureValidator()
        
        // Should initialize without errors
        let report = await validator.validate(category: .api)
        XCTAssertNotNil(report)
        XCTAssertGreaterThan(report.results.count, 0)
    }
    
    func testValidateAllCategories() async {
        let validator = ArchitectureValidator()
        let report = await validator.validate()
        
        // Should have results from all categories
        let categories = Set(report.results.map { $0.category })
        XCTAssertTrue(categories.contains(.api))
        XCTAssertTrue(categories.contains(.performance))
        XCTAssertTrue(categories.contains(.memory))
        XCTAssertTrue(categories.contains(.threadSafety))
        XCTAssertTrue(categories.contains(.documentation))
        XCTAssertTrue(categories.contains(.naming))
    }
    
    func testValidateSpecificCategory() async {
        let validator = ArchitectureValidator()
        let report = await validator.validate(category: .performance)
        
        // Should only have performance results
        let categories = Set(report.results.map { $0.category })
        XCTAssertEqual(categories.count, 1)
        XCTAssertTrue(categories.contains(.performance))
    }
    
    // MARK: - Individual Validator Tests
    
    func testAPIValidator() async {
        let validator = APIValidator()
        let results = await validator.validate()
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertTrue(results.allSatisfy { $0.category == .api })
    }
    
    func testPerformanceValidator() async {
        let validator = PerformanceValidator()
        let results = await validator.validate()
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertTrue(results.allSatisfy { $0.category == .performance })
        
        // Should include performance metrics
        let hasMetrics = results.contains { result in
            result.details.keys.contains("time") || 
            result.details.keys.contains("baseline") ||
            result.details.keys.contains("current")
        }
        XCTAssertTrue(hasMetrics)
    }
    
    func testMemoryValidator() async {
        let validator = MemoryValidator()
        let results = await validator.validate()
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertTrue(results.allSatisfy { $0.category == .memory })
        
        // Should include memory info
        let hasMemoryInfo = results.contains { result in
            result.details.keys.contains("resident") || 
            result.details.keys.contains("virtual")
        }
        XCTAssertTrue(hasMemoryInfo)
    }
    
    func testThreadSafetyValidator() async {
        let validator = ThreadSafetyValidator()
        let results = await validator.validate()
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertTrue(results.allSatisfy { $0.category == .threadSafety })
    }
    
    // MARK: - CI/CD Integration Tests
    
    func testJUnitXMLGeneration() {
        let results = [
            ValidationResult(category: .api, severity: .info, message: "Passed test"),
            ValidationResult(category: .performance, severity: .error, message: "Failed test", details: ["reason": "timeout"])
        ]
        
        let report = ValidationReport(
            results: results,
            summary: generateSummary(from: results),
            timestamp: Date(),
            duration: 2.5
        )
        
        let xml = ValidationCI.generateJUnitXML(from: report)
        
        XCTAssertTrue(xml.contains("<?xml version=\"1.0\""))
        XCTAssertTrue(xml.contains("<testsuites>"))
        XCTAssertTrue(xml.contains("tests=\"2\""))
        XCTAssertTrue(xml.contains("failures=\"1\""))
        XCTAssertTrue(xml.contains("<failure"))
        XCTAssertTrue(xml.contains("Failed test"))
    }
    
    // MARK: - Performance Tests
    
    func testValidationPerformance() async {
        let validator = ArchitectureValidator()
        
        let startTime = CACurrentMediaTime()
        _ = await validator.validate()
        let duration = CACurrentMediaTime() - startTime
        
        // Validation should complete quickly
        XCTAssertLessThan(duration, 5.0, "Validation took too long: \(duration)s")
    }
    
    // MARK: - Helper Methods
    
    private func generateSummary(from results: [ValidationResult]) -> ValidationReport.ValidationSummary {
        var categorySummary: [ValidationCategory: ValidationReport.ValidationSummary.CategorySummary] = [:]
        
        for category in ValidationCategory.allCases {
            let categoryResults = results.filter { $0.category == category }
            let passed = categoryResults.filter { $0.severity < .warning }.count
            
            var issues: [ValidationSeverity: Int] = [:]
            for severity in [ValidationSeverity.warning, .error, .critical] {
                issues[severity] = categoryResults.filter { $0.severity == severity }.count
            }
            
            if !categoryResults.isEmpty {
                categorySummary[category] = ValidationReport.ValidationSummary.CategorySummary(
                    total: categoryResults.count,
                    passed: passed,
                    issues: issues
                )
            }
        }
        
        let passed = results.filter { $0.severity < .warning }.count
        let warnings = results.filter { $0.severity == .warning }.count
        let errors = results.filter { $0.severity == .error }.count
        let critical = results.filter { $0.severity == .critical }.count
        
        return ValidationReport.ValidationSummary(
            totalChecks: results.count,
            passed: passed,
            warnings: warnings,
            errors: errors,
            critical: critical,
            categorySummary: categorySummary
        )
    }
}