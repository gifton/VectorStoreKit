import XCTest
@testable import VectorStoreKit

final class ChunkingTests: XCTestCase {
    
    func testChunkingConfiguration() {
        let config = ChunkingConfiguration(
            maxChunkSize: 1000,
            overlapSize: 200,
            minChunkSize: 100
        )
        
        XCTAssertEqual(config.maxChunkSize, 1000)
        XCTAssertEqual(config.overlapSize, 200)
        XCTAssertEqual(config.minChunkSize, 100)
        
        // Test overlap size clamping
        let config2 = ChunkingConfiguration(
            maxChunkSize: 100,
            overlapSize: 200 // Should be clamped to 50
        )
        XCTAssertEqual(config2.overlapSize, 50)
    }
    
    func testTextChunk() {
        let chunk = TextChunk(
            content: "Hello world",
            startOffset: 0,
            endOffset: 11,
            chunkIndex: 0,
            totalChunks: 1
        )
        
        XCTAssertEqual(chunk.characterCount, 11)
        XCTAssertEqual(chunk.estimatedTokens, 3) // (11 + 3) / 4
    }
    
    func testFixedSizeChunking() {
        let chunker = FixedSizeChunking()
        let text = "This is a test text that should be split into multiple chunks. " +
                   "Each chunk should have the specified size with some overlap between them."
        
        let config = ChunkingConfiguration(
            maxChunkSize: 50,
            overlapSize: 10,
            minChunkSize: 20
        )
        
        let chunks = chunker.chunk(text, configuration: config)
        
        // Verify chunks were created
        XCTAssertGreaterThan(chunks.count, 1)
        
        // Verify chunk sizes
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.content.count, config.maxChunkSize + 100) // Allow for sentence boundary
            XCTAssertGreaterThanOrEqual(chunk.content.count, config.minChunkSize)
        }
        
        // Verify overlap
        if chunks.count > 1 {
            for i in 1..<chunks.count {
                let prevEnd = chunks[i-1].content.suffix(config.overlapSize)
                let currStart = chunks[i].content.prefix(config.overlapSize)
                // There should be some overlap (not exact due to sentence boundaries)
                XCTAssertTrue(chunks[i-1].endOffset > chunks[i].startOffset - config.overlapSize)
            }
        }
        
        // Verify total chunks count
        for chunk in chunks {
            XCTAssertEqual(chunk.totalChunks, chunks.count)
        }
    }
    
    func testSemanticChunking() {
        let chunker = SemanticChunking()
        let text = """
        This is the first paragraph. It contains some information.
        
        This is the second paragraph. It has different content.
        
        This is the third paragraph. It's the final one.
        """
        
        let config = ChunkingConfiguration(
            maxChunkSize: 100,
            overlapSize: 20,
            respectParagraphs: true
        )
        
        let chunks = chunker.chunk(text, configuration: config)
        
        // Should create chunks respecting paragraph boundaries
        XCTAssertGreaterThan(chunks.count, 0)
        
        // Each chunk should contain complete paragraphs when possible
        for chunk in chunks {
            // Verify no partial sentences at boundaries (simplified check)
            let trimmed = chunk.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                XCTAssertTrue(
                    trimmed.hasSuffix(".") || 
                    trimmed.hasSuffix("!") || 
                    trimmed.hasSuffix("?") ||
                    chunk.chunkIndex == chunks.count - 1 // Last chunk might not end with punctuation
                )
            }
        }
    }
    
    func testRecursiveChunking() {
        let chunker = RecursiveChunking()
        let text = """
        Section 1. This is a long section with multiple sentences. It should be split appropriately.
        
        Section 2. Another section here. With more content. And even more text to ensure splitting.
        """
        
        let config = ChunkingConfiguration(
            maxChunkSize: 50,
            overlapSize: 10
        )
        
        let chunks = chunker.chunk(text, configuration: config)
        
        XCTAssertGreaterThan(chunks.count, 1)
        
        // Verify all text is included
        let reconstructed = chunks.map { $0.content }.joined()
        XCTAssertTrue(reconstructed.contains("Section 1"))
        XCTAssertTrue(reconstructed.contains("Section 2"))
    }
    
    func testChunkingEstimation() {
        let text = String(repeating: "word ", count: 200) // 1000 characters
        
        let config = ChunkingConfiguration(
            maxChunkSize: 100,
            overlapSize: 20
        )
        
        let fixedChunker = FixedSizeChunking()
        let estimate = fixedChunker.estimateChunkCount(text, configuration: config)
        let actual = fixedChunker.chunk(text, configuration: config).count
        
        // Estimate should be reasonably close
        XCTAssertTrue(abs(estimate - actual) <= 2)
    }
    
    func testLargeDocumentChunking() {
        // Test with a large document
        let sentences = (0..<100).map { "Sentence \($0). This is test content. " }
        let text = sentences.joined()
        
        let config = ChunkingConfiguration(
            maxChunkSize: 200,
            overlapSize: 50,
            respectSentences: true
        )
        
        let chunker = FixedSizeChunking()
        let chunks = chunker.chunk(text, configuration: config)
        
        // Verify all chunks are valid
        for (index, chunk) in chunks.enumerated() {
            XCTAssertEqual(chunk.chunkIndex, index)
            XCTAssertTrue(chunk.content.count >= config.minChunkSize || index == chunks.count - 1)
            XCTAssertTrue(chunk.startOffset < chunk.endOffset)
        }
        
        // Verify coverage
        XCTAssertEqual(chunks.first?.startOffset, 0)
        XCTAssertLessThanOrEqual(chunks.last?.endOffset ?? 0, text.count)
    }
    
    func testChunkingStrategyFactory() {
        let fixed = ChunkingStrategyFactory.create(.fixedSize)
        XCTAssertTrue(fixed is FixedSizeChunking)
        
        let semantic = ChunkingStrategyFactory.create(.semantic)
        XCTAssertTrue(semantic is SemanticChunking)
        
        let recursive = ChunkingStrategyFactory.create(.recursive)
        XCTAssertTrue(recursive is RecursiveChunking)
        
        // Test custom strategy
        let customChunker = MockChunkingStrategy()
        let custom = ChunkingStrategyFactory.create(.custom(customChunker))
        XCTAssertTrue(custom is MockChunkingStrategy)
    }
}

// MARK: - Mock Chunking Strategy

struct MockChunkingStrategy: ChunkingStrategy {
    func chunk(_ text: String, configuration: ChunkingConfiguration) -> [TextChunk] {
        [TextChunk(
            content: text,
            startOffset: 0,
            endOffset: text.count,
            chunkIndex: 0,
            totalChunks: 1
        )]
    }
    
    func estimateChunkCount(_ text: String, configuration: ChunkingConfiguration) -> Int {
        1
    }
}