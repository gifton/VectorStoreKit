import Foundation

/// Configuration for document chunking
public struct ChunkingConfiguration {
    /// Maximum size of each chunk in characters
    public let maxChunkSize: Int
    
    /// Overlap between consecutive chunks in characters
    public let overlapSize: Int
    
    /// Minimum size for a chunk to be considered valid
    public let minChunkSize: Int
    
    /// Whether to respect sentence boundaries
    public let respectSentences: Bool
    
    /// Whether to respect paragraph boundaries
    public let respectParagraphs: Bool
    
    /// Custom delimiters for splitting
    public let customDelimiters: [String]
    
    public init(
        maxChunkSize: Int = 1000,
        overlapSize: Int = 200,
        minChunkSize: Int = 100,
        respectSentences: Bool = true,
        respectParagraphs: Bool = true,
        customDelimiters: [String] = []
    ) {
        self.maxChunkSize = maxChunkSize
        self.overlapSize = min(overlapSize, maxChunkSize / 2)
        self.minChunkSize = minChunkSize
        self.respectSentences = respectSentences
        self.respectParagraphs = respectParagraphs
        self.customDelimiters = customDelimiters
    }
}

/// A chunk of text with metadata
public struct TextChunk {
    /// Unique identifier for the chunk
    public let id: String
    
    /// The actual text content
    public let content: String
    
    /// Start position in original document
    public let startOffset: Int
    
    /// End position in original document
    public let endOffset: Int
    
    /// Index of this chunk in the sequence
    public let chunkIndex: Int
    
    /// Total number of chunks in the document
    public let totalChunks: Int
    
    /// Metadata from the original document
    public var metadata: [String: Any]
    
    /// Character count
    public var characterCount: Int {
        content.count
    }
    
    /// Estimated token count
    public var estimatedTokens: Int {
        (content.count + 3) / 4
    }
    
    public init(
        id: String = UUID().uuidString,
        content: String,
        startOffset: Int,
        endOffset: Int,
        chunkIndex: Int,
        totalChunks: Int,
        metadata: [String: Any] = [:]
    ) {
        self.id = id
        self.content = content
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.chunkIndex = chunkIndex
        self.totalChunks = totalChunks
        self.metadata = metadata
    }
}

/// Protocol for chunking strategies
public protocol ChunkingStrategy {
    /// Split text into chunks
    func chunk(_ text: String, configuration: ChunkingConfiguration) -> [TextChunk]
    
    /// Estimate number of chunks without actually chunking
    func estimateChunkCount(_ text: String, configuration: ChunkingConfiguration) -> Int
}

/// Fixed-size chunking with overlap
public struct FixedSizeChunking: ChunkingStrategy {
    public init() {}
    
    public func chunk(_ text: String, configuration: ChunkingConfiguration) -> [TextChunk] {
        var chunks: [TextChunk] = []
        let textArray = Array(text)
        var startIndex = 0
        
        while startIndex < textArray.count {
            // Calculate end index
            var endIndex = min(startIndex + configuration.maxChunkSize, textArray.count)
            
            // Adjust for sentence boundaries if needed
            if configuration.respectSentences && endIndex < textArray.count {
                endIndex = adjustForSentenceBoundary(
                    textArray: textArray,
                    proposedEnd: endIndex,
                    maxEnd: min(startIndex + configuration.maxChunkSize + 100, textArray.count)
                )
            }
            
            // Extract chunk content
            let chunkContent = String(textArray[startIndex..<endIndex])
            
            // Only add if meets minimum size
            if chunkContent.count >= configuration.minChunkSize {
                chunks.append(TextChunk(
                    content: chunkContent,
                    startOffset: startIndex,
                    endOffset: endIndex,
                    chunkIndex: chunks.count,
                    totalChunks: 0  // Will be updated later
                ))
            }
            
            // Move to next chunk with overlap
            if endIndex >= textArray.count {
                break
            }
            startIndex = endIndex - configuration.overlapSize
        }
        
        // Update total chunk count
        let totalChunks = chunks.count
        chunks = chunks.map { chunk in
            var updated = chunk
            updated = TextChunk(
                id: chunk.id,
                content: chunk.content,
                startOffset: chunk.startOffset,
                endOffset: chunk.endOffset,
                chunkIndex: chunk.chunkIndex,
                totalChunks: totalChunks,
                metadata: chunk.metadata
            )
            return updated
        }
        
        return chunks
    }
    
    public func estimateChunkCount(_ text: String, configuration: ChunkingConfiguration) -> Int {
        let effectiveChunkSize = configuration.maxChunkSize - configuration.overlapSize
        return max(1, (text.count + effectiveChunkSize - 1) / effectiveChunkSize)
    }
    
    private func adjustForSentenceBoundary(
        textArray: [Character],
        proposedEnd: Int,
        maxEnd: Int
    ) -> Int {
        // Look for sentence endings
        let sentenceEndings: Set<Character> = [".", "!", "?"]
        
        // Search backward from proposed end
        for i in stride(from: proposedEnd - 1, to: max(proposedEnd - 100, 0), by: -1) {
            if sentenceEndings.contains(textArray[i]) {
                // Check if followed by space or newline (to avoid abbreviations)
                if i + 1 < textArray.count &&
                   (textArray[i + 1] == " " || textArray[i + 1] == "\n") {
                    return i + 1
                }
            }
        }
        
        // If no sentence boundary found backward, try forward
        for i in proposedEnd..<min(maxEnd, textArray.count) {
            if sentenceEndings.contains(textArray[i]) {
                if i + 1 < textArray.count &&
                   (textArray[i + 1] == " " || textArray[i + 1] == "\n") {
                    return i + 1
                }
            }
        }
        
        return proposedEnd
    }
}

/// Semantic chunking based on paragraph structure
public struct SemanticChunking: ChunkingStrategy {
    public init() {}
    
    public func chunk(_ text: String, configuration: ChunkingConfiguration) -> [TextChunk] {
        // Split by paragraphs first
        let paragraphs = splitIntoParagraphs(text)
        var chunks: [TextChunk] = []
        var currentChunk = ""
        var currentStartOffset = 0
        var offset = 0
        
        for paragraph in paragraphs {
            let paragraphLength = paragraph.count
            
            // If paragraph itself is too large, use fixed-size chunking on it
            if paragraphLength > configuration.maxChunkSize {
                // Flush current chunk if exists
                if !currentChunk.isEmpty && currentChunk.count >= configuration.minChunkSize {
                    chunks.append(TextChunk(
                        content: currentChunk,
                        startOffset: currentStartOffset,
                        endOffset: offset,
                        chunkIndex: chunks.count,
                        totalChunks: 0
                    ))
                    currentChunk = ""
                    currentStartOffset = offset
                }
                
                // Chunk the large paragraph
                let fixedChunker = FixedSizeChunking()
                let subChunks = fixedChunker.chunk(paragraph, configuration: configuration)
                
                for subChunk in subChunks {
                    chunks.append(TextChunk(
                        content: subChunk.content,
                        startOffset: offset + subChunk.startOffset,
                        endOffset: offset + subChunk.endOffset,
                        chunkIndex: chunks.count,
                        totalChunks: 0
                    ))
                }
                
                offset += paragraphLength + 2  // Account for paragraph separator
                continue
            }
            
            // Check if adding this paragraph would exceed max size
            if !currentChunk.isEmpty &&
               currentChunk.count + paragraph.count + 2 > configuration.maxChunkSize {
                // Save current chunk
                chunks.append(TextChunk(
                    content: currentChunk,
                    startOffset: currentStartOffset,
                    endOffset: offset,
                    chunkIndex: chunks.count,
                    totalChunks: 0
                ))
                
                // Start new chunk with overlap
                let overlapText = extractOverlap(
                    currentChunk,
                    overlapSize: configuration.overlapSize
                )
                currentChunk = overlapText + "\n\n" + paragraph
                currentStartOffset = offset - overlapText.count
            } else {
                // Add paragraph to current chunk
                if !currentChunk.isEmpty {
                    currentChunk += "\n\n"
                }
                currentChunk += paragraph
            }
            
            offset += paragraphLength + 2
        }
        
        // Add final chunk
        if !currentChunk.isEmpty && currentChunk.count >= configuration.minChunkSize {
            chunks.append(TextChunk(
                content: currentChunk,
                startOffset: currentStartOffset,
                endOffset: offset,
                chunkIndex: chunks.count,
                totalChunks: 0
            ))
        }
        
        // Update total chunks
        let totalChunks = chunks.count
        return chunks.map { chunk in
            TextChunk(
                id: chunk.id,
                content: chunk.content,
                startOffset: chunk.startOffset,
                endOffset: chunk.endOffset,
                chunkIndex: chunk.chunkIndex,
                totalChunks: totalChunks,
                metadata: chunk.metadata
            )
        }
    }
    
    public func estimateChunkCount(_ text: String, configuration: ChunkingConfiguration) -> Int {
        let paragraphs = splitIntoParagraphs(text)
        var estimatedChunks = 0
        var currentSize = 0
        
        for paragraph in paragraphs {
            if paragraph.count > configuration.maxChunkSize {
                // This paragraph will be split
                let fixedChunker = FixedSizeChunking()
                estimatedChunks += fixedChunker.estimateChunkCount(
                    paragraph,
                    configuration: configuration
                )
                currentSize = 0
            } else if currentSize + paragraph.count > configuration.maxChunkSize {
                estimatedChunks += 1
                currentSize = configuration.overlapSize + paragraph.count
            } else {
                currentSize += paragraph.count + 2
            }
        }
        
        if currentSize > 0 {
            estimatedChunks += 1
        }
        
        return max(1, estimatedChunks)
    }
    
    private func splitIntoParagraphs(_ text: String) -> [String] {
        // Split by double newlines or custom delimiters
        let paragraphs = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        
        return paragraphs
    }
    
    private func extractOverlap(_ text: String, overlapSize: Int) -> String {
        guard text.count > overlapSize else { return text }
        
        let startIndex = text.index(text.endIndex, offsetBy: -overlapSize)
        return String(text[startIndex...])
    }
}

/// Recursive chunking that splits text hierarchically
public struct RecursiveChunking: ChunkingStrategy {
    private let separators: [String] = ["\n\n", "\n", ". ", " "]
    
    public init() {}
    
    public func chunk(_ text: String, configuration: ChunkingConfiguration) -> [TextChunk] {
        var chunks: [TextChunk] = []
        recursiveChunk(
            text: text,
            configuration: configuration,
            separatorIndex: 0,
            startOffset: 0,
            chunks: &chunks
        )
        
        // Update total chunks
        let totalChunks = chunks.count
        return chunks.enumerated().map { index, chunk in
            TextChunk(
                id: chunk.id,
                content: chunk.content,
                startOffset: chunk.startOffset,
                endOffset: chunk.endOffset,
                chunkIndex: index,
                totalChunks: totalChunks,
                metadata: chunk.metadata
            )
        }
    }
    
    public func estimateChunkCount(_ text: String, configuration: ChunkingConfiguration) -> Int {
        // Rough estimate based on average chunk size
        let avgChunkSize = (configuration.maxChunkSize + configuration.minChunkSize) / 2
        return max(1, text.count / avgChunkSize)
    }
    
    private func recursiveChunk(
        text: String,
        configuration: ChunkingConfiguration,
        separatorIndex: Int,
        startOffset: Int,
        chunks: inout [TextChunk]
    ) {
        // Base case: text fits in chunk
        if text.count <= configuration.maxChunkSize {
            if text.count >= configuration.minChunkSize {
                chunks.append(TextChunk(
                    content: text,
                    startOffset: startOffset,
                    endOffset: startOffset + text.count,
                    chunkIndex: chunks.count,
                    totalChunks: 0
                ))
            }
            return
        }
        
        // Try to split with current separator
        guard separatorIndex < separators.count else {
            // No more separators, use fixed chunking
            let fixedChunker = FixedSizeChunking()
            let fixedChunks = fixedChunker.chunk(text, configuration: configuration)
            for chunk in fixedChunks {
                chunks.append(TextChunk(
                    content: chunk.content,
                    startOffset: startOffset + chunk.startOffset,
                    endOffset: startOffset + chunk.endOffset,
                    chunkIndex: chunks.count,
                    totalChunks: 0
                ))
            }
            return
        }
        
        let separator = separators[separatorIndex]
        let parts = text.components(separatedBy: separator)
        
        var currentChunk = ""
        var currentStartOffset = startOffset
        var currentOffset = startOffset
        
        for (index, part) in parts.enumerated() {
            let partWithSeparator = index < parts.count - 1 ? part + separator : part
            
            if currentChunk.count + partWithSeparator.count > configuration.maxChunkSize {
                // Recursively chunk the current accumulated text
                if !currentChunk.isEmpty {
                    recursiveChunk(
                        text: currentChunk,
                        configuration: configuration,
                        separatorIndex: separatorIndex + 1,
                        startOffset: currentStartOffset,
                        chunks: &chunks
                    )
                }
                
                currentChunk = partWithSeparator
                currentStartOffset = currentOffset
            } else {
                currentChunk += partWithSeparator
            }
            
            currentOffset += partWithSeparator.count
        }
        
        // Handle remaining chunk
        if !currentChunk.isEmpty {
            recursiveChunk(
                text: currentChunk,
                configuration: configuration,
                separatorIndex: separatorIndex + 1,
                startOffset: currentStartOffset,
                chunks: &chunks
            )
        }
    }
}

/// Factory for creating chunking strategies
public struct ChunkingStrategyFactory {
    public enum Strategy {
        case fixedSize
        case semantic
        case recursive
        case custom(ChunkingStrategy)
    }
    
    public static func create(_ strategy: Strategy) -> ChunkingStrategy {
        switch strategy {
        case .fixedSize:
            return FixedSizeChunking()
        case .semantic:
            return SemanticChunking()
        case .recursive:
            return RecursiveChunking()
        case .custom(let chunker):
            return chunker
        }
    }
}