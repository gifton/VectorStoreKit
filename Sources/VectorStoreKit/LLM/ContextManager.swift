import Foundation

/// Manages context assembly and token budgeting for LLM interactions
public actor ContextAssembler {
    private let configuration: ContextConfiguration
    private let tokenEstimator: TokenEstimator
    
    public init(
        configuration: ContextConfiguration,
        tokenEstimator: TokenEstimator = DefaultTokenEstimator()
    ) {
        self.configuration = configuration
        self.tokenEstimator = tokenEstimator
    }
    
    /// Assemble context from multiple sources within token budget
    public func assembleContext(
        systemPrompt: String,
        userQuery: String,
        retrievedChunks: [(content: String, metadata: [String: Any])],
        conversationHistory: [ConversationTurn]? = nil,
        examples: [(input: String, output: String)]? = nil
    ) throws -> AssembledContext {
        // Calculate token budgets
        let systemTokens = tokenEstimator.estimate(systemPrompt)
        let queryTokens = tokenEstimator.estimate(userQuery)
        
        guard systemTokens <= configuration.systemPromptTokens else {
            throw ContextError.systemPromptTooLong(
                actual: systemTokens,
                limit: configuration.systemPromptTokens
            )
        }
        
        guard queryTokens <= configuration.queryTokens else {
            throw ContextError.queryTooLong(
                actual: queryTokens,
                limit: configuration.queryTokens
            )
        }
        
        // Calculate remaining budget
        var remainingTokens = configuration.availableContextTokens
        
        // Add conversation history if provided
        var contextParts: [ContextPart] = []
        if let history = conversationHistory {
            let (historyParts, historyTokens) = assembleHistory(
                history,
                maxTokens: remainingTokens / 3  // Reserve 1/3 for history
            )
            contextParts.append(contentsOf: historyParts)
            remainingTokens -= historyTokens
        }
        
        // Add examples if provided
        if let examples = examples {
            let (exampleParts, exampleTokens) = assembleExamples(
                examples,
                maxTokens: min(remainingTokens / 4, 500)  // Limit examples
            )
            contextParts.append(contentsOf: exampleParts)
            remainingTokens -= exampleTokens
        }
        
        // Add retrieved chunks
        let (chunkParts, _) = assembleChunks(
            retrievedChunks,
            maxTokens: remainingTokens
        )
        contextParts.append(contentsOf: chunkParts)
        
        // Calculate total tokens
        let totalTokens = systemTokens + queryTokens + 
            contextParts.reduce(0) { $0 + $1.tokenCount }
        
        return AssembledContext(
            systemPrompt: systemPrompt,
            userQuery: userQuery,
            contextParts: contextParts,
            totalTokens: totalTokens,
            metadata: ContextMetadata(
                systemTokens: systemTokens,
                queryTokens: queryTokens,
                contextTokens: totalTokens - systemTokens - queryTokens,
                responseTokensReserved: configuration.responseTokens
            )
        )
    }
    
    /// Format assembled context into a prompt
    public func formatPrompt(
        _ context: AssembledContext,
        template: PromptTemplate = .default
    ) -> String {
        var formatted = ""
        
        // Add system prompt
        if !context.systemPrompt.isEmpty {
            formatted += template.systemSection(context.systemPrompt)
        }
        
        // Add context parts
        for part in context.contextParts {
            switch part.type {
            case .conversationHistory:
                formatted += template.historySection(part.content)
            case .example:
                formatted += template.exampleSection(part.content)
            case .retrievedContext:
                formatted += template.contextSection(part.content)
            case .custom:
                formatted += part.content
            }
        }
        
        // Add user query
        formatted += template.querySection(context.userQuery)
        
        return formatted
    }
    
    // MARK: - Private Methods
    
    private func assembleHistory(
        _ history: [ConversationTurn],
        maxTokens: Int
    ) -> ([ContextPart], Int) {
        var parts: [ContextPart] = []
        var totalTokens = 0
        
        // Process history in reverse (most recent first)
        for turn in history.reversed() {
            let turnText = formatTurn(turn)
            let turnTokens = tokenEstimator.estimate(turnText)
            
            if totalTokens + turnTokens <= maxTokens {
                parts.insert(
                    ContextPart(
                        type: .conversationHistory,
                        content: turnText,
                        tokenCount: turnTokens,
                        metadata: ["timestamp": turn.timestamp]
                    ),
                    at: 0
                )
                totalTokens += turnTokens
            } else {
                break
            }
        }
        
        return (parts, totalTokens)
    }
    
    private func assembleExamples(
        _ examples: [(input: String, output: String)],
        maxTokens: Int
    ) -> ([ContextPart], Int) {
        var parts: [ContextPart] = []
        var totalTokens = 0
        
        for example in examples {
            let exampleText = "Input: \(example.input)\nOutput: \(example.output)"
            let exampleTokens = tokenEstimator.estimate(exampleText)
            
            if totalTokens + exampleTokens <= maxTokens {
                parts.append(
                    ContextPart(
                        type: .example,
                        content: exampleText,
                        tokenCount: exampleTokens,
                        metadata: [:]
                    )
                )
                totalTokens += exampleTokens
            } else {
                break
            }
        }
        
        return (parts, totalTokens)
    }
    
    private func assembleChunks(
        _ chunks: [(content: String, metadata: [String: Any])],
        maxTokens: Int
    ) -> ([ContextPart], Int) {
        var parts: [ContextPart] = []
        var totalTokens = 0
        
        for chunk in chunks {
            let chunkTokens = tokenEstimator.estimate(chunk.content)
            
            if totalTokens + chunkTokens <= maxTokens {
                parts.append(
                    ContextPart(
                        type: .retrievedContext,
                        content: chunk.content,
                        tokenCount: chunkTokens,
                        metadata: chunk.metadata
                    )
                )
                totalTokens += chunkTokens
            } else {
                // Try to include partial chunk if significant space remains
                let remainingTokens = maxTokens - totalTokens
                if remainingTokens > 100 {
                    let truncated = truncateToTokens(
                        chunk.content,
                        maxTokens: remainingTokens
                    )
                    parts.append(
                        ContextPart(
                            type: .retrievedContext,
                            content: truncated + "...",
                            tokenCount: remainingTokens,
                            metadata: chunk.metadata
                        )
                    )
                    totalTokens += remainingTokens
                }
                break
            }
        }
        
        return (parts, totalTokens)
    }
    
    private func formatTurn(_ turn: ConversationTurn) -> String {
        "\(turn.role.rawValue): \(turn.content)"
    }
    
    private func truncateToTokens(_ text: String, maxTokens: Int) -> String {
        // Simple character-based truncation
        // In production, use proper tokenization
        let estimatedChars = maxTokens * 4
        if text.count <= estimatedChars {
            return text
        }
        
        let endIndex = text.index(text.startIndex, offsetBy: estimatedChars)
        return String(text[..<endIndex])
    }
}

// MARK: - Supporting Types

/// Represents assembled context ready for LLM
public struct AssembledContext {
    public let systemPrompt: String
    public let userQuery: String
    public let contextParts: [ContextPart]
    public let totalTokens: Int
    public let metadata: ContextMetadata
}

/// Part of the assembled context
public struct ContextPart {
    public enum PartType {
        case conversationHistory
        case example
        case retrievedContext
        case custom(String)
    }
    
    public let type: PartType
    public let content: String
    public let tokenCount: Int
    public let metadata: [String: Any]
}

/// Metadata about context assembly
public struct ContextMetadata {
    public let systemTokens: Int
    public let queryTokens: Int
    public let contextTokens: Int
    public let responseTokensReserved: Int
    
    public var totalTokensUsed: Int {
        systemTokens + queryTokens + contextTokens
    }
    
    public var remainingTokens: Int {
        responseTokensReserved
    }
}

/// Conversation turn for history
public struct ConversationTurn {
    public enum Role: String {
        case user = "User"
        case assistant = "Assistant"
        case system = "System"
    }
    
    public let role: Role
    public let content: String
    public let timestamp: Date
    
    public init(role: Role, content: String, timestamp: Date = Date()) {
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}

/// Errors related to context assembly
public enum ContextError: Error, LocalizedError {
    case systemPromptTooLong(actual: Int, limit: Int)
    case queryTooLong(actual: Int, limit: Int)
    case contextExceedsLimit(actual: Int, limit: Int)
    case invalidConfiguration(String)
    
    public var errorDescription: String? {
        switch self {
        case .systemPromptTooLong(let actual, let limit):
            return "System prompt too long: \(actual) tokens exceeds limit of \(limit)"
        case .queryTooLong(let actual, let limit):
            return "Query too long: \(actual) tokens exceeds limit of \(limit)"
        case .contextExceedsLimit(let actual, let limit):
            return "Context exceeds limit: \(actual) tokens exceeds limit of \(limit)"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        }
    }
}

/// Protocol for token estimation
public protocol TokenEstimator {
    func estimate(_ text: String) -> Int
}

/// Default token estimator
public struct DefaultTokenEstimator: TokenEstimator {
    private let charactersPerToken: Int
    
    public init(charactersPerToken: Int = 4) {
        self.charactersPerToken = charactersPerToken
    }
    
    public func estimate(_ text: String) -> Int {
        (text.count + charactersPerToken - 1) / charactersPerToken
    }
}

/// GPT-specific token estimator
public struct GPTTokenEstimator: TokenEstimator {
    public init() {}
    
    public func estimate(_ text: String) -> Int {
        // More accurate estimation for GPT models
        let wordCount = text.split(separator: " ").count
        let punctuationCount = text.filter { ".,!?;:".contains($0) }.count
        
        // GPT typically uses ~1.3 tokens per word
        return Int(Double(wordCount) * 1.3) + punctuationCount
    }
}

/// Template for formatting prompts
public struct PromptTemplate: Sendable {
    public let systemSection: @Sendable (String) -> String
    public let historySection: @Sendable (String) -> String
    public let exampleSection: @Sendable (String) -> String
    public let contextSection: @Sendable (String) -> String
    public let querySection: @Sendable (String) -> String
    
    public static let `default` = PromptTemplate(
        systemSection: { "System: \($0)\n\n" },
        historySection: { "Previous conversation:\n\($0)\n\n" },
        exampleSection: { "Example:\n\($0)\n\n" },
        contextSection: { "Context:\n\($0)\n\n" },
        querySection: { "User: \($0)\n\nAssistant: " }
    )
    
    public static let markdown = PromptTemplate(
        systemSection: { "## System Instructions\n\($0)\n\n" },
        historySection: { "## Conversation History\n\($0)\n\n" },
        exampleSection: { "### Example\n\($0)\n\n" },
        contextSection: { "## Relevant Context\n\($0)\n\n" },
        querySection: { "## User Query\n\($0)\n\n## Response\n" }
    )
    
    public static let xml = PromptTemplate(
        systemSection: { "<system>\n\($0)\n</system>\n\n" },
        historySection: { "<history>\n\($0)\n</history>\n\n" },
        exampleSection: { "<example>\n\($0)\n</example>\n\n" },
        contextSection: { "<context>\n\($0)\n</context>\n\n" },
        querySection: { "<query>\n\($0)\n</query>\n\n<response>\n" }
    )
}