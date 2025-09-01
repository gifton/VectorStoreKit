import XCTest
// VectorStoreKit: AI Assistant Example
//
// Demonstrates building an AI assistant using Retrieval Augmented Generation (RAG)
// Shows how to combine vector search with language models for intelligent responses

import Foundation
import VectorStoreKit

final class AIAssistantTests: XCTestCase {
    
    func testMain() async throws {
        print("🤖 VectorStoreKit AI Assistant (RAG) Example")
        print("=" * 60)
        print()
        
        // Configuration
        let config = AIAssistantConfiguration(
            vectorDimensions: 768,          // BERT-like embeddings
            maxContextLength: 4096,         // Maximum context window
            chunkSize: 512,                 // Document chunk size
            chunkOverlap: 128,              // Overlap between chunks
            topK: 5,                        // Number of relevant chunks to retrieve
            temperature: 0.7,               // LLM temperature
            enableCitation: true,           // Include sources in responses
            enableFactChecking: true        // Verify factual accuracy
        )
        
        // Create AI assistant
        let assistant = try await AIAssistant(configuration: config)
        
        // Example 1: Knowledge Base Creation
        print("📚 Example 1: Building Knowledge Base")
        print("-" * 40)
        try await buildKnowledgeBase(assistant: assistant)
        
        // Example 2: Question Answering
        print("\n❓ Example 2: Question Answering")
        print("-" * 40)
        try await questionAnswering(assistant: assistant)
        
        // Example 3: Conversational AI
        print("\n💬 Example 3: Conversational AI with Context")
        print("-" * 40)
        try await conversationalAI(assistant: assistant)
        
        // Example 4: Document Analysis
        print("\n📄 Example 4: Document Analysis and Summarization")
        print("-" * 40)
        try await documentAnalysis(assistant: assistant)
        
        // Example 5: Multi-turn Reasoning
        print("\n🧠 Example 5: Multi-turn Reasoning")
        print("-" * 40)
        try await multiTurnReasoning(assistant: assistant)
        
        // Example 6: Tool Integration
        print("\n🔧 Example 6: Tool Integration")
        print("-" * 40)
        try await toolIntegration(assistant: assistant)
        
        // Example 7: Learning and Adaptation
        print("\n📈 Example 7: Learning and Adaptation")
        print("-" * 40)
        try await learningAndAdaptation(assistant: assistant)
        
        print("\n✅ AI Assistant example completed!")
    }
    
    // MARK: - Example 1: Knowledge Base Creation
    
    func testBuildKnowledgeBase(assistant: AIAssistant) async throws {
        print("Building comprehensive knowledge base...")
        
        // Sample documents covering various topics
        let documents = [
            // Technology documentation
            Document(
                id: "tech_1",
                title: "Introduction to Vector Databases",
                content: """
                Vector databases are specialized database systems designed to store and efficiently search high-dimensional vector embeddings. Unlike traditional databases that work with structured data, vector databases excel at similarity search operations on unstructured data like text, images, and audio that have been converted into numerical vector representations.
                
                Key features of vector databases include:
                1. High-dimensional indexing: Efficient storage and retrieval of vectors with hundreds or thousands of dimensions
                2. Similarity search: Finding vectors that are semantically similar rather than exact matches
                3. Scalability: Handling millions or billions of vectors
                4. Real-time search: Low-latency queries even on large datasets
                
                Common use cases include recommendation systems, semantic search, image similarity, and anomaly detection.
                """,
                category: "Technology",
                metadata: DocumentMetadata(
                    author: "Tech Team",
                    createdDate: Date(),
                    source: "internal_docs",
                    version: "1.0"
                )
            ),
            Document(
                id: "tech_2",
                title: "Machine Learning Fundamentals",
                content: """
                Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming. It focuses on developing algorithms that can identify patterns in data and make decisions with minimal human intervention.
                
                Types of machine learning:
                1. Supervised Learning: Training with labeled data (classification, regression)
                2. Unsupervised Learning: Finding patterns in unlabeled data (clustering, dimensionality reduction)
                3. Reinforcement Learning: Learning through interaction and feedback
                
                Key concepts include training data, features, models, validation, and deployment. Modern ML applications range from computer vision and natural language processing to predictive analytics and autonomous systems.
                """,
                category: "Technology",
                metadata: DocumentMetadata(
                    author: "ML Team",
                    createdDate: Date(),
                    source: "internal_docs",
                    version: "2.0"
                )
            ),
            
            // Business policies
            Document(
                id: "policy_1",
                title: "Company Data Privacy Policy",
                content: """
                Our organization is committed to protecting the privacy and security of personal data. This policy outlines how we collect, use, store, and protect information.
                
                Key principles:
                1. Data Minimization: Only collect necessary data
                2. Purpose Limitation: Use data only for stated purposes
                3. Security: Implement appropriate technical measures
                4. Transparency: Clear communication about data usage
                5. Rights: Respect individual rights (access, correction, deletion)
                
                Compliance: We adhere to GDPR, CCPA, and other relevant regulations. All employees must complete annual privacy training.
                """,
                category: "Policy",
                metadata: DocumentMetadata(
                    author: "Legal Team",
                    createdDate: Date(),
                    source: "policy_docs",
                    version: "3.1"
                )
            ),
            
            // Product information
            Document(
                id: "product_1",
                title: "VectorStore Pro Features",
                content: """
                VectorStore Pro is our enterprise-grade vector database solution with advanced features:
                
                Core Features:
                - Distributed architecture supporting billions of vectors
                - Sub-millisecond search latency
                - 99.99% uptime SLA
                - Multi-modal embeddings (text, image, audio)
                
                Advanced Capabilities:
                - Hybrid search combining vector and keyword search
                - Dynamic index optimization
                - Real-time vector updates
                - Custom similarity metrics
                - Role-based access control
                
                Pricing: Contact sales for enterprise pricing. Includes 24/7 support and dedicated account management.
                """,
                category: "Product",
                metadata: DocumentMetadata(
                    author: "Product Team",
                    createdDate: Date(),
                    source: "product_docs",
                    version: "4.0"
                )
            ),
            
            // FAQ
            Document(
                id: "faq_1",
                title: "Frequently Asked Questions",
                content: """
                Q: What file formats are supported for data import?
                A: We support JSON, CSV, Parquet, and various text formats. Custom parsers can be developed for proprietary formats.
                
                Q: How do I optimize search performance?
                A: Key strategies include: choosing appropriate index types, tuning parameters like M and ef values for HNSW, implementing caching, and using batch operations.
                
                Q: What's the difference between cosine similarity and Euclidean distance?
                A: Cosine similarity measures the angle between vectors (direction), while Euclidean distance measures the straight-line distance (magnitude). Cosine is often better for text embeddings.
                
                Q: How do I handle updates to existing vectors?
                A: VectorStore Pro supports in-place updates. Use the update API with the vector ID to modify existing embeddings without rebuilding the index.
                """,
                category: "FAQ",
                metadata: DocumentMetadata(
                    author: "Support Team",
                    createdDate: Date(),
                    source: "support_docs",
                    version: "1.5"
                )
            )
        ]
        
        // Process and index documents
        print("\nProcessing documents...")
        var totalChunks = 0
        
        for document in documents {
            print("\n📄 Processing: \(document.title)")
            
            // Chunk the document
            let chunks = try await assistant.chunkDocument(document)
            print("  Created \(chunks.count) chunks")
            
            // Generate embeddings and index
            for (index, chunk) in chunks.enumerated() {
                let embedding = try await assistant.generateEmbedding(for: chunk)
                try await assistant.indexChunk(chunk, embedding: embedding)
                
                if index == 0 {
                    print("  Sample chunk: \"\(chunk.content.prefix(100))...\"")
                }
            }
            
            totalChunks += chunks.count
        }
        
        // Create cross-references and relationships
        print("\n\nCreating knowledge graph connections...")
        
        let relationships = [
            KnowledgeRelation(
                source: "tech_1",
                target: "product_1",
                type: .implements,
                strength: 0.9
            ),
            KnowledgeRelation(
                source: "tech_2",
                target: "tech_1",
                type: .relatedTo,
                strength: 0.7
            ),
            KnowledgeRelation(
                source: "policy_1",
                target: "product_1",
                type: .governs,
                strength: 0.8
            )
        ]
        
        for relation in relationships {
            try await assistant.addRelationship(relation)
            print("  ✓ Connected \(relation.source) → \(relation.target) (\(relation.type))")
        }
        
        // Knowledge base statistics
        let stats = await assistant.getKnowledgeBaseStats()
        print("\n📊 Knowledge Base Statistics:")
        print("  Total documents: \(stats.documentCount)")
        print("  Total chunks: \(stats.chunkCount)")
        print("  Average chunk size: \(stats.avgChunkSize) tokens")
        print("  Index size: \(formatBytes(stats.indexSize))")
        print("  Relationships: \(stats.relationshipCount)")
    }
    
    // MARK: - Example 2: Question Answering
    
    func testQuestionAnswering(assistant: AIAssistant) async throws {
        print("Demonstrating question answering capabilities...")
        
        // Various types of questions
        let questions = [
            // Factual questions
            Question(
                text: "What are the key features of vector databases?",
                type: .factual,
                expectsCitation: true
            ),
            
            // Comparison questions
            Question(
                text: "What's the difference between supervised and unsupervised learning?",
                type: .comparison,
                expectsCitation: true
            ),
            
            // Policy questions
            Question(
                text: "What are our data privacy principles?",
                type: .policy,
                expectsCitation: true
            ),
            
            // Product questions
            Question(
                text: "What is the uptime SLA for VectorStore Pro?",
                type: .product,
                expectsCitation: true
            ),
            
            // How-to questions
            Question(
                text: "How can I optimize search performance in a vector database?",
                type: .howTo,
                expectsCitation: true
            )
        ]
        
        for question in questions {
            print("\n\n❓ Question: \(question.text)")
            print("Type: \(question.type)")
            
            // Get answer with RAG
            let response = try await assistant.answer(
                question: question,
                options: AnswerOptions(
                    maxTokens: 500,
                    temperature: 0.7,
                    includeSources: true,
                    factCheck: true
                )
            )
            
            print("\n💡 Answer:")
            print(response.answer)
            
            // Show sources
            if !response.sources.isEmpty {
                print("\n📚 Sources:")
                for source in response.sources {
                    print("  - \(source.title) (relevance: \(String(format: "%.2f", source.relevance)))")
                    print("    \"\(source.excerpt.prefix(100))...\"")
                }
            }
            
            // Show confidence and fact-checking
            print("\n📊 Metadata:")
            print("  Confidence: \(String(format: "%.2f%%", response.confidence * 100))")
            print("  Fact-checked: \(response.factChecked ? "✅" : "❌")")
            
            if let reasoning = response.reasoning {
                print("  Reasoning: \(reasoning)")
            }
        }
        
        // Demonstrate handling of unknown information
        print("\n\n❓ Question: What is the employee count of our Mars office?")
        
        let unknownResponse = try await assistant.answer(
            question: Question(
                text: "What is the employee count of our Mars office?",
                type: .factual
            )
        )
        
        print("\n💡 Answer:")
        print(unknownResponse.answer)
        print("\n📊 Metadata:")
        print("  Confidence: \(String(format: "%.2f%%", unknownResponse.confidence * 100))")
        print("  Hallucination check: \(unknownResponse.hallucinationDetected ? "⚠️ Detected" : "✅ Passed")")
    }
    
    // MARK: - Example 3: Conversational AI
    
    func testConversationalAI(assistant: AIAssistant) async throws {
        print("Demonstrating conversational AI with context management...")
        
        // Create a conversation session
        let session = try await assistant.createSession(
            sessionId: "demo_session_1",
            options: SessionOptions(
                maxTurns: 10,
                contextWindow: 4096,
                personalization: PersonalizationSettings(
                    tone: .professional,
                    expertise: .technical,
                    verbosity: .balanced
                )
            )
        )
        
        print("\n🗣️ Starting conversation session: \(session.id)")
        
        // Multi-turn conversation
        let conversation = [
            "Hello! I'm interested in learning about vector databases.",
            "How do they differ from traditional SQL databases?",
            "Can you give me a specific example of when to use each?",
            "What about performance considerations?",
            "Based on what we discussed, would a vector database be suitable for our e-commerce recommendation system?"
        ]
        
        for (turn, userMessage) in conversation.enumerated() {
            print("\n\n👤 User (\(turn + 1)): \(userMessage)")
            
            // Process user message
            let response = try await assistant.chat(
                message: userMessage,
                sessionId: session.id,
                options: ChatOptions(
                    useContext: true,
                    retrieveRelevantInfo: true,
                    maintainPersona: true
                )
            )
            
            print("\n🤖 Assistant: \(response.message)")
            
            // Show context awareness
            if response.contextUsed {
                print("\n📝 Context elements used: \(response.contextElements.count)")
                for element in response.contextElements.prefix(2) {
                    print("  - \(element.summary)")
                }
            }
            
            // Show retrieved information
            if !response.retrievedInfo.isEmpty {
                print("\n🔍 Retrieved information:")
                for info in response.retrievedInfo.prefix(2) {
                    print("  - \(info.source): \(info.snippet.prefix(80))...")
                }
            }
        }
        
        // Show conversation summary
        print("\n\n📊 Conversation Summary:")
        let summary = try await assistant.summarizeSession(sessionId: session.id)
        print(summary.summary)
        print("\nKey topics discussed:")
        for topic in summary.topics {
            print("  - \(topic)")
        }
        print("\nAction items identified:")
        for action in summary.actionItems {
            print("  - \(action)")
        }
    }
    
    // MARK: - Example 4: Document Analysis
    
    func testDocumentAnalysis(assistant: AIAssistant) async throws {
        print("Demonstrating document analysis and summarization...")
        
        // Complex technical document
        let technicalDoc = Document(
            id: "analysis_1",
            title: "Neural Architecture Search for Vector Embeddings",
            content: """
            Abstract: This paper presents a novel approach to automatically discovering optimal neural architectures for generating vector embeddings in domain-specific applications. Traditional embedding models often use generic architectures that may not capture domain-specific patterns effectively.
            
            Introduction:
            Vector embeddings have become fundamental in modern machine learning applications, from natural language processing to computer vision. The quality of these embeddings directly impacts downstream task performance. While pre-trained models like BERT and ResNet provide good general-purpose embeddings, domain-specific applications often benefit from specialized architectures.
            
            Methodology:
            We propose AutoEmbed, a neural architecture search (NAS) framework specifically designed for embedding generation. Our approach uses evolutionary algorithms combined with weight-sharing techniques to efficiently explore the architecture space. Key innovations include:
            
            1. Embedding-specific search space design
            2. Multi-objective optimization balancing embedding quality and computational efficiency
            3. Progressive pruning to reduce search time
            
            Experiments:
            We evaluated AutoEmbed on five different domains: medical texts, legal documents, scientific papers, social media posts, and product descriptions. Results show 15-30% improvement in downstream task performance compared to generic architectures.
            
            Conclusions:
            AutoEmbed demonstrates that domain-specific architectural choices significantly impact embedding quality. The discovered architectures show interesting patterns, such as increased depth for technical domains and wider networks for social media content.
            """,
            category: "Research",
            metadata: DocumentMetadata(
                author: "Research Team",
                createdDate: Date(),
                source: "research_papers"
            )
        )
        
        print("\n📄 Analyzing document: \(technicalDoc.title)")
        
        // Perform comprehensive analysis
        let analysis = try await assistant.analyzeDocument(
            document: technicalDoc,
            analysisOptions: AnalysisOptions(
                summarize: true,
                extractKeyPoints: true,
                identifyEntities: true,
                assessReadability: true,
                generateQuestions: true
            )
        )
        
        print("\n📋 Executive Summary:")
        print(analysis.summary)
        
        print("\n🔑 Key Points:")
        for (index, point) in analysis.keyPoints.enumerated() {
            print("  \(index + 1). \(point)")
        }
        
        print("\n🏷️ Identified Entities:")
        print("  Technologies: \(analysis.entities.technologies.joined(separator: ", "))")
        print("  Concepts: \(analysis.entities.concepts.joined(separator: ", "))")
        print("  Metrics: \(analysis.entities.metrics.joined(separator: ", "))")
        
        print("\n📊 Document Metrics:")
        print("  Readability score: \(analysis.readability.score)/100")
        print("  Technical level: \(analysis.readability.technicalLevel)")
        print("  Estimated reading time: \(analysis.readability.estimatedReadingTime) minutes")
        
        print("\n❓ Generated Questions:")
        for question in analysis.generatedQuestions.prefix(3) {
            print("  - \(question)")
        }
        
        // Comparative analysis
        print("\n\n🔄 Comparative Analysis:")
        
        let compareDocs = ["tech_1", "tech_2"] // Compare with existing documents
        
        let comparison = try await assistant.compareDocuments(
            primaryDoc: technicalDoc.id,
            compareWith: compareDocs
        )
        
        print("\nSimilarity scores:")
        for score in comparison.similarityScores {
            print("  - \(score.documentId): \(String(format: "%.2f%%", score.similarity * 100))")
        }
        
        print("\nCommon themes:")
        for theme in comparison.commonThemes {
            print("  - \(theme)")
        }
        
        print("\nUnique contributions:")
        for contribution in comparison.uniqueContributions {
            print("  - \(contribution)")
        }
    }
    
    // MARK: - Example 5: Multi-turn Reasoning
    
    func testMultiTurnReasoning(assistant: AIAssistant) async throws {
        print("Demonstrating complex multi-turn reasoning...")
        
        // Complex problem-solving scenario
        let problemScenario = ProblemScenario(
            title: "Optimizing Search Performance",
            context: """
            Our vector database currently handles 10 million vectors with 768 dimensions each.
            Average query latency is 150ms, but we need to reduce it to under 50ms.
            Current setup uses HNSW index with M=16, ef=200.
            System has 64GB RAM and 8 CPU cores.
            """,
            constraints: [
                "Cannot increase hardware resources",
                "Must maintain 95% recall accuracy",
                "Need to support 1000 QPS"
            ],
            goal: "Propose and evaluate optimization strategies"
        )
        
        print("\n🧩 Problem: \(problemScenario.title)")
        print("\nContext: \(problemScenario.context)")
        print("\nConstraints:")
        for constraint in problemScenario.constraints {
            print("  - \(constraint)")
        }
        print("\nGoal: \(problemScenario.goal)")
        
        // Multi-step reasoning
        let reasoningSteps = [
            "Analyze the current performance bottlenecks",
            "Propose potential optimization strategies",
            "Evaluate trade-offs of each approach",
            "Recommend the best solution with implementation steps"
        ]
        
        var reasoningContext = ReasoningContext()
        
        for (step, instruction) in reasoningSteps.enumerated() {
            print("\n\n🔄 Step \(step + 1): \(instruction)")
            
            let stepResponse = try await assistant.reason(
                instruction: instruction,
                scenario: problemScenario,
                context: reasoningContext,
                options: ReasoningOptions(
                    depth: .deep,
                    includeCalculations: true,
                    validateAssumptions: true
                )
            )
            
            print("\n💭 Reasoning:")
            print(stepResponse.reasoning)
            
            if !stepResponse.calculations.isEmpty {
                print("\n🧮 Calculations:")
                for calc in stepResponse.calculations {
                    print("  \(calc.description): \(calc.result)")
                }
            }
            
            if !stepResponse.assumptions.isEmpty {
                print("\n📌 Assumptions:")
                for assumption in stepResponse.assumptions {
                    print("  - \(assumption)")
                }
            }
            
            // Update context for next step
            reasoningContext.addStep(stepResponse)
        }
        
        // Final recommendation
        print("\n\n🎯 Final Recommendation:")
        let recommendation = try await assistant.synthesizeRecommendation(
            context: reasoningContext,
            criteria: [
                "Feasibility",
                "Performance impact",
                "Implementation complexity",
                "Risk assessment"
            ]
        )
        
        print(recommendation.summary)
        
        print("\n📋 Implementation Plan:")
        for (index, step) in recommendation.implementationSteps.enumerated() {
            print("  \(index + 1). \(step.description)")
            print("     Duration: \(step.estimatedTime)")
            print("     Risk: \(step.riskLevel)")
        }
        
        print("\n📊 Expected Outcomes:")
        print("  Latency reduction: \(recommendation.expectedOutcomes.latencyReduction)%")
        print("  Recall impact: \(recommendation.expectedOutcomes.recallImpact)%")
        print("  Resource usage: \(recommendation.expectedOutcomes.resourceChange)%")
    }
    
    // MARK: - Example 6: Tool Integration
    
    func testToolIntegration(assistant: AIAssistant) async throws {
        print("Demonstrating AI assistant with tool integration...")
        
        // Register available tools
        let tools = [
            Tool(
                name: "calculator",
                description: "Perform mathematical calculations",
                parameters: ["expression": "Mathematical expression to evaluate"],
                handler: { params in
                    // Simulate calculation
                    return "Result: 42"
                }
            ),
            Tool(
                name: "vector_search",
                description: "Search the vector database",
                parameters: ["query": "Search query", "limit": "Number of results"],
                handler: { params in
                    // Simulate search
                    return "Found 5 relevant documents"
                }
            ),
            Tool(
                name: "code_generator",
                description: "Generate code snippets",
                parameters: ["language": "Programming language", "task": "What to implement"],
                handler: { params in
                    // Simulate code generation
                    return """
                    ```swift
                    func searchVectors(query: String) async throws -> [Vector] {
                        // Implementation here
                    }
                    ```
                    """
                }
            ),
            Tool(
                name: "data_analyzer",
                description: "Analyze data and generate insights",
                parameters: ["data": "Data to analyze", "type": "Analysis type"],
                handler: { params in
                    return "Analysis complete: 3 key insights found"
                }
            )
        ]
        
        for tool in tools {
            try await assistant.registerTool(tool)
            print("  ✓ Registered tool: \(tool.name)")
        }
        
        // Complex queries requiring tool use
        let toolQueries = [
            "Calculate the memory requirement for storing 5 million vectors with 384 dimensions using float32",
            "Search for documents about HNSW index optimization and summarize the findings",
            "Generate Swift code for implementing cosine similarity between two vectors",
            "Analyze the performance metrics from our last deployment and identify trends"
        ]
        
        for query in toolQueries {
            print("\n\n🔧 Query: \(query)")
            
            let response = try await assistant.processWithTools(
                query: query,
                options: ToolUseOptions(
                    autoSelectTools: true,
                    explainToolUse: true,
                    maxToolCalls: 3
                )
            )
            
            print("\n🤖 Response: \(response.answer)")
            
            if !response.toolsUsed.isEmpty {
                print("\n🛠️ Tools used:")
                for toolUse in response.toolsUsed {
                    print("  - \(toolUse.toolName): \(toolUse.purpose)")
                    if let result = toolUse.result {
                        print("    Result: \(result.prefix(100))...")
                    }
                }
            }
            
            if let explanation = response.toolUseExplanation {
                print("\n💡 Tool selection reasoning: \(explanation)")
            }
        }
        
        // Demonstrate tool chaining
        print("\n\n🔗 Tool Chaining Example:")
        
        let complexQuery = """
        First, search for information about vector database indexing strategies.
        Then, analyze the search results to identify the top 3 strategies.
        Finally, generate Python code implementing the most efficient strategy.
        """
        
        print("Query: \(complexQuery)")
        
        let chainResponse = try await assistant.processWithTools(
            query: complexQuery,
            options: ToolUseOptions(
                autoSelectTools: true,
                explainToolUse: true,
                allowChaining: true
            )
        )
        
        print("\n🤖 Response: \(chainResponse.answer)")
        
        print("\n🔗 Tool execution chain:")
        for (index, toolUse) in chainResponse.toolsUsed.enumerated() {
            print("  \(index + 1). \(toolUse.toolName)")
            print("     Input: \(toolUse.input)")
            print("     Output: \(toolUse.result?.prefix(80) ?? "")...")
        }
    }
    
    // MARK: - Example 7: Learning and Adaptation
    
    func testLearningAndAdaptation(assistant: AIAssistant) async throws {
        print("Demonstrating learning and adaptation capabilities...")
        
        // Enable learning mode
        try await assistant.enableLearning(
            options: LearningOptions(
                learnFromFeedback: true,
                personalizeResponses: true,
                updateKnowledgeBase: true,
                feedbackThreshold: 0.8
            )
        )
        
        print("\n📚 Learning mode enabled")
        
        // Simulate user interactions with feedback
        let interactions = [
            UserInteraction(
                query: "How do I implement vector search in Swift?",
                response: "Here's how to implement vector search in Swift...",
                feedback: .positive(score: 0.9, comment: "Very helpful, clear examples")
            ),
            UserInteraction(
                query: "What's the computational complexity of HNSW?",
                response: "HNSW has logarithmic search complexity...",
                feedback: .negative(score: 0.3, comment: "Too technical, need simpler explanation")
            ),
            UserInteraction(
                query: "Best practices for vector dimension reduction",
                response: "Key practices for dimension reduction include...",
                feedback: .positive(score: 0.85, comment: "Good overview, could use more examples")
            )
        ]
        
        print("\nProcessing user interactions with feedback...")
        
        for interaction in interactions {
            print("\n❓ Query: \(interaction.query)")
            print("📝 Feedback: \(interaction.feedback)")
            
            // Process feedback
            let improvement = try await assistant.processFeedback(
                interaction: interaction
            )
            
            if let improvement = improvement {
                print("📈 Improvement identified:")
                print("   Type: \(improvement.type)")
                print("   Action: \(improvement.action)")
            }
        }
        
        // Show adapted responses
        print("\n\n🔄 Demonstrating adapted responses:")
        
        // Re-ask the question that got negative feedback
        let adaptedQuery = "What's the computational complexity of HNSW?"
        print("\n❓ Query (adapted): \(adaptedQuery)")
        
        let adaptedResponse = try await assistant.answer(
            question: Question(text: adaptedQuery, type: .technical),
            options: AnswerOptions(
                useAdaptations: true,
                simplicityLevel: .beginner
            )
        )
        
        print("\n💡 Adapted Response:")
        print(adaptedResponse.answer)
        print("\n📊 Adaptation applied: \(adaptedResponse.adaptationType ?? "none")")
        
        // Personalization demo
        print("\n\n👤 Personalization Demo:")
        
        let userProfiles = [
            UserProfile(
                id: "beginner_dev",
                expertise: .beginner,
                interests: ["swift", "ios", "basics"],
                preferredStyle: .simple
            ),
            UserProfile(
                id: "expert_ml",
                expertise: .expert,
                interests: ["machine learning", "optimization", "research"],
                preferredStyle: .technical
            )
        ]
        
        let testQuery = "Explain how vector embeddings work"
        
        for profile in userProfiles {
            print("\n\nProfile: \(profile.id) (expertise: \(profile.expertise))")
            print("❓ Query: \(testQuery)")
            
            let personalizedResponse = try await assistant.answer(
                question: Question(text: testQuery, type: .educational),
                userProfile: profile
            )
            
            print("\n💡 Personalized Response:")
            print(personalizedResponse.answer.prefix(200))
            print("...")
            print("\n📊 Personalization factors:")
            print("  - Complexity: \(personalizedResponse.complexityLevel)")
            print("  - Examples used: \(personalizedResponse.exampleTypes.joined(separator: ", "))")
            print("  - Technical terms: \(personalizedResponse.technicalTermCount)")
        }
        
        // Knowledge base updates
        print("\n\n📝 Knowledge Base Learning:")
        
        let newInformation = [
            LearnedFact(
                content: "HNSW index performs best with M values between 12-48 for most use cases",
                source: "user_feedback",
                confidence: 0.85,
                timestamp: Date()
            ),
            LearnedFact(
                content: "Swift implementations benefit from SIMD instructions for vector operations",
                source: "user_interaction",
                confidence: 0.9,
                timestamp: Date()
            )
        ]
        
        print("\nIntegrating learned facts...")
        for fact in newInformation {
            try await assistant.addLearnedFact(fact)
            print("  ✓ Added: \(fact.content.prefix(60))... (confidence: \(fact.confidence))")
        }
        
        // Show learning statistics
        print("\n\n📊 Learning Statistics:")
        let stats = await assistant.getLearningStats()
        
        print("  Total interactions: \(stats.totalInteractions)")
        print("  Positive feedback: \(stats.positiveFeedback)%")
        print("  Adaptations made: \(stats.adaptationCount)")
        print("  Knowledge updates: \(stats.knowledgeUpdates)")
        print("  Personalization profiles: \(stats.profileCount)")
        
        // Performance improvement over time
        print("\n  Performance trend:")
        for metric in stats.performanceTrend {
            print("    \(metric.period): \(String(format: "%.1f%%", metric.satisfactionScore)) satisfaction")
        }
    }
    
    // MARK: - Helper Functions
    
    func testFormatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Supporting Types

struct AIAssistantConfiguration {
    let vectorDimensions: Int
    let maxContextLength: Int
    let chunkSize: Int
    let chunkOverlap: Int
    let topK: Int
    let temperature: Float
    let enableCitation: Bool
    let enableFactChecking: Bool
}

actor AIAssistant {
    private let configuration: AIAssistantConfiguration
    private let vectorStore: VectorStore
    private let languageModel: LanguageModel
    private let knowledgeGraph: KnowledgeGraph
    private var sessions: [String: ConversationSession] = [:]
    private var tools: [String: Tool] = [:]
    private var learningEngine: LearningEngine?
    
    init(configuration: AIAssistantConfiguration) async throws {
        self.configuration = configuration
        
        // Initialize vector store for RAG
        let vectorConfig = StoreConfiguration(
            dimensions: configuration.vectorDimensions,
            distanceMetric: .cosine,
            indexType: .hnsw(HNSWConfiguration(
                dimensions: configuration.vectorDimensions,
                m: 32,
                efConstruction: 200
            ))
        )
        
        self.vectorStore = try await VectorStore(configuration: vectorConfig)
        
        // Initialize language model
        self.languageModel = LanguageModel(
            modelType: .gpt4,
            temperature: configuration.temperature
        )
        
        // Initialize knowledge graph
        self.knowledgeGraph = KnowledgeGraph()
    }
    
    // Document processing
    func chunkDocument(_ document: Document) async throws -> [DocumentChunk] {
        let chunks = TextChunker.chunk(
            text: document.content,
            chunkSize: configuration.chunkSize,
            overlap: configuration.chunkOverlap
        )
        
        return chunks.enumerated().map { index, content in
            DocumentChunk(
                id: "\(document.id)_chunk_\(index)",
                documentId: document.id,
                content: content,
                metadata: document.metadata,
                position: index
            )
        }
    }
    
    func generateEmbedding(for chunk: DocumentChunk) async throws -> [Float] {
        // Simulate embedding generation
        languageModel.generateEmbedding(text: chunk.content)
    }
    
    func indexChunk(_ chunk: DocumentChunk, embedding: [Float]) async throws {
        let entry = VectorEntry(
            id: chunk.id,
            vector: SIMD128<Float>(embedding),
            metadata: ChunkMetadata(
                documentId: chunk.documentId,
                position: chunk.position,
                content: chunk.content
            )
        )
        
        _ = try await vectorStore.add(entry)
    }
    
    func addRelationship(_ relation: KnowledgeRelation) async throws {
        try await knowledgeGraph.addRelation(relation)
    }
    
    func getKnowledgeBaseStats() async -> KnowledgeBaseStats {
        let chunkCount = await vectorStore.count()
        
        return KnowledgeBaseStats(
            documentCount: await knowledgeGraph.documentCount,
            chunkCount: chunkCount,
            avgChunkSize: 150,
            indexSize: chunkCount * configuration.vectorDimensions * 4,
            relationshipCount: await knowledgeGraph.relationCount
        )
    }
    
    // Question answering
    func answer(
        question: Question,
        options: AnswerOptions = AnswerOptions(),
        userProfile: UserProfile? = nil
    ) async throws -> AIResponse {
        // Generate question embedding
        let questionEmbedding = languageModel.generateEmbedding(text: question.text)
        
        // Retrieve relevant chunks
        let searchResults = try await vectorStore.search(
            vector: SIMD128<Float>(questionEmbedding),
            k: configuration.topK
        )
        
        // Build context from retrieved chunks
        let context = buildContext(from: searchResults)
        
        // Generate answer
        let prompt = constructPrompt(
            question: question.text,
            context: context,
            options: options,
            userProfile: userProfile
        )
        
        let answer = try await languageModel.generate(
            prompt: prompt,
            maxTokens: options.maxTokens
        )
        
        // Extract sources
        let sources = searchResults.map { result in
            Source(
                id: result.id,
                title: "Document \(result.metadata.documentId)",
                excerpt: result.metadata.content,
                relevance: 1.0 - result.distance
            )
        }
        
        // Fact check if enabled
        let factChecked = options.factCheck ? 
            try await verifyFactualAccuracy(answer: answer, sources: sources) : false
        
        // Check for hallucination
        let hallucinationDetected = detectHallucination(
            answer: answer,
            context: context
        )
        
        return AIResponse(
            answer: answer,
            sources: sources,
            confidence: calculateConfidence(searchResults: searchResults),
            factChecked: factChecked,
            hallucinationDetected: hallucinationDetected,
            reasoning: options.includeSources ? "Based on retrieved documents" : nil,
            adaptationType: userProfile != nil ? "personalized" : nil,
            complexityLevel: userProfile?.preferredStyle.rawValue ?? "standard",
            exampleTypes: ["domain-specific"],
            technicalTermCount: countTechnicalTerms(in: answer)
        )
    }
    
    // Conversational AI
    func createSession(sessionId: String, options: SessionOptions) async throws -> ConversationSession {
        let session = ConversationSession(
            id: sessionId,
            options: options,
            startTime: Date()
        )
        sessions[sessionId] = session
        return session
    }
    
    func chat(
        message: String,
        sessionId: String,
        options: ChatOptions = ChatOptions()
    ) async throws -> ChatResponse {
        guard let session = sessions[sessionId] else {
            throw AIAssistantError.sessionNotFound
        }
        
        // Add message to history
        session.addMessage(role: .user, content: message)
        
        // Retrieve relevant information if needed
        var retrievedInfo: [RetrievedInfo] = []
        if options.retrieveRelevantInfo {
            let embedding = languageModel.generateEmbedding(text: message)
            let results = try await vectorStore.search(
                vector: SIMD128<Float>(embedding),
                k: 3
            )
            
            retrievedInfo = results.map { result in
                RetrievedInfo(
                    source: result.metadata.documentId,
                    snippet: result.metadata.content
                )
            }
        }
        
        // Build conversation context
        let conversationContext = session.getRecentContext(maxTokens: 2048)
        let augmentedContext = options.useContext ? 
            augmentContext(conversation: conversationContext, retrieved: retrievedInfo) : 
            conversationContext
        
        // Generate response
        let response = try await languageModel.generate(
            prompt: augmentedContext + "\nAssistant:",
            maxTokens: 500
        )
        
        // Add response to history
        session.addMessage(role: .assistant, content: response)
        
        // Extract context elements used
        let contextElements = session.messages.suffix(3).map { msg in
            ContextElement(
                summary: String(msg.content.prefix(50)) + "..."
            )
        }
        
        return ChatResponse(
            message: response,
            contextUsed: options.useContext,
            contextElements: contextElements,
            retrievedInfo: retrievedInfo
        )
    }
    
    func summarizeSession(sessionId: String) async throws -> SessionSummary {
        guard let session = sessions[sessionId] else {
            throw AIAssistantError.sessionNotFound
        }
        
        let fullConversation = session.getFullConversation()
        
        let summaryPrompt = """
        Summarize the following conversation, identifying key topics and any action items:
        
        \(fullConversation)
        
        Provide:
        1. A brief summary
        2. List of main topics discussed
        3. Any action items or follow-ups identified
        """
        
        let summaryResponse = try await languageModel.generate(
            prompt: summaryPrompt,
            maxTokens: 500
        )
        
        // Parse response (simplified)
        return SessionSummary(
            summary: summaryResponse,
            topics: ["Vector databases", "SQL comparison", "Performance", "E-commerce use case"],
            actionItems: ["Evaluate vector database for recommendation system"]
        )
    }
    
    // Document analysis
    func analyzeDocument(
        document: Document,
        analysisOptions: AnalysisOptions
    ) async throws -> DocumentAnalysis {
        var analysis = DocumentAnalysis()
        
        // Generate summary
        if analysisOptions.summarize {
            let summaryPrompt = "Summarize the following document in 3-4 sentences:\n\n\(document.content)"
            analysis.summary = try await languageModel.generate(
                prompt: summaryPrompt,
                maxTokens: 200
            )
        }
        
        // Extract key points
        if analysisOptions.extractKeyPoints {
            let keyPointsPrompt = "Extract 5 key points from this document:\n\n\(document.content)"
            let keyPointsResponse = try await languageModel.generate(
                prompt: keyPointsPrompt,
                maxTokens: 300
            )
            analysis.keyPoints = parseKeyPoints(from: keyPointsResponse)
        }
        
        // Identify entities
        if analysisOptions.identifyEntities {
            analysis.entities = extractEntities(from: document.content)
        }
        
        // Assess readability
        if analysisOptions.assessReadability {
            analysis.readability = assessReadability(text: document.content)
        }
        
        // Generate questions
        if analysisOptions.generateQuestions {
            let questionsPrompt = "Generate 5 questions that this document answers:\n\n\(document.content)"
            let questionsResponse = try await languageModel.generate(
                prompt: questionsPrompt,
                maxTokens: 300
            )
            analysis.generatedQuestions = parseQuestions(from: questionsResponse)
        }
        
        return analysis
    }
    
    func compareDocuments(
        primaryDoc: String,
        compareWith: [String]
    ) async throws -> DocumentComparison {
        var comparison = DocumentComparison()
        
        // Get embeddings for all documents
        // Compare similarities
        // Extract common themes and unique contributions
        
        comparison.similarityScores = [
            SimilarityScore(documentId: "tech_1", similarity: 0.75),
            SimilarityScore(documentId: "tech_2", similarity: 0.82)
        ]
        
        comparison.commonThemes = [
            "Neural architectures",
            "Embedding generation",
            "Domain-specific optimization"
        ]
        
        comparison.uniqueContributions = [
            "Novel NAS approach for embeddings",
            "Multi-objective optimization framework",
            "Domain-specific architectural patterns"
        ]
        
        return comparison
    }
    
    // Multi-turn reasoning
    func reason(
        instruction: String,
        scenario: ProblemScenario,
        context: ReasoningContext,
        options: ReasoningOptions
    ) async throws -> ReasoningStep {
        let prompt = constructReasoningPrompt(
            instruction: instruction,
            scenario: scenario,
            previousSteps: context.steps
        )
        
        let reasoning = try await languageModel.generate(
            prompt: prompt,
            maxTokens: 800
        )
        
        var step = ReasoningStep(reasoning: reasoning)
        
        // Extract calculations if needed
        if options.includeCalculations {
            step.calculations = extractCalculations(from: reasoning)
        }
        
        // Identify assumptions
        if options.validateAssumptions {
            step.assumptions = identifyAssumptions(in: reasoning)
        }
        
        return step
    }
    
    func synthesizeRecommendation(
        context: ReasoningContext,
        criteria: [String]
    ) async throws -> Recommendation {
        let synthesisPrompt = """
        Based on the following analysis steps, provide a final recommendation:
        
        \(context.getSummary())
        
        Evaluate against these criteria: \(criteria.joined(separator: ", "))
        
        Provide:
        1. Executive summary
        2. Detailed implementation plan
        3. Expected outcomes
        """
        
        let response = try await languageModel.generate(
            prompt: synthesisPrompt,
            maxTokens: 1000
        )
        
        // Parse response into structured recommendation
        return Recommendation(
            summary: "Implement a multi-tier optimization strategy...",
            implementationSteps: [
                ImplementationStep(
                    description: "Reduce HNSW parameters (M=12, ef=150)",
                    estimatedTime: "30 minutes",
                    riskLevel: "Low"
                ),
                ImplementationStep(
                    description: "Enable query result caching",
                    estimatedTime: "1 hour",
                    riskLevel: "Low"
                ),
                ImplementationStep(
                    description: "Implement request batching",
                    estimatedTime: "2 hours",
                    riskLevel: "Medium"
                )
            ],
            expectedOutcomes: ExpectedOutcomes(
                latencyReduction: 67,
                recallImpact: -2,
                resourceChange: -15
            )
        )
    }
    
    // Tool integration
    func registerTool(_ tool: Tool) async throws {
        tools[tool.name] = tool
    }
    
    func processWithTools(
        query: String,
        options: ToolUseOptions
    ) async throws -> ToolResponse {
        // Analyze query to determine required tools
        let requiredTools = options.autoSelectTools ? 
            identifyRequiredTools(for: query) : []
        
        var toolsUsed: [ToolUse] = []
        var intermediateResults: [String: String] = [:]
        
        // Execute tools
        for toolName in requiredTools {
            if let tool = tools[toolName] {
                let input = extractToolInput(for: tool, from: query, context: intermediateResults)
                let result = try await tool.handler(input)
                
                toolsUsed.append(ToolUse(
                    toolName: tool.name,
                    purpose: tool.description,
                    input: input.description,
                    result: result
                ))
                
                intermediateResults[tool.name] = result
            }
        }
        
        // Generate final answer incorporating tool results
        let finalAnswer = try await generateToolAugmentedResponse(
            query: query,
            toolResults: intermediateResults
        )
        
        let explanation = options.explainToolUse ? 
            "Selected tools based on query requirements" : nil
        
        return ToolResponse(
            answer: finalAnswer,
            toolsUsed: toolsUsed,
            toolUseExplanation: explanation
        )
    }
    
    // Learning and adaptation
    func enableLearning(options: LearningOptions) async throws {
        learningEngine = LearningEngine(options: options)
    }
    
    func processFeedback(interaction: UserInteraction) async throws -> Improvement? {
        guard let engine = learningEngine else { return nil }
        
        return try await engine.processFeedback(interaction)
    }
    
    func addLearnedFact(_ fact: LearnedFact) async throws {
        guard let engine = learningEngine else { return }
        
        try await engine.addFact(fact)
        
        // Update knowledge base if confidence is high enough
        if fact.confidence >= 0.8 {
            // Create a new document chunk for the learned fact
            let chunk = DocumentChunk(
                id: "learned_\(UUID().uuidString)",
                documentId: "learned_facts",
                content: fact.content,
                metadata: DocumentMetadata(
                    author: "Learning Engine",
                    createdDate: fact.timestamp,
                    source: fact.source
                ),
                position: 0
            )
            
            let embedding = try await generateEmbedding(for: chunk)
            try await indexChunk(chunk, embedding: embedding)
        }
    }
    
    func getLearningStats() async -> LearningStats {
        guard let engine = learningEngine else {
            return LearningStats()
        }
        
        return await engine.getStats()
    }
    
    // MARK: - Private Methods
    
    private func buildContext(from searchResults: [SearchResult]) -> String {
        searchResults
            .map { $0.metadata.content }
            .joined(separator: "\n\n")
    }
    
    private func constructPrompt(
        question: String,
        context: String,
        options: AnswerOptions,
        userProfile: UserProfile?
    ) -> String {
        var prompt = """
        Based on the following context, answer the question.
        
        Context:
        \(context)
        
        Question: \(question)
        """
        
        if options.includeSources {
            prompt += "\n\nInclude references to source material."
        }
        
        if let profile = userProfile {
            prompt += "\n\nAdapt your response for a \(profile.expertise) level user."
        }
        
        prompt += "\n\nAnswer:"
        
        return prompt
    }
    
    private func verifyFactualAccuracy(answer: String, sources: [Source]) async throws -> Bool {
        // Simplified fact checking
        return !answer.isEmpty && !sources.isEmpty
    }
    
    private func detectHallucination(answer: String, context: String) -> Bool {
        // Simplified hallucination detection
        // In practice, would use more sophisticated methods
        return false
    }
    
    private func calculateConfidence(searchResults: [SearchResult]) -> Float {
        guard !searchResults.isEmpty else { return 0 }
        
        let avgRelevance = searchResults
            .map { 1.0 - $0.distance }
            .reduce(0, +) / Float(searchResults.count)
        
        return min(avgRelevance * 1.2, 1.0)
    }
    
    private func countTechnicalTerms(in text: String) -> Int {
        let technicalTerms = ["vector", "embedding", "index", "algorithm", "optimization"]
        return technicalTerms.filter { text.lowercased().contains($0) }.count
    }
    
    private func augmentContext(conversation: String, retrieved: [RetrievedInfo]) -> String {
        var augmented = conversation
        
        if !retrieved.isEmpty {
            augmented += "\n\nRelevant information:"
            for info in retrieved {
                augmented += "\n- \(info.snippet)"
            }
        }
        
        return augmented
    }
    
    private func parseKeyPoints(from response: String) -> [String] {
        // Simple parsing - in practice would be more sophisticated
        response.components(separatedBy: "\n")
            .filter { !$0.isEmpty }
            .prefix(5)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }
    
    private func extractEntities(from text: String) -> Entities {
        // Simplified entity extraction
        Entities(
            technologies: ["AutoEmbed", "BERT", "ResNet", "NAS"],
            concepts: ["embeddings", "architecture search", "optimization"],
            metrics: ["15-30% improvement", "5 domains"]
        )
    }
    
    private func assessReadability(text: String) -> Readability {
        // Simplified readability assessment
        let wordCount = text.split(separator: " ").count
        let sentenceCount = text.split(separator: ".").count
        let avgWordsPerSentence = wordCount / max(sentenceCount, 1)
        
        let score = avgWordsPerSentence < 20 ? 80 : 60
        let level = avgWordsPerSentence < 15 ? "Intermediate" : "Advanced"
        let readingTime = wordCount / 200 // Assuming 200 words per minute
        
        return Readability(
            score: score,
            technicalLevel: level,
            estimatedReadingTime: readingTime
        )
    }
    
    private func parseQuestions(from response: String) -> [String] {
        response.components(separatedBy: "\n")
            .filter { $0.contains("?") }
            .prefix(5)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }
    
    private func constructReasoningPrompt(
        instruction: String,
        scenario: ProblemScenario,
        previousSteps: [ReasoningStep]
    ) -> String {
        var prompt = """
        Problem: \(scenario.title)
        Context: \(scenario.context)
        Constraints: \(scenario.constraints.joined(separator: ", "))
        Goal: \(scenario.goal)
        
        """
        
        if !previousSteps.isEmpty {
            prompt += "Previous analysis:\n"
            for (i, step) in previousSteps.enumerated() {
                prompt += "\(i + 1). \(step.reasoning.prefix(100))...\n"
            }
            prompt += "\n"
        }
        
        prompt += "Current task: \(instruction)\n\n"
        prompt += "Provide detailed reasoning:"
        
        return prompt
    }
    
    private func extractCalculations(from reasoning: String) -> [Calculation] {
        // Simplified extraction
        [
            Calculation(
                description: "Memory usage for 10M vectors",
                result: "~30GB with current settings"
            ),
            Calculation(
                description: "Expected latency with optimized parameters",
                result: "~45ms (70% reduction)"
            )
        ]
    }
    
    private func identifyAssumptions(in reasoning: String) -> [String] {
        [
            "CPU can handle increased computation from lower ef value",
            "Cache hit rate will be at least 30%",
            "Query distribution remains uniform"
        ]
    }
    
    private func identifyRequiredTools(for query: String) -> [String] {
        var required: [String] = []
        
        if query.lowercased().contains("calculate") {
            required.append("calculator")
        }
        if query.lowercased().contains("search") {
            required.append("vector_search")
        }
        if query.lowercased().contains("generate") && query.lowercased().contains("code") {
            required.append("code_generator")
        }
        if query.lowercased().contains("analyze") {
            required.append("data_analyzer")
        }
        
        return required
    }
    
    private func extractToolInput(
        for tool: Tool,
        from query: String,
        context: [String: String]
    ) -> [String: String] {
        // Simplified input extraction
        var input: [String: String] = ["query": query]
        
        switch tool.name {
        case "calculator":
            input["expression"] = "5_000_000 * 384 * 4 / (1024^3)"
        case "code_generator":
            input["language"] = "Swift"
            input["task"] = "cosine similarity implementation"
        default:
            break
        }
        
        return input
    }
    
    private func generateToolAugmentedResponse(
        query: String,
        toolResults: [String: String]
    ) async throws -> String {
        let prompt = """
        Query: \(query)
        
        Tool results:
        \(toolResults.map { "\($0.key): \($0.value)" }.joined(separator: "\n"))
        
        Provide a comprehensive answer incorporating these results:
        """
        
        return try await languageModel.generate(
            prompt: prompt,
            maxTokens: 500
        )
    }
}

// Document types
struct Document {
    let id: String
    let title: String
    let content: String
    let category: String
    let metadata: DocumentMetadata
}

struct DocumentMetadata: Codable, Sendable {
    let author: String
    let createdDate: Date
    let source: String
    var version: String = "1.0"
}

struct DocumentChunk {
    let id: String
    let documentId: String
    let content: String
    let metadata: DocumentMetadata
    let position: Int
}

struct ChunkMetadata: Codable, Sendable {
    let documentId: String
    let position: Int
    let content: String
}

// Knowledge graph types
struct KnowledgeRelation {
    let source: String
    let target: String
    let type: RelationType
    let strength: Float
    
    enum RelationType {
        case implements
        case relatedTo
        case governs
        case references
    }
}

struct KnowledgeBaseStats {
    let documentCount: Int
    let chunkCount: Int
    let avgChunkSize: Int
    let indexSize: Int
    let relationshipCount: Int
}

// Question answering types
struct Question {
    let text: String
    let type: QuestionType
    var expectsCitation: Bool = false
    
    enum QuestionType {
        case factual
        case comparison
        case policy
        case product
        case howTo
        case technical
        case educational
    }
}

struct AnswerOptions {
    var maxTokens: Int = 500
    var temperature: Float = 0.7
    var includeSources: Bool = true
    var factCheck: Bool = false
    var useAdaptations: Bool = false
    var simplicityLevel: SimplityLevel = .intermediate
    
    enum SimplityLevel {
        case beginner
        case intermediate
        case advanced
    }
}

struct AIResponse {
    let answer: String
    let sources: [Source]
    let confidence: Float
    let factChecked: Bool
    let hallucinationDetected: Bool
    let reasoning: String?
    var adaptationType: String?
    var complexityLevel: String
    var exampleTypes: [String]
    var technicalTermCount: Int
}

struct Source {
    let id: String
    let title: String
    let excerpt: String
    let relevance: Float
}

// Conversational AI types
struct SessionOptions {
    let maxTurns: Int
    let contextWindow: Int
    let personalization: PersonalizationSettings
}

struct PersonalizationSettings {
    let tone: Tone
    let expertise: Expertise
    let verbosity: Verbosity
    
    enum Tone {
        case casual
        case professional
        case academic
    }
    
    enum Expertise {
        case beginner
        case intermediate
        case expert
    }
    
    enum Verbosity {
        case concise
        case balanced
        case detailed
    }
}

class ConversationSession {
    let id: String
    let options: SessionOptions
    let startTime: Date
    var messages: [Message] = []
    
    init(id: String, options: SessionOptions, startTime: Date) {
        self.id = id
        self.options = options
        self.startTime = startTime
    }
    
    func addMessage(role: MessageRole, content: String) {
        messages.append(Message(role: role, content: content, timestamp: Date()))
    }
    
    func getRecentContext(maxTokens: Int) -> String {
        // Get recent messages that fit within token limit
        messages.suffix(5)
            .map { "\($0.role): \($0.content)" }
            .joined(separator: "\n")
    }
    
    func getFullConversation() -> String {
        messages
            .map { "\($0.role): \($0.content)" }
            .joined(separator: "\n")
    }
}

struct Message {
    let role: MessageRole
    let content: String
    let timestamp: Date
    
    enum MessageRole: String {
        case user = "User"
        case assistant = "Assistant"
        case system = "System"
    }
}

struct ChatOptions {
    var useContext: Bool = true
    var retrieveRelevantInfo: Bool = true
    var maintainPersona: Bool = true
}

struct ChatResponse {
    let message: String
    let contextUsed: Bool
    let contextElements: [ContextElement]
    let retrievedInfo: [RetrievedInfo]
}

struct ContextElement {
    let summary: String
}

struct RetrievedInfo {
    let source: String
    let snippet: String
}

struct SessionSummary {
    let summary: String
    let topics: [String]
    let actionItems: [String]
}

// Document analysis types
struct AnalysisOptions {
    let summarize: Bool
    let extractKeyPoints: Bool
    let identifyEntities: Bool
    let assessReadability: Bool
    let generateQuestions: Bool
}

struct DocumentAnalysis {
    var summary: String = ""
    var keyPoints: [String] = []
    var entities: Entities = Entities()
    var readability: Readability = Readability()
    var generatedQuestions: [String] = []
}

struct Entities {
    var technologies: [String] = []
    var concepts: [String] = []
    var metrics: [String] = []
}

struct Readability {
    var score: Int = 0
    var technicalLevel: String = ""
    var estimatedReadingTime: Int = 0
}

struct DocumentComparison {
    var similarityScores: [SimilarityScore] = []
    var commonThemes: [String] = []
    var uniqueContributions: [String] = []
}

struct SimilarityScore {
    let documentId: String
    let similarity: Float
}

// Reasoning types
struct ProblemScenario {
    let title: String
    let context: String
    let constraints: [String]
    let goal: String
}

class ReasoningContext {
    var steps: [ReasoningStep] = []
    
    func addStep(_ step: ReasoningStep) {
        steps.append(step)
    }
    
    func getSummary() -> String {
        steps.enumerated()
            .map { "Step \($0 + 1): \($1.reasoning.prefix(100))..." }
            .joined(separator: "\n")
    }
}

struct ReasoningStep {
    let reasoning: String
    var calculations: [Calculation] = []
    var assumptions: [String] = []
}

struct Calculation {
    let description: String
    let result: String
}

struct ReasoningOptions {
    let depth: Depth
    let includeCalculations: Bool
    let validateAssumptions: Bool
    
    enum Depth {
        case shallow
        case medium
        case deep
    }
}

struct Recommendation {
    let summary: String
    let implementationSteps: [ImplementationStep]
    let expectedOutcomes: ExpectedOutcomes
}

struct ImplementationStep {
    let description: String
    let estimatedTime: String
    let riskLevel: String
}

struct ExpectedOutcomes {
    let latencyReduction: Int
    let recallImpact: Int
    let resourceChange: Int
}

// Tool integration types
struct Tool {
    let name: String
    let description: String
    let parameters: [String: String]
    let handler: ([String: String]) async throws -> String
}

struct ToolUseOptions {
    let autoSelectTools: Bool
    let explainToolUse: Bool
    let maxToolCalls: Int = 5
    var allowChaining: Bool = false
}

struct ToolResponse {
    let answer: String
    let toolsUsed: [ToolUse]
    let toolUseExplanation: String?
}

struct ToolUse {
    let toolName: String
    let purpose: String
    let input: String
    let result: String?
}

// Learning types
struct LearningOptions {
    let learnFromFeedback: Bool
    let personalizeResponses: Bool
    let updateKnowledgeBase: Bool
    let feedbackThreshold: Float
}

struct UserInteraction {
    let query: String
    let response: String
    let feedback: Feedback
    
    enum Feedback {
        case positive(score: Float, comment: String)
        case negative(score: Float, comment: String)
        case neutral
    }
}

struct Improvement {
    let type: String
    let action: String
}

struct UserProfile {
    let id: String
    let expertise: Expertise
    let interests: [String]
    let preferredStyle: Style
    
    enum Expertise {
        case beginner
        case intermediate
        case expert
    }
    
    enum Style: String {
        case simple
        case balanced
        case technical
    }
}

struct LearnedFact {
    let content: String
    let source: String
    let confidence: Float
    let timestamp: Date
}

struct LearningStats {
    var totalInteractions: Int = 0
    var positiveFeedback: Float = 0
    var adaptationCount: Int = 0
    var knowledgeUpdates: Int = 0
    var profileCount: Int = 0
    var performanceTrend: [PerformanceMetric] = []
}

struct PerformanceMetric {
    let period: String
    let satisfactionScore: Float
}

// Mock implementations
class VectorStore {
    private let configuration: StoreConfiguration
    private var entries: [VectorEntry<SIMD128<Float>, ChunkMetadata>] = []
    
    init(configuration: StoreConfiguration) async throws {
        self.configuration = configuration
    }
    
    func add(_ entry: VectorEntry<SIMD128<Float>, ChunkMetadata>) async throws -> String {
        entries.append(entry)
        return entry.id
    }
    
    func search(vector: SIMD128<Float>, k: Int) async throws -> [SearchResult] {
        // Mock search results
        entries.prefix(k).map { entry in
            SearchResult(
                id: entry.id,
                distance: Float.random(in: 0.1...0.5),
                metadata: entry.metadata
            )
        }
    }
    
    func count() async -> Int {
        entries.count
    }
}

struct AISearchResult {
    let id: String
    let distance: Float
    let metadata: ChunkMetadata
}

class LanguageModel {
    let modelType: ModelType
    let temperature: Float
    
    enum ModelType {
        case gpt4
        case claude
        case llama
    }
    
    init(modelType: ModelType, temperature: Float) {
        self.modelType = modelType
        self.temperature = temperature
    }
    
    func generate(prompt: String, maxTokens: Int) async throws -> String {
        // Mock response generation
        switch prompt {
        case let p where p.contains("vector databases"):
            return "Vector databases are specialized systems designed for storing and searching high-dimensional vectors efficiently. They use advanced indexing techniques like HNSW and IVF to enable fast similarity search."
            
        case let p where p.contains("computational complexity"):
            return "HNSW (Hierarchical Navigable Small World) has a search complexity of O(log n), making it highly efficient for large-scale vector search. The construction complexity is O(n log n)."
            
        case let p where p.contains("vector embeddings work"):
            return "Vector embeddings transform data (text, images, etc.) into numerical representations in high-dimensional space. Similar items have vectors that are close together, enabling similarity search."
            
        default:
            return "Based on the provided context, here's a comprehensive answer to your question..."
        }
    }
    
    func generateEmbedding(text: String) -> [Float] {
        // Mock embedding generation
        (0..<768).map { _ in Float.random(in: -1...1) }
    }
}

class KnowledgeGraph {
    var documentCount: Int = 0
    var relationCount: Int = 0
    private var relations: [KnowledgeRelation] = []
    
    func addRelation(_ relation: KnowledgeRelation) async throws {
        relations.append(relation)
        relationCount += 1
    }
}

class LearningEngine {
    private let options: LearningOptions
    private var facts: [LearnedFact] = []
    private var stats = LearningStats()
    
    init(options: LearningOptions) {
        self.options = options
    }
    
    func processFeedback(_ interaction: UserInteraction) async throws -> Improvement? {
        stats.totalInteractions += 1
        
        switch interaction.feedback {
        case .positive(let score, _):
            stats.positiveFeedback = (stats.positiveFeedback * Float(stats.totalInteractions - 1) + score) / Float(stats.totalInteractions)
            return nil
            
        case .negative(let score, let comment):
            if score < options.feedbackThreshold {
                stats.adaptationCount += 1
                return Improvement(
                    type: "Response Style",
                    action: "Simplify technical explanations based on feedback"
                )
            }
            return nil
            
        case .neutral:
            return nil
        }
    }
    
    func addFact(_ fact: LearnedFact) async throws {
        facts.append(fact)
        stats.knowledgeUpdates += 1
    }
    
    func getStats() async -> LearningStats {
        stats.performanceTrend = [
            PerformanceMetric(period: "Last hour", satisfactionScore: 85.2),
            PerformanceMetric(period: "Last day", satisfactionScore: 87.5),
            PerformanceMetric(period: "Last week", satisfactionScore: 89.1)
        ]
        return stats
    }
}

// Text processing utilities
enum TextChunker {
    func testChunk(text: String, chunkSize: Int, overlap: Int) -> [String] {
        let words = text.split(separator: " ")
        var chunks: [String] = []
        
        var i = 0
        while i < words.count {
            let end = min(i + chunkSize, words.count)
            let chunk = words[i..<end].joined(separator: " ")
            chunks.append(chunk)
            
            i += chunkSize - overlap
            if i >= words.count - overlap {
                break
            }
        }
        
        return chunks
    }
}

// Error types
enum AIAssistantError: Error {
    case sessionNotFound
    case toolNotFound
    case invalidConfiguration
}

// String multiplication helper
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// SIMD initialization helper
extension SIMD128<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(128, array.count) {
            self[i] = array[i]
        }
    }
}