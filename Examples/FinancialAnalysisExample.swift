import Foundation
import VectorStoreKit

/// Financial Analysis Example demonstrating time-series similarity search for financial data
///
/// This example showcases:
/// - Financial time-series embeddings (stocks, volumes, indicators)
/// - Pattern recognition (head & shoulders, double bottom, etc.)
/// - Correlation analysis between instruments
/// - Anomaly detection for fraud/market anomalies
/// - Portfolio similarity and risk analysis
@main
struct FinancialAnalysisExample {
    static func main() async throws {
        print("=== VectorStoreKit Financial Analysis Example ===\n")
        
        let example = FinancialAnalysisExample()
        
        // Run all demonstrations
        try await example.demonstrateTimeSeriesEmbeddings()
        try await example.demonstratePatternRecognition()
        try await example.demonstrateCorrelationAnalysis()
        try await example.demonstrateAnomalyDetection()
        try await example.demonstratePortfolioAnalysis()
        try await example.demonstrateRealTimeAnalysis()
    }
    
    // MARK: - Data Structures
    
    /// Represents a financial time series data point
    struct FinancialDataPoint {
        let timestamp: Date
        let open: Double
        let high: Double
        let low: Double
        let close: Double
        let volume: Double
        let rsi: Double?        // Relative Strength Index
        let macd: Double?       // Moving Average Convergence Divergence
        let bollingerBand: (upper: Double, middle: Double, lower: Double)?
    }
    
    /// Represents a financial instrument
    struct FinancialInstrument {
        let symbol: String
        let name: String
        let sector: String
        let marketCap: Double
        let timeSeries: [FinancialDataPoint]
    }
    
    /// Pattern types for technical analysis
    enum TechnicalPattern: String, CaseIterable {
        case headAndShoulders = "Head and Shoulders"
        case doubleBottom = "Double Bottom"
        case ascendingTriangle = "Ascending Triangle"
        case bearishFlag = "Bearish Flag"
        case bullishPennant = "Bullish Pennant"
        case cupAndHandle = "Cup and Handle"
    }
    
    // MARK: - Time Series Embeddings
    
    func demonstrateTimeSeriesEmbeddings() async throws {
        print("1. Time Series Embeddings Demonstration")
        print("=" * 40)
        
        // Create mock financial instruments
        let instruments = createMockFinancialInstruments()
        
        // Initialize vector store with financial-specific configuration
        let config = StoreConfiguration(
            dimensions: 128,  // Embedding dimension for financial features
            similarity: .cosine,
            indexType: .hierarchical(
                levels: 3,
                branchingFactor: 16
            )
        )
        
        let store = try VectorStore(configuration: config)
        
        // Demonstrate different embedding strategies
        print("\nEmbedding Strategies:")
        
        // 1. Sliding Window Embeddings
        print("\n  a) Sliding Window Embeddings (20-day windows)")
        for instrument in instruments {
            let embeddings = createSlidingWindowEmbeddings(
                timeSeries: instrument.timeSeries,
                windowSize: 20
            )
            
            for (idx, embedding) in embeddings.enumerated() {
                let metadata: [String: Any] = [
                    "symbol": instrument.symbol,
                    "strategy": "sliding_window",
                    "window_index": idx,
                    "timestamp": instrument.timeSeries[idx].timestamp.timeIntervalSince1970
                ]
                
                try await store.add(
                    id: "\(instrument.symbol)_sw_\(idx)",
                    vector: embedding,
                    metadata: metadata
                )
            }
        }
        
        // 2. Spectral Embeddings (frequency domain)
        print("  b) Spectral Embeddings (Fourier transform)")
        for instrument in instruments {
            let spectralEmbedding = createSpectralEmbedding(
                timeSeries: instrument.timeSeries
            )
            
            let metadata: [String: Any] = [
                "symbol": instrument.symbol,
                "strategy": "spectral",
                "sector": instrument.sector
            ]
            
            try await store.add(
                id: "\(instrument.symbol)_spectral",
                vector: spectralEmbedding,
                metadata: metadata
            )
        }
        
        // 3. Technical Indicator Embeddings
        print("  c) Technical Indicator Embeddings")
        for instrument in instruments {
            let technicalEmbedding = createTechnicalIndicatorEmbedding(
                timeSeries: instrument.timeSeries
            )
            
            let metadata: [String: Any] = [
                "symbol": instrument.symbol,
                "strategy": "technical_indicators",
                "market_cap": instrument.marketCap
            ]
            
            try await store.add(
                id: "\(instrument.symbol)_technical",
                vector: technicalEmbedding,
                metadata: metadata
            )
        }
        
        // Query similar time series patterns
        print("\n  Finding similar price movements to AAPL last 20 days...")
        let aaplRecent = createSlidingWindowEmbeddings(
            timeSeries: instruments[0].timeSeries,
            windowSize: 20
        ).last!
        
        let results = try await store.search(
            query: aaplRecent,
            k: 5,
            filters: ["strategy": "sliding_window"]
        )
        
        print("  Similar patterns found:")
        for result in results {
            let symbol = result.metadata["symbol"] as? String ?? "Unknown"
            print("    - \(symbol): similarity = \(String(format: "%.3f", result.score))")
        }
        
        print("\n" + "=" * 40 + "\n")
    }
    
    // MARK: - Pattern Recognition
    
    func demonstratePatternRecognition() async throws {
        print("2. Technical Pattern Recognition")
        print("=" * 40)
        
        // Create pattern templates
        let patternTemplates = createPatternTemplates()
        
        // Initialize pattern recognition store
        let config = StoreConfiguration(
            dimensions: 64,  // Pattern feature dimension
            similarity: .euclidean,
            indexType: .ivf(
                nlist: 50,
                nprobe: 5
            )
        )
        
        let patternStore = try VectorStore(configuration: config)
        
        // Index pattern templates
        print("\nIndexing technical patterns:")
        for (pattern, embedding) in patternTemplates {
            let metadata: [String: Any] = [
                "pattern_type": pattern.rawValue,
                "reliability": getPatternReliability(pattern)
            ]
            
            try await patternStore.add(
                id: pattern.rawValue,
                vector: embedding,
                metadata: metadata
            )
            print("  - \(pattern.rawValue)")
        }
        
        // Detect patterns in mock data
        print("\nScanning for patterns in market data...")
        let instruments = createMockFinancialInstruments()
        
        for instrument in instruments {
            print("\n  \(instrument.symbol):")
            
            // Scan time series for patterns
            let detectedPatterns = await scanForPatterns(
                timeSeries: instrument.timeSeries,
                patternStore: patternStore
            )
            
            for detection in detectedPatterns {
                print("    - \(detection.pattern) at day \(detection.dayIndex) " +
                      "(confidence: \(String(format: "%.2f", detection.confidence)))")
            }
        }
        
        print("\n" + "=" * 40 + "\n")
    }
    
    // MARK: - Correlation Analysis
    
    func demonstrateCorrelationAnalysis() async throws {
        print("3. Cross-Asset Correlation Analysis")
        print("=" * 40)
        
        let instruments = createMockFinancialInstruments()
        
        // Create correlation matrix embeddings
        let config = StoreConfiguration(
            dimensions: 256,  // Higher dimension for correlation features
            similarity: .cosine,
            indexType: .hnsw(
                m: 16,
                efConstruction: 200,
                efSearch: 50
            )
        )
        
        let correlationStore = try VectorStore(configuration: config)
        
        print("\nComputing correlation embeddings:")
        
        // Create correlation embeddings for each instrument pair
        for i in 0..<instruments.count {
            for j in (i+1)..<instruments.count {
                let correlation = computeCorrelation(
                    series1: instruments[i].timeSeries,
                    series2: instruments[j].timeSeries
                )
                
                let embedding = createCorrelationEmbedding(
                    instrument1: instruments[i],
                    instrument2: instruments[j],
                    correlation: correlation
                )
                
                let metadata: [String: Any] = [
                    "symbol1": instruments[i].symbol,
                    "symbol2": instruments[j].symbol,
                    "correlation": correlation.value,
                    "rolling_correlation": correlation.rolling,
                    "sector_pair": "\(instruments[i].sector)-\(instruments[j].sector)"
                ]
                
                try await correlationStore.add(
                    id: "\(instruments[i].symbol)_\(instruments[j].symbol)",
                    vector: embedding,
                    metadata: metadata
                )
                
                print("  - \(instruments[i].symbol) <-> \(instruments[j].symbol): " +
                      "ρ = \(String(format: "%.3f", correlation.value))")
            }
        }
        
        // Find highly correlated pairs
        print("\nFinding highly correlated asset pairs:")
        let highCorrelationQuery = createCorrelationQueryVector(minCorrelation: 0.7)
        
        let correlatedPairs = try await correlationStore.search(
            query: highCorrelationQuery,
            k: 10
        )
        
        for pair in correlatedPairs {
            let symbol1 = pair.metadata["symbol1"] as? String ?? ""
            let symbol2 = pair.metadata["symbol2"] as? String ?? ""
            let corr = pair.metadata["correlation"] as? Double ?? 0
            
            print("  - \(symbol1) <-> \(symbol2): ρ = \(String(format: "%.3f", corr))")
        }
        
        print("\n" + "=" * 40 + "\n")
    }
    
    // MARK: - Anomaly Detection
    
    func demonstrateAnomalyDetection() async throws {
        print("4. Anomaly Detection for Fraud & Market Events")
        print("=" * 40)
        
        // Create normal and anomalous transaction patterns
        let transactions = createMockTransactions()
        
        let config = StoreConfiguration(
            dimensions: 96,
            similarity: .euclidean,
            indexType: .hierarchical(
                levels: 4,
                branchingFactor: 8
            )
        )
        
        let anomalyStore = try VectorStore(configuration: config)
        
        print("\nIndexing normal transaction patterns:")
        
        // Index normal patterns
        let normalTransactions = transactions.filter { !$0.isAnomalous }
        for transaction in normalTransactions {
            let embedding = createTransactionEmbedding(transaction)
            
            let metadata: [String: Any] = [
                "type": transaction.type,
                "amount": transaction.amount,
                "time_of_day": transaction.timeOfDay,
                "merchant_category": transaction.merchantCategory
            ]
            
            try await anomalyStore.add(
                id: transaction.id,
                vector: embedding,
                metadata: metadata
            )
        }
        
        print("  Indexed \(normalTransactions.count) normal patterns")
        
        // Detect anomalies
        print("\nScanning for anomalous transactions:")
        let anomalousTransactions = transactions.filter { $0.isAnomalous }
        
        for transaction in anomalousTransactions {
            let embedding = createTransactionEmbedding(transaction)
            
            // Find nearest normal transactions
            let nearestNormal = try await anomalyStore.search(
                query: embedding,
                k: 5
            )
            
            // Calculate anomaly score
            let anomalyScore = calculateAnomalyScore(
                distances: nearestNormal.map { $0.score }
            )
            
            if anomalyScore > 0.7 {
                print("  🚨 ANOMALY DETECTED: Transaction \(transaction.id)")
                print("     Type: \(transaction.anomalyType ?? "Unknown")")
                print("     Score: \(String(format: "%.3f", anomalyScore))")
                print("     Amount: $\(String(format: "%.2f", transaction.amount))")
            }
        }
        
        // Market anomaly detection
        print("\nDetecting market anomalies:")
        let marketEvents = createMarketAnomalyEvents()
        
        for event in marketEvents {
            let embedding = createMarketEventEmbedding(event)
            let normalMarketQuery = createNormalMarketEmbedding()
            
            let distance = euclideanDistance(embedding, normalMarketQuery)
            if distance > 2.5 {
                print("  ⚠️  Market Anomaly: \(event.description)")
                print("     Severity: \(String(format: "%.2f", distance))")
                print("     Affected sectors: \(event.affectedSectors.joined(separator: ", "))")
            }
        }
        
        print("\n" + "=" * 40 + "\n")
    }
    
    // MARK: - Portfolio Analysis
    
    func demonstratePortfolioAnalysis() async throws {
        print("5. Portfolio Similarity & Risk Analysis")
        print("=" * 40)
        
        // Create mock portfolios
        let portfolios = createMockPortfolios()
        
        let config = StoreConfiguration(
            dimensions: 192,  // Rich portfolio features
            similarity: .cosine,
            indexType: .hnsw(
                m: 32,
                efConstruction: 300,
                efSearch: 100
            )
        )
        
        let portfolioStore = try VectorStore(configuration: config)
        
        print("\nIndexing portfolio profiles:")
        
        for portfolio in portfolios {
            let embedding = createPortfolioEmbedding(portfolio)
            
            let riskMetrics = calculateRiskMetrics(portfolio)
            
            let metadata: [String: Any] = [
                "name": portfolio.name,
                "total_value": portfolio.totalValue,
                "sharpe_ratio": riskMetrics.sharpeRatio,
                "beta": riskMetrics.beta,
                "var_95": riskMetrics.valueAtRisk95,
                "max_drawdown": riskMetrics.maxDrawdown,
                "risk_level": portfolio.riskLevel.rawValue
            ]
            
            try await portfolioStore.add(
                id: portfolio.id,
                vector: embedding,
                metadata: metadata
            )
            
            print("  - \(portfolio.name) (Risk: \(portfolio.riskLevel.rawValue))")
        }
        
        // Find similar portfolios
        print("\nFinding portfolios similar to 'Balanced Growth':")
        let targetPortfolio = portfolios.first { $0.name == "Balanced Growth" }!
        let targetEmbedding = createPortfolioEmbedding(targetPortfolio)
        
        let similarPortfolios = try await portfolioStore.search(
            query: targetEmbedding,
            k: 5,
            filters: ["risk_level": ["moderate", "aggressive"]]
        )
        
        for result in similarPortfolios {
            let name = result.metadata["name"] as? String ?? "Unknown"
            let sharpe = result.metadata["sharpe_ratio"] as? Double ?? 0
            
            print("  - \(name)")
            print("    Similarity: \(String(format: "%.3f", result.score))")
            print("    Sharpe Ratio: \(String(format: "%.2f", sharpe))")
        }
        
        // Risk-based portfolio search
        print("\nFinding low-risk portfolios with high Sharpe ratio:")
        let lowRiskHighReturn = createRiskReturnQueryVector(
            maxRisk: 0.15,
            minSharpe: 1.5
        )
        
        let optimalPortfolios = try await portfolioStore.search(
            query: lowRiskHighReturn,
            k: 3,
            filters: ["risk_level": "conservative"]
        )
        
        for result in optimalPortfolios {
            let name = result.metadata["name"] as? String ?? "Unknown"
            let sharpe = result.metadata["sharpe_ratio"] as? Double ?? 0
            let var95 = result.metadata["var_95"] as? Double ?? 0
            
            print("  - \(name)")
            print("    Sharpe: \(String(format: "%.2f", sharpe))")
            print("    VaR (95%): \(String(format: "%.1f%%", var95 * 100))")
        }
        
        print("\n" + "=" * 40 + "\n")
    }
    
    // MARK: - Real-Time Analysis
    
    func demonstrateRealTimeAnalysis() async throws {
        print("6. Real-Time Market Analysis Simulation")
        print("=" * 40)
        
        // Initialize real-time analysis system
        let config = StoreConfiguration(
            dimensions: 128,
            similarity: .cosine,
            indexType: .flat  // Fast for real-time updates
        )
        
        let realtimeStore = try VectorStore(configuration: config)
        
        // Simulate market data stream
        print("\nSimulating real-time market data stream...")
        print("(Processing 10 ticks)\n")
        
        let marketSimulator = MarketDataSimulator()
        
        for tick in 1...10 {
            // Generate market tick
            let marketTick = marketSimulator.generateTick()
            
            print("Tick \(tick) - \(marketTick.timestamp.formatted())")
            
            // Process each instrument update
            for update in marketTick.updates {
                let embedding = createRealTimeEmbedding(update)
                
                // Check for significant events
                let events = await detectMarketEvents(
                    update: update,
                    store: realtimeStore
                )
                
                for event in events {
                    switch event.type {
                    case .priceSpike:
                        print("  📈 PRICE SPIKE: \(update.symbol) " +
                              "(\(String(format: "%+.2f%%", event.magnitude)))")
                    case .volumeSurge:
                        print("  📊 VOLUME SURGE: \(update.symbol) " +
                              "(\(String(format: "%.0fx", event.magnitude)) normal)")
                    case .volatilityAlert:
                        print("  ⚡ VOLATILITY: \(update.symbol) " +
                              "(σ = \(String(format: "%.2f", event.magnitude)))")
                    case .correlationBreak:
                        print("  🔗 CORRELATION BREAK: \(update.symbol)")
                    }
                }
                
                // Store update for pattern detection
                let metadata: [String: Any] = [
                    "symbol": update.symbol,
                    "timestamp": update.timestamp.timeIntervalSince1970,
                    "price": update.price,
                    "volume": update.volume
                ]
                
                try await realtimeStore.add(
                    id: "\(update.symbol)_\(tick)",
                    vector: embedding,
                    metadata: metadata
                )
            }
            
            // Brief pause to simulate real-time
            try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
        }
        
        print("\nReal-time analysis complete.")
        print("\n" + "=" * 40 + "\n")
        
        print("Financial Analysis Example completed successfully!")
    }
}

// MARK: - Helper Functions and Data Generation

extension FinancialAnalysisExample {
    
    // MARK: Mock Data Generators
    
    func createMockFinancialInstruments() -> [FinancialInstrument] {
        let symbols = [
            ("AAPL", "Apple Inc.", "Technology", 3.0e12),
            ("MSFT", "Microsoft Corp.", "Technology", 2.8e12),
            ("JPM", "JPMorgan Chase", "Finance", 4.5e11),
            ("TSLA", "Tesla Inc.", "Automotive", 8.0e11),
            ("AMZN", "Amazon.com", "E-Commerce", 1.7e12)
        ]
        
        return symbols.map { symbol, name, sector, marketCap in
            let timeSeries = generateRealisticTimeSeries(
                days: 252,  // 1 year of trading days
                basePrice: Double.random(in: 100...500),
                volatility: sector == "Technology" ? 0.25 : 0.15
            )
            
            return FinancialInstrument(
                symbol: symbol,
                name: name,
                sector: sector,
                marketCap: marketCap,
                timeSeries: timeSeries
            )
        }
    }
    
    func generateRealisticTimeSeries(days: Int, basePrice: Double, volatility: Double) -> [FinancialDataPoint] {
        var timeSeries: [FinancialDataPoint] = []
        var currentPrice = basePrice
        let calendar = Calendar.current
        var currentDate = calendar.date(byAdding: .day, value: -days, to: Date())!
        
        // Generate technical analysis patterns
        let patternProbability = 0.1
        var inPattern = false
        var patternDaysLeft = 0
        var patternType: TechnicalPattern?
        
        for day in 0..<days {
            // Skip weekends
            let weekday = calendar.component(.weekday, from: currentDate)
            if weekday == 1 || weekday == 7 {
                currentDate = calendar.date(byAdding: .day, value: 1, to: currentDate)!
                continue
            }
            
            // Pattern generation
            if !inPattern && Double.random(in: 0...1) < patternProbability {
                patternType = TechnicalPattern.allCases.randomElement()
                patternDaysLeft = Int.random(in: 10...30)
                inPattern = true
            }
            
            // Calculate daily movement
            var dailyReturn = (Double.random(in: -1...1) * volatility) / sqrt(252.0)
            
            // Apply pattern if active
            if inPattern && patternDaysLeft > 0 {
                dailyReturn += getPatternMovement(
                    pattern: patternType!,
                    dayInPattern: patternDaysLeft
                )
                patternDaysLeft -= 1
                if patternDaysLeft == 0 {
                    inPattern = false
                }
            }
            
            // Update price
            currentPrice *= (1 + dailyReturn)
            
            // Generate OHLC data
            let dailyVolatility = volatility / sqrt(252.0) * 0.5
            let high = currentPrice * (1 + Double.random(in: 0...dailyVolatility))
            let low = currentPrice * (1 - Double.random(in: 0...dailyVolatility))
            let open = currentPrice * (1 + Double.random(in: -dailyVolatility...dailyVolatility))
            
            // Generate volume (higher during patterns)
            let baseVolume = Double.random(in: 1e6...5e6)
            let volume = inPattern ? baseVolume * Double.random(in: 1.5...3.0) : baseVolume
            
            // Calculate technical indicators
            let rsi = calculateRSI(timeSeries: timeSeries, periods: 14)
            let macd = calculateMACD(timeSeries: timeSeries)
            let bollinger = calculateBollingerBands(timeSeries: timeSeries, periods: 20)
            
            let dataPoint = FinancialDataPoint(
                timestamp: currentDate,
                open: open,
                high: high,
                low: low,
                close: currentPrice,
                volume: volume,
                rsi: rsi,
                macd: macd,
                bollingerBand: bollinger
            )
            
            timeSeries.append(dataPoint)
            currentDate = calendar.date(byAdding: .day, value: 1, to: currentDate)!
        }
        
        return timeSeries
    }
    
    func getPatternMovement(pattern: TechnicalPattern, dayInPattern: Int) -> Double {
        switch pattern {
        case .headAndShoulders:
            // Create head and shoulders pattern
            if dayInPattern > 20 {
                return 0.02  // Left shoulder up
            } else if dayInPattern > 15 {
                return -0.01  // Valley
            } else if dayInPattern > 10 {
                return 0.03  // Head up
            } else if dayInPattern > 5 {
                return -0.01  // Valley
            } else {
                return 0.02  // Right shoulder
            }
            
        case .doubleBottom:
            if dayInPattern > 20 {
                return -0.02  // First decline
            } else if dayInPattern > 15 {
                return 0.01  // Small recovery
            } else if dayInPattern > 10 {
                return -0.02  // Second decline
            } else {
                return 0.03  // Breakout
            }
            
        case .ascendingTriangle:
            // Converging pattern with flat top
            let progress = Double(30 - dayInPattern) / 30.0
            return 0.01 * (1 - progress) + Double.random(in: -0.005...0.005)
            
        case .bearishFlag:
            if dayInPattern > 20 {
                return -0.03  // Sharp decline
            } else {
                return 0.005  // Slight upward consolidation
            }
            
        case .bullishPennant:
            if dayInPattern > 20 {
                return 0.03  // Sharp rise
            } else {
                // Converging consolidation
                let volatilityDecay = Double(dayInPattern) / 20.0
                return Double.random(in: -0.01...0.01) * volatilityDecay
            }
            
        case .cupAndHandle:
            if dayInPattern > 25 {
                return -0.02  // Left side of cup
            } else if dayInPattern > 15 {
                return 0.001  // Bottom of cup
            } else if dayInPattern > 5 {
                return 0.02  // Right side of cup
            } else {
                return -0.005  // Handle
            }
        }
    }
    
    // MARK: Embedding Creation Functions
    
    func createSlidingWindowEmbeddings(timeSeries: [FinancialDataPoint], windowSize: Int) -> [[Float]] {
        var embeddings: [[Float]] = []
        
        for i in 0...(timeSeries.count - windowSize) {
            let window = Array(timeSeries[i..<(i + windowSize)])
            var features: [Float] = []
            
            // Price features
            let prices = window.map { $0.close }
            features.append(Float(prices.last! / prices.first! - 1))  // Return
            features.append(Float(prices.max()! / prices.min()! - 1))  // Range
            features.append(Float(standardDeviation(prices)))  // Volatility
            
            // Volume features
            let volumes = window.map { $0.volume }
            features.append(Float(volumes.reduce(0, +) / Double(volumes.count)))  // Avg volume
            features.append(Float(volumes.max()! / volumes.min()!))  // Volume range
            
            // Technical features
            if let rsi = window.last?.rsi {
                features.append(Float(rsi / 100.0))
            }
            
            // Normalize to 128 dimensions with padding
            while features.count < 128 {
                features.append(0)
            }
            
            embeddings.append(features)
        }
        
        return embeddings
    }
    
    func createSpectralEmbedding(timeSeries: [FinancialDataPoint]) -> [Float] {
        let prices = timeSeries.map { $0.close }
        
        // Simple FFT simulation (in real implementation, use Accelerate framework)
        var spectralFeatures: [Float] = []
        
        // Compute frequency components
        for frequency in stride(from: 1.0, to: 65.0, by: 1.0) {
            var real: Double = 0
            var imaginary: Double = 0
            
            for (idx, price) in prices.enumerated() {
                let angle = 2.0 * Double.pi * frequency * Double(idx) / Double(prices.count)
                real += price * cos(angle)
                imaginary += price * sin(angle)
            }
            
            let magnitude = sqrt(real * real + imaginary * imaginary)
            spectralFeatures.append(Float(magnitude / Double(prices.count)))
        }
        
        // Add phase information
        for i in 0..<64 {
            spectralFeatures.append(Float.random(in: -1...1))
        }
        
        return spectralFeatures
    }
    
    func createTechnicalIndicatorEmbedding(timeSeries: [FinancialDataPoint]) -> [Float] {
        var features: [Float] = []
        
        // Moving averages
        let prices = timeSeries.map { $0.close }
        let ma20 = movingAverage(prices, periods: 20)
        let ma50 = movingAverage(prices, periods: 50)
        let ma200 = movingAverage(prices, periods: 200)
        
        features.append(Float(prices.last! / ma20 - 1))
        features.append(Float(prices.last! / ma50 - 1))
        features.append(Float(prices.last! / ma200 - 1))
        
        // RSI distribution
        let rsiValues = timeSeries.compactMap { $0.rsi }
        if !rsiValues.isEmpty {
            features.append(Float(rsiValues.last! / 100.0))
            features.append(Float(rsiValues.min()! / 100.0))
            features.append(Float(rsiValues.max()! / 100.0))
        }
        
        // Bollinger band position
        if let lastBB = timeSeries.last?.bollingerBand,
           let lastPrice = timeSeries.last?.close {
            let bbPosition = (lastPrice - lastBB.lower) / (lastBB.upper - lastBB.lower)
            features.append(Float(bbPosition))
        }
        
        // Volume profile
        let volumes = timeSeries.map { $0.volume }
        let avgVolume = volumes.reduce(0, +) / Double(volumes.count)
        let recentVolume = volumes.suffix(5).reduce(0, +) / 5.0
        features.append(Float(recentVolume / avgVolume))
        
        // Pad to 128 dimensions
        while features.count < 128 {
            features.append(0)
        }
        
        return features
    }
    
    // MARK: Pattern Recognition
    
    func createPatternTemplates() -> [(TechnicalPattern, [Float])] {
        var templates: [(TechnicalPattern, [Float])] = []
        
        for pattern in TechnicalPattern.allCases {
            let template = createPatternTemplate(pattern)
            templates.append((pattern, template))
        }
        
        return templates
    }
    
    func createPatternTemplate(_ pattern: TechnicalPattern) -> [Float] {
        var features: [Float] = []
        
        switch pattern {
        case .headAndShoulders:
            // Characteristic shape: up, down, higher up, down, up
            features = [0.5, 0.3, 0.8, 0.3, 0.5] + Array(repeating: 0.4, count: 59)
            
        case .doubleBottom:
            // Two valleys at similar levels
            features = [0.8, 0.3, 0.5, 0.3, 0.8] + Array(repeating: 0.5, count: 59)
            
        case .ascendingTriangle:
            // Flat top with rising lows
            features = (0..<64).map { i in
                Float(0.8 - 0.3 * (1 - Double(i) / 64.0))
            }
            
        case .bearishFlag:
            // Sharp decline followed by slight upward consolidation
            features = [1.0, 0.3] + (0..<62).map { _ in Float.random(in: 0.3...0.4) }
            
        case .bullishPennant:
            // Sharp rise followed by converging consolidation
            features = [0.3, 1.0] + (0..<62).map { i in
                Float(0.65 + 0.35 * cos(Double(i) / 10.0) * exp(-Double(i) / 30.0))
            }
            
        case .cupAndHandle:
            // U-shape followed by small dip
            features = (0..<64).map { i in
                if i < 50 {
                    // Cup shape
                    let x = Double(i) / 50.0 - 0.5
                    return Float(0.5 + 0.5 * (4 * x * x))
                } else {
                    // Handle
                    return Float(0.9 - 0.1 * sin(Double(i - 50) / 4.0))
                }
            }
        }
        
        return features
    }
    
    func scanForPatterns(timeSeries: [FinancialDataPoint], patternStore: VectorStore) async -> [(pattern: String, dayIndex: Int, confidence: Double)] {
        var detections: [(String, Int, Double)] = []
        
        let windowSize = 30
        for i in 0...(timeSeries.count - windowSize) {
            let window = Array(timeSeries[i..<(i + windowSize)])
            let windowEmbedding = createWindowPatternEmbedding(window)
            
            let results = try? await patternStore.search(
                query: windowEmbedding,
                k: 1
            )
            
            if let topResult = results?.first, topResult.score > 0.8 {
                let patternType = topResult.metadata["pattern_type"] as? String ?? "Unknown"
                detections.append((patternType, i, topResult.score))
            }
        }
        
        return detections
    }
    
    func createWindowPatternEmbedding(_ window: [FinancialDataPoint]) -> [Float] {
        let prices = window.map { $0.close }
        let normalized = normalizeTimeSeries(prices)
        
        // Resample to 64 points
        var resampled: [Float] = []
        let step = Double(normalized.count - 1) / 63.0
        
        for i in 0..<64 {
            let index = Double(i) * step
            let lower = Int(index)
            let upper = min(lower + 1, normalized.count - 1)
            let fraction = index - Double(lower)
            
            let value = normalized[lower] * (1 - fraction) + normalized[upper] * fraction
            resampled.append(Float(value))
        }
        
        return resampled
    }
    
    // MARK: Correlation Analysis
    
    struct CorrelationResult {
        let value: Double
        let rolling: [Double]
        let significance: Double
    }
    
    func computeCorrelation(series1: [FinancialDataPoint], series2: [FinancialDataPoint]) -> CorrelationResult {
        let prices1 = series1.map { $0.close }
        let prices2 = series2.map { $0.close }
        
        // Overall correlation
        let corr = pearsonCorrelation(prices1, prices2)
        
        // Rolling correlation
        let windowSize = 20
        var rolling: [Double] = []
        
        for i in 0...(prices1.count - windowSize) {
            let window1 = Array(prices1[i..<(i + windowSize)])
            let window2 = Array(prices2[i..<(i + windowSize)])
            rolling.append(pearsonCorrelation(window1, window2))
        }
        
        // Significance (simplified)
        let significance = abs(corr) * sqrt(Double(prices1.count)) / 2.0
        
        return CorrelationResult(
            value: corr,
            rolling: rolling,
            significance: significance
        )
    }
    
    func createCorrelationEmbedding(instrument1: FinancialInstrument, instrument2: FinancialInstrument, correlation: CorrelationResult) -> [Float] {
        var features: [Float] = []
        
        // Correlation features
        features.append(Float(correlation.value))
        features.append(Float(correlation.rolling.min() ?? 0))
        features.append(Float(correlation.rolling.max() ?? 0))
        features.append(Float(standardDeviation(correlation.rolling)))
        
        // Sector encoding
        let sectorPairs = [
            (instrument1.sector, instrument2.sector).hashValue,
            instrument1.sector.hashValue,
            instrument2.sector.hashValue
        ]
        
        for hash in sectorPairs {
            features.append(Float(hash % 1000) / 1000.0)
        }
        
        // Market cap ratio
        features.append(Float(log10(instrument1.marketCap / instrument2.marketCap)))
        
        // Price series characteristics
        let prices1 = instrument1.timeSeries.map { $0.close }
        let prices2 = instrument2.timeSeries.map { $0.close }
        
        features.append(Float(standardDeviation(prices1) / standardDeviation(prices2)))
        features.append(Float(skewness(prices1)))
        features.append(Float(skewness(prices2)))
        features.append(Float(kurtosis(prices1)))
        features.append(Float(kurtosis(prices2)))
        
        // Pad to 256 dimensions
        while features.count < 256 {
            features.append(0)
        }
        
        return features
    }
    
    func createCorrelationQueryVector(minCorrelation: Double) -> [Float] {
        var features = [Float](repeating: 0, count: 256)
        features[0] = Float(minCorrelation)
        features[1] = Float(minCorrelation * 0.8)  // Min rolling
        features[2] = Float(minCorrelation * 1.2)  // Max rolling
        features[3] = 0.1  // Low std dev
        return features
    }
    
    // MARK: Anomaly Detection
    
    struct Transaction {
        let id: String
        let timestamp: Date
        let amount: Double
        let type: String
        let merchantCategory: String
        let location: String
        let timeOfDay: Double  // 0-24 hours
        let isAnomalous: Bool
        let anomalyType: String?
    }
    
    func createMockTransactions() -> [Transaction] {
        var transactions: [Transaction] = []
        let calendar = Calendar.current
        
        // Normal transactions
        for i in 0..<1000 {
            let date = calendar.date(byAdding: .hour, value: -i, to: Date())!
            let hour = Double(calendar.component(.hour, from: date))
            
            let transaction = Transaction(
                id: "TXN_\(i)",
                timestamp: date,
                amount: Double.random(in: 10...500),
                type: ["purchase", "withdrawal", "deposit", "transfer"].randomElement()!,
                merchantCategory: ["grocery", "gas", "restaurant", "retail", "online"].randomElement()!,
                location: ["US", "US", "US", "CA", "UK"].randomElement()!,
                timeOfDay: hour,
                isAnomalous: false,
                anomalyType: nil
            )
            
            transactions.append(transaction)
        }
        
        // Anomalous transactions
        let anomalies = [
            Transaction(
                id: "TXN_FRAUD_1",
                timestamp: Date(),
                amount: 5000,  // Unusually large
                type: "purchase",
                merchantCategory: "jewelry",
                location: "RU",  // Unusual location
                timeOfDay: 3,  // Unusual time
                isAnomalous: true,
                anomalyType: "Fraud - Large amount, unusual location"
            ),
            Transaction(
                id: "TXN_FRAUD_2",
                timestamp: Date(),
                amount: 50,
                type: "purchase",
                merchantCategory: "online",
                location: "CN",
                timeOfDay: 4,
                isAnomalous: true,
                anomalyType: "Fraud - Card testing pattern"
            ),
            Transaction(
                id: "TXN_ANOMALY_1",
                timestamp: Date(),
                amount: 10000,
                type: "withdrawal",
                merchantCategory: "atm",
                location: "US",
                timeOfDay: 14,
                isAnomalous: true,
                anomalyType: "Unusual withdrawal amount"
            )
        ]
        
        transactions.append(contentsOf: anomalies)
        return transactions
    }
    
    func createTransactionEmbedding(_ transaction: Transaction) -> [Float] {
        var features: [Float] = []
        
        // Amount features (log scale)
        features.append(Float(log10(transaction.amount + 1)))
        features.append(Float(transaction.amount > 1000 ? 1.0 : 0.0))
        
        // Time features
        features.append(Float(sin(2 * Double.pi * transaction.timeOfDay / 24)))
        features.append(Float(cos(2 * Double.pi * transaction.timeOfDay / 24)))
        
        // Type encoding (one-hot)
        let types = ["purchase", "withdrawal", "deposit", "transfer"]
        for type in types {
            features.append(Float(transaction.type == type ? 1.0 : 0.0))
        }
        
        // Merchant category (one-hot)
        let categories = ["grocery", "gas", "restaurant", "retail", "online", "atm", "jewelry", "other"]
        for category in categories {
            features.append(Float(transaction.merchantCategory == category ? 1.0 : 0.0))
        }
        
        // Location risk score
        let riskLocations = ["RU": 0.9, "CN": 0.8, "NG": 0.7]
        let locationRisk = riskLocations[transaction.location] ?? 0.1
        features.append(Float(locationRisk))
        
        // Behavioral features
        let isNightTime = transaction.timeOfDay < 6 || transaction.timeOfDay > 22
        features.append(Float(isNightTime ? 1.0 : 0.0))
        
        // Pad to 96 dimensions
        while features.count < 96 {
            features.append(0)
        }
        
        return features
    }
    
    func calculateAnomalyScore(distances: [Float]) -> Double {
        // Use average distance to nearest neighbors
        let avgDistance = distances.reduce(0, +) / Float(distances.count)
        
        // Normalize to 0-1 range (assuming distances are typically 0-2)
        return Double(min(avgDistance / 2.0, 1.0))
    }
    
    struct MarketEvent {
        let timestamp: Date
        let description: String
        let affectedSectors: [String]
        let severity: Double
    }
    
    func createMarketAnomalyEvents() -> [MarketEvent] {
        return [
            MarketEvent(
                timestamp: Date(),
                description: "Flash Crash - Technology Sector",
                affectedSectors: ["Technology", "Communications"],
                severity: 0.9
            ),
            MarketEvent(
                timestamp: Date(),
                description: "Circuit Breaker Triggered",
                affectedSectors: ["All"],
                severity: 1.0
            ),
            MarketEvent(
                timestamp: Date(),
                description: "Unusual Options Activity - Finance",
                affectedSectors: ["Finance"],
                severity: 0.6
            )
        ]
    }
    
    func createMarketEventEmbedding(_ event: MarketEvent) -> [Float] {
        var features: [Float] = []
        
        // Severity
        features.append(Float(event.severity))
        
        // Sector impact (one-hot for major sectors)
        let allSectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrial", "Communications", "All"]
        for sector in allSectors {
            features.append(Float(event.affectedSectors.contains(sector) ? 1.0 : 0.0))
        }
        
        // Time features
        let hour = Calendar.current.component(.hour, from: event.timestamp)
        features.append(Float(sin(2 * Double.pi * Double(hour) / 24)))
        features.append(Float(cos(2 * Double.pi * Double(hour) / 24)))
        
        // Market hours indicator
        let isMarketHours = hour >= 9 && hour < 16
        features.append(Float(isMarketHours ? 1.0 : 0.0))
        
        // Pad to match dimension
        while features.count < 128 {
            features.append(0)
        }
        
        return features
    }
    
    func createNormalMarketEmbedding() -> [Float] {
        var features = [Float](repeating: 0, count: 128)
        features[0] = 0.1  // Low severity
        // Normal market hours
        features[10] = Float(sin(2 * Double.pi * 12 / 24))
        features[11] = Float(cos(2 * Double.pi * 12 / 24))
        features[12] = 1.0  // Market hours
        return features
    }
    
    // MARK: Portfolio Analysis
    
    struct Portfolio {
        let id: String
        let name: String
        let holdings: [(symbol: String, weight: Double)]
        let totalValue: Double
        let riskLevel: RiskLevel
        let strategy: String
    }
    
    enum RiskLevel: String {
        case conservative = "conservative"
        case moderate = "moderate"
        case aggressive = "aggressive"
    }
    
    struct RiskMetrics {
        let sharpeRatio: Double
        let beta: Double
        let valueAtRisk95: Double
        let maxDrawdown: Double
        let volatility: Double
    }
    
    func createMockPortfolios() -> [Portfolio] {
        return [
            Portfolio(
                id: "PORT_1",
                name: "Conservative Income",
                holdings: [
                    ("BND", 0.6),   // Bonds
                    ("VIG", 0.3),   // Dividend stocks
                    ("GLD", 0.1)    // Gold
                ],
                totalValue: 1_000_000,
                riskLevel: .conservative,
                strategy: "Income generation with capital preservation"
            ),
            Portfolio(
                id: "PORT_2",
                name: "Balanced Growth",
                holdings: [
                    ("VTI", 0.4),   // Total market
                    ("VXUS", 0.2),  // International
                    ("BND", 0.3),   // Bonds
                    ("VNQ", 0.1)    // REITs
                ],
                totalValue: 500_000,
                riskLevel: .moderate,
                strategy: "Long-term growth with moderate risk"
            ),
            Portfolio(
                id: "PORT_3",
                name: "Tech Growth",
                holdings: [
                    ("QQQ", 0.5),   // Tech heavy
                    ("ARKK", 0.3),  // Innovation
                    ("ICLN", 0.2)   // Clean energy
                ],
                totalValue: 250_000,
                riskLevel: .aggressive,
                strategy: "High growth technology focus"
            ),
            Portfolio(
                id: "PORT_4",
                name: "Value Focused",
                holdings: [
                    ("VTV", 0.4),   // Value stocks
                    ("SCHD", 0.3),  // Dividend value
                    ("VYM", 0.2),   // High yield
                    ("BRK.B", 0.1)  // Berkshire
                ],
                totalValue: 750_000,
                riskLevel: .moderate,
                strategy: "Value investing with dividend focus"
            ),
            Portfolio(
                id: "PORT_5",
                name: "Risk Parity",
                holdings: [
                    ("SPY", 0.25),  // Stocks
                    ("TLT", 0.25),  // Long bonds
                    ("GLD", 0.25),  // Gold
                    ("DBC", 0.25)   // Commodities
                ],
                totalValue: 2_000_000,
                riskLevel: .moderate,
                strategy: "Equal risk contribution across asset classes"
            )
        ]
    }
    
    func createPortfolioEmbedding(_ portfolio: Portfolio) -> [Float] {
        var features: [Float] = []
        
        // Asset allocation features
        let equityWeight = portfolio.holdings.filter { ["VTI", "VTV", "QQQ", "SPY", "ARKK", "VXUS", "SCHD", "VYM", "VIG", "BRK.B"].contains($0.symbol) }
            .reduce(0) { $0 + $1.weight }
        let bondWeight = portfolio.holdings.filter { ["BND", "TLT"].contains($0.symbol) }
            .reduce(0) { $0 + $1.weight }
        let alternativeWeight = portfolio.holdings.filter { ["GLD", "VNQ", "DBC", "ICLN"].contains($0.symbol) }
            .reduce(0) { $0 + $1.weight }
        
        features.append(Float(equityWeight))
        features.append(Float(bondWeight))
        features.append(Float(alternativeWeight))
        
        // Concentration metrics
        let herfindahlIndex = portfolio.holdings.reduce(0) { $0 + pow($1.weight, 2) }
        features.append(Float(herfindahlIndex))
        
        // Risk level encoding
        switch portfolio.riskLevel {
        case .conservative:
            features.append(contentsOf: [1.0, 0.0, 0.0])
        case .moderate:
            features.append(contentsOf: [0.0, 1.0, 0.0])
        case .aggressive:
            features.append(contentsOf: [0.0, 0.0, 1.0])
        }
        
        // Sector exposures (simplified)
        let techExposure = portfolio.holdings.filter { ["QQQ", "ARKK"].contains($0.symbol) }
            .reduce(0) { $0 + $1.weight }
        let valueExposure = portfolio.holdings.filter { ["VTV", "SCHD", "VYM", "BRK.B"].contains($0.symbol) }
            .reduce(0) { $0 + $1.weight }
        
        features.append(Float(techExposure))
        features.append(Float(valueExposure))
        
        // Portfolio size (log scale)
        features.append(Float(log10(portfolio.totalValue)))
        
        // Strategy encoding (simplified hash)
        let strategyHash = portfolio.strategy.hashValue
        features.append(Float(strategyHash % 1000) / 1000.0)
        
        // Simulated factor exposures
        features.append(Float.random(in: -1...1))  // Value factor
        features.append(Float.random(in: -1...1))  // Momentum factor
        features.append(Float.random(in: -1...1))  // Quality factor
        features.append(Float.random(in: -1...1))  // Low volatility factor
        
        // Pad to 192 dimensions
        while features.count < 192 {
            features.append(0)
        }
        
        return features
    }
    
    func calculateRiskMetrics(_ portfolio: Portfolio) -> RiskMetrics {
        // Simulate risk metrics based on holdings
        let baseVolatility: Double
        let baseSharpe: Double
        
        switch portfolio.riskLevel {
        case .conservative:
            baseVolatility = 0.08
            baseSharpe = 0.8
        case .moderate:
            baseVolatility = 0.15
            baseSharpe = 1.2
        case .aggressive:
            baseVolatility = 0.25
            baseSharpe = 1.5
        }
        
        // Add some randomness for realism
        let volatility = baseVolatility * Double.random(in: 0.8...1.2)
        let sharpeRatio = baseSharpe * Double.random(in: 0.9...1.1)
        let beta = volatility / 0.15  // Relative to market volatility
        let valueAtRisk95 = -1.645 * volatility  // 95% VaR
        let maxDrawdown = -2.5 * volatility  // Rough approximation
        
        return RiskMetrics(
            sharpeRatio: sharpeRatio,
            beta: beta,
            valueAtRisk95: valueAtRisk95,
            maxDrawdown: maxDrawdown,
            volatility: volatility
        )
    }
    
    func createRiskReturnQueryVector(maxRisk: Double, minSharpe: Double) -> [Float] {
        var features = [Float](repeating: 0, count: 192)
        
        // Target conservative allocation
        features[0] = 0.3  // Low equity
        features[1] = 0.6  // High bonds
        features[2] = 0.1  // Some alternatives
        
        // Low concentration
        features[3] = 0.2  // Low Herfindahl
        
        // Conservative risk level
        features[4] = 1.0
        features[5] = 0.0
        features[6] = 0.0
        
        // Low tech exposure
        features[7] = 0.1
        
        // High value exposure
        features[8] = 0.4
        
        return features
    }
    
    // MARK: Real-Time Analysis
    
    struct MarketTick {
        let timestamp: Date
        let updates: [InstrumentUpdate]
    }
    
    struct InstrumentUpdate {
        let symbol: String
        let timestamp: Date
        let price: Double
        let volume: Double
        let bid: Double
        let ask: Double
        let volatility: Double
    }
    
    enum MarketEventType {
        case priceSpike
        case volumeSurge
        case volatilityAlert
        case correlationBreak
    }
    
    struct DetectedEvent {
        let type: MarketEventType
        let magnitude: Double
        let timestamp: Date
    }
    
    class MarketDataSimulator {
        private var basePrices: [String: Double] = [
            "AAPL": 180.0,
            "MSFT": 420.0,
            "GOOGL": 150.0,
            "AMZN": 180.0,
            "TSLA": 250.0
        ]
        
        private var baseVolumes: [String: Double] = [
            "AAPL": 50_000_000,
            "MSFT": 30_000_000,
            "GOOGL": 20_000_000,
            "AMZN": 40_000_000,
            "TSLA": 100_000_000
        ]
        
        func generateTick() -> MarketTick {
            var updates: [InstrumentUpdate] = []
            
            for (symbol, basePrice) in basePrices {
                // Simulate price movement
                let priceChange = Double.random(in: -0.02...0.02)
                let newPrice = basePrice * (1 + priceChange)
                basePrices[symbol] = newPrice
                
                // Simulate volume
                let volumeMultiplier = Double.random(in: 0.5...2.0)
                let volume = baseVolumes[symbol]! * volumeMultiplier
                
                // Bid-ask spread
                let spread = newPrice * 0.001
                let bid = newPrice - spread / 2
                let ask = newPrice + spread / 2
                
                // Volatility
                let volatility = abs(priceChange) * sqrt(252.0)
                
                let update = InstrumentUpdate(
                    symbol: symbol,
                    timestamp: Date(),
                    price: newPrice,
                    volume: volume,
                    bid: bid,
                    ask: ask,
                    volatility: volatility
                )
                
                updates.append(update)
            }
            
            return MarketTick(timestamp: Date(), updates: updates)
        }
    }
    
    func createRealTimeEmbedding(_ update: InstrumentUpdate) -> [Float] {
        var features: [Float] = []
        
        // Price features
        features.append(Float(log10(update.price)))
        features.append(Float((update.ask - update.bid) / update.price))  // Spread
        
        // Volume features
        features.append(Float(log10(update.volume)))
        
        // Volatility
        features.append(Float(update.volatility))
        
        // Time features
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: update.timestamp)
        let minute = calendar.component(.minute, from: update.timestamp)
        let timeOfDay = Double(hour) + Double(minute) / 60.0
        
        features.append(Float(sin(2 * Double.pi * timeOfDay / 24)))
        features.append(Float(cos(2 * Double.pi * timeOfDay / 24)))
        
        // Market session indicators
        let isPreMarket = hour < 9 || (hour == 9 && minute < 30)
        let isMarketHours = hour >= 9 && hour < 16 && !isPreMarket
        let isAfterHours = hour >= 16
        
        features.append(Float(isPreMarket ? 1.0 : 0.0))
        features.append(Float(isMarketHours ? 1.0 : 0.0))
        features.append(Float(isAfterHours ? 1.0 : 0.0))
        
        // Symbol encoding (simplified)
        let symbolHash = update.symbol.hashValue
        features.append(Float(symbolHash % 1000) / 1000.0)
        
        // Pad to 128 dimensions
        while features.count < 128 {
            features.append(0)
        }
        
        return features
    }
    
    func detectMarketEvents(update: InstrumentUpdate, store: VectorStore) async -> [DetectedEvent] {
        var events: [DetectedEvent] = []
        
        // Price spike detection (> 1% in a tick)
        let priceChangeThreshold = 0.01
        let embedding = createRealTimeEmbedding(update)
        
        // Get recent updates for the same symbol
        let recentUpdates = try? await store.search(
            query: embedding,
            k: 10,
            filters: ["symbol": update.symbol]
        )
        
        if let recent = recentUpdates, !recent.isEmpty {
            let recentPrices = recent.compactMap { $0.metadata["price"] as? Double }
            if let avgPrice = recentPrices.isEmpty ? nil : recentPrices.reduce(0, +) / Double(recentPrices.count) {
                let priceChange = abs(update.price - avgPrice) / avgPrice
                
                if priceChange > priceChangeThreshold {
                    events.append(DetectedEvent(
                        type: .priceSpike,
                        magnitude: priceChange * 100,  // Percentage
                        timestamp: update.timestamp
                    ))
                }
            }
            
            // Volume surge detection
            let recentVolumes = recent.compactMap { $0.metadata["volume"] as? Double }
            if let avgVolume = recentVolumes.isEmpty ? nil : recentVolumes.reduce(0, +) / Double(recentVolumes.count) {
                let volumeRatio = update.volume / avgVolume
                
                if volumeRatio > 3.0 {
                    events.append(DetectedEvent(
                        type: .volumeSurge,
                        magnitude: volumeRatio,
                        timestamp: update.timestamp
                    ))
                }
            }
        }
        
        // Volatility alert
        if update.volatility > 0.4 {  // 40% annualized
            events.append(DetectedEvent(
                type: .volatilityAlert,
                magnitude: update.volatility,
                timestamp: update.timestamp
            ))
        }
        
        return events
    }
    
    // MARK: Statistical Helper Functions
    
    func standardDeviation(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(values.count)
        return sqrt(variance)
    }
    
    func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count > 1 else { return 0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = x.reduce(0) { $0 + $1 * $1 }
        let sumY2 = y.reduce(0) { $0 + $1 * $1 }
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        return denominator == 0 ? 0 : numerator / denominator
    }
    
    func movingAverage(_ values: [Double], periods: Int) -> Double {
        guard values.count >= periods else { return values.reduce(0, +) / Double(values.count) }
        let recent = values.suffix(periods)
        return recent.reduce(0, +) / Double(periods)
    }
    
    func euclideanDistance(_ v1: [Float], _ v2: [Float]) -> Float {
        guard v1.count == v2.count else { return Float.infinity }
        return sqrt(zip(v1, v2).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
    }
    
    func normalizeTimeSeries(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        let min = values.min()!
        let max = values.max()!
        let range = max - min
        
        guard range > 0 else { return Array(repeating: 0.5, count: values.count) }
        
        return values.map { ($0 - min) / range }
    }
    
    func skewness(_ values: [Double]) -> Double {
        let n = Double(values.count)
        let mean = values.reduce(0, +) / n
        let std = standardDeviation(values)
        
        guard std > 0 else { return 0 }
        
        let sum = values.reduce(0) { $0 + pow(($1 - mean) / std, 3) }
        return sum / n
    }
    
    func kurtosis(_ values: [Double]) -> Double {
        let n = Double(values.count)
        let mean = values.reduce(0, +) / n
        let std = standardDeviation(values)
        
        guard std > 0 else { return 0 }
        
        let sum = values.reduce(0) { $0 + pow(($1 - mean) / std, 4) }
        return sum / n - 3  // Excess kurtosis
    }
    
    // Technical indicator calculations
    func calculateRSI(timeSeries: [FinancialDataPoint], periods: Int) -> Double? {
        guard timeSeries.count >= periods else { return nil }
        
        var gains: [Double] = []
        var losses: [Double] = []
        
        for i in 1..<timeSeries.count {
            let change = timeSeries[i].close - timeSeries[i-1].close
            if change > 0 {
                gains.append(change)
                losses.append(0)
            } else {
                gains.append(0)
                losses.append(-change)
            }
        }
        
        let avgGain = gains.suffix(periods).reduce(0, +) / Double(periods)
        let avgLoss = losses.suffix(periods).reduce(0, +) / Double(periods)
        
        guard avgLoss > 0 else { return 100 }
        
        let rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))
    }
    
    func calculateMACD(timeSeries: [FinancialDataPoint]) -> Double? {
        guard timeSeries.count >= 26 else { return nil }
        
        let prices = timeSeries.map { $0.close }
        let ema12 = exponentialMovingAverage(prices, periods: 12)
        let ema26 = exponentialMovingAverage(prices, periods: 26)
        
        return ema12 - ema26
    }
    
    func exponentialMovingAverage(_ values: [Double], periods: Int) -> Double {
        guard !values.isEmpty else { return 0 }
        guard values.count >= periods else { return values.reduce(0, +) / Double(values.count) }
        
        let alpha = 2.0 / (Double(periods) + 1.0)
        var ema = values[0]
        
        for i in 1..<values.count {
            ema = alpha * values[i] + (1 - alpha) * ema
        }
        
        return ema
    }
    
    func calculateBollingerBands(timeSeries: [FinancialDataPoint], periods: Int) -> (upper: Double, middle: Double, lower: Double)? {
        guard timeSeries.count >= periods else { return nil }
        
        let prices = timeSeries.suffix(periods).map { $0.close }
        let sma = prices.reduce(0, +) / Double(periods)
        let std = standardDeviation(prices)
        
        return (
            upper: sma + 2 * std,
            middle: sma,
            lower: sma - 2 * std
        )
    }
    
    func getPatternReliability(_ pattern: TechnicalPattern) -> Double {
        switch pattern {
        case .headAndShoulders: return 0.85
        case .doubleBottom: return 0.80
        case .ascendingTriangle: return 0.75
        case .bearishFlag: return 0.70
        case .bullishPennant: return 0.72
        case .cupAndHandle: return 0.78
        }
    }
}

// String multiplication operator for separator lines
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}