# Market Analysis

RAGVenture provides powerful market analysis capabilities using real-world data from authoritative sources.

## Command Line Interface

The market analysis feature is integrated into the main CLI command:

```bash
# Generate ideas with market analysis (default)
python -m src.rag_startups.cli generate-all "AI healthcare" --num-ideas 2

# Skip market analysis
python -m src.rag_startups.cli generate-all "fintech" --no-market

# Show relevant examples with market analysis
python -m src.rag_startups.cli generate-all "edtech" --print-examples
```

Example output:
```
╭─────────────────────────── Generated Startup Idea ───────────────────────────╮
│ Name: HealthAI Diagnostics                                                   │
│                                                                             │
│ Problem: Long wait times and high costs in medical diagnostics              │
│ Solution: AI-powered diagnostic platform for rapid disease screening        │
│ Target Market: Healthcare providers and diagnostic labs                     │
│ Unique Value: 90% faster diagnosis with 95% accuracy                       │
╰─────────────────────────────────────────────────────────────────────────────╯

                Market Analysis
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric            ┃ Value                 ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Market Size       │ $84.2B (2023)         │
│ Growth Rate       │ 23.7% CAGR            │
│ Competition       │ Medium                │
│ Opportunity Score │ 8.5/10                │
│ Risk Factors      │ • Regulatory approval │
│                   │ • Data privacy        │
└───────────────────┴───────────────────────┘
```

## Market Analysis Components

### 1. Market Size Estimation
```python
from rag_startups.analysis.market_size import MarketSizeEstimator

estimator = MarketSizeEstimator()
market_size = estimator.estimate(
    "AI healthcare diagnostics",
    region="global",
    year=2023
)

print(f"TAM: ${market_size.tam}B")
print(f"SAM: ${market_size.sam}B")
print(f"SOM: ${market_size.som}B")
```

### 2. Growth Analysis
```python
from rag_startups.analysis.growth import GrowthAnalyzer

analyzer = GrowthAnalyzer()
growth = analyzer.analyze(
    sector="healthcare",
    subsector="diagnostics",
    technology="AI"
)

print(f"CAGR: {growth.cagr}%")
print(f"YoY Growth: {growth.yoy}%")
print(f"Market Stage: {growth.stage}")
```

### 3. Competition Analysis
```python
from rag_startups.analysis.competition import CompetitionAnalyzer

analyzer = CompetitionAnalyzer()
competition = analyzer.analyze(
    idea="AI diagnostics platform",
    region="global"
)

print(f"Competition Level: {competition.level}")
print(f"Key Players: {len(competition.key_players)}")
print(f"Entry Barriers: {competition.barriers}")
```

### 4. Risk Assessment
```python
from rag_startups.analysis.risk import RiskAnalyzer

analyzer = RiskAnalyzer()
risks = analyzer.assess(
    idea="AI diagnostics platform",
    sector="healthcare"
)

for risk in risks:
    print(f"Risk: {risk.name}")
    print(f"Impact: {risk.impact}/10")
    print(f"Mitigation: {risk.mitigation}")
```

### 5. Opportunity Scoring
```python
from rag_startups.analysis.opportunity import OpportunityScorer

scorer = OpportunityScorer()
score = scorer.calculate(
    idea="AI diagnostics platform",
    market_size=market_size,
    growth=growth,
    competition=competition,
    risks=risks
)

print(f"Overall Score: {score.overall}/10")
print(f"Market Potential: {score.market_potential}/10")
print(f"Technical Feasibility: {score.technical_feasibility}/10")
print(f"Risk-Adjusted Return: {score.risk_adjusted_return}/10")
```

## Data Sources

The market analysis uses data from multiple authoritative sources:

1. Industry Reports
   - Gartner
   - IDC
   - Forrester
   - CB Insights

2. Government Data
   - Bureau of Labor Statistics
   - World Bank
   - USPTO (Patent Data)

3. Market Research
   - Crunchbase
   - PitchBook
   - S&P Global

4. Academic Sources
   - Research papers
   - Industry journals
   - Economic databases

## Customization

You can customize the market analysis through environment variables:

```bash
# Optional: Set preferred data sources
MARKET_DATA_SOURCES="gartner,idc,crunchbase"

# Optional: Set analysis region
MARKET_ANALYSIS_REGION="north_america"

# Optional: Set confidence threshold
MARKET_CONFIDENCE_THRESHOLD="0.8"
```

For more configuration options, see the [Configuration Guide](configuration.md).
