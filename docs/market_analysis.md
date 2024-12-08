# Market Analysis

RAGVenture provides powerful market analysis capabilities using real-world data from authoritative sources.

## Market Size Estimation

Get comprehensive market size estimates for your startup idea:

```python
from rag_startups.analysis.market_size import MarketSizeEstimator
from rag_startups.analysis.external_data import get_industry_analysis

# Initialize with your startup data
estimator = MarketSizeEstimator(startup_data)

# Get market size estimation
market_size = estimator.estimate_market_size(
    "AI-powered healthcare diagnostics platform",
    similar_startups  # From your RAG search
)

print(f"Total Addressable Market: ${market_size.total_addressable_market}B")
print(f"Serviceable Addressable Market: ${market_size.serviceable_addressable_market}B")
print(f"Serviceable Obtainable Market: ${market_size.serviceable_obtainable_market}B")
print(f"Market Stage: {market_size.stage.value}")
print(f"Confidence Score: {market_size.confidence_score}")
```

## Data Sources

### World Bank Data
Access global economic indicators:

```python
from rag_startups.analysis.external_data import WorldBankData

wb = WorldBankData()
metrics = wb.get_industry_metrics(country="USA", year=2023)

print(f"GDP: ${metrics['gdp']:,.2f}")
print(f"Industry Percentage: {metrics['industry_percentage']}%")
print(f"Growth Rate: {metrics['growth_rate']}%")
```

### Bureau of Labor Statistics
Get detailed employment data:

```python
from rag_startups.analysis.external_data import BLSData

bls = BLSData()
employment = bls.get_employment_data("5112")  # Software Publishers

print(f"Employment: {employment['employment']:,}")
print(f"Year: {employment['year']}")
```

## Caching

Results are automatically cached to improve performance:

```python
from rag_startups.utils.caching import get_cache_stats, clear_cache

# Get cache statistics
stats = get_cache_stats()
print(f"Cache Type: {stats['type']}")
print(f"Number of Keys: {stats['keys']}")

# Clear specific cache entries
clear_cache(prefix="worldbank")

# Clear all cache
clear_cache()
```

## Best Practices

1. **Use Multiple Data Sources**
   - Combine World Bank and BLS data for comprehensive analysis
   - Consider both global and local market indicators

2. **Handle Data Freshness**
   - Check the `year` field in results
   - Use `clear_cache()` to force fresh data when needed

3. **Consider Confidence Scores**
   - Higher scores indicate more reliable estimates
   - Scores below 0.5 suggest limited data availability

4. **Market Segmentation**
   - Use appropriate industry codes for BLS data
   - Consider both B2B and B2C aspects of your market

## Error Handling

The market analysis tools include robust error handling:

```python
try:
    metrics = get_industry_analysis("5112")
except Exception as e:
    print(f"Error getting industry analysis: {e}")
    # Fall back to startup-based estimation only
```

## Configuration

Set up your environment variables:

```bash
# .env file
BLS_API_KEY=your_key_here
REDIS_HOST=localhost  # Optional: for Redis caching
REDIS_PORT=6379      # Optional: for Redis caching
```

## Extending the Analysis

You can extend the market analysis with custom data sources:

```python
from rag_startups.analysis.external_data import IndustryMetrics

def get_custom_metrics(industry_code: str) -> IndustryMetrics:
    # Add your custom data source integration
    return IndustryMetrics(
        gdp_contribution=your_calculation,
        employment=your_employment_data,
        growth_rate=your_growth_rate,
        market_size=your_market_size,
        confidence_score=your_confidence,
        year=current_year,
        sources=["Your Custom Source"]
    )
```
