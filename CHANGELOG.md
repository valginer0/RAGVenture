# Changelog

All notable changes to RAGVenture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2023-12-30

### Added
- Market analysis feature with comprehensive components:
  - Market Size Estimation
  - Growth Analysis
  - Competition Analysis
  - Risk Assessment
  - Opportunity Scoring
- Environment variables for market analysis customization:
  - `MARKET_DATA_SOURCES`
  - `MARKET_ANALYSIS_REGION`
  - `MARKET_CONFIDENCE_THRESHOLD`
- New documentation:
  - Comprehensive market analysis guide
  - Updated configuration documentation
  - Enhanced examples with new features

### Changed
- CLI interface improvements:
  - Renamed `generate` command to `generate-all` for clarity
  - Updated command parameters from `--num` to `--num-ideas`
  - Added market analysis flags and options
- Documentation structure:
  - Reorganized for better navigation
  - Added detailed configuration options
  - Updated all CLI examples to use new command structure

### Fixed
- Null value handling in data loader
- Mock startup file format in tests
- Documentation inconsistencies in command examples
