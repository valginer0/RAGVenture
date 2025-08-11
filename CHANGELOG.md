# Changelog

All notable changes to RAGVenture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- N/A

### Changed
- N/A

### Fixed
- N/A

---

## [0.9.2] - 2025-08-10

Docs-only release. Published to PyPI.

### Added
- `docs/examples.md`: Added “CLI Auth & Preflight” and “Offline Testing Notes”.
- `README.md`: Documented offline testing policy, token preflight, and updated test counts (178).
- `CONTRIBUTING.md`: Clarified offline testing workflow and sanitized push command.

### Changed
- Minor documentation refinements and consistency across CLI examples and policies.

### Fixed
- None (no code changes).

---

## [0.9.1] - 2025-08-10

Test stabilization and CI reliability improvements. Note: 0.9.1 may not have been published to PyPI; superseded by 0.9.2.

### Added
- Global autouse test fixtures in `tests/conftest.py`:
  - Block outbound HTTP(S) by default; opt-out via `@pytest.mark.allow_network`.
  - Mock `huggingface_hub.InferenceClient` to avoid network calls in CLI/model tests.
  - Mock `transformers`/embeddings paths and `rag_startups.embed_master.calculate_result` for deterministic outputs.
- Scripts for timing and profiling (e.g., `parse_top50.py`); artifacts ignored via `.gitignore`.

### Changed
- CLI preflight check after model selection using `huggingface_hub.model_info` with clear gated-access messages.
- Model selection: retry anonymously on 401 to allow public models without a token.
- Embedding and language model wiring to ensure chosen models are actually used; removed mock fallback.
- Various test speedups via monkeypatching (Redis, backoff, sleep).

### Fixed
- CI failures due to unauthorized HF calls by enforcing offline mode env vars and comprehensive mocks.
- Import-time validation issues by making settings retrieval lazy and fixing absolute imports.
- Migration test flakiness with improved mocking.
- Minor cleanup in `generator.py` (no behavior change).

---

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
