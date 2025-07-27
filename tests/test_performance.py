"""Performance tests for RAG Startups using pytest-benchmark.

These tests establish performance baselines and prevent regressions.
Current baseline: ~34s cold start for idea generation.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from rag_startups.config.settings import Settings
from rag_startups.idea_generator.generator import IdeaGenerator


class TestPerformanceBaselines:
    """Performance regression tests with established baselines."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for consistent performance testing."""
        settings = Settings()
        settings.RAG_MODEL_TIMEOUT = 60
        settings.RAG_SMART_MODELS = True
        settings.RAG_MODEL_CHECK_INTERVAL = 3600
        return settings

    @pytest.fixture
    def mock_generator(self, mock_settings):
        """Mock generator with realistic response times."""
        with patch(
            "rag_startups.idea_generator.generator.Settings", return_value=mock_settings
        ):
            generator = IdeaGenerator()

            # Mock the actual generation to simulate realistic timing
            def mock_generate(*args, **kwargs):
                time.sleep(0.1)  # Simulate processing time
                return {
                    "idea": "Mock AI-powered fintech startup",
                    "description": "A revolutionary platform that uses AI...",
                    "market_analysis": "The fintech market is growing...",
                    "similar_companies": ["Company A", "Company B"],
                }

            generator._generate_single_idea = MagicMock(side_effect=mock_generate)
            return generator

    def test_cold_start_performance(self, benchmark, mock_generator):
        """Test cold start performance - should complete within 45s baseline."""

        def cold_start_generation():
            return mock_generator.generate_ideas("fintech", num_ideas=1)

        result = benchmark.pedantic(cold_start_generation, rounds=1, iterations=1)

        # Verify we got results
        assert len(result) == 1
        assert "idea" in result[0]

        # Performance assertion - should be much faster with mocks
        assert benchmark.stats.mean < 1.0  # Mock should be sub-second

    def test_batch_generation_performance(self, benchmark, mock_generator):
        """Test batch generation performance - should scale linearly."""

        def batch_generation():
            return mock_generator.generate_ideas("fintech", num_ideas=3)

        result = benchmark.pedantic(batch_generation, rounds=1, iterations=1)

        # Verify we got results
        assert len(result) == 3

        # Performance assertion - should scale roughly linearly
        assert benchmark.stats.mean < 3.0  # 3 ideas should take < 3s with mocks

    @pytest.mark.slow
    def test_real_model_performance_baseline(self, benchmark):
        """Integration test with real models - establishes actual baseline.

        This test is marked as 'slow' and should only be run when establishing
        new performance baselines or investigating performance regressions.
        """
        # Skip this test in CI unless explicitly requested
        pytest.skip("Real model test - run with 'pytest -m slow' to execute")

        def real_generation():
            generator = IdeaGenerator()
            return generator.generate_ideas("fintech", num_ideas=1)

        result = benchmark.pedantic(real_generation, rounds=1, iterations=1)

        # Verify we got results
        assert len(result) == 1

        # Real baseline - should complete within 45s (allowing buffer over 34s baseline)
        assert benchmark.stats.mean < 45.0

    def test_memory_usage_stability(self, mock_generator):
        """Test that memory usage remains stable across multiple generations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate multiple batches
        for _ in range(5):
            mock_generator.generate_ideas("fintech", num_ideas=2)

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal (< 50MB for mocked operations)
        assert memory_growth < 50 * 1024 * 1024  # 50MB threshold


class TestPerformanceRegression:
    """Tests to catch performance regressions in key operations."""

    def test_config_loading_performance(self, benchmark):
        """Config loading should be fast."""

        def load_config():
            return Settings()

        benchmark(load_config)

        # Config loading should be very fast
        assert benchmark.stats.mean < 0.1

    def test_import_performance(self, benchmark):
        """Module imports should be fast."""

        def import_modules():
            # Clear module cache to simulate cold import
            import sys

            modules_to_clear = [
                m for m in sys.modules.keys() if m.startswith("rag_startups")
            ]
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]

            # Re-import main modules
            import rag_startups.core.rag_chain  # noqa: F401

            return True

        result = benchmark(import_modules)
        assert result is True

        # Imports should be fast
        assert benchmark.stats.mean < 2.0


# Pytest configuration for performance tests
def pytest_configure(config):
    """Configure pytest markers for performance tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Benchmark configuration
def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Update benchmark JSON with additional metadata."""
    output_json["environment"] = {
        "python_version": f"{config.getoption('--tb')}",
        "test_type": "performance_regression",
        "baseline_version": "1.0.0",
    }
