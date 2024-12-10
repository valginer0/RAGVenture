"""
Startup idea generator using HuggingFace's Inference API with rate limiting.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import backoff  # We'll need to add this to requirements.txt
from huggingface_hub import InferenceClient

from ..analysis.market_analyzer import MarketAnalyzer, MarketInsights
from ..utils.caching import ttl_cache
from .processors import clean_response, parse_ideas
from .prompts import generate_base_prompt


class RateLimitError(Exception):
    """Custom exception for rate limit handling"""

    pass


class StartupIdeaGenerator:
    """
    Generator class that handles API calls with rate limiting.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_requests_per_hour: int = 120,  # Conservative default
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_requests_per_hour = max_requests_per_hour
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("HuggingFace token not provided")

        # Rate limiting state
        self.request_timestamps: List[datetime] = []

        # Initialize clients
        self.client = InferenceClient(token=self.token)
        self.market_analyzer = MarketAnalyzer()

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            bool: True if we can proceed, False if we're at limit
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Clean old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > hour_ago
        ]

        # Check if we're at limit
        return len(self.request_timestamps) < self.max_requests_per_hour

    def _update_rate_limit(self):
        """Record a new request timestamp"""
        self.request_timestamps.append(datetime.now())

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max wait
    )
    def generate(
        self,
        num_ideas: int = 3,
        example_startups: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        include_market_analysis: bool = True,
    ) -> Tuple[Optional[str], Optional[Dict[str, MarketInsights]]]:
        """
        Generate startup ideas with rate limiting and retries.

        Args:
            num_ideas: Number of ideas to generate
            example_startups: Optional list of example startups from database
            temperature: Model temperature (0.0-1.0)
            include_market_analysis: Whether to include market analysis in results

        Returns:
            Tuple of (generated ideas text, market insights dict) or (None, None) if failed

        Raises:
            RateLimitError: If rate limit is hit and retries exhausted
            ValueError: If parameters are invalid
        """
        if not self._check_rate_limit():
            raise RateLimitError("Rate limit exceeded, try again later")

        if num_ideas < 1 or num_ideas > 5:
            raise ValueError("num_ideas must be between 1 and 5")

        try:
            # Generate prompt
            prompt = generate_base_prompt(num_ideas, example_startups)

            # Record start time for logging
            start_time = time.time()

            # Make API call
            response = self.client.text_generation(
                prompt=prompt,
                model=self.model_name,
                max_new_tokens=1000,
                temperature=temperature,
                repetition_penalty=1.2,
                return_full_text=True,
            )

            # Record request
            self._update_rate_limit()

            # Process response
            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.2f} seconds")

            # Clean response
            cleaned_response = clean_response(response)

            # Parse ideas and analyze markets if requested
            market_insights = None
            if cleaned_response and include_market_analysis:
                ideas = parse_ideas(cleaned_response)
                market_insights = {}
                for idea in ideas:
                    try:
                        insights = self._analyze_market(idea)
                        if insights:
                            market_insights[idea["name"]] = insights
                    except Exception as e:
                        print(f"Failed to analyze market for {idea['name']}: {e}")
                        continue

            return cleaned_response, market_insights

        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                raise RateLimitError(str(e))
            raise

    @ttl_cache(ttl=3600)  # Cache market analysis results for 1 hour
    def _analyze_market(self, idea: Dict) -> Optional[MarketInsights]:
        """Analyze market potential for a startup idea."""
        try:
            return self.market_analyzer.analyze_startup_idea(idea)
        except Exception as e:
            print(f"Market analysis failed: {e}")
            return None

    async def generate_async(
        self,
        num_ideas: int = 3,
        example_startups: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        include_market_analysis: bool = True,
    ) -> Tuple[Optional[str], Optional[Dict[str, MarketInsights]]]:
        """
        Async version of generate method.
        TODO: Implement async version when needed
        """
        raise NotImplementedError("Async generation not yet implemented")
