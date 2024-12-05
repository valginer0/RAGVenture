"""
Startup idea generator using HuggingFace's Inference API with rate limiting.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import backoff  # We'll need to add this to requirements.txt
from huggingface_hub import InferenceClient

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

        # Initialize client
        self.client = InferenceClient(token=self.token)

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
    ) -> Optional[str]:
        """
        Generate startup ideas with rate limiting and retries.

        Args:
            num_ideas: Number of ideas to generate
            example_startups: Optional list of example startups from database
            temperature: Model temperature (0.0-1.0)

        Returns:
            Generated and cleaned ideas text, or None if generation failed

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

            # Clean and return response
            return clean_response(response)

        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                raise RateLimitError(str(e))
            raise

    async def generate_async(
        self,
        num_ideas: int = 3,
        example_startups: Optional[List[Dict]] = None,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Async version of generate method.
        TODO: Implement async version when needed
        """
        raise NotImplementedError("Async generation not yet implemented")
