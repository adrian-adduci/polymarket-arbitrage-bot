"""
LLM Event Analysis Signal - Use Claude to estimate market probabilities.

This signal source uses Claude to analyze prediction market questions
and estimate the probability of the YES outcome. It then compares
this estimate to the current market price to detect mispricings.

When the LLM estimate significantly diverges from market price:
- LLM higher than market -> BUY_YES (market is undervaluing YES)
- LLM lower than market -> BUY_NO (market is overvaluing YES)

Example:
    from lib.signals import LLMEventSignal

    signal = LLMEventSignal(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        min_divergence=0.05,  # 5% minimum divergence
        cache_ttl=300,        # 5 minute cache
    )

    async with signal:
        result = await signal.get_signal(market)
        if result:
            print(f"LLM says {result.metadata['llm_probability']:.0%}")
            print(f"Market says {result.metadata['market_probability']:.0%}")
            print(f"Divergence: {result.strength:.0%} -> {result.direction.value}")

Note:
    Requires the 'anthropic' package: pip install anthropic
    Set ANTHROPIC_API_KEY environment variable or pass api_key
"""

import json
import logging
import os
import time
from typing import Optional, Any, Dict, Tuple

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

from lib.signals.base import SignalSource, TradingSignal, SignalDirection

logger = logging.getLogger(__name__)


class LLMEventSignal(SignalSource):
    """
    Use Claude to analyze market questions and estimate probabilities.

    Compares LLM probability estimate vs market price to find mispricings.
    Uses caching to minimize API calls (signals don't change rapidly).

    Attributes:
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Claude model to use (default: claude-sonnet-4-20250514)
        min_divergence: Minimum divergence to generate signal (default 0.05)
        cache_ttl: Cache duration in seconds (default 300)
        max_tokens: Max tokens for API response (default 500)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        min_divergence: float = 0.05,
        cache_ttl: float = 300.0,
        max_tokens: int = 500,
    ):
        """
        Initialize LLM event signal.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
            min_divergence: Minimum probability divergence for signal (default 5%)
            cache_ttl: Cache TTL in seconds (default 5 minutes)
            max_tokens: Max response tokens
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required for LLMEventSignal. "
                "Install with: pip install anthropic"
            )

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.min_divergence = min_divergence
        self.cache_ttl = cache_ttl
        self.max_tokens = max_tokens

        self.client: Optional[anthropic.Anthropic] = None
        self._cache: Dict[str, Tuple[float, TradingSignal]] = {}  # slug -> (timestamp, signal)
        self._analysis_cache: Dict[str, Tuple[float, dict]] = {}  # question -> (timestamp, analysis)

    @property
    def name(self) -> str:
        """Unique identifier for this signal source."""
        return "llm_event_analysis"

    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key to constructor."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(
            f"LLMEventSignal initialized: model={self.model}, "
            f"min_divergence={self.min_divergence:.0%}"
        )

    async def shutdown(self) -> None:
        """Clear cache and client."""
        self._cache.clear()
        self._analysis_cache.clear()
        self.client = None

    def _get_cached_signal(self, market_slug: str) -> Optional[TradingSignal]:
        """Get cached signal if not expired."""
        if market_slug in self._cache:
            ts, signal = self._cache[market_slug]
            if time.time() - ts < self.cache_ttl:
                return signal
        return None

    def _get_cached_analysis(self, question: str) -> Optional[dict]:
        """Get cached analysis if not expired."""
        if question in self._analysis_cache:
            ts, analysis = self._analysis_cache[question]
            if time.time() - ts < self.cache_ttl:
                return analysis
        return None

    def _analyze_market_sync(self, question: str) -> Optional[dict]:
        """
        Call Claude API for market analysis (synchronous).

        Returns:
            Dict with probability, confidence, reasoning or None on error
        """
        if not self.client:
            return None

        # Check analysis cache first
        cached = self._get_cached_analysis(question)
        if cached:
            return cached

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this prediction market question and estimate the probability of YES outcome.

Question: "{question}"

Today's date is {time.strftime('%Y-%m-%d')}.

Consider:
1. Current events and context
2. Historical precedents
3. Base rates for similar events
4. Key uncertainties

Return ONLY valid JSON in this exact format (no other text):
{{
    "probability": <0-100>,
    "confidence": <0-100>,
    "reasoning": "<brief 1-2 sentence explanation>"
}}"""
                }],
            )

            response_text = message.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                # Extract JSON from code block
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.startswith("```")):
                        json_lines.append(line)
                response_text = "\n".join(json_lines).strip()

            analysis = json.loads(response_text)

            # Validate response
            if not all(k in analysis for k in ["probability", "confidence", "reasoning"]):
                logger.warning(f"Invalid LLM response format: {analysis}")
                return None

            # Cache the analysis
            self._analysis_cache[question] = (time.time(), analysis)

            return analysis

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    async def get_signal(
        self,
        market: Any,
        orderbook: Optional[Any] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate signal by comparing LLM estimate to market price.

        Args:
            market: Market object with 'slug', 'question', and price info
            orderbook: Not used for LLM analysis

        Returns:
            TradingSignal if divergence exceeds threshold, None otherwise
        """
        market_slug = getattr(market, "slug", str(market))

        # Check signal cache first
        cached_signal = self._get_cached_signal(market_slug)
        if cached_signal:
            return cached_signal

        # Get market question
        question = getattr(market, "question", None)
        if not question:
            return None

        # Get current market price (YES probability)
        # Try different attribute names
        market_prob = None
        for attr in ["yes_price", "yes_ask", "outcomePrices"]:
            value = getattr(market, attr, None)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    market_prob = float(value[0]) if value else None
                else:
                    market_prob = float(value)
                break

        if market_prob is None:
            logger.debug(f"Could not get market price for {market_slug}")
            return None

        # Get LLM analysis
        analysis = self._analyze_market_sync(question)
        if not analysis:
            return None

        # Calculate divergence
        llm_prob = analysis["probability"] / 100.0
        divergence = llm_prob - market_prob

        # Check if divergence is significant
        if abs(divergence) < self.min_divergence:
            logger.debug(
                f"LLM divergence too small for {market_slug}: "
                f"{divergence:.1%} < {self.min_divergence:.1%}"
            )
            return None

        # Determine direction
        if divergence > 0:
            # LLM thinks YES is more likely than market
            direction = SignalDirection.BUY_YES
        else:
            # LLM thinks NO is more likely than market
            direction = SignalDirection.BUY_NO

        # Create signal
        signal = TradingSignal(
            source=self.name,
            market_slug=market_slug,
            direction=direction,
            strength=min(1.0, abs(divergence) * 2),  # Scale divergence to strength
            confidence=analysis["confidence"] / 100.0,
            timestamp=time.time(),
            metadata={
                "llm_probability": llm_prob,
                "market_probability": market_prob,
                "divergence": divergence,
                "reasoning": analysis["reasoning"],
                "model": self.model,
            },
            expires_at=time.time() + self.cache_ttl,
        )

        # Cache the signal
        self._cache[market_slug] = (time.time(), signal)

        logger.info(
            f"LLM signal for {market_slug}: {direction.value} "
            f"(LLM: {llm_prob:.0%}, Market: {market_prob:.0%}, "
            f"Divergence: {divergence:+.1%})"
        )

        return signal

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._analysis_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "signal_cache_size": len(self._cache),
            "analysis_cache_size": len(self._analysis_cache),
        }
