"""Token / cost usage tracking model.

Shared across both extraction paths (LLM and ACU) so the API can report an
apples-to-apples cost story for the demo.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Aggregated usage for one extraction run.

    For the LLM path these are real token counts. For the ACU path we record
    the number of analyzer calls (batches) and, where the service reports it,
    page/unit counts — kept in the same shape so downstream cost math is
    uniform.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # ACU-specific / batching metadata (0 for a single LLM call).
    api_calls: int = 0
    batch_count: int = 0
    pages_processed: int = 0

    # Wall-clock for the run, filled in by the caller.
    elapsed_seconds: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        """Sum two usage records (used when merging chunk/batch results)."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            api_calls=self.api_calls + other.api_calls,
            batch_count=self.batch_count + other.batch_count,
            pages_processed=self.pages_processed + other.pages_processed,
            elapsed_seconds=max(self.elapsed_seconds, other.elapsed_seconds),
        )

    @classmethod
    def from_tokens(cls, input_tokens: int, output_tokens: int) -> "Usage":
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            api_calls=1,
        )
