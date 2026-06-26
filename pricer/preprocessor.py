import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-4o-mini"   # swap to "gpt-4o" for higher quality

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    """
    Single-item preprocessor: calls OpenAI to convert a raw scraped
    product blob into a clean, structured summary.

    Use this for one-off or small-scale summarisation.
    For bulk processing (thousands of items), use Batch instead —
    it costs 50% less via the OpenAI Batch API.

    Tracks token usage and estimated cost across all calls.
    """

    # gpt-4o-mini pricing (per 1M tokens, as of mid-2025)
    INPUT_COST_PER_M = 0.15
    OUTPUT_COST_PER_M = 0.60

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        """
        Summarise a raw product description into structured text.

        Args:
            text: Item.full — the cleaned raw product blob.

        Returns:
            Structured summary (Title / Category / Brand / Description / Details).
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages_for(text),
            max_tokens=200,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    @property
    def total_cost(self) -> float:
        """Estimated USD cost based on gpt-4o-mini list pricing."""
        return (
            self.total_input_tokens / 1_000_000 * self.INPUT_COST_PER_M
            + self.total_output_tokens / 1_000_000 * self.OUTPUT_COST_PER_M
        )

    def stats(self) -> dict:
        """Return accumulated token usage and estimated cost."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 6),
        }
