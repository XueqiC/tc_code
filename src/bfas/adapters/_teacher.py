"""Per-episode accounting for the shared black-box teacher client."""

from collections.abc import Mapping
from typing import Any

from ..ledger import estimate_response_tokens


class TeacherSession:
    def __init__(self, config: Any):
        self.config = config
        self.response_texts: list[str] = []
        self.tokens_spent = 0
        self.usage: dict[str, int] = {}

    def generate_reply(self, messages: list[dict[str, str]], temperature: float) -> str:
        import appworld_teacher

        reported = False

        def charge(usage: Mapping[str, Any]) -> None:
            nonlocal reported
            # Preserve reported totals, including hidden reasoning. Never add
            # reasoning-token details again or substitute total_tokens.
            for field in ("completion_tokens", "prompt_tokens"):
                count = usage.get(field)
                if type(count) is int and count >= 0:
                    self.usage[field] = self.usage.get(field, 0) + count
            details = usage.get("prompt_tokens_details")
            if isinstance(details, Mapping):
                cached = details.get("cached_tokens")
                if type(cached) is int and cached >= 0:
                    # Cached tokens are a subset of prompt_tokens, not output.
                    self.usage["cached_tokens"] = self.usage.get("cached_tokens", 0) + cached
            count = usage.get("completion_tokens", usage.get("output_tokens"))
            if type(count) is int and count >= 0:
                self.tokens_spent += count
                reported = True

        reply = appworld_teacher.generate_reply(
            self.config, messages, temperature=temperature, usage_callback=charge
        )
        self.response_texts.append(reply)
        if not reported:
            self.tokens_spent += estimate_response_tokens([reply])
        return reply
