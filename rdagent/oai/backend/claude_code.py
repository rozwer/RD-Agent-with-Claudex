"""ClaudeCode compatibility shim — drop-in APIBackend for Claude Code transport."""
from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS

logger = logging.getLogger(__name__)


class ClaudeCodeAPIBackend(APIBackend):
    """
    Compatibility shim that routes chat completions through Claude Code CLI,
    uses LiteLLM for local token counting, and fails fast on embedding requests.
    """

    def supports_response_schema(self) -> bool:
        return False

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        try:
            from litellm import token_counter

            return token_counter(model=LLM_SETTINGS.chat_model, messages=messages)
        except ImportError:
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4

    def _create_embedding_inner_function(
        self, input_content_list: list[str]
    ) -> list[list[float]]:
        raise RuntimeError(
            "Embedding backend not configured. "
            "Set embedding backend or disable knowledge (with_knowledge=False)."
        )

    def _create_chat_completion_inner_function(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        if response_format is not None:
            logger.warning(
                "response_format=%r is not supported by ClaudeCodeAPIBackend; ignoring.",
                response_format,
            )

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]\n{content}")
        combined_prompt = "\n\n".join(prompt_parts)

        try:
            result = subprocess.run(
                ["claude", "--print", "-p", combined_prompt],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude Code CLI failed (exit {result.returncode}): {result.stderr[:500]}"
                )
            content = result.stdout.strip()
            if not content:
                raise RuntimeError("Claude Code CLI returned empty response.")
            return content, "stop"
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI ('claude') not found. "
                "Install it or use a different backend."
            )
