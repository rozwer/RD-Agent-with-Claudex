from __future__ import annotations

from pathlib import Path
from typing import Literal

from rdagent.core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    # backend
    backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"

    chat_model: str = "anthropic/claude-sonnet-4-20250514"
    embedding_model: str = "voyage/voyage-3"

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    enable_response_schema: bool = True
    # Whether to enable response_schema in chat models. may not work for models that do not support it.

    # Handling format
    reasoning_think_rm: bool = False
    """
    Some LLMs include <think>...</think> tags in their responses, which can interfere with the main output.
    Set reasoning_think_rm to True to remove any <think>...</think> content from responses.
    """

    log_llm_chat_content: bool = True

    max_retry: int = 10
    retry_wait_seconds: int = 1
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10
    timeout_fail_limit: int = 10
    violation_fail_limit: int = 1

    # Behavior of returning answers to the same question when caching is enabled
    use_auto_chat_cache_seed_gen: bool = False
    """
    `_create_chat_completion_inner_function` provides a feature to pass in a seed to affect the cache hash key
    We want to enable a auto seed generator to get different default seed for `_create_chat_completion_inner_function`
    if seed is not given.
    So the cache will only not miss you ask the same question on same round.
    """
    init_chat_cache_seed: int = 42

    # Chat configs
    anthropic_api_key: str = ""
    chat_max_tokens: int | None = None
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = 200000  # Claude model context window
    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."
    system_prompt_role: str = "system"
    """Some models (like o1) do not support the 'system' role.
    Therefore, we make the system_prompt_role customizable to ensure successful calls."""

    # Embedding configs
    voyage_api_key: str = ""
    embedding_max_str_num: int = 50
    embedding_max_length: int = 8192

    chat_model_map: dict[str, dict[str, str]] = {}


LLM_SETTINGS = LLMSettings()
