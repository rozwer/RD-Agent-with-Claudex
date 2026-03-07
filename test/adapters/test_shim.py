"""Tests for ClaudeCodeAPIBackend compatibility shim."""
import pytest

from rdagent.oai.backend.claude_code import ClaudeCodeAPIBackend


@pytest.fixture
def backend():
    return ClaudeCodeAPIBackend(
        use_chat_cache=False,
        dump_chat_cache=False,
        use_embedding_cache=False,
        dump_embedding_cache=False,
    )


class TestShimBasics:
    def test_import_succeeds(self):
        from rdagent.oai.backend.claude_code import ClaudeCodeAPIBackend

        assert ClaudeCodeAPIBackend is not None

    def test_isinstance_api_backend(self, backend):
        from rdagent.oai.backend.base import APIBackend

        assert isinstance(backend, APIBackend)

    def test_supports_response_schema_false(self, backend):
        assert backend.supports_response_schema() is False


class TestTokenCount:
    def test_returns_positive_int(self, backend):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]
        count = backend._calculate_token_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0


class TestEmbeddingFailFast:
    def test_embedding_raises_runtime_error(self, backend):
        with pytest.raises(RuntimeError, match="Embedding backend not configured"):
            backend._create_embedding_inner_function(["test input"])


class TestChatCompletion:
    def test_chat_returns_tuple(self, backend):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"},
        ]
        try:
            content, finish_reason = backend._create_chat_completion_inner_function(messages)
            assert isinstance(content, str)
            assert len(content) > 0
            assert finish_reason == "stop"
        except RuntimeError as e:
            if "not found" in str(e) or "Nested sessions" in str(e):
                pytest.skip("Claude CLI not available or nested session")
            raise
