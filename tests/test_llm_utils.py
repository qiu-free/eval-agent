"""LLM 工具函数单元测试"""

import pytest
from unittest.mock import MagicMock, patch
from core.llm_utils import safe_llm_call, get_llm_client


class TestSafeLlmCall:
    def test_success_first_try(self):
        fn = MagicMock(return_value="ok")
        result = safe_llm_call(fn, label="test")
        assert result == "ok"
        assert fn.call_count == 1

    def test_retry_on_failure(self):
        fn = MagicMock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), "ok"])
        result = safe_llm_call(fn, max_retries=3, label="test")
        assert result == "ok"
        assert fn.call_count == 3

    def test_fallback_on_all_failures(self):
        fn = MagicMock(side_effect=ConnectionError("fail"))
        result = safe_llm_call(fn, max_retries=1, fallback="default", label="test")
        assert result == "default"

    def test_raises_without_fallback(self):
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            safe_llm_call(fn, max_retries=0, label="test")

    def test_fallback_type_preserved(self):
        fn = MagicMock(side_effect=ConnectionError("fail"))
        result = safe_llm_call(fn, max_retries=0, fallback=42, label="test")
        assert result == 42
        assert isinstance(result, int)

    def test_args_passed_through(self):
        fn = MagicMock(return_value="ok")
        safe_llm_call(fn, "a", "b", label="test", x=1)
        fn.assert_called_once_with("a", "b", x=1)


class TestGetLlmClient:
    @patch("config.settings")
    @patch("core.llm_utils._shared_clients", {})
    def test_creates_client_with_default_provider(self, mock_settings):
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_api_base = "https://api.test.com"
        client = get_llm_client()
        assert client is not None

    @patch("config.settings")
    @patch("core.llm_utils._shared_clients", {})
    def test_creates_client_with_dashscope(self, mock_settings):
        mock_settings.llm_provider = "dashscope"
        mock_settings.openai_api_key = "sk-test"
        client = get_llm_client("dashscope")
        assert client is not None

    @patch("config.settings")
    @patch("core.llm_utils._shared_clients", {})
    def test_singleton_behavior(self, mock_settings):
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_api_base = "https://api.test.com"
        client1 = get_llm_client("openai")
        client2 = get_llm_client("openai")
        assert client1 is client2
