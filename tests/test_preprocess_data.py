import pytest
import re
from src.preprocess_data import remove_special_tokens


class TestRemoveSpecialTokens:
    def test_remove_url_simple(self):
        text = "Check this <URL> out"
        result = remove_special_tokens(text)
        assert "<URL>" not in result

    def test_remove_url_multiple(self):
        text = "Visit <URL> and <URL> now"
        result = remove_special_tokens(text)
        assert "<URL>" not in result
        assert result.count("<URL>") == 0

    def test_remove_mention(self):
        text = "Hey @user check this out"
        result = remove_special_tokens(text)
        assert "@user" not in result

    def test_remove_newline(self):
        text = "First line\nSecond line"
        result = remove_special_tokens(text)
        assert "\n" not in result
        assert " " in result

    def test_combined(self):
        text = "Hey @user check <URL>\nMore text"
        result = remove_special_tokens(text)
        assert "<URL>" not in result
        assert "@user" not in result
        assert "\n" not in result

    def test_escaped_vs_unescaped_url(self):
        text = "Check <URL> out"

        escaped = re.sub(r"\<URL\>", "", text)
        unescaped = re.sub(r"<URL>", "", text)

        assert "<URL>" not in escaped
        assert "<URL>" not in unescaped

    def test_debug_pattern_matching(self):
        text = "Check <URL> out"
        pattern = r"\<URL\>"
        result = re.sub(pattern, "REMOVED", text)
        print(f"\nPattern: {repr(pattern)}")
        print(f"Text: {repr(text)}")
        print(f"Result: {repr(result)}")
        assert "REMOVED" in result, f"Pattern {repr(pattern)} did not match!"
