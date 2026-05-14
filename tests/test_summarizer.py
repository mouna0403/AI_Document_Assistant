"""
Simple tests for summarizer.py
"""

import os

import pytest

from AI_Document_Assitant.utils.summarizer import summarize_text


class TestSummarizeText:
    """Basic tests for summarize_text function"""

    def test_returns_string(self):
        """Verify the function returns a string"""
        text = "This is a test text for summarization."

        # If no API key, we skip the test
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("No GROQ_API_KEY set; skipping test")

        result = summarize_text(text)
        assert isinstance(result, str)
        assert len(result) > 0  # Summary should not be empty

    def test_empty_text(self):
        """Verify that empty input returns an empty string or summary"""
        text = ""

        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("No GROQ_API_KEY set; skipping test")

        result = summarize_text(text)
        assert isinstance(result, str)
