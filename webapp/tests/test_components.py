"""Tests for UI components."""

from unittest.mock import patch


class TestModelSelector:
    """Tests for model_selector component."""

    def test_available_models_defined(self) -> None:
        """Test that AVAILABLE_MODELS is defined with expected models."""
        from components.model_selector import AVAILABLE_MODELS

        assert isinstance(AVAILABLE_MODELS, list)
        assert "gpt-5" in AVAILABLE_MODELS
        assert "gpt-4-1" in AVAILABLE_MODELS
        assert len(AVAILABLE_MODELS) >= 5

    def test_default_model_in_list(self) -> None:
        """Test that default model gpt-5 is in the available models."""
        from components.model_selector import DEFAULT_MODEL, AVAILABLE_MODELS

        assert DEFAULT_MODEL in AVAILABLE_MODELS
        assert DEFAULT_MODEL == "gpt-5"


class TestSimilarityDisplay:
    """Tests for similarity display components."""

    def test_truncate_text_short(self) -> None:
        """Test that short text is not truncated."""
        from components.similarity_display import _truncate_text

        text = "Short text"
        result = _truncate_text(text, 50)
        assert result == text

    def test_truncate_text_long(self) -> None:
        """Test that long text is truncated with ellipsis."""
        from components.similarity_display import _truncate_text

        text = "This is a very long text that should be truncated"
        result = _truncate_text(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_text_exact_length(self) -> None:
        """Test that text at exact length is not truncated."""
        from components.similarity_display import _truncate_text

        text = "Exact len"
        result = _truncate_text(text, 9)
        assert result == text

    def test_concordance_color_positive(self) -> None:
        """Test concordance color for positive values."""
        from components.similarity_display import _concordance_color

        assert _concordance_color(0.9) == "green"
        assert _concordance_color(0.5) == "green"
        assert _concordance_color(0.3) == "orange"
        assert _concordance_color(0.0) == "orange"

    def test_concordance_color_negative(self) -> None:
        """Test concordance color for negative values."""
        from components.similarity_display import _concordance_color

        assert _concordance_color(-0.1) == "red"
        assert _concordance_color(-0.9) == "red"

    def test_format_match_type_exact(self) -> None:
        """Test match type formatting for exact match."""
        from components.similarity_display import _format_match_type

        result = _format_match_type(exact=True, fuzzy=False, efo=False)
        assert result == "Exact"

    def test_format_match_type_fuzzy(self) -> None:
        """Test match type formatting for fuzzy match."""
        from components.similarity_display import _format_match_type

        result = _format_match_type(exact=False, fuzzy=True, efo=False)
        assert result == "Fuzzy"

    def test_format_match_type_efo(self) -> None:
        """Test match type formatting for EFO match."""
        from components.similarity_display import _format_match_type

        result = _format_match_type(exact=False, fuzzy=False, efo=True)
        assert result == "EFO"

    def test_format_match_type_multiple(self) -> None:
        """Test match type formatting for multiple matches."""
        from components.similarity_display import _format_match_type

        result = _format_match_type(exact=True, fuzzy=True, efo=False)
        assert result == "Exact, Fuzzy"

        result = _format_match_type(exact=True, fuzzy=True, efo=True)
        assert result == "Exact, Fuzzy, EFO"

    def test_format_match_type_none(self) -> None:
        """Test match type formatting when no match type is set."""
        from components.similarity_display import _format_match_type

        result = _format_match_type(exact=False, fuzzy=False, efo=False)
        assert result == "N/A"

    def test_trait_similarity_table_empty(self) -> None:
        """Test trait similarity table with empty list."""
        with patch("components.similarity_display.st") as mock_st:
            from components.similarity_display import trait_similarity_table

            result = trait_similarity_table([])

            assert result is None
            mock_st.info.assert_called_once()

    def test_evidence_similarity_table_empty(self) -> None:
        """Test evidence similarity table with empty list."""
        with patch("components.similarity_display.st") as mock_st:
            from components.similarity_display import evidence_similarity_table

            result = evidence_similarity_table([])

            assert result is None
            mock_st.info.assert_called_once()


class TestStudyTable:
    """Tests for study table component."""

    def test_study_table_empty(self) -> None:
        """Test study table with empty list shows info message."""
        with patch("components.study_table.st") as mock_st:
            from components.study_table import study_table

            result = study_table([])

            assert result is None
            mock_st.info.assert_called_once()


class TestAutocompleteResult:
    """Tests for AutocompleteResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        from services.api_client import AutocompleteResult

        items = ["item1", "item2"]
        result = AutocompleteResult.success_result(items)

        assert result.success is True
        assert result.items == items
        assert result.error_message is None

    def test_error_result(self) -> None:
        """Test creating an error result."""
        from services.api_client import AutocompleteResult

        error_msg = "Connection failed"
        result = AutocompleteResult.error_result(error_msg)

        assert result.success is False
        assert result.items == []
        assert result.error_message == error_msg

    def test_success_result_empty(self) -> None:
        """Test creating a successful result with empty items."""
        from services.api_client import AutocompleteResult

        result = AutocompleteResult.success_result([])

        assert result.success is True
        assert result.items == []
        assert result.error_message is None
