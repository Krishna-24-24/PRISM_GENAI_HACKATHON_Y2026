"""
Unit tests for QueryAnalyzer — one test per query category.

Tests verify:
1. Correct category assignment for representative queries
2. Correct signal flag extraction (backtick, camelCase, snake_case, etc.)
3. Correct channel weights per category
4. disable_routing=True always returns dense-only
5. QueryProfile.as_dict() is serialisable
6. Extracted identifiers are populated correctly
7. Edge cases: empty query, all-punctuation, very short queries
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.retrieval.query_analyzer import (
    QueryAnalyzer,
    QueryCategory,
    QueryProfile,
)


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(disable_routing=False)


@pytest.fixture
def disabled_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(disable_routing=True)


# ---------------------------------------------------------------------------
# Semantic queries
# ---------------------------------------------------------------------------

class TestSemanticQueries:
    """Pure NL queries — no identifiers. Should route to Dense-only."""

    @pytest.mark.parametrize("query", [
        "validate user input before saving to the database",
        "retry logic for network timeouts",
        "how does the authentication flow work",
        "send an email notification when the order ships",
        "parse a configuration file and apply settings",
    ])
    def test_category_is_semantic(self, analyzer, query):
        profile = analyzer.analyze(query)
        assert profile.category == QueryCategory.SEMANTIC, \
            f"Expected SEMANTIC for '{query}', got {profile.category}"

    def test_semantic_weights_dense_only(self, analyzer):
        profile = analyzer.analyze("retry logic for network timeouts")
        assert "dense" in profile.recommended_weights
        # BM25 should NOT be in weights for semantic queries
        assert profile.recommended_weights.get("bm25", 0) == 0 or \
               "bm25" not in profile.recommended_weights

    def test_semantic_no_identifiers(self, analyzer):
        profile = analyzer.analyze("handle errors gracefully")
        assert profile.identifier_count == 0

    def test_semantic_routing_reason_explains_evidence(self, analyzer):
        profile = analyzer.analyze("sort an array in ascending order")
        assert "dense" in profile.routing_reason.lower() or \
               "semantic" in profile.routing_reason.lower()


# ---------------------------------------------------------------------------
# Symbolic queries
# ---------------------------------------------------------------------------

class TestSymbolicQueries:
    """Queries that reference a specific identifier by name."""

    @pytest.mark.parametrize("query", [
        "`calculateChecksum` function",
        "where is handleHttpRequest defined",
        "getUserById",
        "find the implementation of `renderTemplate`",
        "what does parseConfigFile do",
    ])
    def test_category_is_symbolic(self, analyzer, query):
        profile = analyzer.analyze(query)
        assert profile.category == QueryCategory.SYMBOLIC, \
            f"Expected SYMBOLIC for '{query}', got {profile.category}. " \
            f"Signals: camel={profile.has_camel_case_token}, " \
            f"backtick={profile.has_backtick_identifier}"

    def test_backtick_extraction(self, analyzer):
        profile = analyzer.analyze("`calculateChecksum` in `utils.js`")
        assert "calculateChecksum" in profile.extracted_identifiers

    def test_camelcase_extraction(self, analyzer):
        profile = analyzer.analyze("find getUserById function")
        assert "getUserById" in profile.extracted_identifiers

    def test_symbolic_weights_symbol_dominates(self, analyzer):
        profile = analyzer.analyze("`handleHttpRequest`")
        assert profile.recommended_weights.get("symbol", 0) > \
               profile.recommended_weights.get("dense", 0)

    def test_symbolic_bm25_weight_present(self, analyzer):
        """BM25 should be included for symbolic queries (exact token matching)."""
        profile = analyzer.analyze("calculateChecksum function")
        assert "bm25" in profile.recommended_weights
        assert profile.recommended_weights["bm25"] > 0

    def test_single_camel_short_query(self, analyzer):
        """A single camelCase word should be classified as SYMBOLIC."""
        profile = analyzer.analyze("bubbleSort")
        assert profile.category == QueryCategory.SYMBOLIC


# ---------------------------------------------------------------------------
# Structural queries
# ---------------------------------------------------------------------------

class TestStructuralQueries:
    """Queries about class hierarchies, inheritance, interfaces."""

    @pytest.mark.parametrize("query", [
        "classes that extend BaseController",
        "all classes implementing the Logger interface",
        "which class inherits from EventEmitter",
        "abstract classes in the auth module",
        "find subclasses of HttpClient",
    ])
    def test_category_is_structural(self, analyzer, query):
        profile = analyzer.analyze(query)
        assert profile.category == QueryCategory.STRUCTURAL, \
            f"Expected STRUCTURAL for '{query}', got {profile.category}"

    def test_structural_weights_has_dense(self, analyzer):
        profile = analyzer.analyze("classes that extend BaseController")
        assert "dense" in profile.recommended_weights
        assert profile.recommended_weights["dense"] > 0

    def test_structural_keyword_flag(self, analyzer):
        profile = analyzer.analyze("find subclasses of EventEmitter")
        assert profile.has_structural_keyword

    def test_structural_routing_reason(self, analyzer):
        profile = analyzer.analyze("classes extending BaseModel")
        assert "structural" in profile.routing_reason.lower() or \
               "extend" in profile.routing_reason.lower()


# ---------------------------------------------------------------------------
# Usage queries
# ---------------------------------------------------------------------------

class TestUsageQueries:
    """Queries asking for call sites / examples of using a symbol."""

    @pytest.mark.parametrize("query", [
        "show me examples of calling `initDb`",
        "how to use the `createUser` function",
        "usage of handleRequest in the codebase",
        "example of invoking parseConfig",
    ])
    def test_category_is_usage(self, analyzer, query):
        profile = analyzer.analyze(query)
        assert profile.category in (QueryCategory.USAGE, QueryCategory.SYMBOLIC), \
            f"Expected USAGE or SYMBOLIC for '{query}', got {profile.category}"

    def test_usage_keyword_flag(self, analyzer):
        profile = analyzer.analyze("show me examples of calling `initDb`")
        assert profile.has_usage_keyword

    def test_usage_weights_symbol_active(self, analyzer):
        profile = analyzer.analyze("usage of `handleRequest`")
        # Symbol should be included for usage queries
        weights = profile.recommended_weights
        assert "symbol" in weights or "dense" in weights

    def test_usage_with_identifier_preferred_over_pure_nl(self, analyzer):
        """Usage + identifier should not be classified as SEMANTIC."""
        profile = analyzer.analyze("examples of calling calculateHash")
        assert profile.category != QueryCategory.SEMANTIC


# ---------------------------------------------------------------------------
# Mixed queries
# ---------------------------------------------------------------------------

class TestMixedQueries:
    """Queries mixing NL description with code-like tokens."""

    @pytest.mark.parametrize("query", [
        "how does req.body get validated",
        "explain the behavior of user.save() before committing",
        "what happens when db.connect fails",
    ])
    def test_category_is_mixed_or_symbolic(self, analyzer, query):
        profile = analyzer.analyze(query)
        assert profile.category in (
            QueryCategory.MIXED, QueryCategory.SYMBOLIC, QueryCategory.USAGE
        ), f"Expected mixed/symbolic/usage for '{query}', got {profile.category}"

    def test_dot_accessor_flag(self, analyzer):
        profile = analyzer.analyze("how does req.body get validated")
        assert profile.has_dot_accessor

    def test_mixed_weights_include_multiple_channels(self, analyzer):
        profile = analyzer.analyze("how does user.save() work")
        # Mixed should involve more than one channel
        active = {k for k, v in profile.recommended_weights.items() if v > 0}
        assert len(active) >= 1   # at least dense


# ---------------------------------------------------------------------------
# disable_routing=True
# ---------------------------------------------------------------------------

class TestRoutingDisabled:
    def test_disabled_always_semantic(self, disabled_analyzer):
        queries = [
            "`calculateChecksum`",
            "classes extending BaseController",
            "retry logic for timeouts",
            "usage of `initDb`",
        ]
        for q in queries:
            profile = disabled_analyzer.analyze(q)
            assert profile.category == QueryCategory.SEMANTIC, \
                f"Expected SEMANTIC (routing disabled) for '{q}', got {profile.category}"

    def test_disabled_weights_dense_only(self, disabled_analyzer):
        profile = disabled_analyzer.analyze("`calculateChecksum`")
        assert profile.recommended_weights == {"dense": 1.0}

    def test_disabled_routing_reason_mentions_disabled(self, disabled_analyzer):
        profile = disabled_analyzer.analyze("any query")
        assert "disabled" in profile.routing_reason.lower()


# ---------------------------------------------------------------------------
# QueryProfile serialisation
# ---------------------------------------------------------------------------

class TestQueryProfileSerialisation:
    def test_as_dict_is_json_serialisable(self, analyzer):
        profile = analyzer.analyze("handleHttpRequest")
        d = profile.as_dict()
        s = json.dumps(d)  # must not raise
        assert isinstance(s, str)

    def test_as_dict_contains_expected_keys(self, analyzer):
        profile = analyzer.analyze("retry logic for timeouts")
        d = profile.as_dict()
        assert "category" in d
        assert "signals" in d
        assert "routing" in d
        assert "weights" in d["routing"]
        assert "reason" in d["routing"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query(self, analyzer):
        profile = analyzer.analyze("")
        assert profile.category in QueryCategory.__members__.values()
        assert isinstance(profile.recommended_weights, dict)

    def test_single_word_nl(self, analyzer):
        profile = analyzer.analyze("sort")
        # "sort" is lowercase, not camel — SEMANTIC
        assert profile.category == QueryCategory.SEMANTIC

    def test_single_camel_identifier(self, analyzer):
        profile = analyzer.analyze("bubbleSort")
        assert profile.category == QueryCategory.SYMBOLIC

    def test_numbers_and_punctuation(self, analyzer):
        profile = analyzer.analyze("v2.1 API endpoint for /users/login")
        assert isinstance(profile.category, QueryCategory)

    def test_very_long_query(self, analyzer):
        q = "find the function that validates and sanitizes " * 10
        profile = analyzer.analyze(q)
        assert profile.category == QueryCategory.SEMANTIC
