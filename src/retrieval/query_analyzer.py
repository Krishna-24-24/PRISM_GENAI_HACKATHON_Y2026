"""
QueryAnalyzer — deterministic query classification for adaptive retrieval routing.

Design principles:
  1. No LLM calls. All classification is regex + heuristics.
  2. Every decision is observable (QueryProfile carries full signal breakdown).
  3. The classifier never *prevents* a channel — it weights them.
  4. Fast: < 1ms per query on CPU.

Query categories (from researcher review):
  semantic    — natural language description of behavior
  symbolic    — refers to a specific identifier/symbol by name
  structural  — asks about class hierarchies, inheritance, decorators
  usage       — asks for call sites / examples of using something
  mixed       — ambiguous; traits from more than one category

Each category maps to recommended channel weights for wRRF.
Weights are evidence-driven from the ablation (BM25 alone=0.013, Dense=0.050):
  - Dense is the dominant signal on AppsRetrieval (NL queries)
  - BM25 helps only when the query contains exact tokens from identifiers
  - Symbol channel activates for pure symbol queries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class QueryCategory(str, Enum):
    SEMANTIC   = "semantic"
    SYMBOLIC   = "symbolic"
    STRUCTURAL = "structural"
    USAGE      = "usage"
    MIXED      = "mixed"


@dataclass
class QueryProfile:
    """
    Fully observable classification output. Every field that influenced
    the routing decision is recorded here for debugging.
    """
    raw_query: str
    category: QueryCategory

    # --- Raw signal flags ---
    has_backtick_identifier: bool = False   # `someFunc`
    has_camel_case_token: bool = False      # handleHttpRequest
    has_snake_case_token: bool = False      # get_user_by_id
    has_dot_accessor: bool = False          # obj.method
    has_structural_keyword: bool = False    # class/extends/implements/interface
    has_usage_keyword: bool = False         # call/invoke/example/usage/use
    has_api_token: bool = False             # URL-like, dotted-name like stripe.charge
    has_behavioral_keyword: bool = False    # validates/handles/returns/throws
    identifier_count: int = 0
    word_count: int = 0

    # --- Extracted identifiers for Symbol channel ---
    extracted_identifiers: list = field(default_factory=list)

    # --- Recommended weights (populated by classifier) ---
    recommended_weights: Dict[str, float] = field(default_factory=dict)
    recommended_bm25_k: int = 100
    recommended_dense_k: int = 100
    recommended_symbol_k: int = 50

    # --- Human-readable explanation of the routing decision ---
    routing_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "category": self.category.value,
            "signals": {
                "has_backtick_identifier": self.has_backtick_identifier,
                "has_camel_case_token": self.has_camel_case_token,
                "has_snake_case_token": self.has_snake_case_token,
                "has_dot_accessor": self.has_dot_accessor,
                "has_structural_keyword": self.has_structural_keyword,
                "has_usage_keyword": self.has_usage_keyword,
                "has_api_token": self.has_api_token,
                "has_behavioral_keyword": self.has_behavioral_keyword,
                "identifier_count": self.identifier_count,
                "word_count": self.word_count,
                "extracted_identifiers": self.extracted_identifiers,
            },
            "routing": {
                "weights": self.recommended_weights,
                "bm25_k": self.recommended_bm25_k,
                "dense_k": self.recommended_dense_k,
                "symbol_k": self.recommended_symbol_k,
                "reason": self.routing_reason,
            },
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Backtick-quoted identifiers: `myFunc`, `SomeClass`
_RE_BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

# camelCase: starts with lower, contains upper inside
_RE_CAMEL = re.compile(r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b")

# UPPER_snake or lower_snake (at least one underscore + letter on each side)
_RE_SNAKE = re.compile(r"\b[A-Za-z][a-z0-9]*(?:_[A-Za-z][a-z0-9]+)+\b")

# PascalCase standalone token (class names)
_RE_PASCAL = re.compile(r"\b[A-Z][a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")

# obj.method access patterns
_RE_DOT_ACCESSOR = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]+\b")

# API / library tokens (dotted name OR URL path)
_RE_API_TOKEN = re.compile(
    r"\b(?:[A-Za-z]+\.){1,3}[A-Za-z]+\b"          # stripe.charge.create
    r"|/[a-z][a-z0-9/_\-]*"                          # /api/v1/users
)

_STRUCTURAL_KW = frozenset(
    ["class", "extend", "extends", "extending", "implements", "implementing",
     "interface", "inherit", "inherits", "inheriting", "inheritance",
     "subclass", "subclasses", "superclass", "abstract", "decorator", "mixin"]
)

# NL wrapper words that frame a symbol query ("where is X", "what does X do")
_SYMBOLIC_FRAME_KW = frozenset(
    ["where", "what", "find", "show", "locate", "defined", "definition",
     "implementation", "declare", "declared"]
)

_USAGE_KW = frozenset(
    ["call", "calls", "calling", "invoke", "invokes", "use", "uses",
     "using", "example", "examples", "usage", "how to use", "show me"]
)

_BEHAVIORAL_KW = frozenset(
    ["validates", "validate", "handles", "handle", "returns", "return",
     "throws", "throw", "raises", "emits", "sanitize", "sanitizes",
     "converts", "parses", "parse", "computes", "retry", "retries"]
)


class QueryAnalyzer:
    """
    Classifies a query into one of five categories and returns a QueryProfile
    with recommended per-channel weights derived from ablation evidence.

    Weight evidence (from ablation, 200-query sample, AppsRetrieval):
      Dense alone: NDCG@10=0.0498, MRR=0.046
      BM25 alone:  NDCG@10=0.0131, MRR=0.014
      BM25+Dense:  NDCG@10=0.0495  (BM25 *hurts* Dense via RRF dilution)

    Routing strategy:
      - Semantic queries:   Dense-only (BM25 adds latency, no metric gain)
      - Symbolic queries:   Dense=0.5 + Symbol=2.0 + BM25=1.5 (symbol must dominate)
      - Usage queries:      Dense=1.0 + Symbol=1.5 (find call sites via name)
      - Structural queries: Dense=1.0 + BM25=0.5 (structural keywords exist in text)
      - Mixed queries:      Dense=1.0 + BM25=0.5 + Symbol=0.5 (conservative)

    If adaptive routing is disabled (disable_routing=True), returns uniform weights
    equivalent to the baseline hybrid (Dense-only on AppsRetrieval).
    """

    def __init__(self, disable_routing: bool = False) -> None:
        self.disable_routing = disable_routing

    def analyze(self, query: str) -> QueryProfile:
        profile = QueryProfile(raw_query=query, category=QueryCategory.SEMANTIC)

        # --- Signal extraction ---
        backtick_ids = _RE_BACKTICK.findall(query)
        profile.has_backtick_identifier = bool(backtick_ids)
        profile.extracted_identifiers.extend(backtick_ids)

        camel_tokens = _RE_CAMEL.findall(query)
        pascal_tokens = _RE_PASCAL.findall(query)
        profile.has_camel_case_token = bool(camel_tokens)
        profile.extracted_identifiers.extend(camel_tokens)
        profile.extracted_identifiers.extend(pascal_tokens)

        snake_tokens = _RE_SNAKE.findall(query)
        profile.has_snake_case_token = bool(snake_tokens)
        profile.extracted_identifiers.extend(snake_tokens)

        profile.has_dot_accessor = bool(_RE_DOT_ACCESSOR.search(query))
        profile.extracted_identifiers.extend(
            m.group(0) for m in _RE_DOT_ACCESSOR.finditer(query)
        )

        # Deduplicate while preserving order
        seen: set = set()
        deduped = []
        for ident in profile.extracted_identifiers:
            if ident.lower() not in seen:
                seen.add(ident.lower())
                deduped.append(ident)
        profile.extracted_identifiers = deduped

        profile.identifier_count = len(profile.extracted_identifiers)

        words_lower = set(re.findall(r"\b[a-z]+\b", query.lower()))
        profile.word_count = len(query.split())
        profile.has_structural_keyword = bool(_STRUCTURAL_KW & words_lower)
        profile.has_usage_keyword = bool(_USAGE_KW & words_lower)
        profile.has_behavioral_keyword = bool(_BEHAVIORAL_KW & words_lower)
        profile.has_api_token = bool(_RE_API_TOKEN.search(query))

        # --- Category classification ---
        if self.disable_routing:
            profile.category = QueryCategory.SEMANTIC
            profile.recommended_weights = {"dense": 1.0}
            profile.routing_reason = "routing disabled — dense-only (ablation baseline)"
            return profile

        profile.category, profile.recommended_weights, profile.routing_reason = \
            self._classify(profile, words_lower)

        return profile

    def _classify(
        self, p: QueryProfile, words_lower: set
    ) -> tuple[QueryCategory, Dict[str, float], str]:
        """
        Rule-based classification tree.
        Returns (category, weights_dict, reason_string).
        """
        ident_ratio = p.identifier_count / max(p.word_count, 1)
        has_frame_kw = bool(_SYMBOLIC_FRAME_KW & words_lower)

        # --- SYMBOLIC ---
        # Fire when:
        #   - Backtick-quoted identifier present, OR
        #   - Most words are identifiers (ident_ratio > 0.5), OR
        #   - Short query with a clear identifier focus, OR
        #   - NL wrapper ("where is X defined", "what does X do") + camelCase/snake
        if (
            p.has_backtick_identifier
            or ident_ratio > 0.5
            or (p.word_count <= 3 and p.identifier_count >= 1
                and (p.has_camel_case_token or p.has_snake_case_token))
            or (has_frame_kw and p.identifier_count >= 1
                and (p.has_camel_case_token or p.has_snake_case_token
                     or p.has_backtick_identifier))
        ):
            reason = (
                f"symbolic: backtick={p.has_backtick_identifier}, "
                f"ident_ratio={ident_ratio:.2f}, "
                f"ids={p.extracted_identifiers[:3]}"
            )
            # Symbol dominates; BM25 helps for exact token; Dense provides fallback
            return (
                QueryCategory.SYMBOLIC,
                {"dense": 0.5, "bm25": 1.5, "symbol": 2.0},
                reason,
            )

        # --- STRUCTURAL ---
        if p.has_structural_keyword and not p.has_usage_keyword:
            reason = f"structural: keyword match in {_STRUCTURAL_KW & set(p.raw_query.lower().split())}"
            return (
                QueryCategory.STRUCTURAL,
                {"dense": 1.0, "bm25": 0.5},
                reason,
            )

        # --- USAGE ---
        if p.has_usage_keyword and p.identifier_count >= 1:
            reason = (
                f"usage: usage_kw + identifier "
                f"{p.extracted_identifiers[:2]}"
            )
            return (
                QueryCategory.USAGE,
                {"dense": 1.0, "symbol": 1.5},
                reason,
            )

        # --- MIXED (has some identifier signals but mostly NL) ---
        if p.has_camel_case_token or p.has_snake_case_token or p.has_dot_accessor:
            reason = (
                f"mixed: NL+identifier signals, "
                f"camel={p.has_camel_case_token}, snake={p.has_snake_case_token}"
            )
            return (
                QueryCategory.MIXED,
                {"dense": 1.0, "bm25": 0.5, "symbol": 0.5},
                reason,
            )

        # --- SEMANTIC (default — ablation proves dense-only is best here) ---
        reason = "semantic: no identifier signals; dense-only (ablation evidence)"
        return (
            QueryCategory.SEMANTIC,
            {"dense": 1.0},
            reason,
        )
