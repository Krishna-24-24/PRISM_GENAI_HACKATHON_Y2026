import re
from typing import List, Optional


def tokenize_for_bm25(text: str, identifiers: Optional[List[str]] = None) -> List[str]:
    """Tokenize source code or queries for BM25 with camelCase and snake_case sub-word expansion."""
    tokens: List[str] = []
    if not text:
        return tokens

    words = re.split(r"[^a-zA-Z0-9_]+", text)

    for word in words:
        if not word:
            continue
        tokens.append(word.lower())

        if "_" in word:
            parts = word.split("_")
            tokens.extend([p.lower() for p in parts if p])

        camel_split = re.finditer(
            r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", word
        )
        parts = [m.group(0).lower() for m in camel_split if m.group(0)]
        if len(parts) > 1:
            tokens.extend(parts)

    if identifiers:
        for ident in identifiers:
            tokens.append(ident.lower())

    return tokens
