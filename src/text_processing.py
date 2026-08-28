"""Shared text normalization helpers for BM25 indexing and retrieval."""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_CAMEL_BOUNDARY_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_SPLIT_RE = re.compile(r"[_./:\-]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "is", "it", "its", "may", "might", "must", "not",
    "of", "on", "or", "our", "should", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "about", "into", "using", "use", "used",
}


def _split_camel_case(token: str) -> list[str]:
    token = _CAMEL_BOUNDARY_1.sub(r"\1 \2", token)
    token = _CAMEL_BOUNDARY_2.sub(r"\1 \2", token)
    return [part for part in token.split() if part]


@lru_cache(maxsize=4096)
def tokenize_text(text: str) -> list[str]:
    """Split technical text into normalized BM25 tokens."""
    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(text):
        for part in _SPLIT_RE.split(raw_token):
            if not part:
                continue
            for camel_part in _split_camel_case(part):
                normalized = camel_part.lower()
                if not normalized or normalized in _STOPWORDS:
                    continue
                if normalized.isalpha() and len(normalized) > 2:
                    normalized = _STEMMER.stem(normalized)
                tokens.append(normalized)
    return tokens
