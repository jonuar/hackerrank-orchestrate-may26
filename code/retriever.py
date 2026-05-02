from __future__ import annotations
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Configuration
CORPUS_ROOT = Path(__file__).parent.parent / "data"

CHUNK_SIZE   = 600
CHUNK_OVERLAP = 100

TOP_K = 5

COMPANY_DIR: dict[str, str] = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
}

@dataclass
class Chunk:
    text: str
    source: str
    company: str
    chunk_id: int


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    company: str


# Text utilities

def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "may", "now", "own", "see", "who",
    "did", "let", "put", "say", "she", "too", "use", "with", "that",
    "this", "have", "from", "they", "will", "been", "more", "also",
    "your", "their", "when", "what", "there", "some", "than", "into",
    "about", "would", "which", "were", "been",
}


def _filter_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def _chunk_text(text: str, source: str, company: str, start_id: int) -> list[Chunk]:
    """
    Split text into overlapping character-based chunks.
    Overlap preserves sentence context at boundaries.
    """
    chunks: list[Chunk] = []
    start = 0
    chunk_id = start_id

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                company=company,
                chunk_id=chunk_id,
            ))
            chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


class BM25Index:
    """
    Minimal BM25 index over a list of Chunk objects.
    k1 and b are standard BM25 hyperparameters (Robertson et al.).
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build(chunks)

    def _build(self, chunks: list[Chunk]) -> None:
        self.n_docs = len(chunks)
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        df: dict[str, int] = defaultdict(int)

        for chunk in chunks:
            tokens = _filter_tokens(_tokenize(chunk.text))
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            for term in set(tokens):
                df[term] += 1

        self.avg_doc_len = sum(self.doc_len) / max(self.n_docs, 1)
        # IDF for each term
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        tokens = self.doc_tokens[doc_idx]
        dl = self.doc_len[doc_idx]
        tf_map: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
            score += self.idf[term] * (numerator / denominator)
        return score

    def query(self, text: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        q_tokens = _filter_tokens(_tokenize(text))
        if not q_tokens:
            return []

        scores = [
            (i, self.score(q_tokens, i))
            for i in range(self.n_docs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            (self.chunks[i], s)
            for i, s in scores[:top_k]
            if s > 0
        ]


# Corpus loader

def _load_file(path: Path, company: str, start_id: int) -> list[Chunk]:
    """Read a single file and return its chunks."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not text.strip():
        return []

    # Strip markdown headers and HTML tags minimally
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\s{3,}", "\n\n", text)

    relative = str(path.relative_to(CORPUS_ROOT))
    return _chunk_text(text, source=relative, company=company, start_id=start_id)


def load_corpus(company: str | None = None) -> list[Chunk]:
    """
    Load all corpus files. If company is specified, load only that company's
    subdirectory. Otherwise load all three.
    """
    companies_to_load = (
        {company: COMPANY_DIR[company]}
        if company and company in COMPANY_DIR
        else COMPANY_DIR
    )

    all_chunks: list[Chunk] = []
    chunk_id = 0

    for company_name, subdir in companies_to_load.items():
        corp_dir = CORPUS_ROOT / subdir
        if not corp_dir.exists():
            continue

        for path in sorted(corp_dir.rglob("*")):
            if path.is_file() and path.suffix in {".txt", ".md", ".html", ".json", ".csv"}:
                new_chunks = _load_file(path, company_name, chunk_id)
                all_chunks.extend(new_chunks)
                chunk_id += len(new_chunks)

    return all_chunks


# Avoids rebuilding the index on every ticket
_index_cache: dict[str, BM25Index] = {}


def _get_index(company: str) -> BM25Index:
    """Return (cached) BM25 index for the given company scope."""
    if company not in _index_cache:
        chunks = load_corpus(company if company in COMPANY_DIR else None)
        _index_cache[company] = BM25Index(chunks)
    return _index_cache[company]


# Public API

def retrieve(
    query: str,
    company: str,
    top_k: int = TOP_K,
) -> RetrievalResult:
    """
    Retrieve the top_k most relevant corpus chunks for the given query.

    Args:
        query:   Combined ticket text (issue + subject).
        company: Resolved company name from the classifier.
        top_k:   Number of chunks to return.

    Returns:
        RetrievalResult with ranked chunks.
    """
    index = _get_index(company)
    results = index.query(query, top_k=top_k)
    chunks = [chunk for chunk, _score in results]
    return RetrievalResult(chunks=chunks, query=query, company=company)


def format_context(result: RetrievalResult) -> str:
    """
    Format retrieved chunks into a single string to inject into the LLM prompt.
    Each chunk is labeled with its source file for attribution.
    """
    if not result.chunks:
        return "No relevant documentation found in the support corpus."

    parts: list[str] = []
    for i, chunk in enumerate(result.chunks, 1):
        parts.append(f"[Source {i}: {chunk.source}]\n{chunk.text}")

    return "\n\n---\n\n".join(parts)