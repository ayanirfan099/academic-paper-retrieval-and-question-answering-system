"""Untrusted documents are evidence, never executable instructions."""

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer

STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "and",
    "in",
    "for",
    "on",
    "what",
    "how",
    "does",
    "do",
    "which",
    "with",
    "by",
    "was",
    "as",
    "from",
    "it",
}


def tokens(text):
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    text: str
    title: str = ""


def parse_pdf(
    payload, filename, max_bytes=10 * 1024 * 1024, max_pages=200, chunk_words=160, overlap=30
):
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(".pdf")
    ):
        raise ValueError("Safe PDF basename required")
    if not payload.startswith(b"%PDF-") or len(payload) > max_bytes:
        raise ValueError("Invalid PDF signature or upload exceeds 10 MiB")
    if chunk_words < 1 or not 0 <= overlap < chunk_words:
        raise ValueError("Invalid chunk size/overlap")
    try:
        reader = PdfReader(io.BytesIO(payload), strict=True)
    except Exception as exc:
        raise ValueError("Unreadable PDF") from exc
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs unsupported")
    if len(reader.pages) > max_pages:
        raise ValueError("Too many pages")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    chunks = []
    title = str((reader.metadata or {}).get("/Title", ""))[:300]
    for page_no, page in enumerate(reader.pages, 1):
        # Limit extracted text as well as upload size; OCR is intentionally unsupported.
        text = page.extract_text() or ""
        if len(text) > 1_000_000:
            raise ValueError("Page text too large")
        words = text.split()
        for start in range(0, len(words), chunk_words - overlap):
            content = " ".join(words[start : start + chunk_words])
            if content:
                chunks.append(
                    Chunk(f"{digest}:{page_no}:{start}", filename, page_no, content, title)
                )
            if start + chunk_words >= len(words):
                break
    if not chunks:
        raise ValueError("No extractable text; scanned PDFs require OCR")
    return chunks


class EvidenceIndex:
    def __init__(self, backend="lexical-hash", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if backend not in {"lexical-hash", "sentence-transformer"}:
            raise ValueError("Invalid embedding backend")
        self.backend = backend
        self.model_name = model_name
        self.chunks = []
        self.encoder = None
        if backend == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name, device="cpu")
        else:
            self.encoder = HashingVectorizer(n_features=1024, alternate_sign=False, norm="l2")

    def embed(self, texts):
        if self.backend == "sentence-transformer":
            z = self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        else:
            z = self.encoder.transform(texts).toarray()
        return np.ascontiguousarray(z, dtype="float32")

    def add(self, chunks):
        if not chunks:
            raise ValueError("No chunks")
        seen = {c.id for c in self.chunks}
        fresh = []
        for c in chunks:
            if c.id not in seen:
                if c.page < 1 or not c.text.strip():
                    raise ValueError("Invalid chunk metadata")
                fresh.append(c)
                seen.add(c.id)
        self.chunks.extend(fresh)
        self.word_counts = [Counter(tokens(c.text)) for c in self.chunks]
        self.lengths = np.array([sum(c.values()) for c in self.word_counts])
        self.average_length = max(float(self.lengths.mean()), 1)
        df = Counter(term for count in self.word_counts for term in count)
        self.idf = {
            t: math.log(1 + (len(self.chunks) - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }
        z = self.embed([c.text for c in self.chunks])
        self.index = faiss.IndexFlatIP(z.shape[1])
        self.index.add(z)

    def search(self, query, strategy="hybrid", k=5):
        if not self.chunks:
            raise ValueError("Index documents first")
        if not query.strip() or len(query) > 2000:
            raise ValueError("Query must contain 1-2000 characters")
        if strategy not in {"bm25", "dense", "hybrid"} or not 1 <= k <= 50:
            raise ValueError("Invalid strategy or k")
        words = tokens(query)
        bm25 = np.zeros(len(self.chunks))
        for t in set(words):
            tf = np.array([c[t] for c in self.word_counts])
            bm25 += (
                self.idf.get(t, 0)
                * (tf * 2.5)
                / (tf + 1.5 * (0.25 + 0.75 * self.lengths / self.average_length) + 1e-12)
            )
        similarities, ids = self.index.search(self.embed([query]), len(self.chunks))
        dense = np.zeros(len(self.chunks))
        dense[ids[0]] = similarities[0]
        if strategy == "bm25":
            score = bm25
        elif strategy == "dense":
            score = dense
        else:
            score = np.zeros(len(self.chunks))
            for ranking in [np.argsort(-bm25, kind="stable"), np.argsort(-dense, kind="stable")]:
                for rank, i in enumerate(ranking):
                    score[i] += 1 / (60 + rank + 1)
        ranking = np.argsort(-score, kind="stable")[:k]
        return [
            {
                "chunk": asdict(self.chunks[i]),
                "score": float(score[i]),
                "bm25": float(bm25[i]),
                "similarity": float(dense[i]),
            }
            for i in ranking
        ]

    def answer(self, query, strategy="hybrid", k=3):
        hits = self.search(query, strategy, k)
        # Lexical support gate is intentionally conservative and visible, not a probability.
        terms = set(tokens(query))
        supported = []
        for h in hits:
            overlap = terms & set(tokens(h["chunk"]["text"]))
            if len(overlap) >= min(2, len(terms)) and overlap and h["bm25"] > 0:
                supported.append(h)
        if not supported:
            return {
                "answer": "Insufficient evidence in indexed documents.",
                "abstained": True,
                "citations": [],
                "evidence": hits,
                "backend": self.backend,
            }
        chosen = supported[:2]
        citations = [
            {
                "chunk_id": h["chunk"]["id"],
                "document": h["chunk"]["document"],
                "page": h["chunk"]["page"],
            }
            for h in chosen
        ]
        # Verbatim extraction avoids generating unsupported factual claims or following embedded instructions.
        return {
            "answer": "\n\n".join(f"[{i + 1}] {h['chunk']['text']}" for i, h in enumerate(chosen)),
            "abstained": False,
            "citations": citations,
            "evidence": hits,
            "backend": self.backend,
        }
