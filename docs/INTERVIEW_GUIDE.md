# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## Why does hybrid retrieval use reciprocal ranks?

BM25 and vector similarities have incomparable raw scales. Reciprocal rank fusion combines rankings without assuming comparable score magnitudes.

## What is the difference between a valid citation and an entailing citation?

Validity here means the chunk, document and page exist in the index. Entailment requires the passage actually supports the answer; metadata consistency alone cannot prove that.

## Is lexical hashing semantic dense retrieval?

No. It maps lexical features to dense stored vectors for FAISS. The sentence-transformer backend is the implemented semantic path, and its downloaded-model benchmark remains pending.
