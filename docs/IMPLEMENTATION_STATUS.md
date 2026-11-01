# Implementation status

Implemented but not fully benchmarked

An offline-capable PDF retrieval application with BM25, FAISS vector search, reciprocal rank fusion, page citations and conservative extractive responses. FastAPI and Streamlit interfaces operate without paid APIs or language models.

Five pytest tests passed: actual generated PDF extraction/chunking and page provenance, unsafe names/signature/size rejection, repeated-upload deduplication, retrieval/abstention, untrusted-text extraction, FastAPI upload/answer requests, repeated evaluation metrics and CLI help. API TestClient makes real application calls; remote service deployment is not claimed. Streamlit initial page checked separately.

A SciFact abstract-retrieval benchmark now provides external retrieval measurements; see `docs/scifact-benchmark.md` from the repository root. Semantic-embedding and end-to-end PDF/answer benchmarks remain unexecuted. Historical Hugging Face access failures are not a current access guarantee. Authored notes are tiny and queries share vocabulary. Citation validity checks provenance consistency, not entailment or truth. Lexical abstention can reject paraphrases and accept superficially related passages. No OCR, persistent database, authentication, multitenancy or resource-isolated hostile PDF parser. PDF page text bound is checked after extraction; deploy with process isolation before accepting hostile internet uploads. Local research app only. In-memory index rebuilds all vectors on upload and is not designed for large corpora. Do not expose API publicly as a production service.

Exact commands: [reproducibility](reproducibility.md). Push status and commit identity are recorded in the portfolio summary; remote CI success is not implied.

Additional live service verification: uvicorn started on loopback; real HTTP health request, PDF upload, answer retrieval and page-1 citation passed. Only the verification process was stopped afterward.
