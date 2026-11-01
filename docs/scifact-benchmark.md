# BEIR SciFact retrieval benchmark

Executed on 1 November 2026 using `scripts/benchmark_scifact.py`. This evaluates the existing retrieval code on scientific abstracts, separately from `experiments/evaluation.json` and its seven authored diagnostic questions.

## Provenance and terms

- Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
- SHA256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` (2,816,079-byte download).
- SciFact: Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*, EMNLP, https://aclanthology.org/2020.emnlp-main.609/ .
- BEIR: Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets and Benchmarks, https://arxiv.org/abs/2104.08663 .
- Upstream terms verified at https://github.com/allenai/scifact/blob/master/LICENSE.md: claims/evidence annotations CC BY 4.0; S2ORC abstracts ODC-By 1.0; upstream code Apache-2.0. The BEIR archive repackages the dataset; those source-data terms still matter. No dataset text, queries or labels are redistributed in this repository; only aggregate/per-query numeric results and IDs are committed.

## Protocol

Index all 5,183 corpus documents as one title+abstract text each. Evaluate 300 queries with positive official test qrels at cutoff 10, treating positive judgments as binary. Use the existing tokenizer, stopword list, BM25 k1=1.5/b=0.75, normalized 1024-dimensional lexical-hash FAISS vectors and reciprocal-rank fusion constant 60. No training, tuning or model selection using test labels. Report macro means over queries; relevant but unjudged documents receive no credit, as in the provided relevance set.

The adapter uses an internal page=1 sentinel because the ranker stores a page-bearing Chunk. These are not PDFs and that field is never exported as a real paper page or evaluated as a citation. Chunking also differs from the application's 160-word PDF windows. This benchmark measures document ranking, not end-to-end PDF extraction, page citation, scientific correctness, answer quality or abstention.

## Results

| Strategy | Recall@10 | MRR@10 | nDCG@10 | Query-loop seconds |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.785611 | 0.635950 | 0.667301 | 8.650 |
| Lexical-hash FAISS | 0.403333 | 0.283766 | 0.305295 | 16.028 |
| Hybrid RRF | 0.659167 | 0.460033 | 0.498554 | 25.469 |

Total 53.830 seconds including data load/indexing on this shared CPU environment; timings are descriptive, not a stable performance comparison. The search implementation computes both component scores even for individual strategies. BM25 is strongest here; adding weak lexical-hash rankings through RRF reduced retrieval quality. No semantic embedding model ran. This experiment must not be promoted as generative QA performance.

## Reproduce

Review the linked dataset terms, then from the repository root:

```sh
python -m pip install -e '.[dev]'
curl --fail --location --output /tmp/scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
echo '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  /tmp/scifact.zip' | sha256sum --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark_scifact.py --archive /tmp/scifact.zip --output results/scifact/metrics.json
pytest -q
ruff check .
ruff format --check .
```

Dataset downloads are manual and absent from CI. The script reads named zip members without extracting arbitrary paths. Metric unit tests cover multiple judgments, misses and the top-10 cutoff. Future tuning must use separate training/development judgments and disclose that these test results have already been observed.
