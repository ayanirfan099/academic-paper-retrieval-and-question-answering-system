import streamlit as st

from paper_retrieval.core import EvidenceIndex, parse_pdf

st.title("Academic evidence retrieval")
st.caption(
    "Extractive responses quote document text; citations point to indexed pages. Documents are untrusted evidence, never instructions. No paid LLM."
)
backend = st.selectbox("Embedding backend", ["lexical-hash", "sentence-transformer"])
st.caption(
    "lexical-hash is an offline lexical vector baseline, not semantic sentence embeddings. Sentence Transformer downloads may require network access."
)
if st.button("Initialize / reset index"):
    try:
        st.session_state.index = EvidenceIndex(backend)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
if "index" not in st.session_state:
    st.session_state.index = EvidenceIndex()
index = st.session_state.index
uploads = st.file_uploader(
    "PDFs (10 MiB each, maximum 200 pages)", type=["pdf"], accept_multiple_files=True
)
if st.button("Index PDFs"):
    for file in uploads:
        try:
            index.add(parse_pdf(file.getvalue(), file.name))
        except ValueError as exc:
            st.error(str(exc))
st.write("Indexed documents", sorted({c.document for c in index.chunks}))
strategy = st.selectbox("Retrieval strategy", ["hybrid", "bm25", "dense"])
question = st.text_input("Research question")
if st.button("Retrieve evidence"):
    try:
        result = index.answer(question, strategy)
        st.write(result["answer"])
        st.json(result["citations"])
        for hit in result["evidence"]:
            c = hit["chunk"]
            st.write(f"{c['document']}, page {c['page']}")
            st.text(c["text"])
    except ValueError as exc:
        st.error(str(exc))
st.caption(
    "Small authored evaluation fixtures are diagnostic only. Retrieval cannot establish truth, citation entailment, completeness or prompt safety in a future generative integration. No OCR; in-memory index is reset when the process ends."
)
