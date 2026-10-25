# Reproduce

Python >=3.11; executed in Python 3.12 on CPU. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m paper_retrieval.cli
# Requires downloaded sentence model:
python -m paper_retrieval.cli --backend sentence-transformer --output results/semantic.json
uvicorn paper_retrieval.api:app --host 127.0.0.1 --port 8000
# Separate terminal:
streamlit run src/paper_retrieval/app.py
pytest -q
ruff check .
ruff format --check .
```

Use CPU PyTorch wheels from https://download.pytorch.org/whl/cpu if installing the deep learning dependencies on a CPU machine. Remote datasets and weights require separate access. Do not load untrusted pickle/joblib/checkpoint artifacts.
