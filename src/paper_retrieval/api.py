from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import EvidenceIndex, parse_pdf


def create_app(index=None):
    app = FastAPI(title="Page-aware evidence retrieval")
    store = index or EvidenceIndex()

    class Query(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        strategy: str = "hybrid"
        k: int = Field(default=3, ge=1, le=50)

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(store.chunks), "backend": store.backend}

    @app.get("/documents")
    def documents():
        return sorted({c.document for c in store.chunks})

    @app.post("/documents")
    async def upload(file: Annotated[UploadFile, File()]):
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(400, "PDF required")
        payload = await file.read(10 * 1024 * 1024 + 1)
        try:
            chunks = parse_pdf(payload, file.filename or "")
            store.add(chunks)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"added_chunks": len(chunks), "total_chunks": len(store.chunks)}

    @app.post("/answer")
    def answer(query: Query):
        try:
            return store.answer(query.question, query.strategy, query.k)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


app = create_app()
