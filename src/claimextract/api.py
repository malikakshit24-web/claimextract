"""HTTP API.

POST /documents                 upload a PDF or image; it is extracted and validated
GET  /documents                 list documents, optionally ?status=needs_review
GET  /documents/{id}            one document with its claim, issues and status
POST /documents/{id}/review     approve, correct or reject a document in review
GET  /health                    liveness and database check
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from claimextract import db
from claimextract.config import get_settings
from claimextract.documents import UnsupportedDocument
from claimextract.extractors import Extractor, get_extractor
from claimextract.schemas import DocumentOut, DocumentStatus, ReviewRequest
from claimextract.service import ReviewError, apply_review, process_upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.create_tables()
    yield


app = FastAPI(
    title="ClaimExtract",
    description="Extracts structured healthcare claim data from PDFs and scans.",
    version="0.1.0",
    lifespan=lifespan,
)

SessionDep = Annotated[Session, Depends(db.get_session)]


def extractor_dependency(
    extractor: Annotated[str | None, Query(description="'ocr' (rules baseline) or 'llm' (Claude vision)")] = None,
) -> Extractor:
    try:
        return get_extractor(extractor or get_settings().default_extractor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_or_404(session: Session, document_id: int) -> db.Document:
    row = session.get(db.Document, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    return row


@app.get("/health")
def health(session: SessionDep) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(
    session: SessionDep,
    extractor: Annotated[Extractor, Depends(extractor_dependency)],
    file: Annotated[UploadFile, File(description="PDF, PNG, JPEG, TIFF or WebP")],
) -> db.Document:
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"file is larger than {get_settings().max_upload_mb} MB")
    if not data:
        raise HTTPException(status_code=422, detail="file is empty")
    try:
        return process_upload(session, extractor, file.filename or "upload", data, file.content_type)
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    session: SessionDep,
    status: DocumentStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[db.Document]:
    query = select(db.Document).order_by(db.Document.id.desc()).limit(limit).offset(offset)
    if status is not None:
        query = query.where(db.Document.status == status)
    return list(session.scalars(query))


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(session: SessionDep, document_id: int) -> db.Document:
    return _get_or_404(session, document_id)


@app.post("/documents/{document_id}/review", response_model=DocumentOut)
def review_document(session: SessionDep, document_id: int, review: ReviewRequest) -> db.Document:
    row = _get_or_404(session, document_id)
    try:
        return apply_review(session, row, review)
    except ReviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
