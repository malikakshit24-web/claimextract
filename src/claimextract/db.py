"""Database models and session handling (PostgreSQL in production, SQLite in tests)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from claimextract.config import get_settings

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    extractor: Mapped[str | None] = mapped_column(String(20))
    # What the extractor produced, kept unchanged for auditing and evaluation.
    extracted: Mapped[dict | None] = mapped_column(JsonType)
    # The current claim: the extraction, or a reviewer's correction of it.
    claim: Mapped[dict | None] = mapped_column(JsonType)
    issues: Mapped[list] = mapped_column(JsonType, default=list)
    uncertain_fields: Mapped[list] = mapped_column(JsonType, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(100))
    review_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_engine(url: str | None = None, **kwargs) -> Engine:
    global _engine, _session_factory
    _engine = create_engine(url or get_settings().database_url, pool_pre_ping=True, **kwargs)
    _session_factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def create_tables() -> None:
    if _engine is None:
        init_engine()
    Base.metadata.create_all(_engine)


def get_session() -> Iterator[Session]:
    if _session_factory is None:
        init_engine()
    with _session_factory() as session:
        yield session
