"""Persistence layer for Aurexis Systems accounts, usage, and subscriptions.

Uses SQLAlchemy with a SQLite database by default so the app is fully
self-contained and testable without any external service. Point the
``AUREXIS_DATABASE_URL`` (or ``DATABASE_URL``) environment variable at a
managed PostgreSQL instance (e.g. Supabase) for production deployments.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Tuple

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = (
    os.getenv("AUREXIS_DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or "sqlite:///aurexis_app.db"
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    subscription = Column(String(32), nullable=False, default="free")
    analyses_used = Column(Integer, nullable=False, default=0)
    chat_used = Column(Integer, nullable=False, default=0)
    stripe_customer_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


@dataclass
class UserRecord:
    """Detached, immutable snapshot of a user row.

    Returned to the app layer so callers never touch live ORM sessions
    (and so the record can live safely inside ``st.session_state``).
    """

    id: int
    email: str
    subscription: str
    analyses_used: int
    chat_used: int
    stripe_customer_id: Optional[str]


_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
_engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not yet exist. Safe to call repeatedly."""
    Base.metadata.create_all(_engine)


@contextmanager
def _session_scope() -> Iterator["SessionLocal"]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _to_record(user: User) -> UserRecord:
    return UserRecord(
        id=user.id,
        email=user.email,
        subscription=user.subscription,
        analyses_used=user.analyses_used,
        chat_used=user.chat_used,
        stripe_customer_id=user.stripe_customer_id,
    )


def create_user(email: str, password_hash: str) -> UserRecord:
    """Create a new user. Raises ``ValueError`` if the email already exists."""
    email = email.strip().lower()
    with _session_scope() as session:
        existing = session.query(User).filter(User.email == email).one_or_none()
        if existing is not None:
            raise ValueError("An account with this email already exists.")
        user = User(email=email, password_hash=password_hash)
        session.add(user)
        session.flush()
        return _to_record(user)


def get_user_by_email(email: str) -> Optional[UserRecord]:
    email = (email or "").strip().lower()
    with _session_scope() as session:
        user = session.query(User).filter(User.email == email).one_or_none()
        return _to_record(user) if user else None


def get_credentials(email: str) -> Optional[Tuple[UserRecord, str]]:
    """Return ``(record, password_hash)`` for login verification, or ``None``."""
    email = (email or "").strip().lower()
    with _session_scope() as session:
        user = session.query(User).filter(User.email == email).one_or_none()
        if user is None:
            return None
        return _to_record(user), user.password_hash


def get_user(user_id: int) -> Optional[UserRecord]:
    with _session_scope() as session:
        user = session.get(User, user_id)
        return _to_record(user) if user else None


def increment_analyses(user_id: int, amount: int = 1) -> Optional[UserRecord]:
    with _session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return None
        user.analyses_used = (user.analyses_used or 0) + amount
        session.flush()
        return _to_record(user)


def increment_chat(user_id: int, amount: int = 1) -> Optional[UserRecord]:
    with _session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return None
        user.chat_used = (user.chat_used or 0) + amount
        session.flush()
        return _to_record(user)


def set_subscription(
    email: str,
    subscription: str,
    stripe_customer_id: Optional[str] = None,
) -> Optional[UserRecord]:
    email = (email or "").strip().lower()
    with _session_scope() as session:
        user = session.query(User).filter(User.email == email).one_or_none()
        if user is None:
            return None
        user.subscription = subscription
        if stripe_customer_id:
            user.stripe_customer_id = stripe_customer_id
        session.flush()
        return _to_record(user)


def get_user_by_stripe_customer(stripe_customer_id: str) -> Optional[UserRecord]:
    if not stripe_customer_id:
        return None
    with _session_scope() as session:
        user = (
            session.query(User)
            .filter(User.stripe_customer_id == stripe_customer_id)
            .one_or_none()
        )
        return _to_record(user) if user else None


def set_subscription_by_customer(
    stripe_customer_id: str,
    subscription: str,
) -> Optional[UserRecord]:
    """Update a subscription using the Stripe customer id (used by webhooks)."""
    if not stripe_customer_id:
        return None
    with _session_scope() as session:
        user = (
            session.query(User)
            .filter(User.stripe_customer_id == stripe_customer_id)
            .one_or_none()
        )
        if user is None:
            return None
        user.subscription = subscription
        session.flush()
        return _to_record(user)
