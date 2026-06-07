"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║     AUREXIS SYSTEMS — STREAMLIT EDITION v5.5.1                              ║
║                                                                             ║
║    Distributed AI Governance Operating System                               ║
║    Interactive Streamlit dashboard (self-contained, zero-infra)             ║
║                                                                             ║
║  This is a Streamlit port of the original FastAPI service. The heavy        ║
║  external infrastructure (PostgreSQL, async Redis, Uvicorn, Prometheus      ║
║  HTTP server, Kafka/RabbitMQ) has been replaced with lightweight,           ║
║  embedded equivalents so the whole thing runs with a single command:        ║
║                                                                             ║
║      streamlit run app.py                                                   ║
║                                                                             ║
║  Preserved domain logic:                                                    ║
║  • ECDSA document signing + Fernet encryption (CryptoSigner)                ║
║  • Declarative policy engine (EU AI Act / SR 11-7 / ISO 42001)             ║
║  • Immutable, append-only audit trail (event-sourced)                       ║
║  • Evidence vault with content hashing + chain of custody                   ║
║  • Approval workflow with RBAC                                              ║
║  • JWT token issuance + verification                                        ║
║  • In-process metrics counters (Prometheus-style)                           ║
║                                                                             ║
║  Storage: SQLite (embedded) instead of PostgreSQL.                          ║
║                                                                           ║
║     AUREXIS SYSTEMS — PRODUCTION v5.5.1 (ENTERPRISE-HARDENED)            ║
║                                                                           ║
║    Distributed AI Governance Operating System                             ║
║    Production-Validated, Microservices-Ready, Fault-Tolerant              ║
║                                                                           ║
║  Streamlit deployment profile:                                             ║
║  • Single-command dashboard: streamlit run app.py                         ║
║  • Embedded SQLite persistence with immutable/WORM governance tables       ║
║  • Local HSM-ready crypto abstraction with ECDSA/Fernet when available     ║
║  • JWT authentication, RBAC, tenant isolation, rate limiting               ║
║  • Policy engine, audit trail, evidence vault, approvals, model registry   ║
║  • In-process Prometheus-style metrics and health diagnostics              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import sqlite3
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime as dt, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import hmac
import base64
import secrets

import pandas as pd
import streamlit as st

# ── Optional security stack ───────────────────────────────────────────────
# `cryptography` and `PyJWT` are preferred, but some hosted environments
# (e.g. a stale Streamlit Cloud build) may fail to install them. To guarantee
# the app always boots, we fall back to pure-standard-library equivalents
# (HMAC-SHA256 signatures + Fernet-style symmetric encryption + hand-rolled
# JWT) when the wheels are unavailable.
# ---------------------------------------------------------------------------
# Optional security stack
# ---------------------------------------------------------------------------
# The preferred production profile uses cryptography + PyJWT. The standard
# library fallbacks keep Streamlit Cloud/demo environments bootable when wheels
# are unavailable, while keeping the public service surface identical.
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet, InvalidToken

    _HAS_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover - exercised only when wheel is missing
except Exception:  # pragma: no cover - only used when dependency is missing
    _HAS_CRYPTOGRAPHY = False

    class InvalidToken(Exception):
        """Fallback for cryptography.fernet.InvalidToken."""

try:
    import jwt as _pyjwt

    _HAS_PYJWT = True
except Exception:  # pragma: no cover - exercised only when wheel is missing
except Exception:  # pragma: no cover - only used when dependency is missing
    _pyjwt = None
    _HAS_PYJWT = False


# ══════════════════════════════════════════════════════════════════════════
# ===========================================================================
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
# ===========================================================================


class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Config:
    """Application configuration (env-driven with safe local defaults)."""
    """Streamlit-safe production configuration."""

    ENV = Environment(os.getenv("AUREXIS_ENV", "development"))
    DEBUG = ENV == Environment.DEV

    API_TITLE = "Aurexis Systems v5.5.1"
    API_TITLE = "AUREXIS SYSTEMS — PRODUCTION v5.5.1 (ENTERPRISE-HARDENED)"
    API_VERSION = "5.5.1"
    API_DESCRIPTION = "Distributed AI Governance Operating System"

    # Embedded SQLite database (replaces PostgreSQL).
    DB_PATH = os.getenv("AUREXIS_DB_PATH", "aurexis.db")

    # Encryption key (Fernet). Generated per-session if not provided.
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

    # JWT
    ENCRYPTION_KEY_RAW = os.getenv("ENCRYPTION_KEY", "")
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-only-for-local-testing")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "1000"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "3600"))

    KMS_PROVIDER = os.getenv("KMS_PROVIDER", "local")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Streamlit mode never requires Postgres/Redis/Kafka to boot, but the
    # dashboard surfaces these production integration points for transparency.
    DATABASE_PROFILE = "embedded-sqlite"
    STATE_PROFILE = "in-process"
    MESSAGE_QUEUE_PROFILE = "in-process-events"

def _resolve_encryption_key() -> bytes:
    """Resolve a valid symmetric key, generating one for local use if needed."""
    raw = Config.ENCRYPTION_KEY
    if raw:
        return raw.encode() if isinstance(raw, str) else raw
    if _HAS_CRYPTOGRAPHY:
        return Fernet.generate_key()
    # Stdlib fallback: 32 random bytes, url-safe base64 encoded (Fernet-like).
    return base64.urlsafe_b64encode(secrets.token_bytes(32))
    @classmethod
    def production_warnings(cls) -> List[str]:
        warnings: List[str] = []
        if cls.ENV == Environment.PRODUCTION:
            if cls.JWT_SECRET == "dev-secret-only-for-local-testing":
                warnings.append("JWT_SECRET is using the development fallback.")
            if len(cls.JWT_SECRET) < 32:
                warnings.append("JWT_SECRET should be at least 32 characters.")
            if not cls.ENCRYPTION_KEY_RAW:
                warnings.append("ENCRYPTION_KEY is not set; encrypted data will be process-local.")
        return warnings


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger("aurexis")


# ══════════════════════════════════════════════════════════════════════════
# METRICS (lightweight in-process counters; Prometheus replacement)
# ══════════════════════════════════════════════════════════════════════════
def _utcnow() -> dt:
    return dt.utcnow()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalise_fernet_key(raw: str) -> bytes:
    """Return a valid Fernet key from env input or generated key material."""
    if not _HAS_CRYPTOGRAPHY:
        key_material = raw.encode() if raw else secrets.token_bytes(32)
        return base64.urlsafe_b64encode(hashlib.sha256(key_material).digest())

    if raw:
        candidate = raw.encode()
        try:
            Fernet(candidate)
            return candidate
        except Exception:
            # Accept passphrases/secrets from env and derive the required
            # url-safe 32-byte Fernet key deterministically.
            return base64.urlsafe_b64encode(hashlib.sha256(candidate).digest())

    return Fernet.generate_key()


@st.cache_resource
def get_encryption_key() -> bytes:
    return _normalise_fernet_key(Config.ENCRYPTION_KEY_RAW)


# ===========================================================================
# METRICS
# ===========================================================================


class Metrics:
    """Simple thread-local style counters stored in module state."""
    """In-process Prometheus-style counters and latency histograms."""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))

    def inc(self, name: str, amount: int = 1):
        self.counters[name] += amount
    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, str]] = None) -> str:
        if not labels:
            return name
        label_text = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        return f"{name}{{{label_text}}}"

    def get(self, name: str) -> int:
        return self.counters[name]
    def inc(self, name: str, amount: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        self.counters[self._key(name, labels)] += amount

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.latencies[self._key(name, labels)].append(value)

    def get(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        return self.counters[self._key(name, labels)]

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)
        return dict(sorted(self.counters.items()))

    def latency_snapshot(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name, values in sorted(self.latencies.items()):
            if not values:
                continue
            series = list(values)
            rows.append(
                {
                    "metric": name,
                    "samples": len(series),
                    "avg_ms": round((sum(series) / len(series)) * 1000, 2),
                    "max_ms": round(max(series) * 1000, 2),
                }
            )
        return rows


@st.cache_resource
def get_metrics() -> Metrics:
    return Metrics()


# ══════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHY SERVICE (ECDSA signing + Fernet encryption)
# ══════════════════════════════════════════════════════════════════════════
class instrument:
    """Measure a service block as a Prometheus-style latency metric."""

class CryptoSigner:
    """Document signer + symmetric encryption.
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels
        self.started = 0.0

    Uses ECDSA (SECP256R1) + Fernet when `cryptography` is available, and
    transparently falls back to HMAC-SHA256 signatures + an authenticated
    XOR-stream/HMAC envelope built from the standard library otherwise. The
    public method surface is identical for both backends.
    """
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        get_metrics().observe(self.name, time.perf_counter() - self.started, self.labels)
        if exc_type:
            get_metrics().inc("aurexis_errors_total", labels={"error_type": exc_type.__name__})
        return False


# ===========================================================================
# CRYPTOGRAPHY SERVICE
# ===========================================================================


class CryptoSigner:
    """HSM-aware signer/encrypter adapted for single-process Streamlit."""

    def __init__(self, encryption_key: bytes):
        self.kms_provider = Config.KMS_PROVIDER
        self.backend_name = "cryptography" if _HAS_CRYPTOGRAPHY else "stdlib"
        self._key = encryption_key

        if _HAS_CRYPTOGRAPHY:
            self.backend = default_backend()
            self.private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
            self.cipher_suite = Fernet(encryption_key)
        else:
            # Derive a stable 32-byte secret from the provided key material.
            self._secret = hashlib.sha256(encryption_key).digest()

    # ── Signing ────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, str]:
        provider = self.kms_provider
        if provider != "local":
            provider = f"{provider} (configured; local envelope active in Streamlit)"
        return {
            "crypto_backend": self.backend_name,
            "kms_provider": provider,
            "signing": "ECDSA-SECP256R1" if _HAS_CRYPTOGRAPHY else "HMAC-SHA256",
            "encryption": "Fernet" if _HAS_CRYPTOGRAPHY else "HMAC-authenticated XOR stream",
        }

    def sign_document(self, document: str) -> str:
        if _HAS_CRYPTOGRAPHY:
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(document.encode())
            doc_hash = digest.finalize()
            signature = self.private_key.sign(doc_hash, ec.ECDSA(hashes.SHA256()))
            return signature.hex()
            return self.private_key.sign(doc_hash, ec.ECDSA(hashes.SHA256())).hex()
        return hmac.new(self._secret, document.encode(), hashlib.sha256).hexdigest()

    # ── Encryption ───────────────────────────────────────────────────────────
    def encrypt_sensitive(self, data: str) -> str:
        try:
            if _HAS_CRYPTOGRAPHY:
                return self.cipher_suite.encrypt(data.encode()).decode()
            return self._stdlib_encrypt(data.encode())
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
        if _HAS_CRYPTOGRAPHY:
            return self.cipher_suite.encrypt(data.encode()).decode()
        return self._stdlib_encrypt(data.encode())

    def decrypt_sensitive(self, encrypted: str) -> str:
        try:
            if _HAS_CRYPTOGRAPHY:
                return self.cipher_suite.decrypt(encrypted.encode()).decode()
            return self._stdlib_decrypt(encrypted)
        except InvalidToken as e:
            logger.error(f"Decryption failed: {e}")
        except InvalidToken:
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise InvalidToken(str(e))
        except Exception as exc:
            raise InvalidToken(str(exc)) from exc

    # ── Stdlib authenticated-encryption envelope ─────────────────────────────
    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
        return base64.urlsafe_b64encode(nonce + tag + cipher).decode()

    def _stdlib_decrypt(self, token: str) -> str:
        raw = base64.urlsafe_b64decode(token.encode())
        if len(raw) < 48:
            raise InvalidToken("Encrypted token is too short.")
        nonce, tag, cipher = raw[:16], raw[16:48], raw[48:]
        expected = hmac.new(self._secret, nonce + cipher, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise InvalidToken("Authentication tag mismatch")
            raise InvalidToken("Authentication tag mismatch.")
        plaintext = bytes(b ^ k for b, k in zip(cipher, self._keystream(nonce, len(cipher))))
        return plaintext.decode()


@st.cache_resource
def get_crypto_service() -> CryptoSigner:
    return CryptoSigner(_resolve_encryption_key())
    return CryptoSigner(get_encryption_key())


# ══════════════════════════════════════════════════════════════════════════
# DATABASE SERVICE (SQLite, replaces async PostgreSQL)
# ══════════════════════════════════════════════════════════════════════════
# ===========================================================================
# DATABASE SERVICE
# ===========================================================================


class DatabaseService:
    """Embedded SQLite store with the original schema (synchronous)."""
    """SQLite-backed production schema for Streamlit."""

    IMMUTABLE_TABLES = ("audit_events", "approval_records", "evidence_artifacts")

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def get_session(self):
        conn = sqlite3.connect(self.db_path)
    def get_session(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            raise
        finally:
            conn.close()

    def _init_schema(self):
    def _init_schema(self) -> None:
        with self.get_session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    actor             TEXT NOT NULL,
                    action            TEXT NOT NULL,
                    model_metrics     TEXT,
                    digital_signature TEXT,
                    tenant_id         TEXT NOT NULL
                    tenant_id         TEXT NOT NULL,
                    created_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_model_time
                    ON audit_events (model_id, timestamp);
                    ON audit_events (tenant_id, model_id, timestamp);

                CREATE TABLE IF NOT EXISTS approval_records (
                    record_id         TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    reason            TEXT,
                    model_metrics     TEXT,
                    digital_signature TEXT,
                    parent_record_id  TEXT,
                    tenant_id         TEXT NOT NULL
                    tenant_id         TEXT NOT NULL,
                    created_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_approval_model_time
                    ON approval_records (model_id, timestamp);
                    ON approval_records (tenant_id, model_id, timestamp);

                CREATE TABLE IF NOT EXISTS evidence_artifacts (
                    artifact_id       TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    created_by        TEXT NOT NULL,
                    digital_signature TEXT NOT NULL,
                    artifact_metadata TEXT,
                    content_encrypted BLOB NOT NULL,
                    tenant_id         TEXT NOT NULL
                    tenant_id         TEXT NOT NULL,
                    created_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_model
                    ON evidence_artifacts (model_id);
                    ON evidence_artifacts (tenant_id, model_id, timestamp);

                CREATE TABLE IF NOT EXISTS policy_evaluations (
                    evaluation_id     TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    policy_name       TEXT NOT NULL,
                    risk_class        TEXT NOT NULL,
                    compliant         INTEGER NOT NULL,
                    violations        TEXT,
                    requirements      TEXT,
                    timestamp         TEXT NOT NULL,
                    tenant_id         TEXT NOT NULL
                    tenant_id         TEXT NOT NULL,
                    created_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_policy_eval_model
                    ON policy_evaluations (model_id);
                    ON policy_evaluations (tenant_id, model_id, timestamp);

                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id             TEXT PRIMARY KEY,
                    model_id               TEXT NOT NULL,
                    created_at             TEXT NOT NULL,
                    status                 TEXT NOT NULL,
                    model_metrics          TEXT NOT NULL,
                    risk_classification    TEXT NOT NULL,
                    deployment_status      TEXT NOT NULL,
                    model_artifact_hash    TEXT NOT NULL UNIQUE,
                    created_by             TEXT NOT NULL,
                    deployment_approved_by TEXT,
                    deployment_approved_at TEXT,
                    tenant_id              TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_tenant
                    ON model_versions (tenant_id, model_id, created_at);
                """
            )

            for table in self.IMMUTABLE_TABLES:
                conn.executescript(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS prevent_{table}_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END;

                    CREATE TRIGGER IF NOT EXISTS prevent_{table}_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END;
                    """
                )

    def count(self, table: str, tenant_id: Optional[str] = None) -> int:
        if tenant_id:
            query = f"SELECT COUNT(*) c FROM {table} WHERE tenant_id = ?"
            params: Tuple[Any, ...] = (tenant_id,)
        else:
            query = f"SELECT COUNT(*) c FROM {table}"
            params = ()
        with self.get_session() as conn:
            return int(conn.execute(query, params).fetchone()["c"])


@st.cache_resource
def get_db_service() -> DatabaseService:
    return DatabaseService(Config.DB_PATH)


# ══════════════════════════════════════════════════════════════════════════
# AUDIT SERVICE (event-sourced, append-only)
# ══════════════════════════════════════════════════════════════════════════
# ===========================================================================
# GOVERNANCE SERVICES
# ===========================================================================

class AuditService:

class AuditService:
    @staticmethod
    def log_event(conn, event_type, model_id, actor, action,
                  model_metrics=None, digital_signature=None,
                  tenant_id="default") -> str:
    def log_event(
        conn: sqlite3.Connection,
        event_type: str,
        model_id: str,
        actor: str,
        action: str,
        model_metrics: Optional[Dict[str, Any]] = None,
        digital_signature: Optional[str] = None,
        tenant_id: str = "default",
    ) -> str:
        event_id = str(uuid.uuid4())
        timestamp = dt.utcnow().isoformat()
        timestamp = _utcnow().isoformat()
        conn.execute(
            """INSERT INTO audit_events
               (event_id, timestamp, event_type, model_id, actor, action,
                model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, timestamp, event_type, model_id, actor, action,
             json.dumps(model_metrics) if model_metrics else None,
             digital_signature, tenant_id),
                model_metrics, digital_signature, tenant_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                timestamp,
                event_type,
                model_id,
                actor,
                action,
                _json_dumps(model_metrics) if model_metrics else None,
                digital_signature,
                tenant_id,
                timestamp,
            ),
        )
        get_metrics().inc("audit_events_stored_total")
        get_metrics().inc("aurexis_audit_events_stored_total")
        return event_id

    @staticmethod
    def get_audit_trail(conn, model_id, tenant_id="default", limit=1000) -> List[Dict]:
    def get_audit_trail(
        conn: sqlite3.Connection,
        model_id: str,
        tenant_id: str = "default",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT * FROM audit_events
               WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (model_id, tenant_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
        return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════════════════
# POLICY ENGINE SERVICE (declarative)
# ══════════════════════════════════════════════════════════════════════════

class PolicyEngineService:

    POLICIES = {
        "EU_AI_ACT": {
            "name": "EU AI Act High-Risk",
            "rules": {"fairness_max": 0.15, "drift_max": 0.25, "risk_score_max": 0.60},
        },
    }

    @staticmethod
    def evaluate(conn, model_id, model_metrics, risk_class,
                 policy_name, tenant_id="default") -> Dict:
    def evaluate(
        conn: sqlite3.Connection,
        model_id: str,
        model_metrics: Dict[str, float],
        risk_class: str,
        policy_name: str,
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        policy = PolicyEngineService.POLICIES.get(policy_name)
        if not policy:
            return {"compliant": True, "policy": policy_name, "violations": []}
            return {"compliant": True, "policy": policy_name, "policy_label": policy_name, "violations": []}

        violations = []
        rules = policy["rules"]
        checks = (
            ("fairness", "fairness_threshold", "reject"),
            ("drift", "drift_threshold", "escalate"),
            ("risk_score", "risk_score_threshold", "escalate"),
        )

        if model_metrics.get("fairness", 0) > rules.get("fairness_max", 1.0):
            violations.append({
                "rule": "fairness_threshold",
                "value": model_metrics["fairness"],
                "threshold": rules["fairness_max"],
                "action": "reject",
            })
        if model_metrics.get("drift", 0) > rules.get("drift_max", 1.0):
            violations.append({
                "rule": "drift_threshold",
                "value": model_metrics["drift"],
                "threshold": rules["drift_max"],
                "action": "escalate",
            })
        if model_metrics.get("risk_score", 0) > rules.get("risk_score_max", 1.0):
            violations.append({
                "rule": "risk_score_threshold",
                "value": model_metrics["risk_score"],
                "threshold": rules["risk_score_max"],
                "action": "escalate",
            })
        violations = []
        for metric, rule_name, action in checks:
            threshold = rules.get(f"{metric}_max", 1.0)
            value = float(model_metrics.get(metric, 0.0))
            if value > threshold:
                violations.append(
                    {
                        "rule": rule_name,
                        "metric": metric,
                        "value": round(value, 4),
                        "threshold": threshold,
                        "action": action,
                    }
                )

        is_compliant = len(violations) == 0
        is_compliant = not violations
        timestamp = _utcnow().isoformat()

        conn.execute(
            """INSERT INTO policy_evaluations
               (evaluation_id, model_id, policy_name, compliant, violations,
                requirements, timestamp, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), model_id, policy_name, int(is_compliant),
             json.dumps(violations),
             json.dumps({"human_oversight": not is_compliant}),
             dt.utcnow().isoformat(), tenant_id),
               (evaluation_id, model_id, policy_name, risk_class, compliant,
                violations, requirements, timestamp, tenant_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                model_id,
                policy_name,
                risk_class,
                int(is_compliant),
                _json_dumps(violations),
                _json_dumps({"human_oversight": not is_compliant, "risk_class": risk_class}),
                timestamp,
                tenant_id,
                timestamp,
            ),
        )

        get_metrics().inc(
            "aurexis_model_evaluations_total",
            labels={"policy": policy_name, "risk_class": risk_class},
        )
        if violations:
            get_metrics().inc("policy_violations_total")
            for violation in violations:
                get_metrics().inc(
                    "aurexis_policy_violations_total",
                    labels={"policy": policy_name, "violation_type": violation["rule"]},
                )

        return {
            "compliant": is_compliant,
            "policy": policy_name,
            "policy_label": policy["name"],
            "violations": violations,
            "requirements": {"human_oversight": not is_compliant},
        }


# ══════════════════════════════════════════════════════════════════════════
# EVIDENCE VAULT SERVICE (chain of custody)
# ══════════════════════════════════════════════════════════════════════════

class EvidenceVaultService:

    @staticmethod
    def store_artifact(conn, crypto, evidence_type, content, model_id,
                       created_by, metadata=None, tenant_id="default") -> Dict:
    def store_artifact(
        conn: sqlite3.Connection,
        crypto: CryptoSigner,
        evidence_type: str,
        content: str,
        model_id: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        artifact_id = str(uuid.uuid4())
        timestamp = dt.utcnow()
        timestamp = _utcnow().isoformat()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        document = f"{artifact_id}{timestamp.isoformat()}{content_hash}"
        document = f"{artifact_id}{timestamp}{content_hash}{tenant_id}"
        signature = crypto.sign_document(document)
        encrypted_content = crypto.encrypt_sensitive(content)

        conn.execute(
            """INSERT INTO evidence_artifacts
               (artifact_id, model_id, evidence_type, content_hash, timestamp,
                created_by, digital_signature, artifact_metadata,
                content_encrypted, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, model_id, evidence_type, content_hash,
             timestamp.isoformat(), created_by, signature,
             json.dumps(metadata) if metadata else None,
             encrypted_content.encode(), tenant_id),
                content_encrypted, tenant_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                model_id,
                evidence_type,
                content_hash,
                timestamp,
                created_by,
                signature,
                _json_dumps(metadata or {}),
                encrypted_content.encode(),
                tenant_id,
                timestamp,
            ),
        )

        get_metrics().inc("evidence_artifacts_stored_total")

        get_metrics().inc("aurexis_evidence_artifacts_stored_total")
        return {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "signature": signature[:32] + "...",
            "timestamp": timestamp.isoformat(),
            "timestamp": timestamp,
        }

    @staticmethod
    def get_chain_of_custody(conn, model_id, tenant_id="default") -> List[Dict]:
    def get_chain_of_custody(
        conn: sqlite3.Connection,
        model_id: str,
        tenant_id: str = "default",
    ) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT * FROM evidence_artifacts
            """SELECT artifact_id, evidence_type, content_hash, timestamp,
                      created_by, digital_signature
               FROM evidence_artifacts
               WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp ASC""",
            (model_id, tenant_id),
        ).fetchall()
        return [
            {
                "artifact_id": r["artifact_id"],
                "evidence_type": r["evidence_type"],
                "content_hash": r["content_hash"],
                "timestamp": r["timestamp"],
                "created_by": r["created_by"],
                "signature": (r["digital_signature"] or "")[:32] + "...",
                "artifact_id": row["artifact_id"],
                "evidence_type": row["evidence_type"],
                "content_hash": row["content_hash"],
                "timestamp": row["timestamp"],
                "created_by": row["created_by"],
                "signature": (row["digital_signature"] or "")[:32] + "...",
            }
            for r in rows
            for row in rows
        ]

    @staticmethod
    def get_artifact_content(
        conn: sqlite3.Connection,
        crypto: CryptoSigner,
        artifact_id: str,
        tenant_id: str = "default",
    ) -> Optional[str]:
        row = conn.execute(
            """SELECT content_encrypted FROM evidence_artifacts
               WHERE artifact_id = ? AND tenant_id = ?""",
            (artifact_id, tenant_id),
        ).fetchone()
        if not row:
            return None
        return crypto.decrypt_sensitive(row["content_encrypted"].decode())

# ══════════════════════════════════════════════════════════════════════════
# APPROVAL WORKFLOW SERVICE (RBAC)
# ══════════════════════════════════════════════════════════════════════════

class ApprovalWorkflowService:
    ALLOWED_DECISIONS = {"approved", "rejected", "changes_requested", "escalated"}

    @staticmethod
    def submit_approval(conn, crypto, model_id, approver_role, approver_name,
                        decision, reason, model_metrics=None,
                        tenant_id="default") -> str:
    def submit_approval(
        conn: sqlite3.Connection,
        crypto: CryptoSigner,
        model_id: str,
        approver_role: str,
        approver_name: str,
        decision: str,
        reason: str,
        model_metrics: Optional[Dict[str, Any]] = None,
        tenant_id: str = "default",
    ) -> str:
        if decision not in ApprovalWorkflowService.ALLOWED_DECISIONS:
            raise ValueError(f"Unsupported decision: {decision}")

        record_id = str(uuid.uuid4())
        timestamp = dt.utcnow().isoformat()
        document = f"{record_id}{decision}{approver_role}"
        timestamp = _utcnow().isoformat()
        document = f"{record_id}{decision}{approver_role}{approver_name}{tenant_id}"
        signature = crypto.sign_document(document)

        conn.execute(
            """INSERT INTO approval_records
               (record_id, model_id, timestamp, approver_role, approver_name,
                decision, reason, model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, model_id, timestamp, approver_role, approver_name,
             decision, reason, json.dumps(model_metrics or {}),
             signature, tenant_id),
                decision, reason, model_metrics, digital_signature, tenant_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                model_id,
                timestamp,
                approver_role,
                approver_name,
                decision,
                reason,
                _json_dumps(model_metrics or {}),
                signature,
                tenant_id,
                timestamp,
            ),
        )
        get_metrics().inc("approval_decisions_total")
        get_metrics().inc(
            "aurexis_approvals_total",
            labels={"role": approver_role, "decision": decision},
        )
        return record_id

    @staticmethod
    def get_approvals(conn, model_id, tenant_id="default") -> List[Dict]:
    def get_approvals(
        conn: sqlite3.Connection,
        model_id: str,
        tenant_id: str = "default",
    ) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT * FROM approval_records
               WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC""",
            (model_id, tenant_id),
        ).fetchall()
        return [dict(r) for r in rows]
        return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION (JWT)
# ══════════════════════════════════════════════════════════════════════════
class ModelRegistryService:
    @staticmethod
    def upsert_from_evaluation(
        conn: sqlite3.Connection,
        model_id: str,
        model_metrics: Dict[str, Any],
        risk_class: str,
        created_by: str,
        tenant_id: str,
        compliant: bool,
    ) -> str:
        version_id = str(uuid.uuid4())
        timestamp = _utcnow().isoformat()
        artifact_basis = f"{tenant_id}:{model_id}:{timestamp}:{_json_dumps(model_metrics)}"
        artifact_hash = hashlib.sha256(artifact_basis.encode()).hexdigest()
        conn.execute(
            """INSERT INTO model_versions
               (version_id, model_id, created_at, status, model_metrics,
                risk_classification, deployment_status, model_artifact_hash,
                created_by, deployment_approved_by, deployment_approved_at, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                model_id,
                timestamp,
                "evaluated",
                _json_dumps(model_metrics),
                risk_class,
                "eligible" if compliant else "blocked",
                artifact_hash,
                created_by,
                None,
                None,
                tenant_id,
            ),
        )
        return version_id

    @staticmethod
    def recent(conn: sqlite3.Connection, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT version_id, model_id, created_at, status, risk_classification,
                      deployment_status, created_by, model_artifact_hash
               FROM model_versions
               WHERE tenant_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (tenant_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


# ===========================================================================
# AUTHENTICATION, AUTHORIZATION, RATE LIMITING
# ===========================================================================


DEMO_USERS = {
    "demo": {"password": "demo", "role": "Developer"},
    "risk": {"password": "risk", "role": "Risk Officer"},
    "compliance": {"password": "compliance", "role": "Compliance Officer"},
    "approver": {"password": "approver", "role": "Deployment Approver"},
    "admin": {"password": "admin", "role": "Platform Admin"},
}

APPROVAL_ROLES = {"Risk Officer", "Compliance Officer", "Deployment Approver", "Platform Admin"}


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_jwt_token(user_id: str, role: str, tenant_id: str) -> str:
    exp = dt.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)
    now = _utcnow()
    payload = {
        "user_id": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "exp": int(exp.timestamp()),
        "iat": int(dt.utcnow().timestamp()),
        "exp": int((now + timedelta(hours=Config.JWT_EXPIRY_HOURS)).timestamp()),
        "iat": int(now.timestamp()),
    }
    if _HAS_PYJWT:
        return _pyjwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
        token = _pyjwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
        return token.decode() if isinstance(token, bytes) else token

    # Stdlib HS256 JWT fallback.
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()),
        _b64url_encode(_json_dumps(header).encode()),
        _b64url_encode(_json_dumps(payload).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(Config.JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def verify_token(token: str) -> Optional[Dict]:
def verify_token(token: str) -> Optional[Dict[str, Any]]:
    if _HAS_PYJWT:
        try:
            return _pyjwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        except _pyjwt.ExpiredSignatureError:
        except _pyjwt.InvalidTokenError:
            st.error("Invalid token.")
            return None

    # Stdlib HS256 verification.
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = hmac.new(Config.JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(signature_b64), expected):
            st.error("Invalid token.")
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if "exp" in payload and dt.utcnow().timestamp() > payload["exp"]:
        if dt.utcnow().timestamp() > float(payload.get("exp", 0)):
            st.error("Token expired. Please sign in again.")
            return None
        return payload
    except Exception:
        st.error("Invalid token.")
        return None


# Demo user directory (replaces OIDC/SAML integration).
DEMO_USERS = {
    "demo": {"password": "demo", "role": "Developer"},
    "risk": {"password": "risk", "role": "Risk Officer"},
    "compliance": {"password": "compliance", "role": "Compliance Officer"},
    "approver": {"password": "approver", "role": "Deployment Approver"},
}
class RateLimiter:
    @staticmethod
    def check(client_id: str, max_requests: int, period_seconds: int) -> bool:
        bucket_key = f"rate_limit:{client_id}"
        now = time.time()
        if bucket_key not in st.session_state:
            st.session_state[bucket_key] = []

APPROVAL_ROLES = {"Risk Officer", "Compliance Officer", "Deployment Approver"}
        bucket = [stamp for stamp in st.session_state[bucket_key] if now - stamp < period_seconds]
        allowed = len(bucket) < max_requests
        if allowed:
            bucket.append(now)
        st.session_state[bucket_key] = bucket
        if not allowed:
            get_metrics().inc("aurexis_rate_limited_total", labels={"client": client_id})
        return allowed


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class CurrentUser:
    user_id: str
    role: str
    tenant_id: str

    @classmethod
    def from_session(cls) -> Optional["CurrentUser"]:
        user = st.session_state.get("user")
        if not user:
            return None
        return cls(user_id=user["user_id"], role=user["role"], tenant_id=user["tenant_id"])


# ===========================================================================
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════
# ===========================================================================


st.set_page_config(
    page_title=Config.API_TITLE,
    page_icon="🛡️",
    layout="wide",
)


def login_view():
    st.title("🛡️ Aurexis Systems")
    st.caption(f"{Config.API_DESCRIPTION} · v{Config.API_VERSION}")
def render_brand_header() -> None:
    st.markdown(
        """
        <style>
        .aurexis-hero {
            border: 1px solid rgba(120, 144, 180, .35);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            background: linear-gradient(135deg, rgba(7, 20, 45, .96), rgba(20, 52, 86, .90));
            color: white;
            margin-bottom: 1rem;
        }
        .aurexis-hero h1 { margin: 0; font-size: 1.8rem; letter-spacing: .02em; }
        .aurexis-hero p { margin: .35rem 0 0 0; color: rgba(255,255,255,.78); }
        .status-pill {
            display: inline-block;
            padding: .18rem .55rem;
            border-radius: 999px;
            background: rgba(42, 157, 143, .20);
            border: 1px solid rgba(42, 157, 143, .50);
            margin-right: .35rem;
            font-size: .82rem;
        }
        </style>
        <div class="aurexis-hero">
            <h1>AUREXIS SYSTEMS — PRODUCTION v5.5.1 (ENTERPRISE-HARDENED)</h1>
            <p>Distributed AI Governance Operating System · Streamlit operational console</p>
            <p>
                <span class="status-pill">SQLite WORM audit</span>
                <span class="status-pill">JWT + RBAC</span>
                <span class="status-pill">Policy engine</span>
                <span class="status-pill">Evidence vault</span>
                <span class="status-pill">Prometheus-style metrics</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def login_view() -> None:
    render_brand_header()
    st.subheader("Sign in")

    with st.form("login_form"):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="demo", type="password")
        tenant_id = st.text_input("Tenant ID", value="default")
        col1, col2, col3 = st.columns([1, 1, 1])
        username = col1.text_input("Username", value="demo")
        password = col2.text_input("Password", value="demo", type="password")
        tenant_id = col3.text_input("Tenant ID", value="default")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        user = DEMO_USERS.get(username)
        if user and user["password"] == password:
        if user and hmac.compare_digest(user["password"], password):
            token = create_jwt_token(username, user["role"], tenant_id)
            st.session_state["token"] = token
            st.session_state["user"] = {
                "user_id": username,
                "role": user["role"],
                "tenant_id": tenant_id,
            }
            get_metrics().inc("aurexis_logins_total", labels={"role": user["role"]})
            st.rerun()
        else:
            st.error("Invalid credentials.")

            "|---|---|---|\n"
            "| `demo` | `demo` | Developer |\n"
            "| `risk` | `risk` | Risk Officer |\n"
            "| `compliance` | `compliance` | Compliance Officer |\n"
            "| `approver` | `approver` | Deployment Approver |"
            "| `approver` | `approver` | Deployment Approver |\n"
            "| `admin` | `admin` | Platform Admin |"
        )


def sidebar(user: Dict):
def sidebar(user: CurrentUser) -> str:
    with st.sidebar:
        st.markdown("### 🛡️ Aurexis Systems")
        st.markdown(f"**User:** {user['user_id']}")
        st.markdown(f"**Role:** {user['role']}")
        st.markdown(f"**Tenant:** {user['tenant_id']}")
        st.markdown(f"**User:** {user.user_id}")
        st.markdown(f"**Role:** {user.role}")
        st.markdown(f"**Tenant:** {user.tenant_id}")
        st.divider()
        page = st.radio(
            "Navigation",
            [
                "Submit Approval",
                "Upload Evidence",
                "Audit Trail",
                "Chain of Custody",
                "Model Registry",
                "Metrics",
                "System Health",
            ],
        )
        st.divider()
        backend = get_crypto_service().backend_name
        if backend == "cryptography":
            st.caption("Crypto: ECDSA + Fernet (cryptography)")
        else:
            st.caption("Crypto: HMAC-SHA256 (stdlib fallback)")
        crypto_health = get_crypto_service().health()
        st.caption(f"Crypto: {crypto_health['signing']} + {crypto_health['encryption']}")
        st.caption(f"Storage: {Config.DATABASE_PROFILE}")
        if st.button("Sign out"):
            for key in ("token", "user"):
                st.session_state.pop(key, None)
            st.rerun()
    return page


def dashboard_view(user: Dict):
def _status_badge(compliant: bool) -> str:
    return "✅ Compliant" if compliant else "❌ Violation"


def dashboard_view(user: CurrentUser) -> None:
    st.header("Governance Dashboard")
    st.caption("Tenant-isolated operational view of governance activity.")

    db = get_db_service()
    metrics = get_metrics()

    with db.get_session() as conn:
        audit_count = conn.execute("SELECT COUNT(*) c FROM audit_events").fetchone()["c"]
        eval_count = conn.execute("SELECT COUNT(*) c FROM policy_evaluations").fetchone()["c"]
        evidence_count = conn.execute("SELECT COUNT(*) c FROM evidence_artifacts").fetchone()["c"]
        approval_count = conn.execute("SELECT COUNT(*) c FROM approval_records").fetchone()["c"]
        violation_count = conn.execute(
            "SELECT COUNT(*) c FROM policy_evaluations WHERE compliant = 0"
        ).fetchone()["c"]
    audit_count = db.count("audit_events", user.tenant_id)
    eval_count = db.count("policy_evaluations", user.tenant_id)
    evidence_count = db.count("evidence_artifacts", user.tenant_id)
    approval_count = db.count("approval_records", user.tenant_id)
    model_count = db.count("model_versions", user.tenant_id)

    c1, c2, c3, c4 = st.columns(4)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Audit Events", audit_count)
    c2.metric("Policy Evaluations", eval_count)
    c3.metric("Evidence Artifacts", evidence_count)
    c2.metric("Evaluations", eval_count)
    c3.metric("Evidence", evidence_count)
    c4.metric("Approvals", approval_count)
    c5.metric("Model Versions", model_count)

    st.divider()
    c5, c6 = st.columns(2)
    c5.metric("Non-Compliant Evaluations", violation_count)
    c6.metric("Policy Violations (session)", metrics.get("policy_violations_total"))

    st.subheader("Recent Policy Evaluations")
    with db.get_session() as conn:
        violation_count = conn.execute(
            """SELECT COUNT(*) c FROM policy_evaluations
               WHERE compliant = 0 AND tenant_id = ?""",
            (user.tenant_id,),
        ).fetchone()["c"]
        rows = conn.execute(
            "SELECT model_id, policy_name, compliant, timestamp "
            "FROM policy_evaluations ORDER BY timestamp DESC LIMIT 20"
            """SELECT model_id, policy_name, risk_class, compliant, timestamp
               FROM policy_evaluations
               WHERE tenant_id = ?
               ORDER BY timestamp DESC
               LIMIT 20""",
            (user.tenant_id,),
        ).fetchall()

    st.divider()
    c6, c7 = st.columns(2)
    c6.metric("Non-Compliant Evaluations", int(violation_count))
    c7.metric("Policy Violations (session)", sum(v for k, v in metrics.snapshot().items() if "policy_violations" in k))

    st.subheader("Recent Policy Evaluations")
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df = pd.DataFrame([dict(row) for row in rows])
        df["compliant"] = df["compliant"].map({1: "✅ Compliant", 0: "❌ Violation"})
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations yet. Run one from the **Evaluate Model** page.")
        st.info("No evaluations yet. Run one from the Evaluate Model page.")


def evaluate_view(user: Dict):
def evaluate_view(user: CurrentUser) -> None:
    st.header("Evaluate Model")
    st.caption("Run model metrics against a declarative governance policy.")
    st.caption("Run model metrics against declarative governance controls.")

    with st.form("eval_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        policy_name = st.selectbox(
            "Policy",
            list(PolicyEngineService.POLICIES.keys()),
            format_func=lambda k: PolicyEngineService.POLICIES[k]["name"],
            format_func=lambda key: PolicyEngineService.POLICIES[key]["name"],
        )
        risk_class = st.selectbox("Risk Class", ["high", "limited", "minimal"])

        c1, c2 = st.columns(2)
        drift = c1.slider("Drift", 0.0, 1.0, 0.10, 0.01)
        fairness = c2.slider("Fairness Gap", 0.0, 1.0, 0.08, 0.01)
        stability = c1.slider("Stability", 0.0, 1.0, 0.90, 0.01)
        risk_score = c2.slider("Risk Score", 0.0, 1.0, 0.40, 0.01)
        uncertainty = c1.slider("Uncertainty", 0.0, 1.0, 0.20, 0.01)

        submitted = st.form_submit_button("Evaluate", type="primary")

    if submitted:
        model_metrics = {
            "drift": drift, "fairness": fairness, "stability": stability,
            "risk_score": risk_score, "uncertainty": uncertainty,
        }
        db = get_db_service()
    if not submitted:
        return

    if not RateLimiter.check(user.user_id, Config.RATE_LIMIT_REQUESTS, Config.RATE_LIMIT_PERIOD):
        st.error("Rate limit exceeded. Please try again later.")
        return

    model_metrics = {
        "drift": drift,
        "fairness": fairness,
        "stability": stability,
        "risk_score": risk_score,
        "uncertainty": uncertainty,
    }

    db = get_db_service()
    crypto = get_crypto_service()
    with instrument("aurexis_policy_evaluation_duration_seconds", {"policy": policy_name}):
        with db.get_session() as conn:
            result = PolicyEngineService.evaluate(
                conn, model_id, model_metrics, risk_class,
                policy_name, user["tenant_id"],
                conn,
                model_id,
                model_metrics,
                risk_class,
                policy_name,
                user.tenant_id,
            )
            signature = crypto.sign_document(f"{model_id}{policy_name}{_json_dumps(model_metrics)}")
            version_id = ModelRegistryService.upsert_from_evaluation(
                conn,
                model_id,
                model_metrics,
                risk_class,
                user.user_id,
                user.tenant_id,
                bool(result["compliant"]),
            )
            AuditService.log_event(
                conn, "model_evaluated", model_id, user["user_id"],
                f"policy_evaluation_{policy_name}", model_metrics,
                tenant_id=user["tenant_id"],
                conn,
                "model_evaluated",
                model_id,
                user.user_id,
                f"policy_evaluation_{policy_name}",
                model_metrics,
                digital_signature=signature,
                tenant_id=user.tenant_id,
            )
        get_metrics().inc("model_evaluations_total")

        if result["compliant"]:
            st.success(f"✅ COMPLIANT with {result.get('policy_label', policy_name)}")
        else:
            st.error(f"❌ {len(result['violations'])} violation(s) against {result.get('policy_label', policy_name)}")
            st.dataframe(pd.DataFrame(result["violations"]),
                         use_container_width=True, hide_index=True)
    if result["compliant"]:
        st.success(f"{_status_badge(True)} with {result['policy_label']} · version `{version_id}`")
    else:
        st.error(f"{_status_badge(False)}: {len(result['violations'])} violation(s) against {result['policy_label']}")
        st.dataframe(pd.DataFrame(result["violations"]), use_container_width=True, hide_index=True)
        st.warning("Deployment status is blocked until required human oversight is completed.")


def approval_view(user: Dict):
def approval_view(user: CurrentUser) -> None:
    st.header("Submit Approval")
    if user["role"] not in APPROVAL_ROLES:
    if user.role not in APPROVAL_ROLES:
        st.warning(
            f"Your role (**{user['role']}**) cannot submit approvals. "
            "Sign in as `risk`, `compliance`, or `approver`."
            f"Your role ({user.role}) cannot submit approvals. "
            "Sign in as risk, compliance, approver, or admin."
        )
        return

    with st.form("approval_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        decision = st.selectbox(
            "Decision",
            ["approved", "rejected", "changes_requested", "escalated"],
        )
        reason = st.text_area(
            "Reason (min 10 chars)",
            value="Reviewed metrics and documentation.",
        )
        decision = st.selectbox("Decision", sorted(ApprovalWorkflowService.ALLOWED_DECISIONS))
        reason = st.text_area("Reason (min 10 chars)", value="Reviewed metrics and governance evidence.")
        submitted = st.form_submit_button("Submit Decision", type="primary")

    if submitted:
        if len(reason.strip()) < 10:
            st.error("Reason must be at least 10 characters.")
            return

        db = get_db_service()
        crypto = get_crypto_service()
        with db.get_session() as conn:
            record_id = ApprovalWorkflowService.submit_approval(
                conn, crypto, model_id, user["role"], user["user_id"],
                decision, reason, {}, user["tenant_id"],
            )
            AuditService.log_event(
                conn, "approval_submitted", model_id, user["user_id"],
                f"decision_{decision}", tenant_id=user["tenant_id"],
            )
        with instrument("aurexis_approval_duration_seconds", {"decision": decision}):
            with db.get_session() as conn:
                record_id = ApprovalWorkflowService.submit_approval(
                    conn,
                    crypto,
                    model_id,
                    user.role,
                    user.user_id,
                    decision,
                    reason,
                    {},
                    user.tenant_id,
                )
                AuditService.log_event(
                    conn,
                    "approval_submitted",
                    model_id,
                    user.user_id,
                    f"decision_{decision}",
                    tenant_id=user.tenant_id,
                )
        st.success(f"Decision recorded · record_id `{record_id}` · {decision}")

    st.divider()
    st.subheader("Approval History")
    model_lookup = st.text_input("Look up approvals for Model ID", value="credit-risk-v1")
    if model_lookup:
        db = get_db_service()
        with db.get_session() as conn:
            rows = ApprovalWorkflowService.get_approvals(
                conn, model_lookup, user["tenant_id"]
            )
            rows = ApprovalWorkflowService.get_approvals(conn, model_lookup, user.tenant_id)
        if rows:
            df = pd.DataFrame(rows)[
                ["timestamp", "approver_role", "approver_name", "decision", "reason"]
            ]
            df = pd.DataFrame(rows)[["timestamp", "approver_role", "approver_name", "decision", "reason"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No approvals found for this model.")


def evidence_view(user: Dict):
def evidence_view(user: CurrentUser) -> None:
    st.header("Upload Evidence")
    st.caption("Artifacts are SHA-256 hashed, ECDSA-signed, and Fernet-encrypted.")
    crypto_health = get_crypto_service().health()
    st.caption(f"Artifacts are SHA-256 hashed, {crypto_health['signing']} signed, and {crypto_health['encryption']} encrypted.")

    with st.form("evidence_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        evidence_type = st.selectbox(
            "Evidence Type",
            ["model_card", "test_report", "fairness_audit", "data_lineage", "other"],
            ["model_card", "test_report", "fairness_audit", "data_lineage", "validation_report", "other"],
        )
        content = st.text_area("Content", value="Evidence payload...", height=150)
        content = st.text_area("Content", value="Evidence payload...", height=160)
        submitted = st.form_submit_button("Upload", type="primary")

    if submitted:
        if not content.strip():
            st.error("Content cannot be empty.")
            return
        db = get_db_service()
        crypto = get_crypto_service()
        try:
    if not submitted:
        return

    if not content.strip():
        st.error("Content cannot be empty.")
        return

    db = get_db_service()
    crypto = get_crypto_service()
    try:
        with instrument("aurexis_evidence_upload_duration_seconds", {"evidence_type": evidence_type}):
            with db.get_session() as conn:
                result = EvidenceVaultService.store_artifact(
                    conn, crypto, evidence_type, content, model_id,
                    user["user_id"], {"source": "streamlit-ui"}, user["tenant_id"],
                    conn,
                    crypto,
                    evidence_type,
                    content,
                    model_id,
                    user.user_id,
                    {"source": "streamlit-ui", "classification": "governance-evidence"},
                    user.tenant_id,
                )
                AuditService.log_event(
                    conn, "evidence_uploaded", model_id, user["user_id"],
                    f"evidence_{evidence_type}", tenant_id=user["tenant_id"],
                    conn,
                    "evidence_uploaded",
                    model_id,
                    user.user_id,
                    f"evidence_{evidence_type}",
                    tenant_id=user.tenant_id,
                )
            st.success("Evidence stored securely.")
            st.json(result)
        except sqlite3.IntegrityError:
            st.warning("Identical content already stored (duplicate content hash).")
        st.success("Evidence stored securely.")
        st.json(result)
    except sqlite3.IntegrityError:
        st.warning("Identical content already exists in the evidence vault.")


def audit_trail_view(user: Dict):
def audit_trail_view(user: CurrentUser) -> None:
    st.header("Audit Trail")
    st.caption("Immutable, append-only event log.")
    model_id = st.text_input("Model ID", value="credit-risk-v1")
    if model_id:
        db = get_db_service()
        with db.get_session() as conn:
            trail = AuditService.get_audit_trail(conn, model_id, user["tenant_id"])
        if trail:
            df = pd.DataFrame(trail)[
                ["timestamp", "event_type", "actor", "action"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No audit events for this model yet.")
    if not model_id:
        return

    db = get_db_service()
    with db.get_session() as conn:
        trail = AuditService.get_audit_trail(conn, model_id, user.tenant_id)
    if trail:
        df = pd.DataFrame(trail)[["timestamp", "event_type", "actor", "action", "digital_signature"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events for this model yet.")

def chain_of_custody_view(user: Dict):

def chain_of_custody_view(user: CurrentUser) -> None:
    st.header("Chain of Custody")
    st.caption("Ordered evidence lineage with cryptographic proof.")
    model_id = st.text_input("Model ID", value="credit-risk-v1")
    if model_id:
        db = get_db_service()
        with db.get_session() as conn:
            chain = EvidenceVaultService.get_chain_of_custody(
                conn, model_id, user["tenant_id"]
            )
        if chain:
            st.dataframe(pd.DataFrame(chain), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence artifacts for this model yet.")
    if not model_id:
        return

    db = get_db_service()
    crypto = get_crypto_service()
    with db.get_session() as conn:
        chain = EvidenceVaultService.get_chain_of_custody(conn, model_id, user.tenant_id)

def main():
    # Validate session token if present.
    if "token" in st.session_state:
        payload = verify_token(st.session_state["token"])
        if not payload:
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)

    if "user" not in st.session_state:
        login_view()
    if not chain:
        st.info("No evidence artifacts for this model yet.")
        return

    user = st.session_state["user"]
    page = sidebar(user)

    views = {
        "Dashboard": dashboard_view,
        "Evaluate Model": evaluate_view,
        "Submit Approval": approval_view,
        "Upload Evidence": evidence_view,
        "Audit Trail": audit_trail_view,
        "Chain of Custody": chain_of_custody_view,
    }
    views[page](user)


if __name__ == "__main__":
    main()
# ══════════════════════════════════════════════════════════════════════════
# APPROVAL WORKFLOW SERVICE (RBAC)
# ══════════════════════════════════════════════════════════════════════════

class ApprovalWorkflowService:

    @staticmethod
    def submit_approval(conn, crypto, model_id, approver_role, approver_name,
                        decision, reason, model_metrics=None,
                        tenant_id="default") -> str:
        record_id = str(uuid.uuid4())
        timestamp = dt.utcnow().isoformat()
        document = f"{record_id}{decision}{approver_role}"
        signature = crypto.sign_document(document)

        conn.execute(
            """INSERT INTO approval_records
               (record_id, model_id, timestamp, approver_role, approver_name,
                decision, reason, model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, model_id, timestamp, approver_role, approver_name,
             decision, reason, json.dumps(model_metrics or {}),
             signature, tenant_id),
        )
        get_metrics().inc("approval_decisions_total")
        return record_id

    @staticmethod
    def get_approvals(conn, model_id, tenant_id="default") -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM approval_records
               WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC""",
            (model_id, tenant_id),
        ).fetchall()
        return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION (JWT)
# ══════════════════════════════════════════════════════════════════════════

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(user_id: str, role: str, tenant_id: str) -> str:
    exp = dt.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)
    payload = {
        "user_id": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "exp": int(exp.timestamp()),
        "iat": int(dt.utcnow().timestamp()),
    }
    if _HAS_PYJWT:
        return _pyjwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

    # Stdlib HS256 JWT fallback.
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(Config.JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def verify_token(token: str) -> Optional[Dict]:
    if _HAS_PYJWT:
        try:
            return _pyjwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        except _pyjwt.ExpiredSignatureError:
            st.error("Token expired. Please sign in again.")
            return None
        except _pyjwt.InvalidTokenError:
            st.error("Invalid token.")
            return None

    # Stdlib HS256 verification.
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = hmac.new(Config.JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(signature_b64), expected):
            st.error("Invalid token.")
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if "exp" in payload and dt.utcnow().timestamp() > payload["exp"]:
            st.error("Token expired. Please sign in again.")
            return None
        return payload
    except Exception:
        st.error("Invalid token.")
        return None


# Demo user directory (replaces OIDC/SAML integration).
DEMO_USERS = {
    "demo": {"password": "demo", "role": "Developer"},
    "risk": {"password": "risk", "role": "Risk Officer"},
    "compliance": {"password": "compliance", "role": "Compliance Officer"},
    "approver": {"password": "approver", "role": "Deployment Approver"},
}

APPROVAL_ROLES = {"Risk Officer", "Compliance Officer", "Deployment Approver"}


# ══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=Config.API_TITLE,
    page_icon="🛡️",
    layout="wide",
)


def login_view():
    st.title("🛡️ Aurexis Systems")
    st.caption(f"{Config.API_DESCRIPTION} · v{Config.API_VERSION}")

    st.subheader("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="demo", type="password")
        tenant_id = st.text_input("Tenant ID", value="default")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        user = DEMO_USERS.get(username)
        if user and user["password"] == password:
            token = create_jwt_token(username, user["role"], tenant_id)
            st.session_state["token"] = token
            st.session_state["user"] = {
                "user_id": username,
                "role": user["role"],
                "tenant_id": tenant_id,
            }
            st.rerun()
    st.dataframe(pd.DataFrame(chain), use_container_width=True, hide_index=True)
    artifact_id = st.selectbox("Decrypt evidence artifact", [""] + [row["artifact_id"] for row in chain])
    if artifact_id:
        with db.get_session() as conn:
            try:
                content = EvidenceVaultService.get_artifact_content(conn, crypto, artifact_id, user.tenant_id)
            except InvalidToken:
                st.error("Unable to decrypt artifact with the active encryption key.")
                return
        if content is None:
            st.error("Artifact not found for this tenant.")
        else:
            st.error("Invalid credentials.")
            st.text_area("Decrypted content", content, height=150)

    with st.expander("Demo accounts"):
        st.markdown(
            "| Username | Password | Role |\n"
            "|---|---|---|\n"
            "| `demo` | `demo` | Developer |\n"
            "| `risk` | `risk` | Risk Officer |\n"
            "| `compliance` | `compliance` | Compliance Officer |\n"
            "| `approver` | `approver` | Deployment Approver |"
        )


def sidebar(user: Dict):
    with st.sidebar:
        st.markdown("### 🛡️ Aurexis Systems")
        st.markdown(f"**User:** {user['user_id']}")
        st.markdown(f"**Role:** {user['role']}")
        st.markdown(f"**Tenant:** {user['tenant_id']}")
        st.divider()
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Evaluate Model",
                "Submit Approval",
                "Upload Evidence",
                "Audit Trail",
                "Chain of Custody",
            ],
        )
        st.divider()
        backend = get_crypto_service().backend_name
        if backend == "cryptography":
            st.caption("Crypto: ECDSA + Fernet (cryptography)")
        else:
            st.caption("Crypto: HMAC-SHA256 (stdlib fallback)")
        if st.button("Sign out"):
            for key in ("token", "user"):
                st.session_state.pop(key, None)
            st.rerun()
    return page


def dashboard_view(user: Dict):
    st.header("Governance Dashboard")
def model_registry_view(user: CurrentUser) -> None:
    st.header("Model Registry")
    st.caption("Evaluation-created model versions and deployment eligibility.")
    db = get_db_service()
    metrics = get_metrics()

    with db.get_session() as conn:
        audit_count = conn.execute("SELECT COUNT(*) c FROM audit_events").fetchone()["c"]
        eval_count = conn.execute("SELECT COUNT(*) c FROM policy_evaluations").fetchone()["c"]
        evidence_count = conn.execute("SELECT COUNT(*) c FROM evidence_artifacts").fetchone()["c"]
        approval_count = conn.execute("SELECT COUNT(*) c FROM approval_records").fetchone()["c"]
        violation_count = conn.execute(
            "SELECT COUNT(*) c FROM policy_evaluations WHERE compliant = 0"
        ).fetchone()["c"]
        rows = ModelRegistryService.recent(conn, user.tenant_id)
    if not rows:
        st.info("No model versions yet. Evaluations will appear here.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit Events", audit_count)
    c2.metric("Policy Evaluations", eval_count)
    c3.metric("Evidence Artifacts", evidence_count)
    c4.metric("Approvals", approval_count)

    st.divider()
    c5, c6 = st.columns(2)
    c5.metric("Non-Compliant Evaluations", violation_count)
    c6.metric("Policy Violations (session)", metrics.get("policy_violations_total"))
def metrics_view(user: CurrentUser) -> None:
    st.header("Metrics")
    st.caption("In-process Prometheus-style counters for this Streamlit worker.")
    snapshot = get_metrics().snapshot()
    if snapshot:
        st.dataframe(
            pd.DataFrame([{"metric": key, "value": value} for key, value in snapshot.items()]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No counters recorded in this session yet.")

    st.subheader("Recent Policy Evaluations")
    with db.get_session() as conn:
        rows = conn.execute(
            "SELECT model_id, policy_name, compliant, timestamp "
            "FROM policy_evaluations ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["compliant"] = df["compliant"].map({1: "✅ Compliant", 0: "❌ Violation"})
        st.dataframe(df, use_container_width=True, hide_index=True)
    latencies = get_metrics().latency_snapshot()
    st.subheader("Latency Samples")
    if latencies:
        st.dataframe(pd.DataFrame(latencies), use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations yet. Run one from the **Evaluate Model** page.")
        st.info("No latency samples recorded yet.")


def evaluate_view(user: Dict):
    st.header("Evaluate Model")
    st.caption("Run model metrics against a declarative governance policy.")
def system_health_view(user: CurrentUser) -> None:
    st.header("System Health")
    st.caption("Streamlit-compatible status for production-hardened services.")
    db = get_db_service()
    crypto = get_crypto_service()

    with st.form("eval_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        policy_name = st.selectbox(
            "Policy",
            list(PolicyEngineService.POLICIES.keys()),
            format_func=lambda k: PolicyEngineService.POLICIES[k]["name"],
        )
        risk_class = st.selectbox("Risk Class", ["high", "limited", "minimal"])

        c1, c2 = st.columns(2)
        drift = c1.slider("Drift", 0.0, 1.0, 0.10, 0.01)
        fairness = c2.slider("Fairness Gap", 0.0, 1.0, 0.08, 0.01)
        stability = c1.slider("Stability", 0.0, 1.0, 0.90, 0.01)
        risk_score = c2.slider("Risk Score", 0.0, 1.0, 0.40, 0.01)
        uncertainty = c1.slider("Uncertainty", 0.0, 1.0, 0.20, 0.01)

        submitted = st.form_submit_button("Evaluate", type="primary")

    if submitted:
        model_metrics = {
            "drift": drift, "fairness": fairness, "stability": stability,
            "risk_score": risk_score, "uncertainty": uncertainty,
        }
        db = get_db_service()
    checks = []
    try:
        with db.get_session() as conn:
            result = PolicyEngineService.evaluate(
                conn, model_id, model_metrics, risk_class,
                policy_name, user["tenant_id"],
            )
            AuditService.log_event(
                conn, "model_evaluated", model_id, user["user_id"],
                f"policy_evaluation_{policy_name}", model_metrics,
                tenant_id=user["tenant_id"],
            )
        get_metrics().inc("model_evaluations_total")
            conn.execute("SELECT 1").fetchone()
        checks.append({"component": "SQLite database", "status": "healthy", "detail": Config.DB_PATH})
    except Exception as exc:
        checks.append({"component": "SQLite database", "status": "error", "detail": str(exc)})

        if result["compliant"]:
            st.success(f"✅ COMPLIANT with {result.get('policy_label', policy_name)}")
        else:
            st.error(f"❌ {len(result['violations'])} violation(s) against {result.get('policy_label', policy_name)}")
            st.dataframe(pd.DataFrame(result["violations"]),
                         use_container_width=True, hide_index=True)
    crypto_health = crypto.health()
    for key, value in crypto_health.items():
        checks.append({"component": key, "status": "configured", "detail": value})

    checks.extend(
        [
            {"component": "runtime environment", "status": Config.ENV.value, "detail": "Streamlit"},
            {"component": "distributed state", "status": "embedded", "detail": Config.STATE_PROFILE},
            {"component": "message queue", "status": "embedded", "detail": Config.MESSAGE_QUEUE_PROFILE},
            {"component": "PyJWT", "status": "available" if _HAS_PYJWT else "stdlib fallback", "detail": Config.JWT_ALGORITHM},
            {"component": "cryptography", "status": "available" if _HAS_CRYPTOGRAPHY else "stdlib fallback", "detail": ""},
        ]
    )
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

def approval_view(user: Dict):
    st.header("Submit Approval")
    if user["role"] not in APPROVAL_ROLES:
        st.warning(
            f"Your role (**{user['role']}**) cannot submit approvals. "
            "Sign in as `risk`, `compliance`, or `approver`."
        )
        return
    warnings = Config.production_warnings()
    if warnings:
        st.warning("Production configuration warnings:\n\n" + "\n".join(f"- {item}" for item in warnings))
    else:
        st.success("No production configuration warnings detected.")

    with st.form("approval_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        decision = st.selectbox(
            "Decision",
            ["approved", "rejected", "changes_requested", "escalated"],
        )
        reason = st.text_area(
            "Reason (min 10 chars)",
            value="Reviewed metrics and documentation.",
        )
        submitted = st.form_submit_button("Submit Decision", type="primary")

    if submitted:
        if len(reason.strip()) < 10:
            st.error("Reason must be at least 10 characters.")
            return
        db = get_db_service()
        crypto = get_crypto_service()
        with db.get_session() as conn:
            record_id = ApprovalWorkflowService.submit_approval(
                conn, crypto, model_id, user["role"], user["user_id"],
                decision, reason, {}, user["tenant_id"],
            )
            AuditService.log_event(
                conn, "approval_submitted", model_id, user["user_id"],
                f"decision_{decision}", tenant_id=user["tenant_id"],
            )
        st.success(f"Decision recorded · record_id `{record_id}` · {decision}")
def main() -> None:
    get_db_service()

    st.divider()
    st.subheader("Approval History")
    model_lookup = st.text_input("Look up approvals for Model ID", value="credit-risk-v1")
    if model_lookup:
        db = get_db_service()
        with db.get_session() as conn:
            rows = ApprovalWorkflowService.get_approvals(
                conn, model_lookup, user["tenant_id"]
            )
        if rows:
            df = pd.DataFrame(rows)[
                ["timestamp", "approver_role", "approver_name", "decision", "reason"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No approvals found for this model.")


def evidence_view(user: Dict):
    st.header("Upload Evidence")
    st.caption("Artifacts are SHA-256 hashed, signed, and encrypted.")

    with st.form("evidence_form"):
        model_id = st.text_input("Model ID", value="credit-risk-v1")
        evidence_type = st.selectbox(
            "Evidence Type",
            ["model_card", "test_report", "fairness_audit", "data_lineage", "other"],
        )
        content = st.text_area("Content", value="Evidence payload...", height=150)
        submitted = st.form_submit_button("Upload", type="primary")

    if submitted:
        if not content.strip():
            st.error("Content cannot be empty.")
            return
        db = get_db_service()
        crypto = get_crypto_service()
        try:
            with db.get_session() as conn:
                result = EvidenceVaultService.store_artifact(
                    conn, crypto, evidence_type, content, model_id,
                    user["user_id"], {"source": "streamlit-ui"}, user["tenant_id"],
                )
                AuditService.log_event(
                    conn, "evidence_uploaded", model_id, user["user_id"],
                    f"evidence_{evidence_type}", tenant_id=user["tenant_id"],
                )
            st.success("Evidence stored securely.")
            st.json(result)
        except sqlite3.IntegrityError:
            st.warning("Identical content already stored (duplicate content hash).")


def audit_trail_view(user: Dict):
    st.header("Audit Trail")
    st.caption("Immutable, append-only event log.")
    model_id = st.text_input("Model ID", value="credit-risk-v1")
    if model_id:
        db = get_db_service()
        with db.get_session() as conn:
            trail = AuditService.get_audit_trail(conn, model_id, user["tenant_id"])
        if trail:
            df = pd.DataFrame(trail)[
                ["timestamp", "event_type", "actor", "action"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No audit events for this model yet.")


def chain_of_custody_view(user: Dict):
    st.header("Chain of Custody")
    st.caption("Ordered evidence lineage with cryptographic proof.")
    model_id = st.text_input("Model ID", value="credit-risk-v1")
    if model_id:
        db = get_db_service()
        with db.get_session() as conn:
            chain = EvidenceVaultService.get_chain_of_custody(
                conn, model_id, user["tenant_id"]
            )
        if chain:
            st.dataframe(pd.DataFrame(chain), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence artifacts for this model yet.")


def main():
    # Validate session token if present.
    if "token" in st.session_state:
        payload = verify_token(st.session_state["token"])
        if not payload:
        if payload:
            st.session_state["user"] = {
                "user_id": payload["user_id"],
                "role": payload["role"],
                "tenant_id": payload["tenant_id"],
            }
        else:
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)

    if "user" not in st.session_state:
    user = CurrentUser.from_session()
    if user is None:
        login_view()
        return

    user = st.session_state["user"]
    render_brand_header()
    page = sidebar(user)

    views = {
        "Dashboard": dashboard_view,
        "Submit Approval": approval_view,
        "Upload Evidence": evidence_view,
        "Audit Trail": audit_trail_view,
        "Chain of Custody": chain_of_custody_view,
        "Model Registry": model_registry_view,
        "Metrics": metrics_view,
        "System Health": system_health_view,
    }
    views[page](user)

