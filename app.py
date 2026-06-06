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
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import sqlite3
import hashlib
import logging
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime as dt, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

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
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet, InvalidToken
    _HAS_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover - exercised only when wheel is missing
    _HAS_CRYPTOGRAPHY = False

    class InvalidToken(Exception):
        """Fallback for cryptography.fernet.InvalidToken."""

try:
    import jwt as _pyjwt
    _HAS_PYJWT = True
except Exception:  # pragma: no cover - exercised only when wheel is missing
    _pyjwt = None
    _HAS_PYJWT = False


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Config:
    """Application configuration (env-driven with safe local defaults)."""

    ENV = Environment(os.getenv("AUREXIS_ENV", "development"))
    DEBUG = ENV == Environment.DEV

    API_TITLE = "Aurexis Systems v5.5.1"
    API_VERSION = "5.5.1"
    API_DESCRIPTION = "Distributed AI Governance Operating System"

    # Embedded SQLite database (replaces PostgreSQL).
    DB_PATH = os.getenv("AUREXIS_DB_PATH", "aurexis.db")

    # Encryption key (Fernet). Generated per-session if not provided.
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

    # JWT
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-only-for-local-testing")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def _resolve_encryption_key() -> bytes:
    """Resolve a valid symmetric key, generating one for local use if needed."""
    raw = Config.ENCRYPTION_KEY
    if raw:
        return raw.encode() if isinstance(raw, str) else raw
    if _HAS_CRYPTOGRAPHY:
        return Fernet.generate_key()
    # Stdlib fallback: 32 random bytes, url-safe base64 encoded (Fernet-like).
    return base64.urlsafe_b64encode(secrets.token_bytes(32))


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger("aurexis")


# ══════════════════════════════════════════════════════════════════════════
# METRICS (lightweight in-process counters; Prometheus replacement)
# ══════════════════════════════════════════════════════════════════════════

class Metrics:
    """Simple thread-local style counters stored in module state."""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, amount: int = 1):
        self.counters[name] += amount

    def get(self, name: str) -> int:
        return self.counters[name]

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)


@st.cache_resource
def get_metrics() -> Metrics:
    return Metrics()


# ══════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHY SERVICE (ECDSA signing + Fernet encryption)
# ══════════════════════════════════════════════════════════════════════════

class CryptoSigner:
    """Document signer + symmetric encryption.

    Uses ECDSA (SECP256R1) + Fernet when `cryptography` is available, and
    transparently falls back to HMAC-SHA256 signatures + an authenticated
    XOR-stream/HMAC envelope built from the standard library otherwise. The
    public method surface is identical for both backends.
    """

    def __init__(self, encryption_key: bytes):
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
    def sign_document(self, document: str) -> str:
        if _HAS_CRYPTOGRAPHY:
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(document.encode())
            doc_hash = digest.finalize()
            signature = self.private_key.sign(doc_hash, ec.ECDSA(hashes.SHA256()))
            return signature.hex()
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

    def decrypt_sensitive(self, encrypted: str) -> str:
        try:
            if _HAS_CRYPTOGRAPHY:
                return self.cipher_suite.decrypt(encrypted.encode()).decode()
            return self._stdlib_decrypt(encrypted)
        except InvalidToken as e:
            logger.error(f"Decryption failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise InvalidToken(str(e))

    # ── Stdlib authenticated-encryption envelope ─────────────────────────────
    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(self._secret + nonce + counter.to_bytes(8, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def _stdlib_encrypt(self, plaintext: bytes) -> str:
        nonce = secrets.token_bytes(16)
        cipher = bytes(b ^ k for b, k in zip(plaintext, self._keystream(nonce, len(plaintext))))
        tag = hmac.new(self._secret, nonce + cipher, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(nonce + tag + cipher).decode()

    def _stdlib_decrypt(self, token: str) -> str:
        raw = base64.urlsafe_b64decode(token.encode())
        nonce, tag, cipher = raw[:16], raw[16:48], raw[48:]
        expected = hmac.new(self._secret, nonce + cipher, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise InvalidToken("Authentication tag mismatch")
        plaintext = bytes(b ^ k for b, k in zip(cipher, self._keystream(nonce, len(cipher))))
        return plaintext.decode()


@st.cache_resource
def get_crypto_service() -> CryptoSigner:
    return CryptoSigner(_resolve_encryption_key())


# ══════════════════════════════════════════════════════════════════════════
# DATABASE SERVICE (SQLite, replaces async PostgreSQL)
# ══════════════════════════════════════════════════════════════════════════

class DatabaseService:
    """Embedded SQLite store with the original schema (synchronous)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def get_session(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self.get_session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id          TEXT PRIMARY KEY,
                    timestamp         TEXT NOT NULL,
                    event_type        TEXT NOT NULL,
                    model_id          TEXT NOT NULL,
                    actor             TEXT NOT NULL,
                    action            TEXT NOT NULL,
                    model_metrics     TEXT,
                    digital_signature TEXT,
                    tenant_id         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_model_time
                    ON audit_events (model_id, timestamp);

                CREATE TABLE IF NOT EXISTS approval_records (
                    record_id         TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    timestamp         TEXT NOT NULL,
                    approver_role     TEXT NOT NULL,
                    approver_name     TEXT NOT NULL,
                    decision          TEXT NOT NULL,
                    reason            TEXT,
                    model_metrics     TEXT,
                    digital_signature TEXT,
                    parent_record_id  TEXT,
                    tenant_id         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_approval_model_time
                    ON approval_records (model_id, timestamp);

                CREATE TABLE IF NOT EXISTS evidence_artifacts (
                    artifact_id       TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    evidence_type     TEXT NOT NULL,
                    content_hash      TEXT NOT NULL UNIQUE,
                    timestamp         TEXT NOT NULL,
                    created_by        TEXT NOT NULL,
                    digital_signature TEXT NOT NULL,
                    artifact_metadata TEXT,
                    content_encrypted BLOB NOT NULL,
                    tenant_id         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_model
                    ON evidence_artifacts (model_id);

                CREATE TABLE IF NOT EXISTS policy_evaluations (
                    evaluation_id     TEXT PRIMARY KEY,
                    model_id          TEXT NOT NULL,
                    policy_name       TEXT NOT NULL,
                    compliant         INTEGER NOT NULL,
                    violations        TEXT,
                    requirements      TEXT,
                    timestamp         TEXT NOT NULL,
                    tenant_id         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_policy_eval_model
                    ON policy_evaluations (model_id);
                """
            )


@st.cache_resource
def get_db_service() -> DatabaseService:
    return DatabaseService(Config.DB_PATH)


# ══════════════════════════════════════════════════════════════════════════
# AUDIT SERVICE (event-sourced, append-only)
# ══════════════════════════════════════════════════════════════════════════

class AuditService:

    @staticmethod
    def log_event(conn, event_type, model_id, actor, action,
                  model_metrics=None, digital_signature=None,
                  tenant_id="default") -> str:
        event_id = str(uuid.uuid4())
        timestamp = dt.utcnow().isoformat()
        conn.execute(
            """INSERT INTO audit_events
               (event_id, timestamp, event_type, model_id, actor, action,
                model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, timestamp, event_type, model_id, actor, action,
             json.dumps(model_metrics) if model_metrics else None,
             digital_signature, tenant_id),
        )
        get_metrics().inc("audit_events_stored_total")
        return event_id

    @staticmethod
    def get_audit_trail(conn, model_id, tenant_id="default", limit=1000) -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM audit_events
               WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (model_id, tenant_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# POLICY ENGINE SERVICE (declarative)
# ══════════════════════════════════════════════════════════════════════════

class PolicyEngineService:

    POLICIES = {
        "EU_AI_ACT": {
            "name": "EU AI Act High-Risk",
            "rules": {"fairness_max": 0.15, "drift_max": 0.25, "risk_score_max": 0.60},
        },
        "US_BANKING_SR11_7": {
            "name": "SR 11-7 Model Risk Management",
            "rules": {"fairness_max": 0.10, "drift_max": 0.20, "risk_score_max": 0.50},
        },
        "ISO_42001": {
            "name": "ISO/IEC 42001",
            "rules": {"fairness_max": 0.20, "drift_max": 0.30, "risk_score_max": 0.70},
        },
    }

    @staticmethod
    def evaluate(conn, model_id, model_metrics, risk_class,
                 policy_name, tenant_id="default") -> Dict:
        policy = PolicyEngineService.POLICIES.get(policy_name)
        if not policy:
            return {"compliant": True, "policy": policy_name, "violations": []}

        violations = []
        rules = policy["rules"]

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

        is_compliant = len(violations) == 0

        conn.execute(
            """INSERT INTO policy_evaluations
               (evaluation_id, model_id, policy_name, compliant, violations,
                requirements, timestamp, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), model_id, policy_name, int(is_compliant),
             json.dumps(violations),
             json.dumps({"human_oversight": not is_compliant}),
             dt.utcnow().isoformat(), tenant_id),
        )

        if violations:
            get_metrics().inc("policy_violations_total")

        return {
            "compliant": is_compliant,
            "policy": policy_name,
            "policy_label": policy["name"],
            "violations": violations,
        }


# ══════════════════════════════════════════════════════════════════════════
# EVIDENCE VAULT SERVICE (chain of custody)
# ══════════════════════════════════════════════════════════════════════════

class EvidenceVaultService:

    @staticmethod
    def store_artifact(conn, crypto, evidence_type, content, model_id,
                       created_by, metadata=None, tenant_id="default") -> Dict:
        artifact_id = str(uuid.uuid4())
        timestamp = dt.utcnow()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        document = f"{artifact_id}{timestamp.isoformat()}{content_hash}"
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
        )

        get_metrics().inc("evidence_artifacts_stored_total")

        return {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "signature": signature[:32] + "...",
            "timestamp": timestamp.isoformat(),
        }

    @staticmethod
    def get_chain_of_custody(conn, model_id, tenant_id="default") -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM evidence_artifacts
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
            }
            for r in rows
        ]


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
        else:
            st.error("Invalid credentials.")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit Events", audit_count)
    c2.metric("Policy Evaluations", eval_count)
    c3.metric("Evidence Artifacts", evidence_count)
    c4.metric("Approvals", approval_count)

    st.divider()
    c5, c6 = st.columns(2)
    c5.metric("Non-Compliant Evaluations", violation_count)
    c6.metric("Policy Violations (session)", metrics.get("policy_violations_total"))

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
    else:
        st.info("No evaluations yet. Run one from the **Evaluate Model** page.")


def evaluate_view(user: Dict):
    st.header("Evaluate Model")
    st.caption("Run model metrics against a declarative governance policy.")

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

        if result["compliant"]:
            st.success(f"✅ COMPLIANT with {result.get('policy_label', policy_name)}")
        else:
            st.error(f"❌ {len(result['violations'])} violation(s) against {result.get('policy_label', policy_name)}")
            st.dataframe(pd.DataFrame(result["violations"]),
                         use_container_width=True, hide_index=True)


def approval_view(user: Dict):
    st.header("Submit Approval")
    if user["role"] not in APPROVAL_ROLES:
        st.warning(
            f"Your role (**{user['role']}**) cannot submit approvals. "
            "Sign in as `risk`, `compliance`, or `approver`."
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
    st.caption("Artifacts are SHA-256 hashed, ECDSA-signed, and Fernet-encrypted.")

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
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)

    if "user" not in st.session_state:
        login_view()
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
        else:
            st.error("Invalid credentials.")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit Events", audit_count)
    c2.metric("Policy Evaluations", eval_count)
    c3.metric("Evidence Artifacts", evidence_count)
    c4.metric("Approvals", approval_count)

    st.divider()
    c5, c6 = st.columns(2)
    c5.metric("Non-Compliant Evaluations", violation_count)
    c6.metric("Policy Violations (session)", metrics.get("policy_violations_total"))

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
    else:
        st.info("No evaluations yet. Run one from the **Evaluate Model** page.")


def evaluate_view(user: Dict):
    st.header("Evaluate Model")
    st.caption("Run model metrics against a declarative governance policy.")

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

        if result["compliant"]:
            st.success(f"✅ COMPLIANT with {result.get('policy_label', policy_name)}")
        else:
            st.error(f"❌ {len(result['violations'])} violation(s) against {result.get('policy_label', policy_name)}")
            st.dataframe(pd.DataFrame(result["violations"]),
                         use_container_width=True, hide_index=True)


def approval_view(user: Dict):
    st.header("Submit Approval")
    if user["role"] not in APPROVAL_ROLES:
        st.warning(
            f"Your role (**{user['role']}**) cannot submit approvals. "
            "Sign in as `risk`, `compliance`, or `approver`."
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
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)

    if "user" not in st.session_state:
        login_view()
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
