"""
╔═══════════════════════════════════════════════════════════════════════════╗
║     AUREXIS SYSTEMS — STREAMLIT EDITION v5.5.1                              ║
║     Distributed AI Governance Operating System                              ║
║     Run with:  streamlit run app.py                                         ║
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

import pandas as pd
import streamlit as st

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet, InvalidToken
import jwt


# ───────────────────────── CONFIGURATION ─────────────────────────
class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Config:
    ENV = Environment(os.getenv("AUREXIS_ENV", "development"))
    DEBUG = ENV == Environment.DEV
    API_TITLE = "Aurexis Systems v5.5.1"
    API_VERSION = "5.5.1"
    API_DESCRIPTION = "Distributed AI Governance Operating System"
    DB_PATH = os.getenv("AUREXIS_DB_PATH", "aurexis.db")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-only-for-local-testing")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def _resolve_encryption_key() -> bytes:
    raw = Config.ENCRYPTION_KEY
    if raw:
        return raw.encode() if isinstance(raw, str) else raw
    return Fernet.generate_key()


logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger("aurexis")


# ───────────────────────── METRICS ─────────────────────────
class Metrics:
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


# ───────────────────────── CRYPTO (ECDSA + Fernet) ─────────────────────────
class CryptoSigner:
    def __init__(self, encryption_key: bytes):
        self.backend = default_backend()
        self.private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        self.cipher_suite = Fernet(encryption_key)

    def sign_document(self, document: str) -> str:
        digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        digest.update(document.encode())
        doc_hash = digest.finalize()
        signature = self.private_key.sign(doc_hash, ec.ECDSA(hashes.SHA256()))
        return signature.hex()

    def encrypt_sensitive(self, data: str) -> str:
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive(self, encrypted: str) -> str:
        try:
            return self.cipher_suite.decrypt(encrypted.encode()).decode()
        except InvalidToken as e:
            logger.error(f"Decryption failed: {e}")
            raise


@st.cache_resource
def get_crypto_service() -> CryptoSigner:
    return CryptoSigner(_resolve_encryption_key())


# ───────────────────────── DATABASE (SQLite) ─────────────────────────
class DatabaseService:
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
                    event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL, model_id TEXT NOT NULL,
                    actor TEXT NOT NULL, action TEXT NOT NULL,
                    model_metrics TEXT, digital_signature TEXT,
                    tenant_id TEXT NOT NULL );
                CREATE INDEX IF NOT EXISTS idx_audit_model_time
                    ON audit_events (model_id, timestamp);

                CREATE TABLE IF NOT EXISTS approval_records (
                    record_id TEXT PRIMARY KEY, model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, approver_role TEXT NOT NULL,
                    approver_name TEXT NOT NULL, decision TEXT NOT NULL,
                    reason TEXT, model_metrics TEXT, digital_signature TEXT,
                    parent_record_id TEXT, tenant_id TEXT NOT NULL );
                CREATE INDEX IF NOT EXISTS idx_approval_model_time
                    ON approval_records (model_id, timestamp);

                CREATE TABLE IF NOT EXISTS evidence_artifacts (
                    artifact_id TEXT PRIMARY KEY, model_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL, created_by TEXT NOT NULL,
                    digital_signature TEXT NOT NULL, artifact_metadata TEXT,
                    content_encrypted BLOB NOT NULL, tenant_id TEXT NOT NULL );
                CREATE INDEX IF NOT EXISTS idx_evidence_model
                    ON evidence_artifacts (model_id);

                CREATE TABLE IF NOT EXISTS policy_evaluations (
                    evaluation_id TEXT PRIMARY KEY, model_id TEXT NOT NULL,
                    policy_name TEXT NOT NULL, compliant INTEGER NOT NULL,
                    violations TEXT, requirements TEXT, timestamp TEXT NOT NULL,
                    tenant_id TEXT NOT NULL );
                CREATE INDEX IF NOT EXISTS idx_policy_eval_model
                    ON policy_evaluations (model_id);
                """
            )


@st.cache_resource
def get_db_service() -> DatabaseService:
    return DatabaseService(Config.DB_PATH)


# ───────────────────────── AUDIT (append-only) ─────────────────────────
class AuditService:
    @staticmethod
    def log_event(conn, event_type, model_id, actor, action,
                  model_metrics=None, digital_signature=None, tenant_id="default") -> str:
        event_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO audit_events
               (event_id, timestamp, event_type, model_id, actor, action,
                model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, dt.utcnow().isoformat(), event_type, model_id, actor, action,
             json.dumps(model_metrics) if model_metrics else None,
             digital_signature, tenant_id),
        )
        get_metrics().inc("audit_events_stored_total")
        return event_id

    @staticmethod
    def get_audit_trail(conn, model_id, tenant_id="default", limit=1000) -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM audit_events WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (model_id, tenant_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ───────────────────────── POLICY ENGINE ─────────────────────────
class PolicyEngineService:
    POLICIES = {
        "EU_AI_ACT": {"name": "EU AI Act High-Risk",
                      "rules": {"fairness_max": 0.15, "drift_max": 0.25, "risk_score_max": 0.60}},
        "US_BANKING_SR11_7": {"name": "SR 11-7 Model Risk Management",
                              "rules": {"fairness_max": 0.10, "drift_max": 0.20, "risk_score_max": 0.50}},
        "ISO_42001": {"name": "ISO/IEC 42001",
                      "rules": {"fairness_max": 0.20, "drift_max": 0.30, "risk_score_max": 0.70}},
    }

    @staticmethod
    def evaluate(conn, model_id, model_metrics, risk_class, policy_name, tenant_id="default") -> Dict:
        policy = PolicyEngineService.POLICIES.get(policy_name)
        if not policy:
            return {"compliant": True, "policy": policy_name, "violations": []}

        violations = []
        rules = policy["rules"]
        if model_metrics.get("fairness", 0) > rules.get("fairness_max", 1.0):
            violations.append({"rule": "fairness_threshold", "value": model_metrics["fairness"],
                               "threshold": rules["fairness_max"], "action": "reject"})
        if model_metrics.get("drift", 0) > rules.get("drift_max", 1.0):
            violations.append({"rule": "drift_threshold", "value": model_metrics["drift"],
                               "threshold": rules["drift_max"], "action": "escalate"})
        if model_metrics.get("risk_score", 0) > rules.get("risk_score_max", 1.0):
            violations.append({"rule": "risk_score_threshold", "value": model_metrics["risk_score"],
                               "threshold": rules["risk_score_max"], "action": "escalate"})

        is_compliant = len(violations) == 0
        conn.execute(
            """INSERT INTO policy_evaluations
               (evaluation_id, model_id, policy_name, compliant, violations,
                requirements, timestamp, tenant_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), model_id, policy_name, int(is_compliant),
             json.dumps(violations), json.dumps({"human_oversight": not is_compliant}),
             dt.utcnow().isoformat(), tenant_id),
        )
        if violations:
            get_metrics().inc("policy_violations_total")
        return {"compliant": is_compliant, "policy": policy_name,
                "policy_label": policy["name"], "violations": violations}


# ───────────────────────── EVIDENCE VAULT ─────────────────────────
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
                content_encrypted, tenant_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, model_id, evidence_type, content_hash, timestamp.isoformat(),
             created_by, signature, json.dumps(metadata) if metadata else None,
             encrypted_content.encode(), tenant_id),
        )
        get_metrics().inc("evidence_artifacts_stored_total")
        return {"artifact_id": artifact_id, "content_hash": content_hash,
                "signature": signature[:32] + "...", "timestamp": timestamp.isoformat()}

    @staticmethod
    def get_chain_of_custody(conn, model_id, tenant_id="default") -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM evidence_artifacts WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp ASC""", (model_id, tenant_id),
        ).fetchall()
        return [{"artifact_id": r["artifact_id"], "evidence_type": r["evidence_type"],
                 "content_hash": r["content_hash"], "timestamp": r["timestamp"],
                 "created_by": r["created_by"],
                 "signature": (r["digital_signature"] or "")[:32] + "..."} for r in rows]


# ───────────────────────── APPROVAL WORKFLOW (RBAC) ─────────────────────────
class ApprovalWorkflowService:
    @staticmethod
    def submit_approval(conn, crypto, model_id, approver_role, approver_name,
                        decision, reason, model_metrics=None, tenant_id="default") -> str:
        record_id = str(uuid.uuid4())
        document = f"{record_id}{decision}{approver_role}"
        signature = crypto.sign_document(document)
        conn.execute(
            """INSERT INTO approval_records
               (record_id, model_id, timestamp, approver_role, approver_name,
                decision, reason, model_metrics, digital_signature, tenant_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, model_id, dt.utcnow().isoformat(), approver_role, approver_name,
             decision, reason, json.dumps(model_metrics or {}), signature, tenant_id),
        )
        get_metrics().inc("approval_decisions_total")
        return record_id

    @staticmethod
    def get_approvals(conn, model_id, tenant_id="default") -> List[Dict]:
        rows = conn.execute(
            """SELECT * FROM approval_records WHERE model_id = ? AND tenant_id = ?
               ORDER BY timestamp DESC""", (model_id, tenant_id),
        ).fetchall()
        return [dict(r) for r in rows]


# ───────────────────────── AUTH (JWT) ─────────────────────────
def create_jwt_token(user_id: str, role: str, tenant_id: str) -> str:
    payload = {"user_id": user_id, "role": role, "tenant_id": tenant_id,
               "exp": dt.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS),
               "iat": dt.utcnow()}
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


def verify_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        st.error("Token expired. Please sign in again.")
        return None
    except jwt.InvalidTokenError:
        st.error("Invalid token.")
        return None


DEMO_USERS = {
    "demo": {"password": "demo", "role": "Developer"},
    "risk": {"password": "risk", "role": "Risk Officer"},
    "compliance": {"password": "compliance", "role": "Compliance Officer"},
    "approver": {"password": "approver", "role": "Deployment Approver"},
}
APPROVAL_ROLES = {"Risk Officer", "Compliance Officer", "Deployment Approver"}


# ───────────────────────── STREAMLIT UI ─────────────────────────
st.set_page_config(page_title=Config.API_TITLE, page_icon="🛡️", layout="wide")
# ... (login_view, sidebar, dashboard_view, evaluate_view, approval_view,
#      evidence_view, audit_trail_view, chain_of_custody_view, main)
