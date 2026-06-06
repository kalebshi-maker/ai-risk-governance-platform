The file hasn't changed since I last showed it. Let me read the current version to display it exactly as it stands.

Read app.pyRead rest of app.pyThe tool reported the content is already loaded, but I need to display lines 505–937. Let me retrieve that range specifically.

Read remainderHere is the **complete, final, fixed `app.py`** exactly as it stands in your project (937 lines). The first half (lines 1–504) is shown above; here is the remainder (lines 505–937):

```python
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
```

This is the final, complete `app.py` — no further changes were made since the last fix. The critical correction that resolves your `ModuleNotFoundError` is the optional-import block at the top (lines 54–71): `cryptography` and `PyJWT` are wrapped in `try/except` with pure standard-library fallbacks, so the app boots even when those wheels fail to install on Streamlit Cloud. Run it with `streamlit run app.py`, and commit `requirements.txt` at your repo root then reboot the app for the full crypto backend.
