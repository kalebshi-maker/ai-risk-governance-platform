"""Authentication for Aurexis Systems.

Provides a self-contained email/password account system backed by the
local database layer, plus the Streamlit login/sign-up gate. Passwords are
stored using PBKDF2-HMAC-SHA256 with a per-user random salt.

For production you can swap this module for a managed identity provider
(Supabase Auth, Auth0, Clerk, ...) without touching the rest of the app, as
long as it returns a ``database.UserRecord`` and sets ``st.session_state``.
"""

import hashlib
import hmac
import os
import re
from typing import Optional, Tuple

import streamlit as st

import database

_PBKDF2_ITERATIONS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LENGTH = 6


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, hash_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


def _validate_credentials(email: str, password: str) -> Optional[str]:
    if not _EMAIL_RE.match((email or "").strip()):
        return "Please enter a valid email address."
    if len(password or "") < _MIN_PASSWORD_LENGTH:
        return f"Password must be at least {_MIN_PASSWORD_LENGTH} characters."
    return None


def signup(email: str, password: str) -> Tuple[bool, str, Optional[database.UserRecord]]:
    error = _validate_credentials(email, password)
    if error:
        return False, error, None
    try:
        record = database.create_user(email.strip().lower(), hash_password(password))
    except ValueError as exc:
        return False, str(exc), None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Could not create account: {exc}", None
    return True, "Account created.", record


def login(email: str, password: str) -> Tuple[bool, str, Optional[database.UserRecord]]:
    if not email or not password:
        return False, "Email and password are required.", None
    credentials = database.get_credentials(email)
    if credentials is None:
        return False, "Invalid email or password.", None
    record, password_hash = credentials
    if not verify_password(password, password_hash):
        return False, "Invalid email or password.", None
    return True, "Signed in.", record


def current_user() -> Optional[database.UserRecord]:
    return st.session_state.get("auth_user")


def set_current_user(record: Optional[database.UserRecord]) -> None:
    st.session_state["auth_user"] = record


def refresh_current_user() -> Optional[database.UserRecord]:
    """Reload the logged-in user's row from the database (e.g. after upgrade)."""
    user = current_user()
    if user is None:
        return None
    fresh = database.get_user(user.id)
    if fresh is not None:
        set_current_user(fresh)
    return fresh


def logout() -> None:
    st.session_state["auth_user"] = None
    st.session_state["messages"] = []


def render_auth_gate() -> database.UserRecord:
    """Render login/sign-up UI. Stops the script until the user is signed in."""
    user = current_user()
    if user is not None:
        return user

    st.subheader("🔐 Sign in to Aurexis Systems")
    st.caption(
        "Create a free account to run governance analyses and chat with the AI advisor. "
        "Free accounts include a limited number of analyses and advisor messages."
    )

    login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            ok, message, record = login(email, password)
            if ok and record is not None:
                set_current_user(record)
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message, record = signup(email, password)
                if ok and record is not None:
                    set_current_user(record)
                    st.success("Account created. Welcome to Aurexis Systems.")
                    st.rerun()
                else:
                    st.error(message)

    st.stop()
