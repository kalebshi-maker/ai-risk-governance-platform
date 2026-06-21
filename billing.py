"""Stripe billing for Aurexis Systems.

Implements the ChatGPT-style flow: free usage, then a Stripe Checkout
subscription unlocks unlimited access. Payments are processed entirely on
Stripe's hosted, PCI-compliant Checkout page; this app never handles card
data. After a successful payment Stripe redirects back with the checkout
``session_id``, which is verified server-side via the Stripe API before the
account is upgraded (no separate webhook server required for this flow).

If ``STRIPE_SECRET_KEY`` is not configured the pricing page still renders and
clearly explains that live payments are disabled, so the rest of the app and
the free-tier gating remain fully functional and testable.
"""

import os
from typing import Optional

import streamlit as st

import database
import usage

try:
    import stripe

    _STRIPE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    stripe = None
    _STRIPE_AVAILABLE = False


PRO_PRICE_LABEL = os.getenv("AUREXIS_PRO_PRICE_LABEL", "$29/month")
PRO_AMOUNT_CENTS = int(os.getenv("AUREXIS_PRO_AMOUNT_CENTS", "2900"))
PRO_CURRENCY = os.getenv("AUREXIS_PRO_CURRENCY", "usd")
APP_URL = os.getenv("AUREXIS_APP_URL", "http://localhost:8501")


def _secret_key() -> Optional[str]:
    try:
        value = st.secrets.get("STRIPE_SECRET_KEY")
        if value:
            return value
    except Exception:
        pass
    return os.getenv("STRIPE_SECRET_KEY") or None


def stripe_enabled() -> bool:
    return _STRIPE_AVAILABLE and bool(_secret_key())


def _configure_stripe() -> bool:
    key = _secret_key()
    if not (_STRIPE_AVAILABLE and key):
        return False
    stripe.api_key = key
    return True


def _line_item() -> dict:
    """Use a pre-created recurring price if provided, else inline price_data."""
    price_id = os.getenv("STRIPE_PRICE_ID") or _secret_safe("STRIPE_PRICE_ID")
    if price_id:
        return {"price": price_id, "quantity": 1}
    return {
        "price_data": {
            "currency": PRO_CURRENCY,
            "product_data": {"name": "Aurexis Pro Subscription"},
            "unit_amount": PRO_AMOUNT_CENTS,
            "recurring": {"interval": "month"},
        },
        "quantity": 1,
    }


def _secret_safe(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def create_checkout_session(user: database.UserRecord) -> Optional[str]:
    """Create a Stripe Checkout subscription session and return its URL."""
    if not _configure_stripe():
        return None
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[_line_item()],
            customer_email=user.email,
            client_reference_id=str(user.id),
            success_url=f"{APP_URL}?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_URL}?checkout=cancel",
        )
        return session.url
    except Exception as exc:
        st.error(f"Could not start checkout: {exc}")
        return None


def verify_and_apply_checkout(session_id: str) -> Optional[database.UserRecord]:
    """Verify a completed Checkout session and upgrade the matching account."""
    if not _configure_stripe():
        return None
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as exc:
        st.warning(f"Could not verify payment: {exc}")
        return None

    paid = getattr(session, "payment_status", None) == "paid" or getattr(session, "status", None) == "complete"
    if not paid:
        return None

    email = None
    details = getattr(session, "customer_details", None)
    if details is not None:
        email = getattr(details, "email", None) or (details.get("email") if isinstance(details, dict) else None)
    email = email or getattr(session, "customer_email", None)
    if not email:
        return None

    customer_id = getattr(session, "customer", None)
    return database.set_subscription(email, "pro", stripe_customer_id=customer_id)


def render_pricing(user: database.UserRecord, *, context: str = "") -> None:
    """Render pricing tiers and the upgrade call-to-action."""
    st.subheader("💳 Plans & Pricing")
    if context:
        st.warning(context)

    free_col, pro_col, ent_col = st.columns(3)
    with free_col:
        st.markdown("### Free")
        st.markdown("**$0**")
        st.markdown(
            f"- {usage.FREE_ANALYSIS_LIMIT} governance analyses\n"
            f"- {usage.FREE_CHAT_LIMIT} AI advisor messages\n"
            "- Dataset upload & basic reports"
        )
        if not usage.is_pro(user):
            st.info("Your current plan")
    with pro_col:
        st.markdown("### Professional")
        st.markdown(f"**{PRO_PRICE_LABEL}**")
        st.markdown(
            "- Unlimited analyses\n"
            "- Unlimited AI advisor\n"
            "- Governance PDF reports\n"
            "- SHAP explainability\n"
            "- Priority access to updates"
        )
        if usage.is_pro(user):
            st.success("✅ Active — thank you!")
        else:
            _render_upgrade_button(user)
    with ent_col:
        st.markdown("### Enterprise")
        st.markdown("**$199+/month**")
        st.markdown(
            "- Team accounts & SSO\n"
            "- Full audit logs & API access\n"
            "- Compliance features\n"
            "- Dedicated support"
        )
        st.caption("Contact sales@aurexis.example")


def _render_upgrade_button(user: database.UserRecord) -> None:
    if not stripe_enabled():
        st.button("Upgrade to Pro", disabled=True, use_container_width=True, key="upgrade_disabled")
        st.caption(
            "Live payments are not configured in this environment. Set `STRIPE_SECRET_KEY` "
            "(and optionally `STRIPE_PRICE_ID`, `AUREXIS_APP_URL`) to enable Stripe Checkout."
        )
        return

    if st.button("Upgrade to Pro", type="primary", use_container_width=True, key="upgrade_pro"):
        url = create_checkout_session(user)
        if url:
            st.session_state["checkout_url"] = url

    checkout_url = st.session_state.get("checkout_url")
    if checkout_url:
        st.link_button("Continue to secure Stripe checkout →", checkout_url, use_container_width=True)
        st.caption("You'll be redirected to Stripe's secure payment page.")
