"""Stripe webhook endpoint for Aurexis Systems (out-of-band subscription sync).

The Streamlit app already verifies the Checkout session on redirect, so this
service is optional. It exists to handle subscription lifecycle events that
happen *outside* an active browser session — e.g. renewals, cancellations,
or payment failures — by updating the account's subscription in the database.

Run it as a standalone process (it cannot live inside Streamlit, which does
not expose arbitrary HTTP endpoints):

    uvicorn webhook:app --host 0.0.0.0 --port 8502

Required environment variables:
    STRIPE_SECRET_KEY      Your Stripe secret key (sk_test_... / sk_live_...)
    STRIPE_WEBHOOK_SECRET  The signing secret for this webhook endpoint
                           (whsec_...), shown when you create the endpoint in
                           the Stripe dashboard or via the Stripe CLI.

Point a Stripe webhook at:  https://<your-host>/webhook
Subscribed events used:
    checkout.session.completed          -> upgrade account to "pro"
    customer.subscription.deleted       -> downgrade account to "free"
    invoice.payment_failed              -> downgrade account to "free"
"""

import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

import database

try:
    import stripe

    _STRIPE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    stripe = None
    _STRIPE_AVAILABLE = False


app = FastAPI(title="Aurexis Stripe Webhook")


def _webhook_secret() -> Optional[str]:
    return os.getenv("STRIPE_WEBHOOK_SECRET") or None


def _secret_key() -> Optional[str]:
    return os.getenv("STRIPE_SECRET_KEY") or None


@app.on_event("startup")
def _configure() -> None:
    database.init_db()
    if _STRIPE_AVAILABLE and _secret_key():
        stripe.api_key = _secret_key()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "stripe_library": _STRIPE_AVAILABLE,
        "secret_key_configured": bool(_secret_key()),
        "webhook_secret_configured": bool(_webhook_secret()),
    }


@app.post("/webhook")
async def stripe_webhook(request: Request) -> dict:
    if not _STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stripe library is not installed.")

    webhook_secret = _webhook_secret()
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="STRIPE_WEBHOOK_SECRET is not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    try:
        # Verifies the Stripe signature; raises on tampered/forged payloads.
        stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature.")

    # Work with the verified payload as plain JSON (avoids StripeObject quirks).
    event = json.loads(payload)
    event_type = event.get("type", "")
    data = (event.get("data") or {}).get("object") or {}
    handled = _process_event(event_type, data)
    return {"received": True, "event": event_type, "handled": handled}


def _process_event(event_type: str, data: dict) -> bool:
    """Apply a Stripe event to the database. Returns whether it was handled."""
    if event_type == "checkout.session.completed":
        email = (data.get("customer_details") or {}).get("email") or data.get("customer_email")
        customer_id = data.get("customer")
        if email:
            database.set_subscription(email, "pro", stripe_customer_id=customer_id)
            return True
        if customer_id:
            return database.set_subscription_by_customer(customer_id, "pro") is not None
        return False

    if event_type in {"customer.subscription.deleted", "invoice.payment_failed"}:
        customer_id = data.get("customer")
        if customer_id:
            return database.set_subscription_by_customer(customer_id, "free") is not None
        return False

    # Unhandled event types are acknowledged but not acted upon.
    return False
