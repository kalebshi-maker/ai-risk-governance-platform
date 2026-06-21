"""Usage limits and quota tracking for Aurexis Systems.

Free accounts get a fixed number of governance analyses and AI advisor
messages. Paid plans (``pro`` / ``enterprise``) are unmetered. Limits are
configurable via environment variables so they can be tuned per deployment.
"""

import os

import database

FREE_ANALYSIS_LIMIT = int(os.getenv("AUREXIS_FREE_ANALYSIS_LIMIT", "5"))
FREE_CHAT_LIMIT = int(os.getenv("AUREXIS_FREE_CHAT_LIMIT", "10"))

PAID_PLANS = {"pro", "enterprise"}


def is_pro(user: database.UserRecord) -> bool:
    return bool(user) and user.subscription in PAID_PLANS


def analyses_remaining(user: database.UserRecord) -> int:
    if is_pro(user):
        return -1  # unlimited
    return max(0, FREE_ANALYSIS_LIMIT - (user.analyses_used or 0))


def chat_remaining(user: database.UserRecord) -> int:
    if is_pro(user):
        return -1  # unlimited
    return max(0, FREE_CHAT_LIMIT - (user.chat_used or 0))


def can_run_analysis(user: database.UserRecord) -> bool:
    return is_pro(user) or analyses_remaining(user) > 0


def can_chat(user: database.UserRecord) -> bool:
    return is_pro(user) or chat_remaining(user) > 0


def record_analysis(user: database.UserRecord) -> database.UserRecord:
    """Increment analysis usage for free users; paid users are not metered."""
    if is_pro(user):
        return user
    updated = database.increment_analyses(user.id)
    return updated or user


def record_chat(user: database.UserRecord) -> database.UserRecord:
    if is_pro(user):
        return user
    updated = database.increment_chat(user.id)
    return updated or user
