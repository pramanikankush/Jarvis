"""Signed, expiring URLs for resources fetched without auth headers.

Charts are rendered in <img> tags, which cannot send Authorization or
X-Guest-Id headers. A signature binds the resource id + owner uid + expiry, so
the URL cannot be replayed with a different identity.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

_EPHEMERAL_SECRET = secrets.token_hex(32)


def _secret() -> bytes:
    raw = (os.environ.get("CHART_SIGNING_KEY") or "").strip()
    return (raw or _EPHEMERAL_SECRET).encode("utf-8")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def sign_chart(sheet_id: int, uid: str, ttl: int = 3600) -> str:
    """Return a tamper-evident token for one user's chart."""
    payload = json.dumps(
        {"sid": int(sheet_id), "uid": uid or "", "exp": int(time.time()) + max(1, ttl)},
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    mac = hmac.new(_secret(), payload, hashlib.sha256).hexdigest().encode("ascii")
    return _b64encode(payload + b"\n" + mac)


def verify_chart(token: str) -> dict | None:
    """Return {sid, uid, exp} for a valid, unexpired token; otherwise None."""
    if not token:
        return None
    try:
        payload, mac = _b64decode(token).split(b"\n", 1)
    except (ValueError, TypeError):
        return None
    expected = hmac.new(_secret(), payload, hashlib.sha256).hexdigest().encode("ascii")
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return {"sid": int(data["sid"]), "uid": str(data.get("uid") or ""),
                "exp": int(data["exp"])}
    except (ValueError, TypeError, KeyError):
        return None
