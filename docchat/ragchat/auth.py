"""Identity layer: Clerk session verification + guest/local fallback.

Three identity modes, one stable `user_id` each:

  * clerk : a signed-in Clerk user. The `Authorization: Bearer <jwt>` header
            carries Clerk's session token; we verify its RS256 signature
            against Clerk's JWKS (public keys, fetched once and cached) and
            use the JWT `sub` claim as the user_id. This is the documented
            FastAPI integration (no per-request API call needed; tokens are
            standard RS256 JWTs).
  * guest : a browser-generated UUID sent as `X-Guest-Id`. No account needed;
            the workspace is tied to that device.
  * local : no identity headers — the legacy single-user workspace (user_id "").

Order of precedence: clerk > guest > local.

Failure handling: if Clerk is configured but the token is invalid/expired,
we fall back to guest/local rather than crashing; if JWKS fetch fails we
fall back to Clerk's backend token-verification API when a secret key is
configured, otherwise we treat the token as unverifiable and degrade to
guest/local with a logged warning.
"""
import base64
import json
import logging
import os
import re
import time

import httpx
import jwt
from jwt import PyJWKClient

log = logging.getLogger("jarvis.auth")

# optional config (all via environment variables — never hardcoded)
CLERK_PUBLISHABLE_KEY = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")
CLERK_FRONTEND_API = os.environ.get("CLERK_FRONTEND_API", "").rstrip("/")

_GUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_jwks_client = None
_jwks_checked_at = 0.0
_JWKS_TTL = 3600.0  # re-fetch public keys at most once per hour


class User:
    """A resolved identity. `uid` is the stable per-workspace key used to
    scope every store query; `""` is the local/legacy workspace."""

    __slots__ = ("uid", "source", "name", "email")

    def __init__(self, uid: str, source: str, name: str = "", email: str = ""):
        self.uid = uid
        self.source = source  # "clerk" | "guest" | "local"
        self.name = name
        self.email = email

    def to_dict(self) -> dict:
        return {"uid": self.uid, "source": self.source,
                "name": self.name, "email": self.email}


def clerk_enabled() -> bool:
    """Clerk is only active when a publishable key is configured."""
    return bool(CLERK_PUBLISHABLE_KEY)


def _publishable_domain() -> str:
    """Derive the Clerk frontend API domain from the publishable key if
    CLERK_FRONTEND_API is not set.

    Two encodings exist (see clerk.com/docs/guides/how-clerk-works/overview):
      * current: pk_<base64url('<fapi-url>$')> — the payload is the domain
        string itself, suffixed with a `$` delimiter. e.g. decoding
        `pk_test_ZXhhbXBsZS5hY2NvdW50cy5kZXYk` gives `example.accounts.dev$`.
      * legacy: pk_<base64url(JSON)> where JSON is {"d": "<domain>"}.
    We accept both."""
    if CLERK_FRONTEND_API:
        return CLERK_FRONTEND_API
    key = CLERK_PUBLISHABLE_KEY or ""
    if not key.startswith("pk_"):
        return ""
    try:
        payload = key.split("_", 2)[2]
        payload += "=" * (-len(payload) % 4)  # pad base64url
        raw = base64.urlsafe_b64decode(payload).decode("utf-8", "replace")
        raw = raw.rstrip("$").rstrip("/")
        # legacy JSON form: {"d": "<domain>"}
        if raw.startswith("{"):
            data = json.loads(raw)
            return (data.get("d") or "").rstrip("/")
        return raw  # modern form: the domain string itself
    except Exception:
        return ""


def _jwks_url() -> str:
    domain = _publishable_domain()
    if not domain:
        return ""
    # Clerk hosts instance JWKS at the frontend API root
    base = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"
    return f"{base}/.well-known/jwks.json"


def _get_jwks_client() -> PyJWKClient | None:
    """Cached PyJWKClient for Clerk's public keys. Returns None when Clerk
    is not configured or the JWKS URL cannot be determined."""
    global _jwks_client, _jwks_checked_at
    if not clerk_enabled():
        return None
    url = _jwks_url()
    if not url:
        return None
    now = time.time()
    if _jwks_client is not None and now - _jwks_checked_at < _JWKS_TTL:
        return _jwks_client
    try:
        _jwks_client = PyJWKClient(url, cache_keys=True)
        # force a fetch now so a misconfigured domain fails loudly once
        _jwks_client.fetch_data()
        _jwks_checked_at = now
        return _jwks_client
    except Exception as e:
        log.warning("could not load Clerk JWKS from %s: %s", url, e)
        _jwks_client = None
        return None


def _verify_via_api(token: str) -> dict | None:
    """Fallback verification through Clerk's backend API (needs secret key).
    Returns the session's user payload or None."""
    if not CLERK_SECRET_KEY:
        return None
    try:
        resp = httpx.post(
            "https://api.clerk.com/v1/tokens/verify",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            json={"token": token},
            timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0),
        )
        if resp.status_code != 200:
            log.warning("Clerk token verify API returned %s", resp.status_code)
            return None
        return resp.json()
    except Exception as e:
        log.warning("Clerk token verify API failed: %s", e)
        return None


def verify_clerk_token(token: str) -> User | None:
    """Verify a Clerk session JWT and return the user, or None if invalid."""
    if not clerk_enabled():
        return None
    claims = None
    client = _get_jwks_client()
    if client is not None:
        try:
            claims = jwt.decode(
                token,
                client.get_signing_key_from_jwt(token).key,
                algorithms=["RS256"],
                options={"verify_aud": False},  # audience varies by JWT template
            )
        except jwt.PyJWTError as e:
            log.info("Clerk JWT verify failed locally: %s", e)
    if claims is None:
        data = _verify_via_api(token)
        if data:
            claims = data.get("session") or data
    if not claims:
        return None
    uid = str(claims.get("sub") or "").strip()
    if not uid:
        return None
    name = " ".join(x for x in (
        claims.get("firstName") or claims.get("first_name"),
        claims.get("lastName") or claims.get("last_name"),
    ) if x).strip()
    email = str(claims.get("email") or claims.get("email_address") or "").strip()
    return User(uid=f"clerk:{uid}", source="clerk", name=name or "Clerk user", email=email)


def resolve_user(authorization: str = "", guest_id: str = "") -> User:
    """Resolve identity from request headers. Precedence: clerk > guest > local.
    Never raises: an unverifiable Clerk token degrades to guest/local with a log."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            u = verify_clerk_token(token)
            if u is not None:
                return u
            log.info("unverifiable Clerk token -> degrading to guest/local")
    gid = (guest_id or "").strip()
    if gid and _GUEST_ID_RE.match(gid):
        return User(uid=f"guest:{gid}", source="guest", name="Guest")
    return User(uid="", source="local", name="Local")
