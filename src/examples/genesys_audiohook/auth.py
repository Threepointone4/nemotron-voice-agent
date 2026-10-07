# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication for Genesys Cloud AudioHook and Audio Connector WebSocket connections.

Genesys Cloud sends an ``X-API-KEY`` header when it opens the WebSocket. If the
Audio Connector integration also has a client secret, Genesys signs the upgrade
request with an HTTP message signature (``Signature`` and ``Signature-Input``)
over ``@request-target``, ``@authority``, the ``audiohook-*`` headers, and
``x-api-key``, using HMAC-SHA256 keyed with the base64-decoded client secret.

Signature checks need the request path and ``Host`` header exactly as Genesys
sent them, so a proxy in front of the server must preserve both. When it cannot,
configure only the API key here and verify signatures at the proxy.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

API_KEY_ENV = "GENESYS_AUDIOHOOK_API_KEY"
CLIENT_SECRET_ENV = "GENESYS_AUDIOHOOK_CLIENT_SECRET"
ALLOW_UNAUTHENTICATED_ENV = "GENESYS_AUDIOHOOK_ALLOW_UNAUTHENTICATED"

SIGNED_COMPONENTS: tuple[str, ...] = (
    "@request-target",
    "@authority",
    "audiohook-organization-id",
    "audiohook-session-id",
    "audiohook-correlation-id",
    "x-api-key",
)
MAX_SIGNATURE_AGE_SECS = 10.0
_CLOCK_SKEW_SECS = 5.0
_MIN_NONCE_LENGTH = 22
_MAX_TRACKED_NONCES = 100_000

_KEY = r"[a-z*][a-z0-9_\-.*]*"
_BARE_ITEM = r'"(?:[^"\\]|\\.)*"|-?\d{1,15}|[A-Za-z*][A-Za-z0-9!#$%&\'*+\-.^_`|~:/]*'
_INPUT_MEMBER = re.compile(rf"\s*({_KEY})=\(([^()]*)\)((?:;{_KEY}(?:=(?:{_BARE_ITEM}))?)*)\s*(?:,|$)")
_SIGNATURE_MEMBER = re.compile(rf"\s*({_KEY})=:([A-Za-z0-9+/]*={{0,2}}):\s*(?:,|$)")
_COMPONENT = re.compile(r'\s*"([^"\\]+)"')
_PARAMETER = re.compile(rf";({_KEY})(?:=({_BARE_ITEM}))?")
_INTEGER = re.compile(r"-?\d+")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


class SignatureError(ValueError):
    """Raised when an AudioHook request signature is missing or invalid."""


@dataclass(frozen=True)
class AudioHookAuthConfig:
    """Credentials expected from Genesys Cloud.

    Attributes:
        api_key: Value Genesys sends in ``X-API-KEY``. Required unless
            ``allow_unauthenticated`` is set.
        client_secret: Base64 client secret. When set, requests must also carry a
            valid HMAC-SHA256 HTTP message signature.
        allow_unauthenticated: Accept connections without credentials. Use only for
            local testing on a private network.
        max_signature_age_secs: Maximum age of the signature ``created`` timestamp.
    """

    api_key: str = ""
    client_secret: str = ""
    allow_unauthenticated: bool = False
    max_signature_age_secs: float = MAX_SIGNATURE_AGE_SECS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AudioHookAuthConfig:
        """Read the configuration from the ``GENESYS_AUDIOHOOK_*`` environment variables."""
        env = os.environ if environ is None else environ
        allow = env.get(ALLOW_UNAUTHENTICATED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            api_key=env.get(API_KEY_ENV, "").strip(),
            client_secret=env.get(CLIENT_SECRET_ENV, "").strip(),
            allow_unauthenticated=allow,
        )


@dataclass(frozen=True)
class AuthResult:
    """Outcome of an authentication check.

    Attributes:
        ok: Whether the connection may proceed.
        reason: Why the connection was rejected, or a warning when it was accepted.
    """

    ok: bool
    reason: str = ""


@dataclass
class NonceCache:
    """Recently accepted signature nonces, used to reject replayed requests."""

    max_entries: int = _MAX_TRACKED_NONCES
    _expiry: dict[str, float] = field(default_factory=dict)

    def add(self, nonce: str, expires_at: float, now: float) -> bool:
        """Record ``nonce`` and return ``False`` if it was already seen and is unexpired."""
        if nonce in self._expiry or len(self._expiry) >= self.max_entries:
            for key, expiry in list(self._expiry.items()):
                if expiry <= now:
                    del self._expiry[key]
        if nonce in self._expiry:
            return False
        if len(self._expiry) >= self.max_entries:
            self._expiry.pop(next(iter(self._expiry)))
        self._expiry[nonce] = expires_at
        return True


_NONCES = NonceCache()


def _parse_dictionary(value: str, pattern: re.Pattern[str]) -> dict[str, re.Match[str]]:
    """Parse the subset of RFC 8941 dictionaries used by AudioHook signatures."""
    members: dict[str, re.Match[str]] = {}
    text = value.strip()
    position = 0
    while position < len(text):
        match = pattern.match(text, position)
        if match is None or match.end() == position:
            raise SignatureError("malformed structured header")
        members[match.group(1)] = match
        position = match.end()
    return members


def _parse_components(raw: str) -> list[str]:
    """Parse the covered component names from a ``Signature-Input`` inner list."""
    components: list[str] = []
    position = 0
    while raw[position:].strip():
        match = _COMPONENT.match(raw, position)
        if match is None:
            raise SignatureError("unsupported signature component syntax")
        components.append(match.group(1))
        position = match.end()
    return components


def _parameter_value(raw: str | None) -> str | int | bool:
    """Decode a structured-field bare item (string, integer, token, or boolean flag)."""
    if raw is None:
        return True
    if raw.startswith('"'):
        return re.sub(r"\\(.)", r"\1", raw[1:-1])
    if _INTEGER.fullmatch(raw):
        return int(raw)
    return raw


def _signature_params(components: Sequence[str], parameters: Sequence[tuple[str, str | None]]) -> str:
    """Serialize the ``@signature-params`` value canonically."""
    serialized = "(" + " ".join(f'"{name}"' for name in components) + ")"
    for key, raw in parameters:
        if raw is None:
            serialized += f";{key}"
        elif _INTEGER.fullmatch(raw):
            serialized += f";{key}={int(raw)}"
        else:
            serialized += f";{key}={raw}"
    return serialized


def _signature_base(
    components: Sequence[str],
    parameters: Sequence[tuple[str, str | None]],
    *,
    headers: Mapping[str, str],
    request_target: str,
    authority: str,
) -> bytes:
    """Build the RFC 9421 signature base for the covered components."""
    lines = []
    for name in components:
        if name == "@request-target":
            value = request_target
        elif name == "@authority":
            value = authority.strip().lower()
        elif name.startswith("@"):
            raise SignatureError(f"unsupported derived component {name}")
        else:
            header = headers.get(name)
            if header is None:
                raise SignatureError(f"missing signed header {name}")
            value = header.strip()
        lines.append(f'"{name}": {value}')
    lines.append(f'"@signature-params": {_signature_params(components, parameters)}')
    return "\n".join(lines).encode()


def _verify_member(
    member: re.Match[str],
    signature_b64: str,
    *,
    headers: Mapping[str, str],
    request_target: str,
    authority: str,
    api_key: str,
    key: bytes,
    now: float,
    max_age_secs: float,
) -> tuple[str, int]:
    """Verify one labeled signature and return its ``(nonce, created)`` values."""
    components = _parse_components(member.group(2))
    missing = [name for name in SIGNED_COMPONENTS if name not in components]
    if missing:
        raise SignatureError(f"signature does not cover {', '.join(missing)}")
    parameters = [(match.group(1), match.group(2)) for match in _PARAMETER.finditer(member.group(3))]
    values = {name: _parameter_value(raw) for name, raw in parameters}
    keyid, alg = values.get("keyid"), values.get("alg")
    nonce, created, expires = values.get("nonce"), values.get("created"), values.get("expires")
    if not isinstance(keyid, str) or not hmac.compare_digest(keyid.encode(), api_key.encode()):
        raise SignatureError("keyid does not match X-API-KEY")
    if alg is not None and alg != "hmac-sha256":
        raise SignatureError(f"unsupported algorithm {alg!r}")
    if not isinstance(nonce, str) or len(nonce) < _MIN_NONCE_LENGTH:
        raise SignatureError("missing or too short nonce")
    if not isinstance(created, int) or isinstance(created, bool):
        raise SignatureError("missing created timestamp")
    if created > now + _CLOCK_SKEW_SECS:
        raise SignatureError("created timestamp is in the future")
    if now - created > max_age_secs:
        raise SignatureError("signature is too old")
    if expires is not None and (not isinstance(expires, int) or isinstance(expires, bool) or expires <= now):
        raise SignatureError("signature has expired")
    base = _signature_base(components, parameters, headers=headers, request_target=request_target, authority=authority)
    try:
        provided = base64.b64decode(signature_b64, validate=True)
    except ValueError as exc:
        raise SignatureError("malformed signature value") from exc
    if not hmac.compare_digest(hmac.new(key, base, hashlib.sha256).digest(), provided):
        raise SignatureError("signature mismatch")
    return nonce, created


def verify_signature(
    headers: Mapping[str, str],
    *,
    request_target: str,
    authority: str,
    api_key: str,
    client_secret: str,
    now: float | None = None,
    max_age_secs: float = MAX_SIGNATURE_AGE_SECS,
    nonces: NonceCache | None = None,
) -> None:
    """Verify the HTTP message signature on a Genesys AudioHook upgrade request.

    Args:
        headers: Request headers with lowercase names.
        request_target: Request path and query, as sent by Genesys.
        authority: ``Host`` header value, as sent by Genesys.
        api_key: Expected API key, which Genesys also uses as the ``keyid``.
        client_secret: Base64-encoded client secret configured in Genesys.
        now: Current UNIX time; defaults to ``time.time()``.
        max_age_secs: Maximum accepted signature age.
        nonces: Replay cache; defaults to a process-wide cache.

    Raises:
        SignatureError: If the signature is missing, stale, replayed, or invalid.
    """
    now = time.time() if now is None else now
    signature_input, signature = headers.get("signature-input", ""), headers.get("signature", "")
    if not signature_input or not signature:
        raise SignatureError("missing Signature or Signature-Input header")
    try:
        key = base64.b64decode(client_secret, validate=True)
    except ValueError as exc:
        raise SignatureError(f"{CLIENT_SECRET_ENV} is not valid base64") from exc
    inputs = _parse_dictionary(signature_input, _INPUT_MEMBER)
    signatures = _parse_dictionary(signature, _SIGNATURE_MEMBER)
    failure = SignatureError("no Signature label matches Signature-Input")
    for label, member in inputs.items():
        if label not in signatures:
            continue
        try:
            nonce, created = _verify_member(
                member,
                signatures[label].group(2),
                headers=headers,
                request_target=request_target,
                authority=authority,
                api_key=api_key,
                key=key,
                now=now,
                max_age_secs=max_age_secs,
            )
        except SignatureError as exc:
            failure = exc
            continue
        if not (nonces or _NONCES).add(nonce, created + max_age_secs + _CLOCK_SKEW_SECS, now):
            raise SignatureError("replayed nonce")
        return
    raise failure


def sign_request(
    headers: Mapping[str, str],
    *,
    request_target: str,
    authority: str,
    api_key: str,
    client_secret: str,
    created: int | None = None,
    nonce: str | None = None,
    expires_in_secs: int = 60,
    label: str = "sig1",
) -> dict[str, str]:
    """Create ``Signature-Input`` and ``Signature`` headers the way Genesys Cloud does.

    Used by the local simulator and tests.

    Args:
        headers: Request headers with lowercase names, including ``x-api-key`` and the
            ``audiohook-*`` headers.
        request_target: Request path and query.
        authority: Host the request is sent to.
        api_key: API key, used as the ``keyid``.
        client_secret: Base64-encoded client secret.
        created: Signature creation time; defaults to now.
        nonce: Unique nonce; defaults to a random value.
        expires_in_secs: Lifetime added to ``created`` for the ``expires`` parameter.
        label: Signature label.

    Returns:
        The two signature headers to add to the request.
    """
    created = int(time.time()) if created is None else created
    nonce = nonce or secrets.token_urlsafe(18)
    parameters = [
        ("keyid", json.dumps(api_key)),
        ("nonce", json.dumps(nonce)),
        ("alg", '"hmac-sha256"'),
        ("created", str(created)),
        ("expires", str(created + expires_in_secs)),
    ]
    base = _signature_base(
        SIGNED_COMPONENTS, parameters, headers=headers, request_target=request_target, authority=authority
    )
    digest = hmac.new(base64.b64decode(client_secret), base, hashlib.sha256).digest()
    return {
        "signature-input": f"{label}={_signature_params(SIGNED_COMPONENTS, parameters)}",
        "signature": f"{label}=:{base64.b64encode(digest).decode()}:",
    }


def authenticate_request(
    headers: Mapping[str, str],
    *,
    request_target: str,
    authority: str,
    config: AudioHookAuthConfig,
    now: float | None = None,
    nonces: NonceCache | None = None,
) -> AuthResult:
    """Authenticate a Genesys AudioHook WebSocket upgrade request.

    Args:
        headers: Request headers (any case).
        request_target: Request path and query.
        authority: ``Host`` header value.
        config: Expected credentials.
        now: Current UNIX time; defaults to ``time.time()``.
        nonces: Replay cache for signature nonces.

    Returns:
        Whether the request is authenticated, with a reason.
    """
    normalized = {str(name).lower(): str(value) for name, value in headers.items()}
    if not config.api_key:
        if config.allow_unauthenticated:
            return AuthResult(True, f"{ALLOW_UNAUTHENTICATED_ENV}=true: accepting an unauthenticated connection")
        return AuthResult(False, f"{API_KEY_ENV} is not set; set {ALLOW_UNAUTHENTICATED_ENV}=true only for local tests")
    if not hmac.compare_digest(normalized.get("x-api-key", "").encode(), config.api_key.encode()):
        return AuthResult(False, "missing or invalid X-API-KEY header")
    if config.client_secret:
        try:
            verify_signature(
                normalized,
                request_target=request_target,
                authority=authority or normalized.get("host", ""),
                api_key=config.api_key,
                client_secret=config.client_secret,
                now=now,
                max_age_secs=config.max_signature_age_secs,
                nonces=nonces,
            )
        except SignatureError as exc:
            return AuthResult(False, f"invalid request signature: {exc}")
    return AuthResult(True)


def request_target_from_scope(scope: Mapping[str, Any]) -> str:
    """Return the ``@request-target`` (raw path and query) of an ASGI connection scope."""
    raw_path = scope.get("raw_path")
    path = raw_path.decode("latin-1") if raw_path else str(scope.get("path", "/"))
    query = scope.get("query_string") or b""
    query = query.decode("latin-1") if isinstance(query, bytes) else str(query)
    return f"{path}?{query}" if query else path


def authenticate_websocket(websocket: Any, config: AudioHookAuthConfig | None = None) -> AuthResult:
    """Authenticate a FastAPI/Starlette WebSocket that claims to be Genesys Cloud."""
    return authenticate_request(
        websocket.headers,
        request_target=request_target_from_scope(websocket.scope),
        authority=websocket.headers.get("host", ""),
        config=config or AudioHookAuthConfig.from_env(),
    )


def unauthorized_disconnect_message(session_id: str | None) -> dict[str, Any]:
    """Build the AudioHook ``disconnect`` message that rejects a connection."""
    return {
        "version": "2",
        "type": "disconnect",
        "seq": 1,
        "clientseq": 0,
        "id": session_id if session_id and _UUID.fullmatch(session_id) else str(uuid.uuid4()),
        "parameters": {"reason": "unauthorized"},
    }


async def reject_unauthorized(websocket: Any) -> None:
    """Send ``disconnect`` with reason ``unauthorized``, then close the WebSocket."""
    message = unauthorized_disconnect_message(websocket.headers.get("audiohook-session-id"))
    with contextlib.suppress(Exception):
        await websocket.send_text(json.dumps(message))
    with contextlib.suppress(Exception):
        await websocket.close(code=1008)
