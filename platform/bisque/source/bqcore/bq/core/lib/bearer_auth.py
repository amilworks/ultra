"""Bearer token authentication plugin for repoze.who."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone

import jwt
from paste.httpheaders import AUTHORIZATION  # pylint: disable=no-name-in-module
from paste.deploy.converters import asbool
from repoze.who.interfaces import IAuthenticator, IIdentifier
from zope.interface import implementer
from tg import config

from .oidc_auth import oidc_groups_from_claims, oidc_username_from_claims
from .token_auth import decode_access_token

log = logging.getLogger("bq.auth.bearer")


def _upsert_bq_tag(bq_user, name: str, value: str | None) -> None:
    if not bq_user or not name:
        return
    if value is None:
        return
    try:
        from bq.core.model import DBSession
        from bq.data_service.model.tag_model import Tag

        existing = (
            DBSession.query(Tag)
            .filter(
                Tag.parent == bq_user,
                Tag.resource_name == name,
            )
            .first()
        )
        if existing:
            if existing.resource_value != str(value):
                existing.value = str(value)
            return

        new_tag = Tag(parent=bq_user)
        new_tag.name = name
        new_tag.value = str(value)
        new_tag.owner = bq_user
        DBSession.add(new_tag)
    except Exception:  # noqa: BLE001
        log.debug("Failed to upsert bearer user tag name=%s", name, exc_info=True)


def _ensure_local_oidc_user(username: str, claims: dict) -> str:
    """
    Ensure bearer-authenticated OIDC users exist in local TG/BQ user tables.

    Without this, ``identity.get_user()`` can be None for API-only users,
    which breaks upload/import paths that rely on current user metadata.
    """
    resolved = str(username or "").strip()
    if not resolved:
        return resolved

    try:
        from bq.core.model import DBSession
        from bq.core.model.auth import User as TGUser
        from bq.data_service.model import BQUser

        email = str(claims.get("email") or "").strip() or None
        display_name = str(claims.get("name") or resolved).strip() or resolved

        tg_user = DBSession.query(TGUser).filter_by(user_name=resolved).first()
        if tg_user is None and email:
            tg_user = TGUser.by_email_address(email)

        if tg_user is None:
            base_username = resolved or (email.split("@")[0] if email else "oidc_user")
            candidate = base_username
            counter = 1
            while DBSession.query(TGUser).filter_by(user_name=candidate).first():
                candidate = f"{base_username}_{counter}"
                counter += 1

            tg_user = TGUser(
                user_name=candidate,
                email_address=email or f"{candidate}@local.invalid",
                display_name=display_name or candidate,
                password=f"oidc_auth_{secrets.token_urlsafe(16)}",
            )
            DBSession.add(tg_user)
            DBSession.flush()
            resolved = str(tg_user.user_name or resolved)
            log.info("Bearer auth auto-provisioned local user=%s", resolved)
        else:
            if display_name and tg_user.display_name != display_name:
                tg_user.display_name = display_name
            if email and tg_user.email_address != email:
                tg_user.email_address = email
            DBSession.flush()
            resolved = str(tg_user.user_name or resolved)

        bq_user = DBSession.query(BQUser).filter(BQUser.resource_name == resolved).first()
        if bq_user is not None:
            _upsert_bq_tag(bq_user, "username", resolved)
            _upsert_bq_tag(bq_user, "display_name", display_name)
            _upsert_bq_tag(bq_user, "fullname", display_name)
            if email:
                _upsert_bq_tag(bq_user, "email_verified", "true")
                _upsert_bq_tag(
                    bq_user,
                    "email_verified_at",
                    datetime.now(timezone.utc).isoformat(),
                )
            if asbool(config.get("bisque.oidc.auto_approve_users", False)):
                _upsert_bq_tag(bq_user, "is_approved", "true")
            _upsert_bq_tag(bq_user, "oidc_sub", claims.get("sub"))
            _upsert_bq_tag(bq_user, "oidc_issuer", claims.get("iss"))
            _upsert_bq_tag(bq_user, "oidc_provider", "oidc")

            groups = claims.get("groups") or oidc_groups_from_claims(claims)
            if isinstance(groups, list):
                group_value = ",".join(str(group) for group in groups if group)
                _upsert_bq_tag(bq_user, "oidc_groups", group_value)
            elif isinstance(groups, str):
                _upsert_bq_tag(bq_user, "oidc_groups", groups)
            DBSession.flush()
    except Exception as exc:  # noqa: BLE001
        message = str(exc or "")
        if "No object (name: context)" in message:
            log.debug("Bearer pre-context provisioning deferred for %s", resolved)
        else:
            log.warning("Bearer user provisioning failed for %s: %s", resolved, exc)

    return resolved


@implementer(IIdentifier, IAuthenticator)
class BearerAuthPlugin:
    """Authenticate API requests using ``Authorization: Bearer <token>``."""

    def identify(self, environ):
        authorization = AUTHORIZATION(environ) or ""
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1].strip()
        if not token:
            return None
        return {"bisque.bearer_token": token}

    def remember(self, environ, identity):
        return []

    def forget(self, environ, identity):
        return []

    def authenticate(self, environ, identity):
        token = identity.get("bisque.bearer_token")
        if not token:
            return None

        try:
            claims = decode_access_token(token, verify_exp=True)
        except jwt.ExpiredSignatureError:
            log.warning("Bearer authentication failed: expired token")
            return None
        except jwt.InvalidTokenError as exc:
            log.warning("Bearer authentication failed: invalid token (%s)", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.exception("Bearer authentication failed unexpectedly: %s", exc)
            return None

        username = (
            claims.get("preferred_username")
            or claims.get("sub")
            or oidc_username_from_claims(claims)
        )
        if not username:
            log.warning("Bearer authentication failed: token has no subject")
            return None
        username = _ensure_local_oidc_user(str(username), claims)
        if not username:
            log.warning("Bearer authentication failed: unable to resolve local user")
            return None

        identity["repoze.who.userid"] = username
        identity["bisque.bearer_claims"] = claims
        identity["bisque.auth_type"] = "bearer"

        groups = claims.get("groups") or oidc_groups_from_claims(claims)
        if isinstance(groups, list):
            credentials = environ.setdefault("repoze.what.credentials", {})
            credentials["groups"] = tuple(str(group) for group in groups if group)
        return username


def make_plugin(**kwargs):
    return BearerAuthPlugin()
