from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class BisqueAuthContext:
    username: str | None = None
    password: str | None = None
    bisque_root: str | None = None
    access_token: str | None = None
    id_token: str | None = None
    bisque_cookie_header: str | None = None


_REQUEST_BISQUE_AUTH: ContextVar[BisqueAuthContext | None] = ContextVar(
    "request_bisque_auth",
    default=None,
)


def get_request_bisque_auth() -> BisqueAuthContext | None:
    return _REQUEST_BISQUE_AUTH.get()


def set_request_bisque_auth(context: BisqueAuthContext | None) -> Token[BisqueAuthContext | None]:
    return _REQUEST_BISQUE_AUTH.set(context)


def reset_request_bisque_auth(token: Token[BisqueAuthContext | None]) -> None:
    _REQUEST_BISQUE_AUTH.reset(token)
