"""Bootstrap helpers for modules that need the local bqapi checkout on sys.path."""

from src.utils import ensure_local_bqapi

ensure_local_bqapi()

LOCAL_BQAPI_READY = True
