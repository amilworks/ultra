import base64
import io
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
from PIL import Image

from bq.client_service.controllers import auth_service
from bq.client_service.controllers import client_service
from bq.client_service.controllers import dn_service


pytestmark = pytest.mark.unit


def _png_bytes(mode="RGB", color=(0, 0, 0)):
    image = Image.new(mode, (2, 2), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _data_uri_png(mode="RGBA", color=(0, 0, 0, 0)):
    payload = base64.b64encode(_png_bytes(mode=mode, color=color)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_login_map_defaults_to_local_when_providers_missing(monkeypatch):
    auth_service.AuthenticationServer.providers = {}

    def _config_get(key, default=None):
        if key == "bisque.login.providers":
            return ""
        if key == "bisque.login.local.url":
            return "/auth_service/login"
        return default

    monkeypatch.setattr(auth_service, "config", SimpleNamespace(get=_config_get))
    providers = auth_service.AuthenticationServer.login_map()
    assert "local" in providers
    assert providers["local"]["url"] == "/auth_service/login"


def test_login_check_uses_provider_url_not_encoded_dict(monkeypatch):
    server = auth_service.AuthenticationServer("/auth_service")
    monkeypatch.setattr(server, "login_map", lambda: OrderedDict([("local", {"url": "/auth_service/login"})]))

    def _redirect(location):
        raise RuntimeError(location)

    monkeypatch.setattr(auth_service, "redirect", _redirect)
    with pytest.raises(RuntimeError) as exc:
        server.login_check(came_from="/client_service/", login="")
    location = str(exc.value)
    assert location.startswith("/auth_service/login")
    assert "%7B'url'" not in location


def test_admin_bypass_requires_exact_username_or_admin_group(monkeypatch):
    server = auth_service.AuthenticationServer("/auth_service")
    monkeypatch.setattr(
        auth_service,
        "config",
        SimpleNamespace(get=lambda key, default=None: "admin,administrator" if key == "bisque.auth.admin_bypass_users" else default),
    )
    normal_user = SimpleNamespace(groups=[SimpleNamespace(group_name="users")])
    monkeypatch.setattr(auth_service, "request", SimpleNamespace(identity={"user": normal_user}))

    assert server._is_admin_bypass_user("admin")
    assert server._is_admin_bypass_user("administrator")
    assert not server._is_admin_bypass_user("myadminuser")

    admin_group_user = SimpleNamespace(groups=[SimpleNamespace(group_name="admin")])
    monkeypatch.setattr(auth_service, "request", SimpleNamespace(identity={"user": admin_group_user}))
    assert server._is_admin_bypass_user("custom-user")


def test_client_default_handles_empty_path():
    server = client_service.ClientServer("/client_service")
    assert server._default() is None


def test_client_create_no_nameerror_on_success(monkeypatch):
    server = client_service.ClientServer("/client_service")
    monkeypatch.setattr(client_service.data_service, "new_resource", lambda *_a, **_k: {"uri": "/data_service/00-abc"})
    payload = server.create(type="image")
    assert payload["resource"]["uri"] == "/data_service/00-abc"


def test_mask_receiver_no_mask_path_sets_default_tiff_meta(monkeypatch, tmp_path):
    server = client_service.ClientServer("/client_service")

    request_obj = SimpleNamespace(
        body=json.dumps({"orig_urls": [""], "stack": [_data_uri_png()]}).encode("utf-8"),
        headers={},
        url="http://localhost/client_service/mask_receiver",
    )
    response_obj = SimpleNamespace(status_int=200, headers={})
    monkeypatch.setattr(client_service, "request", request_obj)
    monkeypatch.setattr(client_service.tg, "response", response_obj, raising=False)
    monkeypatch.setattr(client_service, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(client_service.tifffile, "imsave", lambda *_a, **_k: None, raising=False)

    class _FakeTmp(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.name = str(tmp_path / "mask.tiff")

    monkeypatch.setattr(client_service.tempfile, "NamedTemporaryFile", lambda *a, **k: _FakeTmp())
    monkeypatch.setattr(
        client_service.service_registry,
        "find_service",
        lambda _name: SimpleNamespace(transfer_internal=lambda **_kw: "<response><resource uri='/data_service/00-new'/></response>"),
    )

    out = server.mask_receiver()
    assert out == "/data_service/00-new"


def test_mask_receiver_missing_fuse_returns_400(monkeypatch):
    server = client_service.ClientServer("/client_service")
    image_bytes = _png_bytes(mode="RGB", color=(0, 0, 0))

    request_obj = SimpleNamespace(
        body=json.dumps(
            {"orig_urls": ["http://example.org/image_service/00-src"], "stack": [_data_uri_png()]}
        ).encode("utf-8"),
        headers={},
        url="http://localhost/client_service/mask_receiver",
    )
    response_obj = SimpleNamespace(status_int=200, headers={})
    monkeypatch.setattr(client_service, "request", request_obj)
    monkeypatch.setattr(client_service.tg, "response", response_obj, raising=False)

    class _InternalReq:
        def __init__(self, path):
            self.path = path
            self.headers = {}

        def get_response(self, _app):
            return SimpleNamespace(body=image_bytes)

    monkeypatch.setattr(client_service.Request, "blank", lambda path: _InternalReq(path))
    import bq.config.middleware as middleware

    monkeypatch.setattr(middleware, "bisque_app", object())
    out = server.mask_receiver()
    assert response_obj.status_int == 400
    assert "missing 'fuse'" in out


def test_dn_savefile_rejects_path_traversal(monkeypatch, tmp_path):
    server = dn_service.DNServer("/notebook_service")
    response_obj = SimpleNamespace(status_int=200)
    monkeypatch.setattr(dn_service, "response", response_obj)
    monkeypatch.setattr(dn_service, "get_username", lambda: "alice")
    monkeypatch.setattr(dn_service, "anonymous", lambda: False)
    monkeypatch.setattr(
        dn_service,
        "config",
        SimpleNamespace(get=lambda key, default=None: str(tmp_path) if key == "bisque.image_service.upload_dir" else default),
    )
    upload = SimpleNamespace(filename="../escape.tif", file=io.BytesIO(b"abc"))
    out = server.savefile(upload=upload)
    assert response_obj.status_int == 400
    assert "Invalid upload filename" in out


def test_dn_notify_rejects_unsafe_uploaddir(monkeypatch, tmp_path):
    server = dn_service.DNServer("/notebook_service")
    monkeypatch.setattr(dn_service, "get_username", lambda: "alice")

    def _abort(status, message=None):
        raise RuntimeError(status, message)

    monkeypatch.setattr(dn_service, "abort", _abort)
    monkeypatch.setattr(
        dn_service,
        "config",
        SimpleNamespace(get=lambda key, default=None: str(tmp_path) if key == "bisque.image_service.upload_dir" else default),
    )
    with pytest.raises(RuntimeError) as exc:
        server.notify(bixfiles="", imagefiles="", uploaddir="/tmp")
    assert exc.value.args[0] == 400


def test_dn_notify_uses_list_lengths_and_valid_html_join(monkeypatch, tmp_path):
    server = dn_service.DNServer("/notebook_service")
    user_root = tmp_path / "alice"
    user_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dn_service, "get_username", lambda: "alice")
    monkeypatch.setattr(
        dn_service,
        "config",
        SimpleNamespace(
            get=lambda key, default=None: (
                str(tmp_path)
                if key == "bisque.image_service.upload_dir"
                else False
                if key == "bisque.image_service.remove_uploads"
                else default
            )
        ),
    )
    monkeypatch.setattr(dn_service, "abort", lambda *_a, **_k: None)

    class _Importer:
        def __init__(self, upload_dir):
            self.upload_dir = upload_dir

        def process_bix(self, bix):
            return ("included.png", "/data_service/included")

    monkeypatch.setattr(dn_service, "BIXImporter", _Importer)
    html = server.notify(bixfiles="scan.bix", imagefiles="included.png:extra.png")
    assert "Uploaded but not included as images 1" in html
    assert "<table><tr><td>extra.png</td></tr></table>" in html


def test_dn_initialize_sets_global_server():
    dn_service.dn_server = None
    service = dn_service.initialize("/notebook_service")
    assert dn_service.dn_server is service


def test_dn_wrappers_return_explicit_not_initialized_message():
    dn_service.dn_server = None
    assert dn_service.savefile(None) == "<failure msg='not initialized'/>"
    assert dn_service.notify() == "<failure msg='not initialized'/>"
    assert dn_service.test_uploaded() == "<failure msg='not initialized'/>"
