from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from lxml import etree

import bqapi.services as bq_services
from bq.engine.commands import module_admin as engine_module_admin
from bq.exceptions import RequestError
from bq.module_service import api as module_api
from bq.module_service.commands import module_admin as server_module_admin
from bq.module_service.controllers import module_server


pytestmark = pytest.mark.unit


class AbortCalled(Exception):
    def __init__(self, status, detail=None):
        super().__init__(status, detail)
        self.status = status
        self.detail = detail


def _abort(status, detail=None):
    raise AbortCalled(status, detail)


def test_read_xml_body_missing_content_type_returns_none(monkeypatch):
    request = SimpleNamespace(headers={"Content-Length": "18"}, body_file=BytesIO(b"<mex name='demo'/>"))
    monkeypatch.setattr(module_server.tg, "request", request, raising=False)
    assert module_server.read_xml_body() is None


def test_read_xml_body_malformed_xml_aborts(monkeypatch):
    request = SimpleNamespace(
        headers={"Content-Length": "8", "Content-Type": "text/xml"},
        body_file=BytesIO(b"<broken>"),
    )
    monkeypatch.setattr(module_server.tg, "request", request, raising=False)
    monkeypatch.setattr(module_server, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        module_server.read_xml_body()
    assert exc.value.status == 400


def test_create_mex_without_formal_inputs_is_still_created():
    module = etree.Element("module", name="demo", uri="http://bisque/module/00-mod")
    mex = module_server.create_mex(module, "demo", image_url="http://bisque/image/1")
    assert mex.tag == "mex"
    inputs = mex.xpath('./tag[@name="inputs"]')
    assert len(inputs) == 1
    assert len(inputs[0]) == 0


def test_create_mex_iterable_missing_inputs_returns_original_mex():
    module = etree.XML(
        """
        <module name="demo" uri="http://bisque/module/00-mod">
          <tag name="inputs">
            <tag name="items" type="list" />
          </tag>
        </module>
        """
    )
    mex = etree.XML(
        """
        <mex>
          <tag name="execute_options">
            <tag name="iterable" value="items" type="list" />
          </tag>
        </mex>
        """
    )

    result = module_server.create_mex(module, "demo", mex=mex)
    assert result is mex
    assert result.xpath("./mex") == []


def test_register_engine_rejects_malformed_xml(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    monkeypatch.setattr(module_server, "abort", _abort)
    monkeypatch.setattr(module_server, "request", SimpleNamespace(method="POST", body=b"<broken"))

    with pytest.raises(AbortCalled) as exc:
        module_server.ModuleServer.register_engine(controller)
    assert exc.value.status == 400


def test_register_engine_rejects_wrong_root(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    monkeypatch.setattr(module_server, "abort", _abort)
    monkeypatch.setattr(module_server, "request", SimpleNamespace(method="POST", body=b"<engine/>"))

    with pytest.raises(AbortCalled) as exc:
        module_server.ModuleServer.register_engine(controller)
    assert exc.value.status == 400


def test_register_engine_accepts_valid_module_definition(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    controller.service_list = {}
    controller.load_services = lambda *args, **kwargs: {}
    definition = etree.Element("module", name="demo")

    monkeypatch.setattr(module_server, "request", SimpleNamespace(method="POST", body=etree.tostring(definition)))
    monkeypatch.setattr(module_server.data_service, "new_resource", lambda resource, **kw: resource)

    result = module_server.ModuleServer.register_engine(controller)
    assert "<module" in result
    assert "demo" in result


def test_module_lookup_missing_aborts_404(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    controller.service_list = {}
    controller.load_services = lambda **kw: {}
    monkeypatch.setattr(module_server, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        module_server.ModuleServer._lookup(controller, "missing")
    assert exc.value.status == 404


def test_unregister_engine_by_resource_uniq_refreshes_cache(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    controller.service_list = {"before": object()}
    refreshed = {"done": False}

    def _load_services(**kwargs):
        refreshed["done"] = True
        return {"after": object()}

    module = etree.Element("module", resource_uniq="00-mod", value="http://engine/module")
    monkeypatch.setattr(module_server.data_service, "resource_load", lambda *args, **kwargs: module)
    monkeypatch.setattr(module_server.data_service, "update", lambda resource, **kwargs: resource)
    controller.load_services = _load_services

    result = module_server.ModuleServer.unregister_engine(controller, resource_uniq="00-mod")
    assert "module" in result
    assert refreshed["done"] is True
    assert "after" in controller.service_list


def test_unregister_engine_legacy_params_resolve_module(monkeypatch):
    controller = object.__new__(module_server.ModuleServer)
    controller.service_list = {}
    controller.load_services = lambda **kw: {}
    modules = etree.XML(
        """
        <resource>
          <module resource_uniq="00-mod" uri="http://bisque/data_service/00-mod" value="http://engine/module" />
        </resource>
        """
    )
    updated = []

    monkeypatch.setattr(module_server.data_service, "query", lambda *args, **kwargs: modules)
    monkeypatch.setattr(module_server.data_service, "update", lambda resource, **kwargs: updated.append(resource) or resource)

    result = module_server.ModuleServer.unregister_engine(
        controller,
        engine_uri="http://engine/module/",
        module_uri="http://bisque/data_service/00-mod",
    )
    assert "00-mod" in result
    assert updated[0].get("value") == ""


def test_engine_resource_new_requires_module_definitions(monkeypatch):
    engine_resource = object.__new__(module_server.EngineResource)
    monkeypatch.setattr(module_server, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        module_server.EngineResource.new(engine_resource, None, "<engine/>")
    assert exc.value.status == 400


def test_register_module_updates_only_when_newer(monkeypatch):
    engine_resource = object.__new__(module_server.EngineResource)
    module_def = etree.XML(
        """
        <module name="demo" value="http://engine/module" ts="2026-03-09T12:00:00">
          <tag name="module_options">
            <tag name="version" value="1.0" />
          </tag>
        </module>
        """
    )
    existing = etree.XML(
        """
        <resource>
          <module name="demo" uri="http://bisque/data_service/00-mod" ts="2026-03-09T11:00:00">
            <tag name="module_options">
              <tag name="version" value="1.0" />
            </tag>
          </module>
        </resource>
        """
    )
    updated = []

    monkeypatch.setattr(module_server.data_service, "query", lambda *args, **kwargs: existing)
    monkeypatch.setattr(
        module_server.data_service,
        "update_resource",
        lambda resource, new_resource, **kwargs: updated.append((resource, new_resource)) or new_resource,
    )
    monkeypatch.setattr(module_server.data_service, "new_resource", lambda resource: resource)

    result = module_server.EngineResource.register_module(engine_resource, module_def)
    assert updated
    assert result.get("uri") == "http://bisque/data_service/00-mod"


def test_register_module_does_not_replace_newer_existing_definition(monkeypatch):
    engine_resource = object.__new__(module_server.EngineResource)
    module_def = etree.XML(
        """
        <module name="demo" value="http://engine/module" ts="2026-03-09T10:00:00">
          <tag name="module_options">
            <tag name="version" value="1.0" />
          </tag>
        </module>
        """
    )
    existing = etree.XML(
        """
        <resource>
          <module name="demo" uri="http://bisque/data_service/00-mod" ts="2026-03-09T11:00:00">
            <tag name="module_options">
              <tag name="version" value="1.0" />
            </tag>
          </module>
        </resource>
        """
    )
    updated = []

    monkeypatch.setattr(module_server.data_service, "query", lambda *args, **kwargs: existing)
    monkeypatch.setattr(
        module_server.data_service,
        "update_resource",
        lambda resource, new_resource, **kwargs: updated.append((resource, new_resource)) or new_resource,
    )
    monkeypatch.setattr(module_server.data_service, "new_resource", lambda resource: resource)

    result = module_server.EngineResource.register_module(engine_resource, module_def)
    assert updated == []
    assert result.get("uri") == "http://bisque/data_service/00-mod"


@pytest.mark.parametrize(
    "helper,args",
    [
        (module_api.uri, ()),
        (module_api.begin_internal_mex, ()),
        (module_api.end_internal_mex, ("00-mex",)),
        (module_api.heartbeat, ("<hb/>",)),
        (module_api.engines, ()),
    ],
)
def test_module_api_helpers_require_server(monkeypatch, helper, args):
    monkeypatch.setattr(module_api.service_registry, "find_service", lambda _name: None)
    with pytest.raises(RequestError):
        helper(*args)


def test_module_proxy_unregister_prefers_resource_uniq(monkeypatch):
    proxy = bq_services.ModuleProxy(
        SimpleNamespace(service_map={"module_service": "http://bisque/module_service/"}, c=SimpleNamespace(headers={})),
        "module_service",
    )
    captured = {}

    def _request(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(proxy, "request", _request)
    proxy.unregister(engine_url="http://engine/module", resource_uniq="00-mod")

    assert captured["params"] == {"resource_uniq": "00-mod"}


def test_server_module_admin_unregister_uses_resource_uniq(monkeypatch):
    monkeypatch.setattr(server_module_admin.sys, "argv", ["module-admin", "unregister", "http://engine/module/", "http://bisque/data_service/00-mod"])
    admin = server_module_admin.module_admin("1.0")
    admin.root = "http://bisque/"
    captured = {}

    class _Response(dict):
        status = "200"

    def _xmlrequest(url, method=None, userpass=None):
        captured["url"] = url
        return _Response(status="200"), ""

    monkeypatch.setattr(server_module_admin.http, "xmlrequest", _xmlrequest)
    admin.unregister_one("http://engine/module/")

    params = parse_qs(urlparse(captured["url"]).query)
    assert params["resource_uniq"] == ["00-mod"]
    assert "engine_uri" not in params


def test_engine_module_admin_unregister_uses_resource_uniq(monkeypatch):
    monkeypatch.setattr(engine_module_admin.sys, "argv", ["module-admin", "unregister", "http://engine/module/", "http://bisque/data_service/00-mod"])
    admin = engine_module_admin.module_admin("1.0")
    admin.root = "http://bisque/"
    captured = {}

    class _Session:
        def fetchxml(self, url, **params):
            captured["url"] = url
            captured["params"] = params
            return etree.Element("resource")

    admin.session = _Session()
    admin.unregister_one("http://engine/module/")

    assert captured["params"] == {"resource_uniq": "00-mod"}
