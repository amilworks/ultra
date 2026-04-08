from io import BytesIO
from types import SimpleNamespace

import pytest
from lxml import etree

from bq.engine.controllers import engine_service


pytestmark = pytest.mark.unit


class AbortCalled(Exception):
    def __init__(self, status, detail=None):
        super().__init__(status, detail)
        self.status = status
        self.detail = detail


def _abort(status, detail=None):
    raise AbortCalled(status, detail)


def _module_tree(name, module_dir, *, value=None):
    module = etree.XML(
        """
        <module name="{name}" type="runtime">
          <tag name="inputs">
            <tag name="Input Image" type="resource" />
            <tag name="mex_url" type="system-input_resource" />
            <tag name="bisque_token" type="system-input_resource" />
          </tag>
          <tag name="outputs">
            <tag name="Output Image" type="image" />
          </tag>
        </module>
        """.format(name=name)
    )
    module.set("path", str(module_dir))
    module.set("value", value or name)
    return module


def _make_module_dir(tmp_path, name):
    module_dir = tmp_path / name
    module_dir.mkdir()
    public_dir = module_dir / "public"
    public_dir.mkdir()
    (public_dir / "asset.js").write_text("console.log('asset');", encoding="utf-8")
    (module_dir / "secret.txt").write_text("secret", encoding="utf-8")
    return module_dir


@pytest.fixture
def reset_public_file_filter(monkeypatch):
    monkeypatch.setattr(engine_service.public_file_filter, "files", {}, raising=False)


def test_read_xml_body_missing_content_type_returns_none(monkeypatch):
    request = SimpleNamespace(headers={"Content-Length": "18"}, body_file=BytesIO(b"<mex name='demo'/>"))
    monkeypatch.setattr(engine_service.tg, "request", request, raising=False)

    assert engine_service.read_xml_body() is None


def test_read_xml_body_invalid_content_length_aborts(monkeypatch):
    request = SimpleNamespace(
        headers={"Content-Length": "abc", "Content-Type": "text/xml"},
        body_file=BytesIO(b"<mex name='demo'/>"),
    )
    monkeypatch.setattr(engine_service.tg, "request", request, raising=False)
    monkeypatch.setattr(engine_service, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        engine_service.read_xml_body()

    assert exc.value.status == 400


def test_read_xml_body_malformed_xml_aborts(monkeypatch):
    request = SimpleNamespace(
        headers={"Content-Length": "8", "Content-Type": "text/xml"},
        body_file=BytesIO(b"<broken>"),
    )
    monkeypatch.setattr(engine_service.tg, "request", request, raising=False)
    monkeypatch.setattr(engine_service, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        engine_service.read_xml_body()

    assert exc.value.status == 400


def test_refresh_rebuilds_module_routes_and_static_assets(tmp_path, monkeypatch, reset_public_file_filter):
    old_public_key = "/engine_service/Old/asset.js"
    engine_service.public_file_filter.files[old_public_key] = ("/tmp/old.js", None)

    controller = object.__new__(engine_service.EngineServer)
    controller.mpool = object()
    controller.engines = {}
    controller.modules = []
    controller.module_by_name = {"Old": etree.Element("module", name="Old", path="/tmp/old")}
    controller.module_resources = {"Old": object()}
    controller.resource_by_name = {}
    controller.unavailable = set()
    controller.duplicate_modules = {}
    setattr(controller, "Old", object())

    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    new_module = _module_tree("EdgeDetection", module_dir)
    monkeypatch.setattr(
        engine_service,
        "initialize_available_modules",
        lambda engines: ([new_module], []),
    )

    controller.refresh()

    assert not hasattr(controller, "Old")
    assert "Old" not in controller.module_by_name
    assert "EdgeDetection" in controller.module_by_name
    assert all(not key.startswith("/engine_service/Old/") for key in engine_service.public_file_filter.files)
    assert "/engine_service/EdgeDetection/asset.js" in engine_service.public_file_filter.files


def test_refresh_keeps_first_duplicate_module_name(tmp_path, monkeypatch, reset_public_file_filter):
    controller = object.__new__(engine_service.EngineServer)
    controller.mpool = object()
    controller.engines = {}
    controller.modules = []
    controller.module_by_name = {}
    controller.module_resources = {}
    controller.resource_by_name = {}
    controller.unavailable = set()
    controller.duplicate_modules = {}

    first_dir = _make_module_dir(tmp_path, "dup_first")
    second_dir = _make_module_dir(tmp_path, "dup_second")
    first = _module_tree("Duplicated", first_dir)
    second = _module_tree("Duplicated", second_dir)
    monkeypatch.setattr(
        engine_service,
        "initialize_available_modules",
        lambda engines: ([first, second], []),
    )

    controller.refresh()

    assert controller.module_by_name["Duplicated"].get("path") == str(first_dir)
    assert controller.duplicate_modules["Duplicated"] == [str(second_dir)]
    assert len(controller.modules) == 1


def test_definition_does_not_mutate_cached_module_xml(tmp_path, monkeypatch):
    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    module = _module_tree("EdgeDetection", module_dir)
    resource = engine_service.EngineModuleResource(module, object())
    monkeypatch.setattr(
        engine_service.tg,
        "request",
        SimpleNamespace(host_url="http://example.test"),
        raising=False,
    )

    response = resource.definition()

    assert 'value="http://example.test/engine_service/EdgeDetection"' in response
    assert resource.module_xml.get("value") == "EdgeDetection"


def test_define_io_hides_system_input_variants(tmp_path):
    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    resource = engine_service.EngineModuleResource(_module_tree("EdgeDetection", module_dir), object())

    input_names = [item["name"] for item in resource.inputs]

    assert input_names == ["Input Image"]


def test_public_rejects_path_traversal(tmp_path, monkeypatch):
    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    resource = engine_service.EngineModuleResource(_module_tree("EdgeDetection", module_dir), object())
    monkeypatch.setattr(engine_service, "abort", _abort)

    with pytest.raises(AbortCalled) as exc:
        resource.public("..", "secret.txt")

    assert exc.value.status == 404


def test_start_execution_omits_empty_execution_id(tmp_path):
    class NullAdapter:
        def execute(self, module, mextree, mpool, command="start"):
            del module, mextree, mpool, command
            return None

    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    resource = engine_service.EngineModuleResource(_module_tree("EdgeDetection", module_dir), object())
    resource.adapters = {"runtime": NullAdapter()}
    mex = etree.Element("mex", uri="/mex/00-MEX1")

    result = resource.start_execution(mex)

    assert result.xpath('./tag[@name="execution_id"]') == []


def test_start_execution_replaces_duplicate_error_messages(tmp_path, monkeypatch):
    class FailingAdapter:
        def execute(self, module, mextree, mpool, command="start"):
            del module, mpool, command
            mextree.append(etree.Element("tag", name="error_message", value="internal detail"))
            raise engine_service.EngineError("engine failed")

    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    resource = engine_service.EngineModuleResource(_module_tree("EdgeDetection", module_dir), object())
    resource.adapters = {"runtime": FailingAdapter()}
    response = SimpleNamespace(status_int=200)
    monkeypatch.setattr(engine_service.tg, "response", response, raising=False)

    result = resource.start_execution(etree.Element("mex", uri="/mex/00-MEX2"))
    errors = result.xpath('./tag[@name="error_message"]')

    assert response.status_int == 500
    assert len(errors) == 1
    assert errors[0].get("value") == "engine failed"


def test_start_execution_hides_tracebacks_from_generic_failures(tmp_path, monkeypatch):
    class CrashingAdapter:
        def execute(self, module, mextree, mpool, command="start"):
            del module, mextree, mpool, command
            raise RuntimeError("boom")

    module_dir = _make_module_dir(tmp_path, "EdgeDetection")
    resource = engine_service.EngineModuleResource(_module_tree("EdgeDetection", module_dir), object())
    resource.adapters = {"runtime": CrashingAdapter()}
    response = SimpleNamespace(status_int=200)
    monkeypatch.setattr(engine_service.tg, "response", response, raising=False)

    result = resource.start_execution(etree.Element("mex", uri="/mex/00-MEX3"))
    errors = result.xpath('./tag[@name="error_message"]')

    assert response.status_int == 500
    assert len(errors) == 1
    assert errors[0].get("value") == "Module execution failed"
