from types import SimpleNamespace

import pytest

from bq.data_service.controllers import bisquik_resource
from bq.data_service.controllers import data_service
from bq.data_service.controllers import formats
from bq.data_service.controllers import resource
from bq.data_service.controllers import resource_auth
from bq.data_service.controllers import resource_query


pytestmark = pytest.mark.unit


class _QueryRecorder:
    def __init__(self):
        self.filters = []
        self.orders = []

    def filter(self, *exprs):
        self.filters.extend(exprs)
        return self

    def filter_by(self, **kw):
        self.filters.append(("filter_by", kw))
        return self

    def order_by(self, *exprs):
        self.orders.extend(exprs)
        return self

    def offset(self, value):
        self.offset_value = value
        return self

    def limit(self, value):
        self.limit_value = value
        return self


def _configure_minimal_resource_query(monkeypatch, query):
    monkeypatch.setattr(resource_query, "prepare_type", lambda rt: ("image", object, query))
    monkeypatch.setattr(resource_query, "prepare_parent", lambda rt, q, parent: q)
    monkeypatch.setattr(resource_query, "prepare_permissions", lambda q, user_id, with_public, action: q)
    monkeypatch.setattr(resource_query, "prepare_tag_expr", lambda q, tag_query=None: q)
    monkeypatch.setattr(resource_query, "prepare_attributes", lambda q, dbtype, attribs: q)
    monkeypatch.setattr(resource_query, "tags_special", lambda dbtype, q, params: None)
    monkeypatch.setattr(resource_query, "prepare_order_expr", lambda q, tag_order: q)


def test_prepare_tag_expr_invalid_query_raises():
    query = _QueryRecorder()
    with pytest.raises(resource_query.QuerySyntaxError):
        resource_query.prepare_tag_expr(query, "foo::bar")


def test_prepare_tag_expr_valid_query_still_applies_filter():
    query = _QueryRecorder()
    resource_query.prepare_tag_expr(query, "healthy")
    assert query.filters


def test_prepare_order_expr_resets_direction_per_term():
    query = _QueryRecorder()
    resource_query.prepare_order_expr(query, "@ts:desc,@name")
    assert len(query.orders) == 2
    assert str(query.orders[0]).endswith("DESC")
    assert str(query.orders[1]).endswith("ASC")


def test_bisquik_dir_maps_query_syntax_errors_to_400(monkeypatch):
    controller = bisquik_resource.BisquikResource("image", url="http://localhost/data_service")
    req = SimpleNamespace(url="http://localhost/data_service/image", path="/data_service/image")
    monkeypatch.setattr(bisquik_resource, "request", req)
    monkeypatch.setattr(bisquik_resource.identity, "get_user_id", lambda: 1)

    def _raise_query_error(*_args, **_kw):
        raise resource_query.QuerySyntaxError("Malformed tag_query")

    def _abort(status_code, message=None):
        raise RuntimeError(status_code, message)

    monkeypatch.setattr(bisquik_resource, "resource_query", _raise_query_error)
    monkeypatch.setattr(bisquik_resource, "abort", _abort)

    with pytest.raises(RuntimeError) as exc:
        controller.dir(None, tag_query="foo::bar")
    assert exc.value.args[0] == 400
    assert "Malformed tag_query" in exc.value.args[1]


def test_data_server_load_preserves_equals_in_query_values(monkeypatch):
    controller = data_service.DataServerController("http://localhost/data_service/")
    captured = {}

    monkeypatch.setattr(data_service, "dbtype_from_name", lambda token: ("image", object))

    def _resource_count(resource_type, parent=None, **kwargs):
        captured["resource_type"] = resource_type
        captured["kwargs"] = kwargs
        return 3

    monkeypatch.setattr(data_service, "resource_count", _resource_count)

    response = controller.load(
        "http://localhost/data_service/image?view=count&tag_query=a%3Ab%3D%3D1&empty=",
        astree=True,
    )

    assert response[0].tag == "image"
    assert response[0].get("count") == "3"
    assert captured["kwargs"]["tag_query"] == "a:b==1"
    assert captured["kwargs"]["empty"] == ""


def test_find_formatter_unknown_accept_defaults_to_xml():
    formatter, content_type = formats.find_formatter(accept_header="application/x-unknown")
    assert formatter is formats.format_xml
    assert content_type == "text/xml"


def test_find_formatter_type_unknown_accept_defaults_to_xml_type():
    assert formats.find_formatter_type("application/x-unknown") == "xml"


def test_find_inputer_handles_missing_content_type():
    assert formats.find_inputer(None) is None


def test_input_csv_current_stub_returns_empty_response_root():
    parsed = formats.input_csv("name,value\nhealthy,true\n")
    assert parsed.tag == "response"
    assert len(parsed) == 0


def test_resource_auth_load_missing_acl_row_is_safe(monkeypatch):
    class _NoRowsQuery:
        def filter(self, *_args, **_kw):
            return self

        def first(self):
            return None

    class _DummySession:
        def flush(self):
            return None

        def query(self, *_args, **_kw):
            return _NoRowsQuery()

    auth = resource_auth.ResourceAuth("http://localhost/data_service")
    parent_resource = SimpleNamespace(id=11)
    request_stub = SimpleNamespace(bisque=SimpleNamespace(parent=parent_resource), url="http://localhost")

    monkeypatch.setattr(resource_auth, "request", request_stub)
    monkeypatch.setattr(resource_auth, "check_access", lambda _resource, _action: parent_resource)
    monkeypatch.setattr(resource_auth, "DBSession", _DummySession())

    resource_obj, user_obj, acl_obj = auth.load("00-missing-user")
    assert resource_obj is parent_resource
    assert user_obj is None
    assert acl_obj is None


def test_resource_auth_get_with_missing_user_returns_empty_response(monkeypatch):
    auth = resource_auth.ResourceAuth("http://localhost/data_service")
    request_stub = SimpleNamespace(url="http://localhost/data_service/00-x/auth/00-y")
    monkeypatch.setattr(resource_auth, "request", request_stub)
    monkeypatch.setattr(resource_auth.tg, "response", SimpleNamespace(headers={}), raising=False)
    monkeypatch.setattr(resource_auth, "find_formatter", lambda _fmt: (lambda node: node, "text/xml"))

    called = {"acl_query": False}

    def _acl_query(*_args, **_kw):
        called["acl_query"] = True

    monkeypatch.setattr(resource_auth, "resource_acl_query", _acl_query)
    result = auth.get((SimpleNamespace(id=1), None, None))
    assert result.tag == "resource"
    assert called["acl_query"] is False


def test_response_cache_fetch_handles_valid_and_malformed_headers(tmp_path):
    cache = resource.ResponseCache(str(tmp_path))._setup()
    url = "http://localhost/data_service/image"
    user = "1"

    cache.save(url, {"Content-Type": "text/xml"}, "<resource/>", user)
    headers, cached = cache.fetch(url, user)
    assert headers["Content-Type"] == "text/xml"
    assert "<resource/>" in cached

    bad_url = "http://localhost/data_service/image?bad=1"
    bad_cache_file = tmp_path / cache._cache_name(bad_url, user)
    bad_cache_file.write_text("not a dict\n\npayload", encoding="utf-8")
    bad_headers, bad_payload = cache.fetch(bad_url, user)
    assert bad_headers is None
    assert bad_payload is None


def test_response_cache_etag_is_python3_safe(tmp_path):
    cache = resource.ResponseCache(str(tmp_path))._setup()
    url = "http://localhost/data_service/image"
    user = "1"
    cache.save(url, {"Content-Type": "text/xml"}, "<resource/>", user)
    etag_value = cache.etag(url, user)
    assert etag_value is not None
    assert len(etag_value) == 32


def test_parse_http_date_invalid_strings_return_none():
    assert resource.parse_http_date(None) is None
    assert resource.parse_http_date("bad") is None
    assert resource.parse_http_date("not-a-date") is None


def test_parse_http_date_valid_rfc1123_value_still_parses():
    parsed = resource.parse_http_date("Thu, 01 Jan 1970 00:00:00 GMT")
    assert parsed is not None
    assert parsed.year == 1970


@pytest.mark.parametrize(
    "hidden_value, expected_snippets",
    [
        (None, ["IS NULL"]),
        ("true", ["= true"]),
        ("false", ["IS NULL", "= false"]),
    ],
)
def test_resource_query_hidden_filters(monkeypatch, hidden_value, expected_snippets):
    query = _QueryRecorder()
    _configure_minimal_resource_query(monkeypatch, query)
    kwargs = {"permcheck": False}
    if hidden_value is not None:
        kwargs["hidden"] = hidden_value

    resource_query.resource_query(("image", object), **kwargs)
    hidden_filter = str(query.filters[0]).lower()
    for snippet in expected_snippets:
        assert snippet.lower() in hidden_filter
