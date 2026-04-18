from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from lxml import etree
from src import tools


def test_extract_bisque_user_from_xml_ignores_resource_uri_values() -> None:
    session_xml = etree.fromstring(
        """
        <session uri="https://bisque.example.org/auth_service/session">
          <tag name="user" value="https://bisque.example.org/data_service/00-user-resource"/>
        </session>
        """
    )
    whoami_xml = etree.fromstring(
        """
        <user>
          <tag name="name" value="bisque-user"/>
          <tag name="uri" value="https://bisque.example.org/data_service/00-user-resource"/>
        </user>
        """
    )

    assert tools._extract_bisque_user_from_xml(session_xml) is None
    assert tools._extract_bisque_user_from_xml(whoami_xml) == "bisque-user"


def test_bisque_ping_prefers_whoami_name_over_session_resource_uri(monkeypatch) -> None:
    session_xml = etree.fromstring(
        """
        <session uri="https://bisque.example.org/auth_service/session">
          <tag name="user" value="https://bisque.example.org/data_service/00-user-resource"/>
        </session>
        """
    )
    whoami_xml = etree.fromstring(
        """
        <user>
          <tag name="name" value="bisque-user"/>
          <tag name="uri" value="https://bisque.example.org/data_service/00-user-resource"/>
        </user>
        """
    )

    fetch_calls: list[str] = []

    class FakeBQSession:
        pass

    monkeypatch.setattr("bqapi.comm.BQSession", FakeBQSession)
    monkeypatch.setattr(
        tools,
        "_init_bisque_session_with_runtime_auth",
        lambda *, bq, **_kwargs: ("https://bisque.example.org", "token", True),
    )

    def fake_fetchxml(_bq, url, **_kwargs):
        fetch_calls.append(url)
        if url.endswith("/auth_service/session"):
            return session_xml
        if url.endswith("/auth_service/whoami"):
            return whoami_xml
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(tools, "_session_fetchxml_safe", fake_fetchxml)

    result = tools.bisque_ping()

    assert result["success"] is True
    assert result["user"] == "bisque-user"
    assert fetch_calls == [
        "https://bisque.example.org/auth_service/session",
        "https://bisque.example.org/auth_service/whoami",
    ]


def test_loaded_bisque_resource_type_uses_xmltag_for_bq_dataset() -> None:
    resource = SimpleNamespace(
        uri="https://bisque.example.org/data_service/00-dataset",
        name="dataset",
        ts="2026-04-17T00:00:00Z",
        type=None,
        xmltag="dataset",
        xmltree=etree.fromstring('<dataset uri="https://bisque.example.org/data_service/00-dataset"/>'),
        tags=[],
    )

    assert tools._loaded_bisque_resource_type(resource=resource, resource_xml=None) == "dataset"


def test_load_bisque_resource_reports_dataset_type_when_bqapi_object_lacks_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBQSession:
        pass

    resource = SimpleNamespace(
        uri="https://bisque.example.org/data_service/00-dataset",
        name="dataset",
        ts="2026-04-17T00:00:00Z",
        type=None,
        xmltag="dataset",
        xmltree=etree.fromstring('<dataset uri="https://bisque.example.org/data_service/00-dataset"/>'),
        tags=[],
    )

    monkeypatch.setattr("bqapi.comm.BQSession", FakeBQSession)
    monkeypatch.setattr(
        tools,
        "_init_bisque_session_with_runtime_auth",
        lambda *, bq, **_kwargs: ("https://bisque.example.org", "token", True),
    )
    monkeypatch.setattr(
        tools,
        "_load_bisque_resource_with_probe",
        lambda **_kwargs: (resource, None, None),
    )

    result = tools.load_bisque_resource("00-dataset", view="deep")

    assert result["success"] is True
    assert result["resource"]["resource_type"] == "dataset"


def test_init_bq_session_surfaces_clean_error_when_basic_auth_is_disabled() -> None:
    class FakeBQSession:
        def init_local(self, *_args, **_kwargs) -> None:
            return None

        def fetchxml(self, _url):
            raise RuntimeError("basic_auth_disabled")

    with pytest.raises(ValueError, match="BisQue basic authentication failed: basic_auth_disabled"):
        tools._init_bq_session(
            FakeBQSession(),
            username="bisque",
            password="secret",
            access_token=None,
            bisque_root="https://bisque.example.org",
            cookie_header=None,
        )


def test_infer_bisque_dataset_target_requires_explicit_dataset_or_collection() -> None:
    assert (
        tools._infer_bisque_dataset_target_from_text(
            "Upload this image to BisQue and tell me the resource URI."
        )
        is None
    )

    assert tools._infer_bisque_dataset_target_from_text(
        'Upload this image to the dataset "Frontier Screens".'
    ) == {
        "dataset_name": "Frontier Screens",
        "create_dataset_if_missing": False,
    }


def test_execute_tool_call_does_not_infer_dataset_for_plain_bisque_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_upload_to_bisque(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setitem(tools.AVAILABLE_TOOLS, "upload_to_bisque", fake_upload_to_bisque)

    result = tools.execute_tool_call(
        "upload_to_bisque",
        arguments={},
        uploaded_files=["/tmp/example.png"],
        user_text="Upload this image to BisQue and tell me the resource URI.",
    )

    assert json.loads(result)["success"] is True
    assert captured["file_paths"] == ["/tmp/example.png"]
    assert "dataset_name" not in captured
    assert "dataset_uri" not in captured
