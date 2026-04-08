import os
from datetime import datetime

import pytest
import requests
from lxml import etree

from bqapi import BQSession


pytestmark = pytest.mark.functional


def _resource_suffix(uri):
    if not uri:
        return uri
    marker = "/data_service/"
    if marker in uri:
        return uri.split(marker, 1)[1]
    return uri.lstrip("/")


def _parse_uploaded_uri(upload_response):
    payload = upload_response
    if isinstance(upload_response, str):
        payload = upload_response.encode("utf-8")
    try:
        root = etree.XML(payload)
    except Exception as exc:
        pytest.fail(f"Upload response was not XML: {exc}")
    if len(root) == 0:
        pytest.fail("Upload response did not include uploaded resource metadata")
    uri = root[0].get("uri")
    if not uri:
        pytest.fail("Upload response metadata missing uploaded resource URI")
    return uri


def _request_access_token(config):
    host = config.get("host.root")
    user = config.get("host.user")
    password = config.get("host.password")
    token_url = host.rstrip("/") + "/auth_service/token"

    try:
        response = requests.post(
            token_url,
            json={"username": user, "password": password, "grant_type": "password"},
            timeout=10,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise RuntimeError(f"Token endpoint unreachable: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise AssertionError(f"Token endpoint request failed: {exc}") from exc

    if response.status_code >= 400:
        raise AssertionError(
            f"Token endpoint returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError("Token endpoint did not return JSON") from exc

    token = payload.get("access_token")
    if not token:
        raise AssertionError("Token endpoint response missing access_token")
    return token


@pytest.fixture(scope="module")
def token_session(config):
    try:
        token = _request_access_token(config)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except AssertionError as exc:
        pytest.fail(str(exc))

    session = BQSession().init_token(token, bisque_root=config.get("host.root"), create_mex=False)
    expected_user = config.get("host.user")
    whoami = session.fetchxml("/auth_service/whoami")
    user_tag = whoami.find("./tag[@name='name']")
    if user_tag is None:
        session.close()
        pytest.fail("Bearer session whoami response did not include user name")
    if user_tag.get("value") != expected_user:
        session.close()
        pytest.fail(
            f"Bearer token authenticated as unexpected user: {user_tag.get('value')} != {expected_user}"
        )

    yield session
    session.close()


def _upload_image(session, local_file, resource_name):
    resource = etree.Element("image", name=resource_name)
    content = session.postblob(local_file, xml=resource)
    return _parse_uploaded_uri(content)


def _download_blob(session, resource_uri, dest_path):
    resource = session.load(resource_uri)
    if resource is None or not resource.resource_uniq:
        pytest.fail(f"Unable to resolve uploaded image metadata for {resource_uri}")
    blob_url = session.service_url("blob_service", path=resource.resource_uniq)
    session.fetchblob(blob_url, path=dest_path)
    if not os.path.exists(dest_path):
        pytest.fail(f"Blob download did not create expected file: {dest_path}")
    if os.path.getsize(dest_path) == 0:
        pytest.fail(f"Blob download returned empty file: {dest_path}")


def test_notebook_token_upload_download_tag_and_search(token_session, tmp_path):
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    created_uris = []

    source_local = tmp_path / f"{run_id}_source_input.png"
    source_download = tmp_path / f"{run_id}_source_download.png"
    grayscale_local = tmp_path / f"{run_id}_gray.jpg"
    grayscale_download = tmp_path / f"{run_id}_gray_download.jpg"

    try:
        Image = pytest.importorskip("PIL.Image")
        Image.new("RGB", (32, 32), color=(90, 140, 210)).save(source_local, format="PNG")

        source_uri = _upload_image(
            token_session,
            str(source_local),
            f"api_notebook_{run_id}_source.png",
        )
        created_uris.append(source_uri)
        _download_blob(token_session, source_uri, str(source_download))

        with Image.open(source_download) as source_image:
            source_image.convert("L").save(grayscale_local, format="JPEG")
        assert grayscale_local.exists(), "Grayscale conversion did not produce a JPEG file"

        gray_uri = _upload_image(
            token_session,
            str(grayscale_local),
            f"api_notebook_{run_id}_healthy.jpg",
        )
        created_uris.append(gray_uri)

        token_session.postxml(f"{gray_uri}/tag", etree.Element("tag", name="healthy", value="true"))
        token_session.postxml(f"{gray_uri}/tag", etree.Element("tag", name="flow_id", value=run_id))

        tagged_image = token_session.fetchxml(gray_uri, view="deep")
        assert tagged_image.find("./tag[@name='healthy'][@value='true']") is not None

        query_expr = f"healthy:true & flow_id:{run_id} & @name:*.jpg"
        results = token_session.query("image", tag_query=query_expr, limit=25)
        found_uris = {_resource_suffix(getattr(item, "uri", None)) for item in results}
        assert _resource_suffix(gray_uri) in found_uris

        _download_blob(token_session, gray_uri, str(grayscale_download))
    finally:
        for uri in reversed(created_uris):
            try:
                token_session.deletexml(uri)
            except Exception:
                pass
