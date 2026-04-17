import os
from pathlib import Path
from uuid import uuid4
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import pytest


pytestmark = pytest.mark.functional


def _first_resource(root):
    children = list(root)
    if children:
        return children[0]
    return root


def _url2localpath(url):
    parsed = urlparse(url)
    return os.path.normpath(unquote(parsed.path))


def _local_store_path(admin_session):
    stores = admin_session.fetchxml("/blob_service/store")
    for store in stores.findall("store"):
        if store.get("name") == "local":
            return Path(_url2localpath(store.get("value"))).resolve()
    pytest.skip("local store is not available in this BisQue test environment")


def test_zero_copy_path_registration_round_trip(admin_session):
    store_root = _local_store_path(admin_session)
    relative_path = Path("nfs_sync_tests") / ("%s.txt" % uuid4().hex)
    crawl_root = store_root / relative_path.parent
    try:
        crawl_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        pytest.skip("local store path is not writable from this test runner: %s" % exc)
    file_path = (store_root / relative_path).resolve()
    contents = b"zero-copy registration works\n"
    file_path.write_bytes(contents)

    resource_uri = None
    try:
        resource = ET.Element("file", value=file_path.as_uri())
        ET.SubElement(resource, "tag", name="relative_path", value=relative_path.as_posix())
        ET.SubElement(resource, "tag", name="crawl_root", value=str(crawl_root))
        ET.SubElement(resource, "tag", name="source_store", value="local")

        created_root = admin_session.postxml("/blob_service/paths/insert", resource)
        created = _first_resource(created_root)
        resource_uri = created.get("uri")
        resource_uniq = created.get("resource_uniq")

        assert resource_uniq
        assert (created.get("resource_value") or created.get("value", "")).startswith("file://")

        listed_root = admin_session.fetchxml("/blob_service/paths/list", path=file_path.as_uri())
        listed = list(listed_root)
        assert len(listed) == 1
        assert listed[0].get("resource_uniq") == resource_uniq

        deep = admin_session.fetchxml(resource_uri, view="deep")
        tags = {tag.get("name"): tag.get("value") for tag in deep.findall("tag")}
        assert tags["relative_path"] == relative_path.as_posix()
        assert tags["crawl_root"] == str(crawl_root)
        assert tags["source_store"] == "local"

        localpath_doc = admin_session.fetchxml("/blob_service/%s" % resource_uniq, localpath="1")
        assert Path(localpath_doc.get("value")).resolve() == file_path

        fetched = admin_session.fetchblob("/blob_service/%s" % resource_uniq)
        assert fetched == contents
    finally:
        if resource_uri:
            admin_session.deletexml(resource_uri)
        if file_path.exists():
            file_path.unlink()
        if crawl_root.exists():
            try:
                crawl_root.rmdir()
            except OSError:
                pass
