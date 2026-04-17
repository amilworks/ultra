import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import bisque_paths  # noqa: E402


class FakeResponse(object):
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        return None


def make_args(tmp_path, crawl_root, dry_run=False):
    return SimpleNamespace(
        alias=None,
        command="sync",
        crawl_root=str(crawl_root),
        credentials="admin:admin",
        dry_run=dry_run,
        follow_symlinks=False,
        hidden=None,
        host="http://example.test",
        permission="private",
        report=str(tmp_path / "sync-report.jsonl"),
        resource=None,
        skip_existing=True,
        source_store=None,
        tag_file=None,
        verbose=False,
        workers=2,
    )


def test_iter_sync_files_skips_ignored_files_and_symlinks(tmp_path):
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir()
    nested = crawl_root / "nested"
    nested.mkdir()

    keep = nested / "keep.txt"
    keep.write_text("ok", encoding="utf-8")
    (nested / ".DS_Store").write_text("ignore", encoding="utf-8")
    (nested / "Thumbs.db").write_text("ignore", encoding="utf-8")
    (nested / "._hidden").write_text("ignore", encoding="utf-8")

    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    symlink = nested / "linked.txt"
    try:
        symlink.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are not supported in this test environment")

    discovered = list(bisque_paths.iter_sync_files(crawl_root))

    assert discovered == [keep.resolve()]


def test_bisque_sync_reports_registered_and_skipped_files(tmp_path, monkeypatch):
    crawl_root = tmp_path / "crawl"
    nested = crawl_root / "nested"
    nested.mkdir(parents=True)

    existing_file = nested / "existing.txt"
    existing_file.write_text("already there", encoding="utf-8")
    new_file = nested / "new.txt"
    new_file.write_text("register me", encoding="utf-8")
    ignored_file = nested / ".DS_Store"
    ignored_file.write_text("ignore me", encoding="utf-8")

    external = tmp_path / "elsewhere.txt"
    external.write_text("outside", encoding="utf-8")
    symlink = nested / "linked.txt"
    try:
        symlink.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are not supported in this test environment")

    args = make_args(tmp_path, crawl_root)
    created_resources = []

    monkeypatch.setattr(
        bisque_paths,
        "fetch_store_mounts",
        lambda _session, _args: [("legacy_nfs", crawl_root.as_uri() + "/")],
    )
    monkeypatch.setattr(bisque_paths, "get_thread_session", lambda _args: object())
    monkeypatch.setattr(
        bisque_paths,
        "path_already_registered",
        lambda _session, _args, path_url: path_url == existing_file.resolve().as_uri(),
    )

    def fake_post_path_link(_session, _args, resource):
        created_resources.append(resource)
        linked = ET.Element(
            "file",
            uri="/data_service/%s" % resource.find("./tag[@name='relative_path']").get("value"),
            resource_uniq="00-test-%d" % len(created_resources),
            resource_value=resource.get("value"),
        )
        return FakeResponse(ET.tostring(linked, encoding="unicode"))

    monkeypatch.setattr(bisque_paths, "post_path_link", fake_post_path_link)

    bisque_paths.bisque_sync(object(), args)

    report_lines = [json.loads(line) for line in Path(args.report).read_text(encoding="utf-8").splitlines()]
    statuses = {entry["relative_path"]: entry["status"] for entry in report_lines}

    assert statuses["nested/existing.txt"] == "skipped"
    assert statuses["nested/new.txt"] == "registered"
    assert "nested/linked.txt" not in statuses
    assert "nested/.DS_Store" not in statuses

    assert len(created_resources) == 1
    tags = {tag.get("name"): tag.get("value") for tag in created_resources[0].findall("tag")}
    assert tags["relative_path"] == "nested/new.txt"
    assert tags["crawl_root"] == str(crawl_root.resolve())
    assert tags["source_store"] == "legacy_nfs"
    assert created_resources[0].get("name") is None


def test_bisque_sync_dry_run_emits_planned_records(tmp_path, monkeypatch):
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir()
    file_path = crawl_root / "sample.txt"
    file_path.write_text("dry run", encoding="utf-8")

    args = make_args(tmp_path, crawl_root, dry_run=True)

    monkeypatch.setattr(
        bisque_paths,
        "fetch_store_mounts",
        lambda _session, _args: [("legacy_nfs", crawl_root.as_uri() + "/")],
    )
    monkeypatch.setattr(bisque_paths, "get_thread_session", lambda _args: object())
    monkeypatch.setattr(bisque_paths, "path_already_registered", lambda *_args, **_kwargs: False)

    called = {"post": 0}

    def fake_post_path_link(*_args, **_kwargs):
        called["post"] += 1
        raise AssertionError("dry-run should not call post_path_link")

    monkeypatch.setattr(bisque_paths, "post_path_link", fake_post_path_link)

    bisque_paths.bisque_sync(object(), args)

    report = [json.loads(line) for line in Path(args.report).read_text(encoding="utf-8").splitlines()]
    assert report == [
        {
            "crawl_root": str(crawl_root.resolve()),
            "dry_run": True,
            "file_url": file_path.resolve().as_uri(),
            "path": str(file_path.resolve()),
            "relative_path": "sample.txt",
            "source_store": "legacy_nfs",
            "status": "planned",
        }
    ]
    assert called["post"] == 0
