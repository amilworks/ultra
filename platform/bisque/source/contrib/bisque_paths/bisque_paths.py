#!/usr/bin/env python

"""Manipulate BisQue resources by path, including zero-copy sync for mounted files."""

__author__ = "Center for Bioimage Informatics"
__version__ = "1.0"
__copyright__ = "Center for BioImage Informatics, University California, Santa Barbara"

import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

import requests
import six
from six.moves.configparser import NoSectionError, SafeConfigParser


DEFAULTS = dict(
    logfile="/tmp/bisque_insert.log",
    bisque_host="https://bisque.example.com",
    bisque_user="admin",
    bisque_pass="admin",
    irods_host="irods://mokie.iplantcollaborative.org",
)

IGNORED_BASENAMES = {".DS_Store", "Thumbs.db"}
IGNORED_PREFIXES = ("._",)
STDIO_REPORT = "-"
_THREAD_LOCAL = threading.local()


def build_resource(args, srcpath=None, extra_tags=None, set_name=True):
    """Create a compatible resource element for posting or linking."""
    if getattr(args, "tag_file", None):
        try:
            resource = ET.parse(args.tag_file).getroot()
        except ParseError as exc:
            six.print_("Parse failure: aborting:", exc, file=sys.stderr)
            raise SystemExit(2)
    else:
        resource = ET.Element(args.resource or "resource")

    for fld in ("permission", "hidden"):
        if getattr(args, fld, None) is not None:
            resource.set(fld, getattr(args, fld))

    if srcpath:
        resource.set("value", srcpath)
        if set_name:
            resource.set("name", os.path.basename(srcpath))
    elif getattr(args, "srcpath", None):
        resource.set("value", args.srcpath[0])
        if set_name:
            resource.set("name", os.path.basename(args.srcpath[0]))

    for name, value in extra_tags or []:
        ET.SubElement(resource, "tag", name=name, value=value)

    return resource


def create_session(args):
    requests.packages.urllib3.disable_warnings()
    session = requests.Session()
    session.log = logging.getLogger("rods2bq")
    session.auth = tuple(args.credentials.split(":", 1))
    return session


def get_thread_session(args):
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = create_session(args)
        _THREAD_LOCAL.session = session
    return session


def response_resources(xml_text):
    root = ET.fromstring(xml_text)
    children = list(root)
    if children:
        return root, children
    if root.get("resource_uniq") or root.get("uri"):
        return root, [root]
    return root, []


def _path_params(args, path):
    params = {"path": path}
    if args.alias:
        params["user"] = args.alias
    return params


def bisque_delete(session, args):
    """Delete a file based on the path."""
    session.log.info("delete %s", args)
    url = args.host + "/blob_service/paths/remove"
    response = session.get(url, params=_path_params(args, args.srcpath[0]))
    if response.status_code == requests.codes.ok:
        six.print_(response.text)
    response.raise_for_status()


def post_path_link(session, args, resource):
    url = args.host + "/blob_service/paths/insert"
    params = {}
    if args.alias:
        params["user"] = args.alias
    payload = ET.tostring(resource)
    response = session.post(
        url,
        data=payload,
        params=params,
        headers={"content-type": "application/xml"},
    )
    response.raise_for_status()
    return response


def bisque_link(session, args):
    """Insert a file based on the path."""
    session.log.info("link %s", args)
    resource = build_resource(args)
    response = post_path_link(session, args, resource)
    if response.status_code == requests.codes.ok:
        if args.compatible:
            _, resources = response_resources(response.text)
            linked = resources[0] if resources else ET.fromstring(response.text)
            six.print_(linked.get("resource_uniq"), linked.get("uri"))
        else:
            six.print_(response.text)


def bisque_copy(session, args):
    """Copy a file through the upload/import path."""
    session.log.info("insert %s", args)

    url = args.host + "/import/transfer"
    params = {}
    resource = build_resource(args)
    if "value" in resource.attrib:
        del resource.attrib["value"]

    with open(args.srcpath[0], "rb") as src:
        files = {
            "file": (os.path.basename(args.srcpath[0]), src),
            "file_resource": (None, ET.tostring(resource), "text/xml"),
        }

        if args.alias:
            params["user"] = args.alias
        response = session.post(url, files=files, params=params)

    if response.status_code == requests.codes.ok:
        six.print_(response.text)
    response.raise_for_status()


def bisque_rename(session, args):
    """Rename based on paths."""
    session.log.info("rename %s", args)

    url = args.host + "/blob_service/paths/move"
    params = {"path": args.srcpath[0], "destination": args.dstpath}
    if args.alias:
        params["user"] = args.alias

    response = session.get(url, params=params)
    if response.status_code == requests.codes.ok:
        six.print_(response.text)
    response.raise_for_status()


def bisque_list(session, args):
    """List resources at a path."""
    session.log.info("list %s", args)

    url = args.host + "/blob_service/paths/list"
    params = {}
    if len(args.srcpath) > 0:
        params["path"] = args.srcpath[0]
    if args.alias:
        params["user"] = args.alias
    response = session.get(url, params=params)
    if response.status_code == requests.codes.ok:
        if args.compatible:
            for resource in ET.fromstring(response.text):
                six.print_(resource.get("resource_uniq"))
            return
        if args.unique:
            for resource in ET.fromstring(response.text):
                six.print_(resource.get("resource_uniq"), resource.get("resource_value"))
            return
        six.print_(response.text)
    response.raise_for_status()


def should_ignore_path(path):
    name = path.name
    if name in IGNORED_BASENAMES:
        return True
    return any(name.startswith(prefix) for prefix in IGNORED_PREFIXES)


def iter_sync_files(crawl_root, follow_symlinks=False):
    """Yield files from crawl_root while preserving a stable traversal order."""
    for current_root, dirnames, filenames in os.walk(str(crawl_root), followlinks=follow_symlinks):
        current_path = Path(current_root)

        keep_dirs = []
        for dirname in sorted(dirnames):
            dirpath = current_path / dirname
            if should_ignore_path(dirpath):
                continue
            if not follow_symlinks and dirpath.is_symlink():
                continue
            keep_dirs.append(dirname)
        dirnames[:] = keep_dirs

        for filename in sorted(filenames):
            filepath = current_path / filename
            if should_ignore_path(filepath):
                continue
            if not follow_symlinks and filepath.is_symlink():
                continue
            if filepath.is_file():
                yield filepath.resolve()


def fetch_store_mounts(session, args):
    """Return configured blob stores as (name, mount_url) pairs."""
    params = {}
    if args.alias:
        params["user"] = args.alias
    response = session.get(args.host + "/blob_service/store", params=params)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    stores = []
    for store in root.findall("store"):
        name = store.get("name")
        value = store.get("value")
        if name and value:
            stores.append((name, value))
    stores.sort(key=lambda item: len(item[1]), reverse=True)
    return stores


def detect_source_store(file_url, stores):
    normalized_file = file_url.rstrip("/")
    for name, mount_url in stores:
        normalized_mount = mount_url.rstrip("/")
        prefix = mount_url if mount_url.endswith("/") else mount_url + "/"
        if normalized_file == normalized_mount or file_url.startswith(prefix):
            return name
    return None


def path_already_registered(session, args, path_url):
    response = session.get(args.host + "/blob_service/paths/list", params=_path_params(args, path_url))
    response.raise_for_status()
    _, resources = response_resources(response.text)
    return len(resources) > 0


def make_sync_record(status, file_path, file_url, relative_path, crawl_root, source_store, **extra):
    record = {
        "status": status,
        "path": str(file_path),
        "file_url": file_url,
        "relative_path": relative_path,
        "crawl_root": str(crawl_root),
        "source_store": source_store,
    }
    record.update(extra)
    return record


class JsonlReporter(object):
    """Thread-safe JSONL reporter."""

    def __init__(self, target):
        self.target = target
        self._lock = threading.Lock()
        self._owns_stream = target != STDIO_REPORT
        if self._owns_stream:
            report_path = Path(target)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = report_path.open("w", encoding="utf-8")
        else:
            self._stream = sys.stdout

    def write(self, record):
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self):
        if self._owns_stream:
            self._stream.close()


def sync_file(task):
    """Register a single file by path."""
    args = task["args"]
    session = get_thread_session(args)
    file_path = task["file_path"]
    crawl_root = task["crawl_root"]
    relative_path = task["relative_path"]
    source_store = task["source_store"]
    file_url = task["file_url"]

    try:
        if args.skip_existing and path_already_registered(session, args, file_url):
            return make_sync_record(
                "skipped",
                file_path,
                file_url,
                relative_path,
                crawl_root,
                source_store,
                reason="existing",
            )

        if args.dry_run:
            return make_sync_record(
                "planned",
                file_path,
                file_url,
                relative_path,
                crawl_root,
                source_store,
                dry_run=True,
            )

        resource = build_resource(
            args,
            srcpath=file_url,
            extra_tags=[
                ("relative_path", relative_path),
                ("crawl_root", str(crawl_root)),
                ("source_store", source_store),
            ],
            set_name=False,
        )
        response = post_path_link(session, args, resource)
        _, resources = response_resources(response.text)
        linked = resources[0] if resources else ET.fromstring(response.text)
        return make_sync_record(
            "registered",
            file_path,
            file_url,
            relative_path,
            crawl_root,
            source_store,
            uri=linked.get("uri"),
            resource_uniq=linked.get("resource_uniq"),
            resource_value=linked.get("resource_value") or linked.get("value"),
        )
    except requests.exceptions.HTTPError as exc:
        message = exc.response.text if exc.response is not None else str(exc)
        return make_sync_record(
            "failed",
            file_path,
            file_url,
            relative_path,
            crawl_root,
            source_store,
            error=message,
            status_code=exc.response.status_code if exc.response is not None else None,
        )
    except Exception as exc:  # pragma: no cover - defensive catch for CLI reporting
        return make_sync_record(
            "failed",
            file_path,
            file_url,
            relative_path,
            crawl_root,
            source_store,
            error=str(exc),
        )


def bisque_sync(session, args):
    """Bulk-register existing files by path without copying their bytes."""
    crawl_root = Path(args.crawl_root).resolve()
    if not crawl_root.exists():
        raise SystemExit("crawl root does not exist: %s" % crawl_root)
    if not crawl_root.is_dir():
        raise SystemExit("crawl root must be a directory: %s" % crawl_root)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    crawl_root_url = crawl_root.as_uri()
    stores = fetch_store_mounts(session, args)
    source_store = args.source_store or detect_source_store(crawl_root_url, stores)
    if source_store is None:
        raise SystemExit(
            "crawl root %s is not under a configured blob store mount" % crawl_root
        )

    reporter = JsonlReporter(args.report)
    counts = {"registered": 0, "skipped": 0, "failed": 0, "planned": 0}

    def make_tasks():
        for file_path in iter_sync_files(crawl_root, follow_symlinks=args.follow_symlinks):
            yield {
                "args": args,
                "crawl_root": crawl_root,
                "file_path": file_path,
                "file_url": file_path.as_uri(),
                "relative_path": file_path.relative_to(crawl_root).as_posix(),
                "source_store": source_store,
            }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(sync_file, make_tasks()):
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                reporter.write(result)
    finally:
        reporter.close()

    summary = ", ".join("%s=%s" % (key, counts.get(key, 0)) for key in sorted(counts.keys()))
    six.print_("sync summary: %s" % summary, file=sys.stderr)


DESCRIPTION = """Manipulate BisQue resources with store paths.

Insert, link, move, remove, or bulk-register resources by their existing path.
"""


def main():
    config = SafeConfigParser()
    config.add_section("main")
    for key, value in list(DEFAULTS.items()):
        config.set("main", key, value)

    config.read([".bisque", os.path.expanduser("~/.bisque"), "/etc/bisque/bisque_config"])
    defaults = dict(config.items("main"))
    try:
        defaults.update(config.items("bqpath"))
    except NoSectionError:
        pass

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--alias", help="do action on behalf of user specified")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="log debugging")
    parser.add_argument("-H", "--host", default=defaults["bisque_host"], help="bisque host")
    parser.add_argument(
        "-c",
        "--credentials",
        default="%s:%s" % (defaults["bisque_user"], defaults["bisque_pass"]),
        help="user credentials",
    )
    parser.add_argument("-C", "--compatible", action="store_true", help="Make compatible with old script")
    parser.add_argument("-V", "--verbose", action="store_true", help="print stuff")

    sp = parser.add_subparsers(dest="command")

    lsp = sp.add_parser("ls")
    lsp.add_argument("-u", "--unique", default=None, action="store_true", help="return unique codes")
    lsp.add_argument("paths", nargs="+")
    lsp.set_defaults(func=bisque_list)

    lnp = sp.add_parser("ln")
    lnp.add_argument("-T", "--tag_file", default=None, help="tag document for insert")
    lnp.add_argument("-P", "--permission", default="private", help="Set resource permission (compatibility)")
    lnp.add_argument("-R", "--resource", default=None, help="force resource type")
    lnp.add_argument("--hidden", default=None, help="Set resource visibility (hidden)")
    lnp.add_argument("paths", nargs="+")
    lnp.set_defaults(func=bisque_link)

    cpp = sp.add_parser("cp")
    cpp.add_argument("paths", nargs="+")
    cpp.add_argument("-T", "--tag_file", default=None, help="tag document for insert")
    cpp.add_argument("-R", "--resource", default=None, help="force resource type")
    cpp.add_argument("-P", "--permission", default="private", help="Set resource permission (compatibility)")
    cpp.add_argument("--hidden", default=None, help="Set resource visibility (hidden)")
    cpp.set_defaults(func=bisque_copy)

    mvp = sp.add_parser("mv")
    mvp.add_argument("paths", nargs="+")
    mvp.set_defaults(func=bisque_rename)

    rmp = sp.add_parser("rm")
    rmp.add_argument("paths", nargs="+")
    rmp.set_defaults(func=bisque_delete)

    synp = sp.add_parser("sync")
    synp.add_argument("crawl_root", help="local directory to crawl and register by path")
    synp.add_argument("-P", "--permission", default="private", help="Set resource permission")
    synp.add_argument("-R", "--resource", default=None, help="force resource type")
    synp.add_argument("--hidden", default=None, help="Set resource visibility (hidden)")
    synp.add_argument("--dry-run", action="store_true", help="report what would be registered without inserting")
    synp.add_argument("--report", default=STDIO_REPORT, help="JSONL report path or '-' for stdout")
    synp.add_argument("--workers", type=int, default=8, help="number of parallel registration workers")
    synp.add_argument(
        "--source-store",
        default=None,
        help="optional explicit blob store name; otherwise infer from the crawl root path",
    )
    synp.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="follow symlinked directories and files during crawl",
    )
    synp.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="skip files that are already registered at the same path",
    )
    synp.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="always attempt path registration even if the path already exists in BisQue",
    )
    synp.set_defaults(func=bisque_sync)

    logging.basicConfig(
        filename=defaults.get("logfile", DEFAULTS["logfile"]),
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5.5s [%(name)s] %(message)s",
    )

    log = logging.getLogger("rods2bq")

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command != "sync":
        if len(args.paths) > 1:
            args.dstpath = args.paths.pop()
            args.srcpath = args.paths
        else:
            args.srcpath = args.paths

        if args.compatible:
            paths = []
            irods_host = defaults.get("irods_host")
            for element in args.srcpath:
                if not element.startswith("irods://"):
                    paths.append(irods_host + element)
                else:
                    paths.append(element)
            args.srcpath = paths
            if getattr(args, "dstpath", None) and not args.dstpath.startswith("irods://"):
                args.dstpath = irods_host + args.dstpath

    if args.debug:
        six.print_(args, file=sys.stderr)

    try:
        session = create_session(args)
        args.func(session, args)
    except requests.exceptions.HTTPError as exc:
        log.exception("exception occurred %s : %s", exc, exc.response.text if exc.response is not None else "")
        six.print_("ERROR:", exc.response and exc.response.status_code)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
