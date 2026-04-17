
import os
import importlib.util
import sys
import types
import pytest
from io import StringIO, BytesIO
import shutil
from uuid import uuid4
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "controllers" / "blob_drivers.py"
ROOT = Path(__file__).resolve().parents[4]
for extra_path in [ROOT / "bqserver", ROOT / "bqcore"]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


def _install_test_import_stubs():
    if "shortuuid" not in sys.modules:
        shortuuid = types.ModuleType("shortuuid")
        shortuuid.uuid = lambda: "00000000000000000000000000000000"
        sys.modules["shortuuid"] = shortuuid

    if "tg" not in sys.modules:
        tg = types.ModuleType("tg")

        class _Config:
            def get(self, *_args, **_kwargs):
                return None

        tg.config = _Config()
        sys.modules["tg"] = tg

    if "paste" not in sys.modules:
        paste = types.ModuleType("paste")
        sys.modules["paste"] = paste
        deploy = types.ModuleType("paste.deploy")
        converters = types.ModuleType("paste.deploy.converters")

        def asbool(value):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        converters.asbool = asbool
        deploy.converters = converters
        paste.deploy = deploy
        sys.modules["paste.deploy"] = deploy
        sys.modules["paste.deploy.converters"] = converters


_install_test_import_stubs()
SPEC = importlib.util.spec_from_file_location("blob_drivers_test_module", MODULE_PATH)
blob_drivers = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(blob_drivers)
make_storage_driver = blob_drivers.make_storage_driver



pytestmark = pytest.mark.unit

local_driver = {
    'mount_url' : 'file://tests/tests',
    'top' : 'file://tests/tests',
}
MSG=b'A'*10

TSTDIR="tests/tests"



@pytest.fixture(scope='module')
def test_dir ():
    tstdir = TSTDIR + uuid4().hex
    if not os.path.exists (tstdir):
        os.makedirs (tstdir)
    yield tstdir
    shutil.rmtree (tstdir)




def test_local_valid(tmp_path):
    mount = tmp_path / "mount"
    mount.mkdir()
    inside = mount / "a.jpg"
    inside.write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside" / "a.jpg"
    outside.parent.mkdir()
    outside.write_text("nope", encoding="utf-8")

    drv = make_storage_driver(mount_url=f"file://{mount}")

    assert drv.valid(f"file://{inside}") is not None
    assert drv.valid(f"file://{outside}") is None



def test_local_write (test_dir):
    drv = make_storage_driver (**local_driver)

    sf = BytesIO(MSG)
    sf.name = 'none'
    storeurl, localpath = drv.push (sf, 'file://%s/msg.txt' % test_dir)
    print("GOT", storeurl, localpath)

    assert os.path.exists (localpath), "Created file exists"


def test_local_valid_rejects_normalized_escape(tmp_path):
    mount = tmp_path / "mount"
    outside = tmp_path / "outside"
    mount.mkdir()
    outside.mkdir()

    inside_file = mount / "inside.txt"
    inside_file.write_text("ok", encoding="utf-8")
    escaped_file = outside / "escaped.txt"
    escaped_file.write_text("nope", encoding="utf-8")

    drv = make_storage_driver(mount_url=f"file://{mount}")

    assert drv.valid(f"file://{inside_file}") is not None
    assert drv.valid(f"file://{mount / '..' / 'outside' / 'escaped.txt'}") is None


def test_local_valid_allows_relative_paths_only_within_mount(tmp_path):
    top = tmp_path / "imagedir"
    mount = top / "admin"
    mount.mkdir(parents=True)
    inside_file = mount / "inside.txt"
    inside_file.write_text("ok", encoding="utf-8")

    drv = make_storage_driver(mount_url=f"file://{mount}", top=f"file://{top}")

    assert drv.valid("admin/inside.txt") is not None
    assert drv.valid("../inside.txt") is None


def test_local_valid_rejects_symlink_escape(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are not supported on this platform")

    mount = tmp_path / "mount"
    outside = tmp_path / "outside"
    mount.mkdir()
    outside.mkdir()

    secret = outside / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    symlink = mount / "linked_secret.txt"
    symlink.symlink_to(secret)

    drv = make_storage_driver(mount_url=f"file://{mount}")

    assert symlink.is_symlink()
    assert drv.valid(f"file://{symlink}") is None
