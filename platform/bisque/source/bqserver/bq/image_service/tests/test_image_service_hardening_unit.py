import json
import importlib
import types

from bq.image_service.controllers import service as service_controller
from bq.image_service.controllers.converters import converter_ffmpeg as ffmpeg_module
from bq.image_service.controllers.converters import converter_openslide as openslide_module
from bq.image_service.controllers.converters.converter_bioformats import ConverterBioformats
from bq.image_service.controllers.converters.converter_imaris import ConverterImaris
from bq.image_service.controllers.operations.thumbnail import ThumbnailOperation
from bq.image_service.controllers.process_token import ProcessToken
from bq.image_service.controllers.resource_cache import ResourceCache

image_api = importlib.import_module("bq.image_service.api")


class _DummyLockContext:
    def __init__(self, locked=True):
        self.locked = locked

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_imaris_thumbnail_preproc_builds_valid_command(monkeypatch, tmp_path):
    calls = {}

    def _fake_run(cls, ifnm, ofnm, command, **kw):
        calls["command"] = list(command)
        return ofnm

    monkeypatch.setattr(ConverterImaris, "run", classmethod(_fake_run))

    token = ProcessToken(ifnm=str(tmp_path / "input.ims"), series=0)
    result = ConverterImaris.thumbnail(
        token,
        str(tmp_path / "thumb.tif"),
        128,
        128,
        preproc="mid",
    )
    assert result == str(tmp_path / "thumb.tif")
    assert "-tm" in calls["command"]
    idx = calls["command"].index("-tm")
    assert calls["command"][idx + 1] == "MiddleSlice"


def test_openslide_histogram_returns_none_on_slide_error(monkeypatch, tmp_path):
    class _OpenSlideError(RuntimeError):
        pass

    class _FakeOpenSlideNS:
        OpenSlideError = _OpenSlideError
        OpenSlideUnsupportedFormatError = _OpenSlideError

        @staticmethod
        def OpenSlide(_path):
            raise _OpenSlideError("cannot open slide")

    monkeypatch.setattr(openslide_module, "openslide", _FakeOpenSlideNS, raising=False)
    monkeypatch.setattr(openslide_module.ConverterOpenSlide, "supported", classmethod(lambda cls, token, **kw: True))
    monkeypatch.setattr(openslide_module.misc, "start_nounicode_win", lambda ifnm, args: (args, None))
    monkeypatch.setattr(openslide_module.misc, "end_nounicode_win", lambda tmp: None)

    token = ProcessToken(ifnm=str(tmp_path / "input.svs"))
    token.dims = {"image_num_x": 1024, "image_num_y": 1024}
    assert openslide_module.ConverterOpenSlide.writeHistogram(token, str(tmp_path / "hist.bin")) is None


def test_openslide_histogram_writes_python3_safe_header(monkeypatch, tmp_path):
    class _OpenSlideError(RuntimeError):
        pass

    class _FakeSlide:
        level_count = 1

        def close(self):
            return None

    class _FakeImage:
        def histogram(self):
            return [1] * 256

    class _FakeDeepZoomGenerator:
        def __init__(self, slide, tile_size=1024, overlap=0):
            self.level_count = 1

        def get_tile(self, level, tile):
            return _FakeImage()

    class _FakeOpenSlideNS:
        OpenSlideError = _OpenSlideError
        OpenSlideUnsupportedFormatError = _OpenSlideError

        @staticmethod
        def OpenSlide(_path):
            return _FakeSlide()

    monkeypatch.setattr(openslide_module, "openslide", _FakeOpenSlideNS, raising=False)
    monkeypatch.setattr(
        openslide_module,
        "deepzoom",
        types.SimpleNamespace(DeepZoomGenerator=_FakeDeepZoomGenerator),
        raising=False,
    )
    monkeypatch.setattr(openslide_module.ConverterOpenSlide, "supported", classmethod(lambda cls, token, **kw: True))
    monkeypatch.setattr(openslide_module.misc, "start_nounicode_win", lambda ifnm, args: (args, None))
    monkeypatch.setattr(openslide_module.misc, "end_nounicode_win", lambda tmp: None)

    token = ProcessToken(ifnm=str(tmp_path / "input.svs"))
    token.dims = {"image_num_x": 1024, "image_num_y": 1024}
    out = tmp_path / "hist.bin"
    assert openslide_module.ConverterOpenSlide.writeHistogram(token, str(out)) == str(out)
    data = out.read_bytes()
    assert data.startswith(b"BIM1IHS1")


def test_thumbnail_dryrun_handles_zero_dimensions_without_unbound_error(tmp_path):
    class _DummyConverters:
        @staticmethod
        def defaultExtension(_fmt):
            return "jpg"

    class _DummyServer:
        converters = _DummyConverters()

    op = ThumbnailOperation(server=_DummyServer())
    token = ProcessToken(ifnm=str(tmp_path / "input.tif"))
    token.setFile(str(tmp_path / "workfile"))
    token.dims = {"image_num_x": 1, "image_num_y": 0}

    out = op.dryrun(token, "0,0")
    assert out is token
    assert out.getDim("image_num_x", None) == 0
    assert out.getDim("image_num_y", None) == 0


def test_local_dispatch_handles_numeric_and_uniq_ids(monkeypatch):
    calls = []

    class _DummyToken:
        data = "ok"

        @staticmethod
        def isHttpError():
            return False

        @staticmethod
        def isText():
            return True

        @staticmethod
        def isFile():
            return False

    def _fake_process(request, ident, **kw):
        calls.append((request, ident, kw))
        return _DummyToken()

    dispatch = image_api.proxy_dispatch()
    dispatch.baseurl = "/image_service"
    dispatch.server = types.SimpleNamespace(srv=types.SimpleNamespace(process=_fake_process))
    monkeypatch.setattr(image_api.identity, "get_username", lambda: "alice")

    assert dispatch.local_dispatch("/image_service/123?dims") == "ok"
    assert dispatch.local_dispatch("/image_service/00-abc?dims") == "ok"
    assert calls[0][1] == 123
    assert calls[1][1] == "00-abc"
    assert calls[0][2]["user_name"] == "alice"
    assert calls[1][2]["user_name"] == "alice"


def test_build_content_disposition_handles_unicode_filename():
    disposition = service_controller.build_content_disposition("café.png", is_inline=False)
    assert disposition.startswith("attachment; ")
    assert "filename*=UTF-8''" in disposition
    assert "b'" not in disposition


def test_ffmpeg_info_returns_empty_for_malformed_or_missing_video(monkeypatch):
    monkeypatch.setattr(ffmpeg_module, "Locks", lambda *a, **kw: _DummyLockContext(True))
    monkeypatch.setattr(ffmpeg_module.ConverterFfmpeg, "supported", classmethod(lambda cls, token, **kw: True))
    token = ProcessToken(ifnm="/tmp/video.mp4")

    class _BadJsonPopen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd

        def communicate(self):
            return b"{bad json", b""

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", _BadJsonPopen)
    assert ffmpeg_module.ConverterFfmpeg.info(token) == {}

    payload = {"streams": [{"codec_type": "audio"}], "format": {"format_name": "mp4"}}

    class _NoVideoPopen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd

        def communicate(self):
            if self.cmd[0] == "ffprobe":
                return json.dumps(payload).encode("utf-8"), b""
            return b"", b""

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", _NoVideoPopen)
    assert ffmpeg_module.ConverterFfmpeg.info(token) == {}


def test_ffmpeg_info_defaults_frame_count_on_invalid_rate(monkeypatch):
    monkeypatch.setattr(ffmpeg_module, "Locks", lambda *a, **kw: _DummyLockContext(True))
    monkeypatch.setattr(ffmpeg_module.ConverterFfmpeg, "supported", classmethod(lambda cls, token, **kw: True))
    token = ProcessToken(ifnm="/tmp/video.mp4")

    ffprobe_payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 512,
                "height": 256,
                "avg_frame_rate": "0/0",
            }
        ],
        "format": {
            "format_name": "mp4",
            "size": "10",
            "duration": "10.0",
        },
    }

    class _Popen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd

        def communicate(self):
            if self.cmd[0] == "ffprobe":
                return json.dumps(ffprobe_payload).encode("utf-8"), b""
            if self.cmd[0] == "file":
                return b"video data", b""
            return b"", b""

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", _Popen)
    info = ffmpeg_module.ConverterFfmpeg.info(token)
    assert info.get("image_num_t") == 1
    assert info.get("image_num_x") == 512
    assert info.get("image_num_y") == 256


def test_bioformats_meta_returns_empty_without_pixels(monkeypatch):
    monkeypatch.setattr(ConverterBioformats, "installed", True)

    payload = (
        'Checking file format [OME]\n'
        'Series count = 1\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<StructuredAnnotations xmlns="http://www.openmicroscopy.org/Schemas/SA/2016-06" />'
        '<Image ID="Image:0"><AcquisitionDate>2020-01-01T00:00:00</AcquisitionDate></Image>'
        '</OME>'
    )
    monkeypatch.setattr(ConverterBioformats, "run_read", classmethod(lambda cls, ifnm, cmd: payload))

    token = ProcessToken(ifnm="/tmp/file.fake", series=0)
    monkeypatch.setattr("os.path.exists", lambda _path: True)
    assert ConverterBioformats.meta(token) == {}


def test_bioformats_convert_skips_rename_when_conversion_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ConverterBioformats, "installed_formats", {"jpeg": types.SimpleNamespace(ext=["jpg"])})
    monkeypatch.setattr(ConverterBioformats, "run", classmethod(lambda cls, ifnm, ofnm, cmd: None))

    called = {"rename": 0}
    monkeypatch.setattr("os.rename", lambda *_args, **_kw: called.__setitem__("rename", called["rename"] + 1))
    token = ProcessToken(ifnm="/tmp/in.fake", series="not-an-int")
    out = ConverterBioformats.convert(token, str(tmp_path / "out.bin"), fmt="jpeg")
    assert out is None
    assert called["rename"] == 0


def test_resource_cache_applies_bounds(monkeypatch):
    current_user = {"id": "user-1"}
    monkeypatch.setattr("bq.image_service.controllers.resource_cache.identity.get_user_id", lambda: current_user["id"])

    cache = ResourceCache(max_users=1, max_entries_per_user=1)
    cache.get_descriptor("resource-1")
    cache.get_descriptor("resource-2")
    assert "resource-1" not in cache.d["user-1"]
    assert "resource-2" in cache.d["user-1"]

    current_user["id"] = "user-2"
    cache.get_descriptor("resource-3")
    assert "user-1" not in cache.d
    assert "user-2" in cache.d
