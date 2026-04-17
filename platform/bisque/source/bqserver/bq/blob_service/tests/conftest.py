import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
for extra_path in [ROOT / "bqserver", ROOT / "bqcore"]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


def _install_legacy_import_stubs():
    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.iter_entry_points = lambda *args, **kwargs: []
        sys.modules["pkg_resources"] = pkg_resources

    if "shortuuid" not in sys.modules:
        shortuuid = types.ModuleType("shortuuid")
        shortuuid.uuid = lambda: "00000000000000000000000000000000"
        sys.modules["shortuuid"] = shortuuid

    if "tg" not in sys.modules:
        tg = types.ModuleType("tg")

        class _Config:
            def get(self, *_args, **_kwargs):
                return None

        controllers = types.ModuleType("tg.controllers")
        controllers.RestController = type("RestController", (), {})

        render = types.ModuleType("tg.render")
        render.render = lambda *args, **kwargs: None

        tg.config = _Config()
        tg.TGController = type("TGController", (), {})
        tg.expose = lambda *args, **kwargs: (lambda func: func)
        tg.request = types.SimpleNamespace(host_url="http://localhost/")
        tg.tmpl_context = types.SimpleNamespace()
        tg.controllers = controllers
        tg.render = render
        sys.modules["tg"] = tg
        sys.modules["tg.controllers"] = controllers
        sys.modules["tg.render"] = render

    if "pylons" not in sys.modules:
        pylons = types.ModuleType("pylons")
        i18n = types.ModuleType("pylons.i18n")
        i18n.ugettext = lambda text: text
        i18n.ungettext = lambda singular, plural, n: singular if n == 1 else plural
        i18n.N_ = lambda text: text
        pylons.i18n = i18n
        sys.modules["pylons"] = pylons
        sys.modules["pylons.i18n"] = i18n

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


_install_legacy_import_stubs()
