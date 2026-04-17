from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_auth_service_reconciles_missing_bq_user() -> None:
    auth_service = (
        ROOT / "platform/bisque/source/bqserver/bq/client_service/controllers/auth_service.py"
    ).read_text()
    assert "def _ensure_bq_user_record(self, tg_user):" in auth_service
    assert "Missing BQUser for TG user %s; creating it during auth reconciliation" in auth_service
    assert "self._ensure_bq_user_record(tg_user)" in auth_service


def test_bootstrap_recovers_missing_admin_bq_user() -> None:
    bootstrap = (ROOT / "platform/bisque/source/bqcore/bq/websetup/bootstrap.py").read_text()
    assert "Bootstrap found TG admin without BQUser; creating missing BQUser record" in bootstrap
    assert (
        "admin_tg = model.DBSession.query(model.User).filter_by(user_name='admin').first()"
        in bootstrap
    )


def test_admin_ui_blocks_local_user_creation_in_oidc_mode() -> None:
    admin_service = (
        ROOT / "platform/bisque/source/bqserver/bq/admin_service/controllers/service.py"
    ).read_text()
    manager_js = (
        ROOT / "platform/bisque/source/bqcore/bq/core/public/js/admin/BQ.user.Manager.js"
    ).read_text()

    assert (
        "Local BisQue user creation is disabled while OIDC-only auth is enabled." in admin_service
    )
    assert "Create approved users in Keycloak" in admin_service
    assert "def auth_settings(self):" in admin_service
    assert "'local_user_creation_enabled': mode != 'oidc'" in admin_service
    assert "loadAuthSettings: function()" in manager_js
    assert "url: '/admin/auth_settings'" in manager_js
    assert (
        "Create approved users in Keycloak, then have them sign in once to provision their BisQue profile."
        in manager_js
    )
    assert "message = response.responseText;" in manager_js
