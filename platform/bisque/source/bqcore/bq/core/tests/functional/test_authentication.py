# -*- coding: utf-8 -*-
"""
Integration tests for the :mod:`repoze.who`-powered authentication sub-system.

As bqcore grows and the authentication method changes, only these tests
should be updated.

"""
# !!! modern way with pytest

import pytest
class TestAuthentication:
    """Tests for the default authentication setup."""

    def test_forced_login(self, app):
        """Anonymous users are forced to login and redirected properly."""
        # Request a protected area
        # !!! /client_service/upload is not valid anymore
        # resp = app.get('/client_service/upload', status=302)
        resp = app.get('/import/transfer', status=302)
        assert resp.location.startswith('http://localhost/auth_service/login')

        # Follow redirect to login form
        resp = resp.follow(status=200)
        form = resp.form

        # Submit login form
        form['login'] = 'admin'
        form['password'] = 'admin'
        post_login = form.submit(status=302)

        # Should redirect to post_login
        assert post_login.location.startswith('http://localhost/auth_service/post_login')

        # Follow redirect to originally requested page
        initial_page = post_login.follow(status=302)

        cookies = {c.name: c.value for c in app.cookiejar}
        assert 'authtkt' in cookies, \
            f"Session cookie wasn't defined: {cookies}"

        # assert initial_page.location.startswith('http://localhost/client_service/upload')
        assert initial_page.location.startswith('http://localhost/import/transfer')

    def test_voluntary_login(self, app):
        """Voluntary logins must work correctly."""
        resp = app.get('/auth_service/login', status=200)
        form = resp.form

        form['login'] = 'admin'
        form['password'] = 'admin'
        post_login = form.submit(status=302)

        assert post_login.location.startswith('http://localhost/auth_service/post_login')

        home_page = post_login.follow(status=302)
        cookies = {c.name: c.value for c in app.cookiejar}

        assert 'authtkt' in cookies, \
            f'Session cookie was not defined: {cookies}'

        assert home_page.location == 'http://localhost/'

    def test_token_issue_and_bearer_whoami(self, app):
        """Token endpoint should issue bearer tokens usable on auth_service APIs."""
        token_response = app.post_json(
            '/auth_service/token',
            {'username': 'admin', 'password': 'admin', 'grant_type': 'password'},
            status=200,
        )
        payload = token_response.json
        assert payload.get('token_type') == 'Bearer'
        token = payload.get('access_token')
        assert token, "Token response missing access_token"

        whoami = app.get(
            '/auth_service/whoami',
            headers={'Authorization': f'Bearer {token}'},
            status=200,
        )
        assert 'name' in whoami.text
        assert 'admin' in whoami.text

    def test_token_invalid_credentials(self, app):
        """Token endpoint should reject bad credentials with 401."""
        token_response = app.post_json(
            '/auth_service/token',
            {'username': 'admin', 'password': 'not-the-password', 'grant_type': 'password'},
            status=400,
        )
        payload = token_response.json
        assert payload.get('error') == 'invalid_grant'

#!!! old way with nose
# from bq.core.tests import TestController


# class TestAuthentication(TestController):
#     """Tests for the default authentication setup.

#     By default in TurboGears 2, :mod:`repoze.who` is configured with the same
#     plugins specified by repoze.what-quickstart (which are listed in
#     http://code.gustavonarea.net/repoze.what-quickstart/#repoze.what.plugins.quickstart.setup_sql_auth).

#     As the settings for those plugins change, or the plugins are replaced,
#     these tests should be updated.

#     """

#     application_under_test = 'main'

#     def test_forced_login(self):
#         """Anonymous users are forced to login

#         Test that anonymous users are automatically redirected to the login
#         form when authorization is denied. Next, upon successful login they
#         should be redirected to the initially requested page.

#         """
#         # Requesting a protected area
#         resp = self.app.get('/client_service/upload', status=302)
#         assert resp.location.startswith('http://localhost/auth_service/login')
#         # Getting the login form:
#         resp = resp.follow(status=200)
#         form = resp.form
#         # Submitting the login form:
#         form['login'] = 'admin'
#         form['password'] = 'admin'
#         post_login = form.submit(status=302)
#         # Being redirected to the initially requested page:
#         assert post_login.location.startswith('http://localhost/auth_service/post_login')
#         initial_page = post_login.follow(status=302)
#         assert 'auth_tkt' in initial_page.request.cookies, \
#                "Session cookie wasn't defined: %s" % initial_page.request.cookies
#         assert initial_page.location.startswith('http://localhost/client_service/upload'), \
#                initial_page.location

#     def test_voluntary_login(self):
#         """Voluntary logins must work correctly"""
#         # Going to the login form voluntarily:
#         resp = self.app.get('/auth_service/login', status=200)
#         form = resp.form
#         # Submitting the login form:
#         form['login'] = 'admin'
#         form['password'] = 'admin'
#         post_login = form.submit(status=302)
#         # Being redirected to the home page:
#         assert post_login.location.startswith('http://localhost/auth_service/post_login')
#         home_page = post_login.follow(status=302)
#         cookies = {c.name: c.value for c in self.app.cookiejar}
#         # assert 'auth_tkt' in home_page.request.cookies, \
#         #        'Session cookie was not defined: %s' % home_page.request.cookies
#         # !!! modern way
#         assert 'authtkt' in cookies, \
#                'Session cookie was not defined: %s' % cookies

#         assert home_page.location == 'http://localhost/'
# !!! this was commented out before
#     def test_logout(self):
#         """Logouts must work correctly"""
#         # Logging in voluntarily the quick way:
#         resp = self.app.get('/auth_service/login?login=admin&password=admin',
#                             status=302)
#         resp = resp.follow(status=302)
#         assert 'authtkt' in resp.request.cookies, \
#                'Session cookie was not defined: %s' % resp.request.cookies
#         # Logging out:
#         resp = self.app.get('/auth_service/logout_handler', status=302)
#         assert resp.location.startswith('http://localhost/post_logout')
#         # Finally, redirected to the home page:
#         home_page = resp.follow(status=302)
#         authtkt = home_page.request.cookies.get('authtkt')
#         assert not authtkt or authtkt == 'INVALID', \
#                'Session cookie was not deleted: %s' % home_page.request.cookies
#         assert home_page.location == 'http://localhost/', home_page.location
