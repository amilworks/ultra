# -*- coding: utf-8 -*-

import pytest
import logging


def pytest_addoption(parser):
    group = parser.getgroup('bisque')
    #group.addoption(
    #    '--foo',
    #    action='store',
    #    dest='dest_foo',
    #    default='2016',
    #    help='Set the value for the fixture "bar".'
    #)

    #parser.addini('HELLO', 'Dummy pytest.ini setting')


import os
import pytest
from bqapi import BQSession
from bq.util.configfile import ConfigFile
from bq.util.bunch import Bunch

DEFAULT_TEST_INI = "config/test.ini"


def resolve_test_ini():
    env_override = os.environ.get("BISQUE_TEST_INI")
    if env_override:
        candidate = os.path.abspath(os.path.expanduser(env_override))
        if not os.path.exists(candidate):
            raise RuntimeError(
                "BISQUE_TEST_INI points to a missing file: %s" % candidate
            )
        return candidate

    cwd_candidate = os.path.abspath(DEFAULT_TEST_INI)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    source_candidate = os.path.abspath(
        os.path.join(os.getcwd(), "source", "config", "test.ini")
    )
    if os.path.exists(source_candidate):
        return source_candidate

    raise RuntimeError(
        "Could not locate test configuration file.\n"
        "Checked:\n"
        "  - %s\n"
        "  - %s\n"
        "Provide BISQUE_TEST_INI or run scripts/dev/init_test_config.sh."
        % (cwd_candidate, source_candidate)
    )

def load_application_config(filename):
    # setup resources before any test is executed
    from paste.deploy import appconfig, load_app
    from bq.config.environment import load_environment
    from webtest import TestApp

    conf = appconfig('config:' + os.path.abspath(filename))
    load_environment(conf.global_conf, conf.local_conf)
    return conf

def load_test_application(filename):
    # setup resources before any test is executed
    from paste.deploy import appconfig, loadapp
    from bq.config.environment import load_environment
    from webtest import TestApp
    from paste.script.appinstall import SetupCommand


    print("pytest_bisque:load_test_application:", filename)
    wsgiapp = loadapp('config:' + os.path.abspath(filename))
    # Note: logging.config.fileConfig is already called by site.cfg during loadapp
    # logging.config.fileConfig (filename)  # This line causes KeyError: 'formatters'
    app = TestApp(wsgiapp)
    app.authorization = ('Basic', ('admin', 'admin'))
    #KGK Following lines are required to create database tables.. but somehow turn off logging??
    #KGKcmd = SetupCommand('setup-app')
    #KGKcmd.run([filename])
    #KGK Just run this command externally before tests:
    #KGK   paster setup-app config/test.ini

    return app

@pytest.fixture(scope="session") #once per run
def test_ini_path():
    return resolve_test_ini()


@pytest.fixture(scope="session") #once per run
def application (test_ini_path):
    return load_test_application (test_ini_path)



def load_api_config(filename):
    config = ConfigFile (filename)
    try:
        cfg = Bunch(config.get ('test', asdict=True))
    except Exception as exc:
        raise RuntimeError(
            "Invalid test config '%s': missing [test] section or required keys"
            % filename
        ) from exc
    try:
        cfg.store = Bunch (config.get ('store', asdict=True))
    except Exception:
        cfg.store = Bunch()
    return cfg


@pytest.fixture(scope="session") # once per run
def config(test_ini_path):
    "Load the bisque test config/test.ini"
    cfg =  load_api_config (test_ini_path)
    print("CFG", cfg)
    return cfg



@pytest.fixture(scope="module") # once per module
def session(config):
    "Create a BQApi BQSession object based on config"
    host = config.get ( 'host.root')
    user = config.get ( 'host.user')
    passwd = config.get ( 'host.password')

    bq = BQSession()
    bq.config = config
    bq.init_local (user, passwd, bisque_root = host, create_mex = False)
    yield  bq
    bq.close()


@pytest.fixture(scope="module") # once per module
def mexsession(config):
    "Create a BQApi BQSession object based on config"
    host = config.get ( 'host.root')
    user = config.get ( 'host.user')
    passwd = config.get ( 'host.password')

    bq = BQSession()
    bq.config = config
    bq.init_local (user, passwd, bisque_root = host, create_mex = True)
    yield  bq
    bq.close()
