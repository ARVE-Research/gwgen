
def pytest_addoption(parser):
    group = parser.getgroup("gwgen", "GWGEN specific options")
    group.addoption('--offline', help="Block outgoing internet connections",
                    action='store_true')
    group.addoption('--no-db', help="Don't test postgres databases",
                    action='store_true')
    group.addoption('--database',
                    help=("The name of the postgres database to use. Default: "
                          "%(default)s"), default='travis_ci_test')
    group.addoption('--user',
                    help="The username to access the postgres database")
    group.addoption('--host',
                    help=("The hostname for the postgres database. Default: "
                          "%(default)s"), default='127.0.0.1')
    group.addoption('--port', help="The port of the postgres database.")
    group.addoption('--no-remove', action='store_true',
                    help=("Do not remove the test directory at the end of the "
                          "tests."))


def pytest_configure(config):
    import _base_testing as bt
    if config.getoption('offline'):
        bt.online = False
    if config.getoption('no_db'):
        bt.BaseTest.use_db = False
    for option in ['database', 'user', 'host', 'port']:
        bt.db_config[option] = config.getoption(option)
    if config.getoption('no_remove'):
        bt.BaseTest.remove_at_cleanup = False
