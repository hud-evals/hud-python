"""Example coding tasks."""

from env import coding_task
from env import env as env

_flask_config_from_file = coding_task(
    description="""Add a file mode parameter to flask.Config.from_file()
Python 3.11 introduced native TOML support with the `tomllib` package. This could work nicely with the `flask.Config.from_file()` method as an easy way to load TOML config files:

```python
app.config.from_file("config.toml", tomllib.load)
```

However, `tomllib.load()` takes an object readable in binary mode, while `flask.Config.from_file()` opens a file in text mode, resulting in this error:

```
TypeError: File must be opened in binary mode, e.g. use `open('foo.toml', 'rb')`
```

We can get around this with a more verbose expression, like loading from a file opened with the built-in `open()` function and passing the `dict` to `app.Config.from_mapping()`:

```python
# We have to repeat the path joining that from_file() does
with open(os.path.join(app.config.root_path, "config.toml"), "rb") as file:
    app.config.from_mapping(tomllib.load(file))
```

But adding a file mode parameter to `flask.Config.from_file()` would enable the use of a simpler expression. E.g.:

```python
app.config.from_file("config.toml", tomllib.load, mode="b")
```
""",
    test_command=(
        "PYTHONPATH=src python -m pytest -q -W ignore::DeprecationWarning tests/test_config.py --junitxml={junit_path}"
    ),
    test_patch="""diff --git a/tests/static/config.toml b/tests/static/config.toml
new file mode 100644
--- /dev/null
+++ b/tests/static/config.toml
@@ -0,0 +1,2 @@
+TEST_KEY="foo"
+SECRET_KEY="config"
diff --git a/tests/test_config.py b/tests/test_config.py
--- a/tests/test_config.py
+++ b/tests/test_config.py
@@ -6,7 +6,6 @@
\x20
 import flask
\x20
-
 # config keys used for the TestConfig
 TEST_KEY = "foo"
 SECRET_KEY = "config"
@@ -30,13 +29,23 @@ def test_config_from_object():
     common_object_test(app)
\x20
\x20
-def test_config_from_file():
+def test_config_from_file_json():
     app = flask.Flask(__name__)
     current_dir = os.path.dirname(os.path.abspath(__file__))
     app.config.from_file(os.path.join(current_dir, "static", "config.json"), json.load)
     common_object_test(app)
\x20
\x20
+def test_config_from_file_toml():
+    tomllib = pytest.importorskip("tomllib", reason="tomllib added in 3.11")
+    app = flask.Flask(__name__)
+    current_dir = os.path.dirname(os.path.abspath(__file__))
+    app.config.from_file(
+        os.path.join(current_dir, "static", "config.toml"), tomllib.load, text=False
+    )
+    common_object_test(app)
+
+
 def test_from_prefixed_env(monkeypatch):
     monkeypatch.setenv("FLASK_STRING", "value")
     monkeypatch.setenv("FLASK_BOOL", "true")
""",
    test_path="tests",
    base_ref="origin/flask_4992_baseline",
    fail_to_pass=["tests.test_config.test_config_from_file_toml"],
    pass_to_pass=[
        "tests.test_config.test_config_from_pyfile",
        "tests.test_config.test_config_from_object",
        "tests.test_config.test_config_from_file_json",
        "tests.test_config.test_from_prefixed_env",
        "tests.test_config.test_from_prefixed_env_custom_prefix",
        "tests.test_config.test_from_prefixed_env_nested",
        "tests.test_config.test_config_from_mapping",
        "tests.test_config.test_config_from_class",
        "tests.test_config.test_config_from_envvar",
        "tests.test_config.test_config_from_envvar_missing",
        "tests.test_config.test_config_missing",
        "tests.test_config.test_config_missing_file",
        "tests.test_config.test_custom_config_class",
        "tests.test_config.test_session_lifetime",
        "tests.test_config.test_get_namespace",
        "tests.test_config.test_from_pyfile_weird_encoding[utf-8]",
        "tests.test_config.test_from_pyfile_weird_encoding[iso-8859-15]",
        "tests.test_config.test_from_pyfile_weird_encoding[latin-1]",
    ],
    binary=True,
)
_flask_config_from_file.slug = "flask-4992"


_flask_routes_domains = coding_task(
    description="""Flask routes to return domain/sub-domains information
Currently when checking **flask routes** it provides all routes but **it is no way to see which routes are assigned to which subdomain**.

**Default server name:**
SERVER_NAME: 'test.local'

**Domains (sub-domains):**
test.test.local
admin.test.local
test.local

**Adding blueprints:**
app.register_blueprint(admin_blueprint,url_prefix='',subdomain='admin')
app.register_blueprint(test_subdomain_blueprint,url_prefix='',subdomain='test')

```
$ flask routes
 * Tip: There are .env or .flaskenv files present. Do "pip install python-dotenv" to use them.
Endpoint                                                 Methods    Rule
-------------------------------------------------------  ---------  ------------------------------------------------
admin_blueprint.home                                      GET        /home
test_subdomain_blueprint.home                             GET        /home
static                                                    GET        /static/<path:filename>
...
```

**Feature request**
It will be good to see something like below (that will make more clear which route for which subdomain, because now need to go and check configuration).
**If it is not possible to fix routes**, can you add or tell which method(s) should be used to get below information from flask?

```
$ flask routes
 * Tip: There are .env or .flaskenv files present. Do "pip install python-dotenv" to use them.
Domain                Endpoint                                             Methods    Rule
-----------------   ----------------------------------------------------  ----------  ------------------------------------------------
admin.test.local     admin_blueprint.home                                  GET        /home
test.test.local      test_subdomain_blueprint.home                         GET        /home
test.local           static                                                GET        /static/<path:filename>
...
```
""",
    test_command=(
        "PYTHONPATH=src python -m pytest -q -W ignore::DeprecationWarning tests/test_cli.py --junitxml={junit_path}"
    ),
    test_patch="""diff --git a/tests/test_cli.py b/tests/test_cli.py
--- a/tests/test_cli.py
+++ b/tests/test_cli.py
@@ -433,16 +433,12 @@ class TestRoutes:
     @pytest.fixture
     def app(self):
         app = Flask(__name__)
-        app.testing = True
-
-        @app.route("/get_post/<int:x>/<int:y>", methods=["GET", "POST"])
-        def yyy_get_post(x, y):
-            pass
-
-        @app.route("/zzz_post", methods=["POST"])
-        def aaa_post():
-            pass
-
+        app.add_url_rule(
+            "/get_post/<int:x>/<int:y>",
+            methods=["GET", "POST"],
+            endpoint="yyy_get_post",
+        )
+        app.add_url_rule("/zzz_post", methods=["POST"], endpoint="aaa_post")
         return app
\x20
     @pytest.fixture
@@ -450,17 +446,6 @@ def invoke(self, app, runner):
         cli = FlaskGroup(create_app=lambda: app)
         return partial(runner.invoke, cli)
\x20
-    @pytest.fixture
-    def invoke_no_routes(self, runner):
-        def create_app():
-            app = Flask(__name__, static_folder=None)
-            app.testing = True
-
-            return app
-
-        cli = FlaskGroup(create_app=create_app)
-        return partial(runner.invoke, cli)
-
     def expect_order(self, order, output):
         # skip the header and match the start of each row
         for expect, line in zip(order, output.splitlines()[2:]):
@@ -493,11 +478,31 @@ def test_all_methods(self, invoke):
         output = invoke(["routes", "--all-methods"]).output
         assert "GET, HEAD, OPTIONS, POST" in output
\x20
-    def test_no_routes(self, invoke_no_routes):
-        result = invoke_no_routes(["routes"])
+    def test_no_routes(self, runner):
+        app = Flask(__name__, static_folder=None)
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
         assert result.exit_code == 0
         assert "No routes were registered." in result.output
\x20
+    def test_subdomain(self, runner):
+        app = Flask(__name__, static_folder=None)
+        app.add_url_rule("/a", subdomain="a", endpoint="a")
+        app.add_url_rule("/b", subdomain="b", endpoint="b")
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Subdomain" in result.output
+
+    def test_host(self, runner):
+        app = Flask(__name__, static_folder=None, host_matching=True)
+        app.add_url_rule("/a", host="a", endpoint="a")
+        app.add_url_rule("/b", host="b", endpoint="b")
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Host" in result.output
+
\x20
 def dotenv_not_available():
     try:
""",
    test_path="tests",
    base_ref="origin/flask_5063_baseline",
    fail_to_pass=[
        "tests.test_cli.TestRoutes.test_subdomain",
        "tests.test_cli.TestRoutes.test_host",
    ],
    pass_to_pass=[
        "tests.test_cli.test_cli_name",
        "tests.test_cli.test_find_best_app",
        "tests.test_cli.test_prepare_import[test-path0-test]",
        "tests.test_cli.test_prepare_import[test.py-path1-test]",
        "tests.test_cli.test_prepare_import[a/test-path2-test]",
        "tests.test_cli.test_prepare_import[test/__init__.py-path3-test]",
        "tests.test_cli.test_prepare_import[test/__init__-path4-test]",
        "tests.test_cli.test_prepare_import[value5-path5-cliapp.inner1]",
        "tests.test_cli.test_prepare_import[value6-path6-cliapp.inner1.inner2]",
        "tests.test_cli.test_prepare_import[test.a.b-path7-test.a.b]",
        "tests.test_cli.test_prepare_import[value8-path8-cliapp.app]",
        "tests.test_cli.test_prepare_import[value9-path9-cliapp.message.txt]",
        "tests.test_cli.test_locate_app[cliapp.app-None-testapp]",
        "tests.test_cli.test_locate_app[cliapp.app-testapp-testapp]",
        "tests.test_cli.test_locate_app[cliapp.factory-None-app]",
        "tests.test_cli.test_locate_app[cliapp.factory-create_app-app]",
        "tests.test_cli.test_locate_app[cliapp.factory-create_app()-app]",
        'tests.test_cli.test_locate_app[cliapp.factory-create_app2("foo",',
        "tests.test_cli.test_locate_app[cliapp.factory-",
        "tests.test_cli.test_locate_app_raises[notanapp.py-None]",
        "tests.test_cli.test_locate_app_raises[cliapp/app-None]",
        "tests.test_cli.test_locate_app_raises[cliapp.app-notanapp]",
        'tests.test_cli.test_locate_app_raises[cliapp.factory-create_app2("foo")]',
        "tests.test_cli.test_locate_app_raises[cliapp.factory-create_app(]",
        "tests.test_cli.test_locate_app_raises[cliapp.factory-no_app]",
        "tests.test_cli.test_locate_app_raises[cliapp.importerrorapp-None]",
        "tests.test_cli.test_locate_app_raises[cliapp.message.txt-None]",
        "tests.test_cli.test_locate_app_suppress_raise",
        "tests.test_cli.test_get_version",
        "tests.test_cli.test_scriptinfo",
        "tests.test_cli.test_app_cli_has_app_context",
        "tests.test_cli.test_with_appcontext",
        "tests.test_cli.test_appgroup_app_context",
        "tests.test_cli.test_flaskgroup_app_context",
        "tests.test_cli.test_flaskgroup_debug[True]",
        "tests.test_cli.test_flaskgroup_debug[False]",
        "tests.test_cli.test_flaskgroup_nested",
        "tests.test_cli.test_no_command_echo_loading_error",
        "tests.test_cli.test_help_echo_loading_error",
        "tests.test_cli.test_help_echo_exception",
        "tests.test_cli.TestRoutes.test_simple",
        "tests.test_cli.TestRoutes.test_sort",
        "tests.test_cli.TestRoutes.test_all_methods",
        "tests.test_cli.TestRoutes.test_no_routes",
        "tests.test_cli.test_load_dotenv",
        "tests.test_cli.test_dotenv_path",
        "tests.test_cli.test_dotenv_optional",
        "tests.test_cli.test_disable_dotenv_from_env",
        "tests.test_cli.test_run_cert_path",
        "tests.test_cli.test_run_cert_adhoc",
        "tests.test_cli.test_run_cert_import",
        "tests.test_cli.test_run_cert_no_ssl",
        "tests.test_cli.test_cli_blueprints",
        "tests.test_cli.test_cli_empty",
    ],
    binary=True,
)
_flask_routes_domains.slug = "flask-5063"


tasks = [_flask_config_from_file, _flask_routes_domains]
