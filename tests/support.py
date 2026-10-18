"""Test plumbing.

* A throwaway `config` module pointing at a synthetic library (tests/fixtures.py)
  is registered BEFORE any app module is imported — every app module does
  `from config import ...` at import time.
* load_page() builds a namespace from a Streamlit page file WITHOUT running the
  page: it executes the imports, constants and function/class definitions, and
  skips everything else (widgets, layout, page_setup()). The page's own
  references to `st` are pointed at a small fake with a plain-dict
  session_state, so its functions can be called directly from a test.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PLANNER_TEST_DATA", REPO / "tests" / ".data"))

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _install_config() -> None:
    if "config" in sys.modules and getattr(sys.modules["config"], "_IS_TEST_CONFIG", False):
        return
    from tests.fixtures import write_library
    paths = write_library(DATA)
    cfg = types.ModuleType("config")
    cfg._IS_TEST_CONFIG = True
    for k, v in paths.items():
        setattr(cfg, k, v)
    cfg.APP_ENV = "test"
    cfg.DRIVE_WRITES = False
    cfg.WHITELISTED_EMAILS = ["tester@example.com"]
    cfg.CURVE_CACHE_DIR = DATA / ".curve_cache"
    sys.modules["config"] = cfg


_install_config()


# ---------------------------------------------------------------------------
# Fake streamlit for page namespaces
# ---------------------------------------------------------------------------

class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _decorator(*args, **kwargs):
    """Works as @x, @x() and @x(...): returns the function unchanged."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        fn = args[0]
        fn.clear = lambda *a, **k: None
        return fn

    def wrap(fn):
        fn.clear = lambda *a, **k: None
        return fn
    return wrap


class FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.session_state = SessionState()
        self.cache_data = _decorator
        self.cache_resource = _decorator
        self.fragment = _decorator
        self.experimental_fragment = _decorator

    def __getattr__(self, name):          # any widget / layout call is a no-op
        def _noop(*a, **k):
            return None
        return _noop


_CONST = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


def load_page(path, session_state: dict | None = None) -> dict:
    """Namespace of a page's definitions (see module docstring)."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    fake = FakeStreamlit()
    if session_state:
        fake.session_state.update(session_state)
    ns = {"__name__": f"page_{path.stem}", "__file__": str(path)}

    def run(node):
        code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
        exec(code, ns)

    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))]
    for node in imports:
        run(node)
    ns["st"] = fake
    ns["fragment"] = _decorator
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            run(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names and all(_CONST.match(n) for n in names):
                try:
                    run(node)
                except Exception:      # a constant built from UI state — not needed by tests
                    pass
    ns["st"] = fake          # an import above may have rebound it
    return ns


PLANNER = REPO / "pages" / "1_Compilation_Planner.py"
