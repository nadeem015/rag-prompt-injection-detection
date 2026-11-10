"""STUB-BASED smoke test of app.py -- this is NOT a launch of the Streamlit application.

The real `streamlit` package could not be installed where this project was prepared (the package index was
unreachable), so the Streamlit UI has never been started in a browser.  This test only checks that the
script `app.py` executes top to bottom, including a "Run pipeline" click, against a minimal fake `streamlit`
module, i.e. that the application logic and its calls into `src/` are not broken.  Layout, widgets,
reruns, file upload and browser behaviour are NOT tested.
"""
from __future__ import annotations

import contextlib
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, *a, **k):
        _Stub.calls.append("metric")


class _Session(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Stub(types.ModuleType):
    calls: list = []

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _Session()
        self.sidebar = self          # st.sidebar.<widget> behaves like st.<widget>

    # widgets with return values
    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def checkbox(self, label, value=False, **k):
        return value

    def slider(self, label, lo, hi, value=None, step=None, **k):
        return value if value is not None else lo

    def text_input(self, label, value="", **k):
        return value

    def text_area(self, label, value="", **k):
        return value

    def file_uploader(self, *a, **k):
        return []

    def button(self, label, **k):
        return label == "Run pipeline"   # click only the main button

    def tabs(self, names):
        return [_Ctx() for _ in names]

    def columns(self, n, **k):
        return [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]

    def expander(self, *a, **k):
        return _Ctx()

    def rerun(self):
        raise AssertionError("st.rerun must not be reached in the smoke run")

    def __getattr__(self, name):         # st.title, st.write, st.json, st.dataframe, ... -> recorded no-ops
        if name.startswith("__"):
            raise AttributeError(name)

        def _f(*a, **k):
            _Stub.calls.append(name)
        return _f


def test_app_executes_against_stub_streamlit(monkeypatch, tmp_path):
    stub = _Stub()
    _Stub.calls = []
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    # redirect the event log so the test does not append to the repository's log file
    from src import events
    _orig_init = events.EventLogger.__init__
    monkeypatch.setattr(events.EventLogger, "__init__", lambda self, path: _orig_init(self, tmp_path / "events.jsonl"))
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(ROOT))
    with contextlib.redirect_stdout(None):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    calls = _Stub.calls
    assert "title" in calls and "metric" in calls          # dashboard rendered
    assert "success" in calls or "error" in calls          # the pipeline ran and produced an answer or a refusal
    assert (tmp_path / "events.jsonl").exists()            # events were logged (to the redirected path)
    assert ns["cfg"].detector_mode == "hybrid_fusion"
