"""Real Streamlit run of app.py (headless AppTest), complementing the stub smoke test.
Skipped when streamlit is not installed; it runs in CI."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_renders_without_exceptions():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) >= 4
