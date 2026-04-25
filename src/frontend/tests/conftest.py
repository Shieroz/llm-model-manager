"""Shared fixtures for frontend E2E tests."""
import pytest
import subprocess
import time
import httpx
import os
import sys
import shutil

BASE_URL = "http://127.0.0.1:18976"
_TMP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tmp")


@pytest.fixture(scope="session")
def backend_server():
    """Start the FastAPI backend server for E2E tests."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    test_server = os.path.join(os.path.dirname(__file__), "run_test_server.py")

    # Use the .venv Python to ensure all dependencies are available
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    proc = subprocess.Popen(
        [venv_python, test_server],
        cwd=project_root,
        env={**os.environ, "LOG_LEVEL": "WARNING"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(30):
        try:
            with httpx.Client(timeout=2) as client:
                r = client.get(f"{BASE_URL}/")
                if r.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        stderr = proc.stderr.read().decode()
        proc.terminate()
        raise RuntimeError(f"Backend server failed to start:\n{stderr}")

    yield proc

    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(autouse=True)
def wait_for_server(backend_server):
    """Ensure server is running before each test."""
    time.sleep(0.5)
    yield


@pytest.fixture
def http():
    """Provide an httpx client for API testing."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as client:
        yield client


@pytest.fixture
def html(http):
    """Provide parsed HTML from the main page."""
    from bs4 import BeautifulSoup
    response = http.get("/")
    assert response.status_code == 200
    return BeautifulSoup(response.text, "html.parser")


def pytest_sessionfinish(session, exitstatus):
    """Clean up tmp directory after tests complete."""
    if os.path.isdir(_TMP_DIR):
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
