import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from apps.api.main import app

@pytest.fixture
def client():
    return TestClient(app)
