from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.apps.inference import create_app
from app.inference.runtime.served import ServedModel


@pytest.fixture
def app(served: ServedModel) -> FastAPI:
    return create_app(served)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def unstarted_client(app: FastAPI) -> TestClient:
    """A client whose app never ran its lifespan, like a worker that hasn't started."""
    return TestClient(app, raise_server_exceptions=False)
