import pytest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sample_file_path = os.path.join(os.path.dirname(__file__), "../sample_input.json")


from main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Flask App is Running!" in response.data


def test_predict(client):
    """Test prediction using sample_input.json"""
    with open(sample_file_path, "r") as file:
        sample_data = json.load(file)

    response = client.post("/predict", json=sample_data)
    json_data = response.get_json()

    assert response.status_code == 200
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)  # Ensure it's a list
    assert len(json_data["predictions"]) > 0  # Ensure predictions exist
