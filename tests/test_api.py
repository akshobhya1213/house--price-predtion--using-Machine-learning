import pytest
from src.predict import DEFAULTS, MODEL_PATH
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")

@pytest.fixture
def client():
    from app import app
    return app.test_client()

def test_health(client):
    assert client.get("/health").json == {"status": "ok"}

def test_predict_ok(client):
    r = client.post("/predict", json=DEFAULTS)
    assert r.status_code == 200 and r.json["price_usd"] > 0

def test_predict_missing_field(client):
    assert client.post("/predict", json={"MedInc": 3}).status_code == 400
