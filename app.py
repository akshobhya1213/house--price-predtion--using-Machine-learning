"""Flask API + tiny web form. Run: python app.py"""
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from src.data import FEATURES
from src.predict import DEFAULTS, MODEL_PATH

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>House Price Predictor</title>
<style>body{font-family:system-ui;max-width:420px;margin:2rem auto;padding:0 1rem}label{display:block;margin:.6rem 0 .2rem;font-size:.9rem}input{width:100%;padding:.5rem;box-sizing:border-box}button{margin-top:1rem;padding:.7rem;width:100%;font-size:1rem}#out{font-size:1.5rem;margin-top:1rem;font-weight:600}</style>
<h2>House Price Predictor</h2><form id=f>%FIELDS%<button>Predict</button></form><div id=out></div>
<script>f.onsubmit=async e=>{e.preventDefault();const d=Object.fromEntries(new FormData(f));const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});const j=await r.json();out.textContent=j.error?j.error:'$'+Math.round(j.price_usd).toLocaleString()}</script>"""

@app.get("/")
def index():
    fields = "".join(f'<label>{k}</label><input name="{k}" type="number" step="any" value="{DEFAULTS[k]}" required>' for k in FEATURES)
    return PAGE.replace("%FIELDS%", fields)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    missing = [k for k in FEATURES if k not in data]
    if missing:
        return jsonify(error=f"Missing fields: {missing}"), 400
    try:
        row = pd.DataFrame([{k: float(data[k]) for k in FEATURES}])
    except (TypeError, ValueError):
        return jsonify(error="All fields must be numeric"), 400
    return jsonify(price_usd=float(model.predict(row)[0]) * 100_000)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
