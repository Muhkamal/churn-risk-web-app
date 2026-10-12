"""Churn Risk Checker: a small web app that serves the churn model.

    python train.py      # once, creates model.joblib
    python app.py        # then open http://127.0.0.1:5055

Endpoints
  GET  /              a form that calls the API
  POST /api/predict   JSON in, risk score and the main reasons out
  GET  /api/health    is the model loaded?
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "model.joblib")
CARD = json.loads((BASE / "model_card.json").read_text())
THRESHOLD = CARD["alert_threshold"]

PRE = MODEL.named_steps["columntransformer"]
LR = MODEL.named_steps["logisticregression"]
FEATURE_NAMES = [n.split("__", 1)[1] for n in PRE.get_feature_names_out()]

NUMERIC_RULES = {  # field: (minimum, maximum, whole number only)
    "tenure_months": (1, 72, True),
    "support_calls": (0, 30, True),
    "late_payments": (0, 30, True),
    "has_bundle": (0, 1, True),
    "monthly_charge_ngn": (1000, 100000, False),
}
CHOICES = {k: CARD["inputs"][k] for k in ("contract", "plan", "payment")}
SHARES = CARD["level_shares"]
NUM_STATS = {n: (m, s) for n, m, s in zip(
    [f for f in FEATURE_NAMES if f in NUMERIC_RULES],
    PRE.named_transformers_["n"].mean_, PRE.named_transformers_["n"].scale_)}


def describe(field, value):
    """The customer's own value, in words."""
    if field == "tenure_months":
        return f"{value} months as a customer"
    if field == "support_calls":
        return f"{value} support calls"
    if field == "late_payments":
        return f"{value} late payments"
    if field == "has_bundle":
        return "has a bundle" if value else "no bundle"
    if field == "monthly_charge_ngn":
        return f"NGN {value:,.0f} monthly charge"
    if field == "payment":
        return f"pays by {value.lower()}"
    return f"{value} {field}"


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096  # a prediction request is tiny


def validate(payload):
    """Return (clean_row, errors). Every problem is reported, not just the first."""
    errors, row = {}, {}
    if not isinstance(payload, dict):
        return None, {"body": "send a JSON object"}
    for field, (lo, hi, whole) in NUMERIC_RULES.items():
        if field not in payload:
            errors[field] = "required"
            continue
        v = payload[field]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors[field] = "must be a number"
        elif whole and v != int(v):
            errors[field] = "must be a whole number"
        elif not lo <= v <= hi:
            errors[field] = f"must be between {lo} and {hi}"
        else:
            row[field] = int(v) if whole else float(v)
    for field, allowed in CHOICES.items():
        if field not in payload:
            errors[field] = "required"
        elif payload[field] not in allowed:
            errors[field] = "must be one of: " + ", ".join(allowed)
        else:
            row[field] = payload[field]
    return (row, errors) if not errors else (None, errors)


def factors(row):
    """The three inputs that moved this customer's score most, measured against the average customer."""
    coefs = dict(zip(FEATURE_NAMES, LR.coef_[0]))
    effects = {}
    for field in NUMERIC_RULES:
        mean, scale = NUM_STATS[field]
        effects[field] = coefs[field] * (row[field] - mean) / scale
    for field in CHOICES:
        level_coef = lambda lvl: coefs.get(f"{field}_{lvl}", 0.0)  # the baseline level has coefficient 0
        average = sum(share * level_coef(lvl) for lvl, share in SHARES[field].items())
        effects[field] = level_coef(row[field]) - average
    ranked = sorted(effects.items(), key=lambda kv: -abs(kv[1]))[:3]
    return [
        {"factor": describe(k, row[k]) if k != "contract" and k != "plan" else f"{row[k]} {k}",
         "effect": "raises risk" if v > 0 else "lowers risk", "strength": round(abs(v), 2)}
        for k, v in ranked
    ]


@app.get("/")
def home():
    return render_template("index.html", card=CARD, choices=CHOICES, rules=NUMERIC_RULES)


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_test_auc=CARD["test_auc"], simulated_data=True)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    row, errors = validate(payload)
    if errors:
        return jsonify(errors=errors), 400
    frame = pd.DataFrame([row])
    score = float(MODEL.predict_proba(frame)[0, 1])
    return jsonify(
        risk_score=round(score, 4),
        flagged=score >= THRESHOLD,
        threshold=THRESHOLD,
        top_factors=factors(row),
        compared_with="the average customer in the training data",
        note="Score ranks risk on simulated data. It is not a real probability of leaving.",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=False)
