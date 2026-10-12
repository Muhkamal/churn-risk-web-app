import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as appmod  # noqa: E402

GOOD = {"tenure_months": 6, "contract": "Monthly", "plan": "Basic", "payment": "Cash",
        "support_calls": 4, "late_payments": 2, "has_bundle": 0, "monthly_charge_ngn": 12000}


@pytest.fixture()
def client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def post(client, payload):
    return client.post("/api/predict", json=payload)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Churn Risk Checker" in r.data


def test_valid_request_returns_score_and_reasons(client):
    d = post(client, GOOD).get_json()
    assert 0 <= d["risk_score"] <= 1
    assert isinstance(d["flagged"], bool)
    assert 1 <= len(d["top_factors"]) <= 3
    assert d["flagged"] == (d["risk_score"] >= d["threshold"])


def test_missing_fields_are_all_reported(client):
    r = post(client, {"tenure_months": 5})
    assert r.status_code == 400
    assert {"contract", "plan", "payment", "support_calls"} <= set(r.get_json()["errors"])


@pytest.mark.parametrize("field,value", [
    ("tenure_months", 0), ("tenure_months", 500), ("tenure_months", 3.5), ("tenure_months", "6"),
    ("contract", "Weekly"), ("payment", "Bitcoin"), ("monthly_charge_ngn", -5), ("has_bundle", 2), ("support_calls", True),
])
def test_bad_values_are_rejected(client, field, value):
    r = post(client, {**GOOD, field: value})
    assert r.status_code == 400 and field in r.get_json()["errors"]


def test_not_json_is_rejected(client):
    r = client.post("/api/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_risk_orders_sensibly(client):
    risky = {**GOOD, "tenure_months": 2, "contract": "Monthly", "support_calls": 8, "late_payments": 4, "has_bundle": 0}
    safe = {**GOOD, "tenure_months": 60, "contract": "Two year", "support_calls": 0, "late_payments": 0, "has_bundle": 1, "payment": "Auto-debit"}
    assert post(client, risky).get_json()["risk_score"] > post(client, safe).get_json()["risk_score"] + 0.3


def test_api_matches_the_model_directly(client):
    rng = random.Random(1)
    for _ in range(40):
        row = {"tenure_months": rng.randint(1, 72), "contract": rng.choice(appmod.CHOICES["contract"]),
               "plan": rng.choice(appmod.CHOICES["plan"]), "payment": rng.choice(appmod.CHOICES["payment"]),
               "support_calls": rng.randint(0, 10), "late_payments": rng.randint(0, 6), "has_bundle": rng.randint(0, 1),
               "monthly_charge_ngn": rng.randrange(2000, 40000, 100)}
        direct = float(appmod.MODEL.predict_proba(pd.DataFrame([row]))[0, 1])
        assert abs(post(client, row).get_json()["risk_score"] - direct) < 1e-4


def test_exported_demo_parameters_reproduce_the_model():
    """The portfolio page recomputes the score in JavaScript from demo_params.json.
    This does the same arithmetic in Python and compares it with the real model."""
    p = json.loads((Path(__file__).resolve().parent.parent / "demo_params.json").read_text())
    rng = random.Random(2)
    worst = 0.0
    for _ in range(300):
        row = {"tenure_months": rng.randint(1, 72), "contract": rng.choice(appmod.CHOICES["contract"]),
               "plan": rng.choice(appmod.CHOICES["plan"]), "payment": rng.choice(appmod.CHOICES["payment"]),
               "support_calls": rng.randint(0, 10), "late_payments": rng.randint(0, 6), "has_bundle": rng.randint(0, 1),
               "monthly_charge_ngn": rng.randrange(2000, 60000, 100)}
        z = p["intercept"]
        for n in p["numeric"]:
            z += n["coef"] * (row[n["name"]] - n["mean"]) / n["scale"]
        for col, levels in p["categorical"].items():
            z += sum(l["coef"] for l in levels if l["level"] == row[col])
        mine = 1 / (1 + np.exp(-z))
        direct = float(appmod.MODEL.predict_proba(pd.DataFrame([row]))[0, 1])
        worst = max(worst, abs(mine - direct))
    assert worst < 1e-9


def test_explanation_describes_the_customer(client):
    no_bundle = post(client, {**GOOD, "has_bundle": 0, "tenure_months": 30, "support_calls": 1, "late_payments": 0,
                              "contract": "Monthly", "monthly_charge_ngn": 11000, "payment": "Auto-debit"}).get_json()
    text = " ".join(f["factor"] for f in no_bundle["top_factors"])
    assert "having a bundle" not in text
    two_year = post(client, {**GOOD, "contract": "Two year"}).get_json()
    contract = [f for f in two_year["top_factors"] if f["factor"] == "Two year contract"]
    assert contract and contract[0]["effect"] == "lowers risk"
