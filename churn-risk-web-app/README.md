# Churn Risk Checker (web app)

A small Flask web app that serves the churn model from the `churn-prediction` project. Type in a customer, get a risk score and the main reasons behind it.

> **The model is trained on simulated customers**, so scores say nothing about real people. The page says this on screen.

## What is in it
| Part | What it does |
|---|---|
| `train.py` | Rebuilds the churn model (same data and split as `churn-prediction`, test AUC 0.751) and saves `model.joblib`, `model_card.json` and `demo_params.json` |
| `app.py` | Flask app: `GET /` form, `POST /api/predict`, `GET /api/health` |
| `templates/index.html` | The form. It calls the API and shows the result |
| `tests/test_app.py` | 18 tests |

## API
```
POST /api/predict
{"tenure_months": 6, "contract": "Monthly", "plan": "Basic", "payment": "Cash",
 "support_calls": 4, "late_payments": 2, "has_bundle": 0, "monthly_charge_ngn": 12000}

200 {"risk_score": 0.9079, "flagged": true, "threshold": 0.575,
     "top_factors": [{"factor": "4 support calls", "effect": "raises risk", "strength": 0.54}, ...],
     "compared_with": "the average customer in the training data", ...}
400 {"errors": {"tenure_months": "must be between 1 and 72", ...}}
```
- Every input is checked. All problems are reported together, not one at a time.
- The three reasons are measured against the average customer, because a score is only meaningful relative to something.
- The alert threshold 0.575 was chosen by naira cost on a separate validation set in `churn-prediction`.

## Tests (18 passing)
Health and home page, a valid request, missing fields, nine bad-value cases, non-JSON input, that a risky customer scores well above a safe one, that the API equals the model called directly on 40 random customers, that the explanations describe the customer correctly, and that the numbers exported for the browser demo reproduce the real model to 9 decimal places.

## Limits
- Simulated training data. A real deployment would need real data and a fresh evaluation.
- The model uses balanced class weights, so the score **ranks** risk and is **not** a true probability of leaving.
- AUC 0.751 is a modest model. The page does not hide that.
- Runs locally with Flask's built-in server. For a public deployment use a production server such as gunicorn and add authentication.
- The model file depends on the scikit-learn version, so `requirements.txt` pins it.

## Run
```bash
pip install -r requirements.txt
python train.py
python app.py          # open http://127.0.0.1:5055
pytest
```
