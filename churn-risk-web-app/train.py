"""Train the churn model that the web app serves.

The customer data is SIMULATED. The generator below is a copy of the one in
the churn-prediction project (same seed, same split), so this is the same
model, now packaged to be served. It writes:
  model.joblib      the fitted scikit-learn pipeline
  model_card.json   inputs, limits, metrics and the alert threshold
  demo_params.json  the model's numbers, so the same score can be computed
                    in a browser without a server
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 7
N = 7000
rng = np.random.default_rng(SEED)
THRESHOLD = 0.575  # chosen by naira cost on a validation set in churn-prediction

# ------------------------------------------------------------------ simulate
df = pd.DataFrame(
    {
        "tenure_months": np.clip(rng.exponential(22, N), 1, 72).round().astype(int),
        "contract": rng.choice(["Monthly", "One year", "Two year"], N, p=[0.58, 0.27, 0.15]),
        "plan": rng.choice(["Basic", "Standard", "Premium"], N, p=[0.45, 0.35, 0.20]),
        "payment": rng.choice(["Auto-debit", "Card", "Transfer", "Cash"], N, p=[0.30, 0.25, 0.30, 0.15]),
        "support_calls": rng.poisson(1.3, N),
        "late_payments": rng.poisson(0.5, N),
        "has_bundle": rng.choice([0, 1], N, p=[0.6, 0.4]),
    }
)
base = df["plan"].map({"Basic": 5000, "Standard": 12000, "Premium": 24000})
df["monthly_charge_ngn"] = (base * rng.normal(1.0, 0.12, N)).round(-2).astype(int)

logit = (
    -0.035 * df["tenure_months"]
    + 1.15 * (df["contract"] == "Monthly")
    - 0.85 * (df["contract"] == "Two year")
    + 0.22 * df["support_calls"]
    + 0.30 * df["late_payments"]
    - 0.55 * df["has_bundle"]
    + 0.00004 * (df["monthly_charge_ngn"] - 12000)
    + 0.35 * (df["payment"] == "Cash")
    - 0.30 * (df["payment"] == "Auto-debit")
    + rng.normal(0, 0.8, N)
)
lo, hi = -6.0, 6.0
for _ in range(60):  # tune the intercept so about 26% of customers churn
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-(logit + mid)))).mean() > 0.26:
        hi = mid
    else:
        lo = mid
p = 1 / (1 + np.exp(-(logit + mid)))
df["churned"] = (rng.random(N) < p).astype(int)
df.to_csv("customers_simulated.csv", index=False)

# ------------------------------------------------------------------ model
y = df["churned"]
X = df.drop(columns="churned")
cat = ["contract", "plan", "payment"]
num = [c for c in X.columns if c not in cat]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)


model = make_pipeline(
    ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(drop="first"), cat)]),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
model.fit(X_tr, y_tr)
test_auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

joblib.dump(model, "model.joblib")

pre = model.named_steps["columntransformer"]
lr = model.named_steps["logisticregression"]
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
scaler = pre.named_transformers_["n"]
ohe = pre.named_transformers_["c"]

shares = {c: {str(l): round(float((X_tr[c] == l).mean()), 4) for l in ohe.categories_[i]} for i, c in enumerate(cat)}

card = {
    "simulated_data": True,
    "level_shares": shares,
    "model": "Logistic regression with balanced class weights",
    "test_auc": round(test_auc, 3),
    "alert_threshold": THRESHOLD,
    "training_rows": int(len(X_tr)),
    "inputs": {
        "tenure_months": "whole number, 1 to 72",
        "contract": list(ohe.categories_[0]),
        "plan": list(ohe.categories_[1]),
        "payment": list(ohe.categories_[2]),
        "support_calls": "whole number, 0 to 30 (calls in the last 6 months)",
        "late_payments": "whole number, 0 to 30",
        "has_bundle": "0 or 1",
        "monthly_charge_ngn": "number, 1000 to 100000",
    },
    "limits": [
        "Trained on simulated customers, so scores say nothing about real people.",
        "The model uses balanced class weights, so the score ranks risk. It is not a true probability of leaving.",
        "Only the alert threshold was tuned for cost. The cost assumptions are in churn-prediction.",
    ],
}
json.dump(card, open("model_card.json", "w"), indent=2)

demo = {
    "threshold": THRESHOLD,
    "intercept": float(lr.intercept_[0]),
    "numeric": [
        {"name": n, "mean": float(m), "scale": float(s), "coef": float(c)}
        for n, m, s, c in zip(num, scaler.mean_, scaler.scale_, lr.coef_[0][: len(num)])
    ],
    "categorical": {
        col: [{"level": str(level), "coef": float(c)} for level, c in zip(cats[1:], lr.coef_[0][start: start + len(cats) - 1])]
        for col, cats, start in [
            ("contract", ohe.categories_[0], len(num)),
            ("plan", ohe.categories_[1], len(num) + len(ohe.categories_[0]) - 1),
            ("payment", ohe.categories_[2], len(num) + len(ohe.categories_[0]) + len(ohe.categories_[1]) - 2),
        ]
    },
    "shares": shares,
    "baselines": {"contract": str(ohe.categories_[0][0]), "plan": str(ohe.categories_[1][0]), "payment": str(ohe.categories_[2][0])},
    "test_auc": round(test_auc, 3),
}
json.dump(demo, open("demo_params.json", "w"), indent=2)
print("test AUC:", round(test_auc, 3), "| features:", names)
