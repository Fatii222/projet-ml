"""Income (> $50K / ≤ $50K): Logistic Regression + Random Forest (full sklearn pipelines)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import COLS, EDUCATION_NUM, Target, clean_income, load_or_train, read_adult

NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
INPUT_COLS = [c for c in COLS if c != "income"]
FNLWGT_DEFAULT = 178356  # dataset median, not entered by the user


def _load(name, skiprows=0, dedupe=False):
    df = read_adult(name, skiprows)
    if dedupe:
        df = df.drop_duplicates()
    for c in ("workclass", "occupation", "native-country"):
        df[c] = df[c].fillna("Unknown")
    df["income"] = clean_income(df["income"]).map({"<=50K": 0, ">50K": 1})
    return df.dropna(subset=["income"]).astype({"income": int})


def _metrics(y, pred):
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "precision": round(float(precision_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4), "f1": round(float(f1_score(y, pred)), 4)}


def _train():
    df = _load("adult.data", dedupe=True)
    X, y = df.drop(columns="income"), df["income"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test = _load("adult.test", skiprows=1)
    X_te, y_te = test.drop(columns="income"), test["income"]
    specs = {
        "logreg": ("Logistic Regression", "StandardScaler + OneHotEncoder, class_weight='balanced'.",
                   LogisticRegression(max_iter=1000, class_weight="balanced")),
        "rf": ("Random Forest", "200 trees, class_weight='balanced'.",
               RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    }
    models = {}
    for key, (name, desc, clf) in specs.items():
        prep = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)]).fit(X_tr, y_tr)
        models[key] = {"name": name, "description": desc, "pipeline": pipe,
                       "metrics": _metrics(y_te, pipe.predict(X_te)), "note": f"official adult.test, {len(y_te)} lignes"}
    return {"models": models}


def load():
    return load_or_train("income", _train)


def predict(bundle, model_id, values):
    row = {**values, "fnlwgt": FNLWGT_DEFAULT, "education-num": EDUCATION_NUM[values["education"]]}
    p_high = float(bundle["models"][model_id]["pipeline"].predict_proba(pd.DataFrame([row])[INPUT_COLS])[0][1])
    return {"label": "Income > $50,000 per year" if p_high >= 0.5 else "Income ≤ $50,000 per year",
            "description": "Decision threshold at 50%.",
            "probas": [("> $50,000 per year", p_high), ("≤ $50,000 per year", 1 - p_high)]}


TARGET = Target(
    slug="income", title="Income", icon="💰",
    question="Does this profile earn more than $50,000 per year?",
    detail="Logistic Regression and Random Forest: pick a model or compare them.",
    fields=("age", "race", "workclass", "sex", "education", "capital-gain", "marital-status", "capital-loss",
            "occupation", "hours-per-week", "relationship", "native-country"),
    unknown_ok=("workclass", "occupation", "native-country"), load=load, predict=predict)
