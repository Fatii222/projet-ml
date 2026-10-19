"""Family role (6 classes): Random Forest (notebook \"relationship\")."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import EDUCATION_NUM, Target, load_or_train, read_adult

NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]

LABELS = {k: k for k in ("Husband", "Wife", "Own-child", "Not-in-family", "Unmarried", "Other-relative")}
DESCR = {"Husband": "Male spouse, married.", "Wife": "Female spouse, married.",
         "Own-child": "Child of the head of household.", "Not-in-family": "Lives alone or outside a family unit.",
         "Unmarried": "Not married, lives in a household with a partner or family.",
         "Other-relative": "Other family tie. Rare class: the least well predicted."}


def _train():
    df = read_adult("adult.data").drop_duplicates().dropna().reset_index(drop=True)  # rows with "?" dropped
    y, X = df["relationship"], df[FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", rf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
               "precision": round(float(precision_score(y_te, pred, average="weighted")), 4),
               "recall": round(float(recall_score(y_te, pred, average="weighted")), 4),
               "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
               "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4)}
    return {"models": {"rf": {
        "name": "Random Forest", "description": "100 trees (max depth 20), class_weight='balanced'.",
        "pipeline": pipe, "metrics": metrics, "note": f"20% test set, {len(y_te)} rows (weighted averages)."}}}


def load():
    return load_or_train("relationship", _train)


def predict(bundle, model_id, values):
    row = {**values, "education-num": EDUCATION_NUM[values["education"]]}
    pipe = bundle["models"][model_id]["pipeline"]
    proba = pipe.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    classes = list(pipe.classes_)
    best = classes[int(proba.argmax())]
    return {"label": LABELS[best], "description": DESCR[best],
            "probas": sorted(((LABELS[c], float(p)) for c, p in zip(classes, proba)), key=lambda x: -x[1])}


TARGET = Target(
    slug="relationship", title="Relationship", icon="👪",
    question="What is this person's family role in the household?",
    detail="Random Forest: Husband, Wife, Own-child, Not-in-family, Unmarried or Other-relative.",
    fields=("age", "race", "workclass", "sex", "education", "capital-gain", "marital-status", "capital-loss",
            "occupation", "hours-per-week", "native-country"),
    unknown_ok=(), load=load, predict=predict)
