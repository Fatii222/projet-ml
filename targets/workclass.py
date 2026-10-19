"""Workclass (8 classes): Random Forest (notebook \"TP_Adult_Prediction_Workclass\").

Imbalanced target (\"Private\" ≈ 74%, \"Without-pay\" and \"Never-worked\" almost absent). The notebook's full
class_weight='balanced' pushes accuracy down to ≈ 0.49 (below the 0.74 \"always Private\" baseline), so here the
class weights are softened (inverse frequency ** 0.5): accuracy stays ≈ 0.73 and macro F1 stays ≈ 0.24.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core import EDUCATION_NUM, OPTIONS, Target, load_or_train, read_adult

NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
FEATURES = ["age", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]  # without fnlwgt or income (as in the notebook)

LABELS = dict(OPTIONS["workclass"])
DESCR = {"Private": "Employee of a private company. Overwhelming majority class (≈ 74% of people).",
         "Self-emp-not-inc": "Self-employed, not incorporated (craftsperson, freelancer, farmer…).",
         "Self-emp-inc": "Self-employed, with an incorporated business.",
         "Federal-gov": "Employee of the federal government.",
         "State-gov": "Employee of a state government.",
         "Local-gov": "Employee of a local government (city, county).",
         "Without-pay": "Unpaid work (family help…). Very rare class: almost never predicted.",
         "Never-worked": "Has never worked. Very rare class: almost never predicted."}

WEIGHT_POWER = 0.5  # 0 = no class weights, 1 = fully "balanced" (the notebook's choice)
PARAMS = dict(n_estimators=300, min_samples_leaf=10)


def _class_weights(y):
    """Softened 'balanced' weights: (n / (k * n_c)) ** WEIGHT_POWER, so rare classes count more but do not dominate."""
    counts = y.value_counts()
    return {c: (len(y) / (len(counts) * n)) ** WEIGHT_POWER for c, n in counts.items()}


def _train():
    df = read_adult("adult.data").drop_duplicates().dropna(subset=["workclass"]).reset_index(drop=True)
    y, X = df["workclass"].astype(str), df[FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=_class_weights(y_tr), **PARAMS)
    pipe = Pipeline([("prep", prep), ("model", rf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
               "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
               "f1_macro": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
               "f1": round(float(f1_score(y_te, pred, average="weighted", zero_division=0)), 4)}
    return {"models": {"rf": {
        "name": "Random Forest",
        "description": "300 trees, leaves ≥ 10 rows, softened class weights (inverse frequency ** 0.5).",
        "pipeline": pipe, "metrics": metrics,
        "note": f"20% test set, {len(y_te)} rows. Always predicting \"Private\" would score 74% accuracy but would never find the other "
                f"classes: macro F1 and balanced accuracy show how well the minority classes are spotted."}}}


def load():
    return load_or_train("workclass", _train)


def predict(bundle, model_id, values):
    row = {**values, "education-num": EDUCATION_NUM[values["education"]]}
    for c in ("occupation", "native-country"):  # "Not specified" -> blank, filled in by the model
        if row[c] == "Unknown":
            row[c] = np.nan
    pipe = bundle["models"][model_id]["pipeline"]
    proba = pipe.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    classes = list(pipe.classes_)
    best = classes[int(proba.argmax())]
    return {"label": LABELS[best], "description": DESCR[best],
            "probas": sorted(((LABELS[c], float(p)) for c, p in zip(classes, proba)), key=lambda x: -x[1])}


TARGET = Target(
    slug="workclass", title="Workclass", icon="🏢",
    question="Which workclass does this person work in?",
    detail="Random Forest: private, self-employed, government (federal, state, local)…",
    fields=("age", "sex", "education", "marital-status", "occupation", "relationship", "race", "native-country",
            "capital-gain", "capital-loss", "hours-per-week"),
    unknown_ok=("occupation", "native-country"), load=load, predict=predict)
