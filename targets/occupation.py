"""Occupation category (Executives / Clerical & service / Blue-collar): Random Forest (notebook \"occupation\")."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import EDUCATION_NUM, Target, clean_income, load_or_train, read_adult

NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "relationship", "race", "sex", "native-country", "income"]
FEATURES = ["age", "workclass", "education-num", "marital-status", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

GROUPES = {
    "Exec-managerial": "Exec_Prof", "Prof-specialty": "Exec_Prof", "Tech-support": "Exec_Prof",
    "Adm-clerical": "Clerical_Service", "Sales": "Clerical_Service", "Other-service": "Clerical_Service",
    "Protective-serv": "Clerical_Service", "Priv-house-serv": "Clerical_Service",
    "Craft-repair": "Blue_collar", "Machine-op-inspct": "Blue_collar", "Transport-moving": "Blue_collar",
    "Handlers-cleaners": "Blue_collar", "Farming-fishing": "Blue_collar", "Armed-Forces": "Blue_collar",
}
LABELS = {"Exec_Prof": "Executives and professionals", "Clerical_Service": "Clerical and service",
          "Blue_collar": "Blue-collar and manual trades"}
DESCR = {"Exec_Prof": "Executives, professional specialties, tech support.",
         "Clerical_Service": "Administrative, sales, services, protective services.",
         "Blue_collar": "Manual trades, transport, farming."}


def _features(df):
    X = df[FEATURES].copy()
    for c in ("capital-gain", "capital-loss"):
        X[c] = np.log1p(X[c])  # many zeros and a few huge amounts
    return X


def _train():
    df = pd.concat([read_adult("adult.data"), read_adult("adult.test", skiprows=1)], ignore_index=True)
    df["income"] = clean_income(df["income"])
    df = df.dropna(subset=["occupation"]).drop_duplicates().reset_index(drop=True)
    y, X = df["occupation"].map(GROUPES), _features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42)
    pipe = Pipeline([("prep", prep), ("model", rf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
               "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
               "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
               "f1": round(float(f1_score(y_te, pred, average="weighted")), 4)}
    return {"models": {"rf": {
        "name": "Random Forest", "description": "300 voting trees (bagging), 14 occupations grouped into 3 categories.",
        "pipeline": pipe, "metrics": metrics, "note": f"20% test set, {len(y_te)} rows. Chance would give ≈ 0.33."}}}


def load():
    return load_or_train("occupation", _train)


def predict(bundle, model_id, values):
    row = {**values, "education-num": EDUCATION_NUM[values["education"]]}
    for c in ("workclass", "native-country"):  # "Not specified" -> blank, filled in by the model
        if row[c] == "Unknown":
            row[c] = np.nan
    pipe = bundle["models"][model_id]["pipeline"]
    proba = pipe.predict_proba(_features(pd.DataFrame([row])))[0]
    classes = list(pipe.classes_)
    best = classes[int(proba.argmax())]
    return {"label": LABELS[best], "description": DESCR[best],
            "probas": sorted(((LABELS[c], float(p)) for c, p in zip(classes, proba)), key=lambda x: -x[1])}


TARGET = Target(
    slug="occupation", title="Occupation", icon="💼",
    question="Which occupation category does this profile belong to?",
    detail="Random Forest: executive / professional, clerical / service, or blue-collar.",
    fields=("age", "race", "workclass", "sex", "education", "capital-gain", "marital-status", "capital-loss",
            "relationship", "hours-per-week", "income", "native-country"),
    unknown_ok=("workclass", "native-country"), load=load, predict=predict)
