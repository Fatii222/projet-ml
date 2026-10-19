"""Education level (3 groups): XGBoost."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from core import Target, load_or_train, read_adult

NUM = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "occupation", "marital-status", "race", "native-country", "income"]
FEATURES = NUM + CAT

GROUPS = {
    **dict.fromkeys(["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad"],
                    "1_Secondary_or_less"),
    **dict.fromkeys(["Some-college", "Assoc-voc", "Assoc-acdm"], "2_Short_post-secondary"),
    **dict.fromkeys(["Bachelors", "Masters", "Prof-school", "Doctorate"], "3_Higher"),
}
LABELS = {"1_Secondary_or_less": "Secondary or less", "2_Short_post-secondary": "Short post-secondary",
          "3_Higher": "Higher education"}
DESCR = {"1_Secondary_or_less": "From preschool to high-school graduate (HS-grad).",
         "2_Short_post-secondary": "Some-college, vocational or associate degrees (1 to 2 years after high school).",
         "3_Higher": "Bachelors, Masters, Prof-school or Doctorate."}


def clean_features(d):
    """Missing values -> 'Unknown', country -> US / Non-US / Unknown, log1p on gains and losses."""
    d = d.copy()
    for c in ["workclass", "occupation", "native-country"]:
        d[c] = d[c].fillna("Unknown")
    d["native-country"] = np.where(d["native-country"] == "United-States", "US",
                                   np.where(d["native-country"] == "Unknown", "Unknown", "Non-US"))
    d["capital-gain"] = np.log1p(d["capital-gain"])
    d["capital-loss"] = np.log1p(d["capital-loss"])
    return d[FEATURES]


def _train():
    train = read_adult("adult.data").drop_duplicates()
    test = read_adult("adult.test", skiprows=1)
    test["income"] = test["income"].str.rstrip(".")
    test = test.drop_duplicates()
    X_tr, y_tr = clean_features(train), train["education"].map(GROUPS)
    X_te, y_te = clean_features(test), test["education"].map(GROUPS)
    le = LabelEncoder().fit(y_tr)
    prep = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pipe = Pipeline([("prep", prep), ("model", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"))])
    pipe.fit(X_tr, le.transform(y_tr))
    pred = pipe.predict(X_te)
    metrics = {"accuracy": round(float(accuracy_score(le.transform(y_te), pred)), 4),
               "f1_macro": round(float(f1_score(le.transform(y_te), pred, average="macro")), 4)}
    return {"label_encoder": le, "models": {"xgb": {
        "name": "XGBoost", "description": "Gradient boosting on 10 socio-professional variables.",
        "pipeline": pipe, "metrics": metrics, "note": f"official adult.test, {len(y_te)} rows. "
        "The \"Short post-secondary\" class is the least reliable."}}}


def load():
    return load_or_train("education", _train)


def predict(bundle, model_id, values):
    proba = bundle["models"][model_id]["pipeline"].predict_proba(clean_features(pd.DataFrame([values])))[0]
    classes = list(bundle["label_encoder"].classes_)
    best = classes[int(proba.argmax())]
    return {"label": LABELS[best], "description": DESCR[best],
            "probas": sorted(((LABELS[c], float(p)) for c, p in zip(classes, proba)), key=lambda x: -x[1])}


TARGET = Target(
    slug="education", title="Education", icon="🎓",
    question="What is the most likely education level for this profile?",
    detail="XGBoost: secondary or less, short post-secondary, or higher education.",
    fields=("age", "race", "workclass", "income", "marital-status", "capital-gain", "occupation", "capital-loss",
            "hours-per-week", "native-country"),
    unknown_ok=("workclass", "occupation", "native-country"), load=load, predict=predict)
