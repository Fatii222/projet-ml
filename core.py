"""Shared pieces: Adult data, form choice lists, saved models."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"]


# ----------------------------------------------------------------------------- data
def read_adult(name="adult.data", skiprows=0):
    """Reads adult.data / adult.test ('?' -> NaN). adult.test starts with a comment line (skiprows=1)."""
    df = pd.read_csv(DATA / name, names=COLS, skipinitialspace=True, na_values="?", skiprows=skiprows)
    return df.dropna(how="all")


def clean_income(s):
    """adult.test writes '<=50K.' with a trailing dot."""
    return s.str.replace(".", "", regex=False).str.strip()


# ----------------------------------------------------------------------------- saved models
SCHEMA = 2  # bump when the text / classes stored in the models change (forces a retrain of old artifacts)
def _versions():
    import xgboost
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "schema": SCHEMA}


def load_or_train(name, trainer):
    """Loads artifacts/<name>.joblib; (re)trains if missing, unreadable or built with other versions."""
    path, meta = ART / f"{name}.joblib", ART / f"{name}.json"
    try:
        if json.loads(meta.read_text()) == _versions():
            return joblib.load(path)
    except Exception:
        pass
    bundle = trainer()
    try:
        ART.mkdir(exist_ok=True)
        joblib.dump(bundle, path, compress=3)
        meta.write_text(json.dumps(_versions()))
    except OSError:  # read-only folder (some hosts): keep the model in memory
        pass
    return bundle


@dataclass(frozen=True)
class Target:
    """One page of the platform (one variable to predict)."""
    slug: str
    title: str
    icon: str
    question: str
    detail: str
    fields: tuple          # form fields, in order (laid out in left/right pairs)
    unknown_ok: tuple      # fields that can be "Not specified"
    load: Callable         # () -> bundle {"models": {id: {name, description, metrics, note, pipeline}}, ...}
    predict: Callable      # (bundle, model_id, values) -> {"label", "description", "probas": [(nom, proba)]}


# ----------------------------------------------------------------------------- form
EDUCATION_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
                 "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
                 "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}

_COUNTRIES = ["Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
              "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
              "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos",
              "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
              "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

# (value sent to the model, label shown): the labels are the dataset's own values
_VALUES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "State-gov", "Local-gov",
                  "Without-pay", "Never-worked"],
    "education": ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
                  "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"],
    "marital-status": ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Prof-specialty", "Exec-managerial", "Tech-support", "Sales", "Adm-clerical", "Craft-repair",
                   "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing",
                   "Protective-serv", "Priv-house-serv", "Other-service", "Armed-Forces"],
    "relationship": ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried", "Other-relative"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
    "income": ["<=50K", ">50K"],
    "native-country": ["United-States"] + sorted(_COUNTRIES),
}
OPTIONS = {k: [(v, v) for v in vals] for k, vals in _VALUES.items()}

FIELD_SPECS = {
    "age": dict(kind="slider", label="Age", min=17, max=90, default=30),
    "workclass": dict(kind="select", label="Workclass", default="Private"),
    "education": dict(kind="select", label="Education", default="Bachelors"),
    "marital-status": dict(kind="select", label="Marital status", default="Married-civ-spouse"),
    "occupation": dict(kind="select", label="Occupation", default="Prof-specialty"),
    "relationship": dict(kind="select", label="Relationship", default="Husband"),
    "race": dict(kind="select", label="Race", default="White"),
    "sex": dict(kind="select", label="Sex", default="Male"),
    "capital-gain": dict(kind="number", label="Capital gain ($)", max=99999, step=100),
    "capital-loss": dict(kind="number", label="Capital loss ($)", max=4356, step=50),
    "hours-per-week": dict(kind="slider", label="Hours per week", min=1, max=99, default=40),
    "native-country": dict(kind="select", label="Native country", default="United-States"),
    "income": dict(kind="select", label="Annual income", default="<=50K"),
}

METRIC_LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                 "recall": "Recall", "f1": "F1", "f1_macro": "Macro F1",
                 "silhouette": "Silhouette", "nmi": "NMI", "ari": "ARI", "purity": "Purity"}
RAW_METRICS = {"silhouette", "nmi", "ari"}  # scores between 0 and 1 shown as is (not in %)


def options_for(name, unknown_ok=()):
    opts = list(OPTIONS[name])
    return opts + [("Unknown", "Not specified")] if name in unknown_ok else opts
