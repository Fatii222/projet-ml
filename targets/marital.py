"""Socio-professional profile (clustering, 3 groups): K-Means (notebook \"projet_ml_clustering\").

Unsupervised: `marital-status` and `relationship` are NOT used to form the groups;
`marital-status` is only used to validate / describe the groups afterwards.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import EDUCATION_NUM, OPTIONS, Target, load_or_train, read_adult

K = 3
NUM = ["age", "education-num", "hours-per-week"]
CAT = ["workclass", "occupation", "sex", "income"]
BIN = ["has-capital-gain", "has-capital-loss"]
FEATURES = ["age", "workclass", "education-num", "occupation", "sex", "hours-per-week", "income",
            "has-capital-gain", "has-capital-loss"]
CLIP = ["age", "hours-per-week"]  # capped at the 1st and 99th percentiles (as in the notebook)

NAMES = {"jeunes": "Young professionals", "cadres": "Graduate executives", "seniors": "Experienced seniors"}
ORDER = ["jeunes", "cadres", "seniors"]  # display order of the segments
MARITAL_LABEL = {k: v for k, v in OPTIONS["marital-status"]}


def load_frame():
    """Data cleaned as in the notebook: duplicates removed, "?" -> "Unknown"."""
    df = read_adult("adult.data").drop_duplicates().reset_index(drop=True)
    for c in ("workclass", "occupation", "native-country"):
        df[c] = df[c].fillna("Unknown")  # missing values -> "Unknown", all rows are kept
    df["education-num"] = df["education"].map(EDUCATION_NUM)
    return df


def prepared_frame(df, bounds):
    """Same transformations as in the notebook: gain / loss flags and capping (all columns)."""
    df = df.copy()
    df["has-capital-gain"] = (df["capital-gain"] > 0).astype(int)
    df["has-capital-loss"] = (df["capital-loss"] > 0).astype(int)
    for c in CLIP:
        df[c] = df[c].clip(*bounds[c])
    return df


def _prepare(df, bounds):
    return prepared_frame(df, bounds)[FEATURES]


def _name_clusters(df, labels):
    """Names each cluster number from its characteristics (the number itself is arbitrary)."""
    d = df.assign(cluster=labels)
    stats = d.groupby("cluster").agg(age=("age", "mean"), edu=("education-num", "mean"),
                                     hours=("hours-per-week", "mean"),
                                     high=("income", lambda s: (s == ">50K").mean()))
    cadres = stats["edu"].idxmax()
    jeunes = stats.drop(index=cadres)["age"].idxmin()
    keys = {cadres: "cadres", jeunes: "jeunes", **{c: "seniors" for c in stats.index if c not in (cadres, jeunes)}}
    out = {}
    for c, s in stats.iterrows():
        marital = d.loc[d["cluster"] == c, "marital-status"].value_counts(normalize=True)
        out[int(c)] = {"key": keys[c], "label": NAMES[keys[c]],
                       "description": f"Typical profile: {s['age']:.0f} years old, {s['hours']:.0f} h per week, "
                                      f"{s['high'] * 100:.0f}% with income > $50,000. "
                                      f"Most common status: {MARITAL_LABEL[marital.index[0]]} "
                                      f"({marital.iloc[0] * 100:.0f}% of the group)."}
    return out


def _train():
    df = load_frame()
    bounds = {c: tuple(df[c].quantile([0.01, 0.99])) for c in CLIP}
    X_df = _prepare(df, bounds)
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                              ("bin", "passthrough", BIN)])
    X = prep.fit_transform(X_df)
    km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X)
    pipe = Pipeline([("prep", prep), ("model", km)])
    ms, lab = df["marital-status"], km.labels_
    metrics = {"silhouette": round(float(silhouette_score(X, lab, sample_size=5000, random_state=42)), 4),
               "nmi": round(float(normalized_mutual_info_score(ms, lab)), 4),
               "ari": round(float(adjusted_rand_score(ms, lab)), 4),
               "purity": round(float(pd.crosstab(lab, ms).max(axis=1).sum() / len(df)), 4)}
    base = float(ms.value_counts(normalize=True).max())
    return {"bounds": bounds, "clusters": _name_clusters(df, lab), "models": {"km": {
        "name": "K-Means", "description": "3 groups formed without marital status (20 initializations).",
        "pipeline": pipe, "metrics": metrics,
        "note": f"Clustering: no test set. Purity vs marital status: {metrics['purity'] * 100:.0f}%, "
                f"versus {base * 100:.0f}% if everyone were in a single group."}}}


def load():
    return load_or_train("marital_kmeans", _train)


def predict(bundle, model_id, values):
    row = {**values, "education-num": EDUCATION_NUM[values["education"]]}
    X = _prepare(pd.DataFrame([row]), bundle["bounds"])
    cluster = int(bundle["models"][model_id]["pipeline"].predict(X)[0])
    return bundle["clusters"][cluster]


TARGET = Target(
    slug="marital", title="Socio-professional profile", icon="🧩",
    question="Which socio-professional profile does this person belong to?",
    detail="K-Means (3 segments): young professionals, graduate executives, experienced seniors. Full dashboard.",
    fields=("age", "sex", "education", "income", "workclass", "occupation", "hours-per-week", "capital-gain",
            "capital-loss"),
    unknown_ok=("workclass", "occupation"), load=load, predict=predict)
