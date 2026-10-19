# Streamlit platform: predictions on the Adult dataset

One application, six pages (one per variable to predict):

| Page | Target | Model(s) |
|---|---|---|
| Income | > $50,000 or ≤ $50,000 per year | Logistic Regression + Random Forest (pick one, or compare) |
| Education | secondary or less / short post-secondary / higher education | XGBoost |
| Occupation | executives and professionals / clerical and service / blue-collar | Random Forest |
| Relationship | Husband, Wife, Own-child, Not-in-family, Unmarried, Other-relative | Random Forest |
| Workclass | Private / Self-emp (2 types) / Government (3 levels) / Without-pay / Never-worked | Random Forest (softened class weights) |
| Socio-professional profile | young professionals / graduate executives / experienced seniors (clustering) | K-Means (k = 3), with a full dashboard |

Each page keeps the layout of the original form (sliders, dropdowns, "Predict" button), shows **a single result**
(the predicted class) and the model's performance. Dropdown options use the dataset's own values.

## Light / dark mode

The `⋮` menu (top right) offers **System / Light / Dark**. Both themes are defined in `.streamlit/config.toml`;
charts use colours that stay readable on either background.

## "Customer profiles" dashboard (Socio-professional profile page)

The clustering page is a small 8-section dashboard (menu in the sidebar) that reuses the notebook
`projet_ml_clustering.ipynb` without changing it:

| Section | Content |
|---|---|
| Overview | key figures, the 3 segments, PCA map |
| Profiles and interpretation | profile heatmap, card for each segment (what sets it apart, in short) |
| Marital status | purity, NMI, ARI, cross-tab: what the segments do or do not recover |
| Factor analysis | variables that best separate the segments, detail per variable |
| Explore a subgroup | filters (age, sex, education, income, occupation): split by segment and marital status |
| Simulator and recommendations | entered profile -> segment, confidence, distances, recommended offers, system limits |
| Choosing k | inertia, silhouette, Davies-Bouldin, stability (notebook values, `data/choix_k.csv`) |
| Link to the notebook | mapping of sections to notebook parts, screening, hierarchical clustering |

Code: `dashboard.py` (display) and `targets/marital.py` (data, K-Means model). The other pages use the shared form
(`ui.py`).

## Run

- **Windows**: double-click `run.bat`
- **Mac / Linux**: `./run.sh`
- **By hand**: `pip install -r requirements.txt` then `streamlit run app.py` (http://localhost:8501)

Models train automatically the first time each page is opened, then are saved in `artifacts/`
(up to ≈ 1 minute for income). `python train_all.py` trains all six at once.
They retrain by themselves if the scikit-learn or XGBoost version changes (or if `SCHEMA` in `core.py` is bumped).

## Structure

```
app.py               navigation: home + one page per target
ui.py                form and result display (shared by the 5 prediction pages)
dashboard.py         "Customer profiles" dashboard (clustering page)
core.py              data, choice lists, model saving
targets/
  income.py  education.py  occupation.py  relationship.py  workclass.py  marital.py   one file per target
train_all.py         trains all models
data/                adult.data, adult.test, adult.names (a single copy), choix_k.csv (notebook tables)
tests/               pytest: opens each page, submits the form, checks the results
```

## Add a page

1. Copy `targets/relationship.py` to `targets/<name>.py`, then adapt: columns, training, class labels,
   and the form's `fields` list (available fields are in `FIELD_SPECS`, `core.py`).
2. Add `<name>.TARGET` to `targets/__init__.py`. The page and its home card appear on their own.

## Deploy

- **Streamlit Community Cloud** (free): push the folder to GitHub, then New app, choose `app.py`.
- **ngrok**: `streamlit run app.py` then `ngrok http 8501` to share a temporary link.
- **Docker**: `docker build -t adult-app . && docker run -p 8501:8501 adult-app` (trains the models at build time).

## Notes

- Training recipes are those of the original projects (same preprocessing and same hyperparameters):
  income 0.808 / 0.841 accuracy, education 0.626, occupation 0.644, relationship 0.765,
  workclass ≈ 0.73 accuracy (macro F1 ≈ 0.24; "always Private" gives 0.74 accuracy but macro F1 0.11);
  profile (clustering): silhouette 0.162, NMI 0.123, purity 0.598 versus 0.460 for a single group.
- The numeric education level (`education-num`) is derived from the "Education" list, to avoid
  inconsistent combinations.
- "Not specified" is only offered where the model was trained with missing values.
- The data dates from 1994 (United States) and contains biases: educational project, not for real decisions.
