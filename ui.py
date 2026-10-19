"""Streamlit page shared by all prediction targets (same layout as the original form)."""
from html import escape

import streamlit as st

from core import FIELD_SPECS, METRIC_LABELS, RAW_METRICS, options_for
from targets import BY_SLUG

# Neutral style (light / dark): translucent backgrounds, text inherited from the theme.
CSS = """
<style>
.block-container { padding-top: 2.5rem; }
h1 { letter-spacing: -0.02em; }
div[data-testid="stForm"] { border-radius: 14px; padding: 1.25rem 1.5rem 1.5rem; }
div[data-testid="stFormSubmitButton"] button { width: 100%; font-weight: 600; padding: .6rem 0; border-radius: 10px; }
.result-card { margin: 1.25rem 0 1rem; padding: 1.3rem 1.6rem; border-radius: 14px;
               border: 1px solid rgba(128,128,128,.25); border-left: 6px solid #2a78d6;
               background: rgba(42,120,214,.10); }
.result-tag { margin: 0; font-size: .72rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; opacity: .65; }
.result-label { margin: .25rem 0 .45rem; font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.result-desc { margin: 0; opacity: .85; }
</style>
"""


@st.cache_resource(show_spinner=False)
def _bundle(slug):
    return BY_SLUG[slug].load()


def _widget(slug, name, unknown_ok):
    spec, key = FIELD_SPECS[name], f"{slug}_{name}"
    if spec["kind"] == "slider":
        return st.slider(spec["label"], spec["min"], spec["max"], spec["default"], key=key)
    if spec["kind"] == "number":
        return st.number_input(spec["label"], min_value=0, max_value=spec["max"], value=0, step=spec["step"], key=key)
    opts = options_for(name, unknown_ok)
    values = [v for v, _ in opts]
    return st.selectbox(spec["label"], values, index=values.index(spec["default"]),
                        format_func=dict(opts).get, key=key)


def _show(entry, res, with_name):
    """One result: the predicted class and its description (no probability list)."""
    tag = "Result" + (f" · {escape(entry['name'])}" if with_name else "")
    st.markdown(f'''<div class="result-card"><p class="result-tag">{tag}</p>
<p class="result-label">{escape(res["label"])}</p><p class="result-desc">{escape(res["description"])}</p></div>''',
                unsafe_allow_html=True)


def render_target(t):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"{t.icon} {t.title}")
    st.write(t.question)
    with st.spinner("Loading the model (on the very first launch, training can take up to a minute)…"):
        bundle = _bundle(t.slug)
    models = bundle["models"]

    with st.form(f"form_{t.slug}"):
        st.caption("Enter the profile details.")
        choice = next(iter(models))
        if len(models) > 1:
            choice = st.selectbox("Model", list(models), key=f"{t.slug}_model",
                                  format_func=lambda k: models[k]["name"])
        values = {}
        fields = list(t.fields)
        for i in range(0, len(fields), 2):  # fields laid out in pairs: left / right
            for col, name in zip(st.columns(2), fields[i:i + 2]):
                with col:
                    values[name] = _widget(t.slug, name, t.unknown_ok)
        submitted = st.form_submit_button("Predict")

    if submitted:
        _show(models[choice], t.predict(bundle, choice, values), with_name=len(models) > 1)

    with st.expander("Model performance"):
        for entry in models.values():
            st.write(f"**{entry['name']}** : {entry['description']}")
            cols = st.columns(len(entry["metrics"]))
            for col, (k, v) in zip(cols, entry["metrics"].items()):
                col.metric(METRIC_LABELS.get(k, k), f"{v:.3f}" if k in RAW_METRICS else f"{v * 100:.1f} %")
            st.caption(entry["note"])

    st.caption("Educational project on a 1994 US census. Predictions are statistical and reflect "
               "the biases in the data (sex, ethnic origin…): they must not be used to make real decisions.")
