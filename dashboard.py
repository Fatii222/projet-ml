"""\"Customer profiles\" dashboard: the notebook's K-Means clustering (k = 3), presented in 8 sections.

The dashboard reuses the notebook without changing it: same data, same variables, same model
(targets/marital.py). It only reads the results and formats them.
"""
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from core import DATA, EDUCATION_NUM, OPTIONS
from targets import BY_SLUG, marital

SECTIONS = ["Overview", "Profiles and interpretation", "Marital status", "Factor analysis",
            "Explore a subgroup", "Simulator and recommendations", "Choosing k", "Link to the notebook"]

ORDER = marital.ORDER
NAMES = marital.NAMES
NAME_LIST = [NAMES[k] for k in ORDER]
COLORS = {"jeunes": "#2a78d6", "cadres": "#eb6834", "seniors": "#1baf7a"}
SEG_SCALE = alt.Scale(domain=NAME_LIST, range=[COLORS[k] for k in ORDER])
LABEL = "#8b93a1"  # neutral text colour readable on both light and dark backgrounds
MARK = "#e5484d"  # highlight colour that stays visible on both light and dark backgrounds

CSS = """
<style>
[data-testid="stMainBlockContainer"], .block-container { max-width: 1100px; padding-top: 2.5rem; }
h1 { letter-spacing: -0.02em; }
</style>
"""

SUMMARY = {
    "jeunes": "Early career, modest income, mostly never married. "
              "Main need: starting simply (low fees, small savings, first loan).",
    "cadres": "Graduates, long working weeks, high income, most often married. "
              "Main need: long-term projects (property, retirement, wealth).",
    "seniors": "Older, manual trades, less educated, more often divorced or widowed. "
               "Main need: securing the end of a career (retirement, protection, controlled budget).",
}

# (offer, reason): typical commercial offers per segment
OFFERS = {
    "jeunes": [("Fee-free youth account + debit card", "Income still modest: keep banking fees low."),
               ("Automatic savings (round-up on purchases, small monthly transfer)",
                "Build the habit of saving small amounts."),
               ("First small loan (equipment, vehicle)",
                "Early career: first financing needs."),
               ("Renters insurance + phone insurance", "Young profile, often a renter.")],
    "cadres": [("Mortgage", "High income and stable situation: borrowing capacity."),
               ("Retirement savings and life insurance", "Long-term projects."),
               ("Wealth management and investments", "High income, capital gains more frequent."),
               ("Family insurance and protection plan", "Often married: protect the household.")],
    "seniors": [("Retirement savings and annuity", "End of career approaching: secure future income."),
                ("Life insurance / funeral cover", "Older audience, more often widowed or divorced."),
                ("Loan consolidation", "Lower income: reduce monthly payments."),
                ("Supplementary health and long-term care", "Health needs that grow with age.")],
}
MARITAL_OFFERS = {  # extra offer if the customer declared their status (never used for the segment)
    "married": ("Family insurance and joint loan", "Declared status: married."),
    "divorced": ("Contract review (insurance, protection, beneficiaries)", "Declared status: change of situation."),
    "widowed": ("Surviving-spouse protection and support", "Declared status: widowed."),
}

PRESETS = {
    "Custom": None,
    "Young clerical employee": dict(age=24, sex="Female", edu="HS-grad", work="Private", job="Adm-clerical",
                                        hours=35, income="<=50K", gain=False, loss=False, marital="Never-married"),
    "Graduate executive": dict(age=42, sex="Male", edu="Masters", work="Private", job="Exec-managerial",
                          hours=50, income=">50K", gain=True, loss=False, marital="Married-civ-spouse"),
    "Senior in a manual trade": dict(age=58, sex="Male", edu="9th", work="Private", job="Craft-repair",
                                    hours=40, income="<=50K", gain=False, loss=False, marital="Divorced"),
}

FACTORS = [("age", "Age", "num"), ("education-num", "Education level", "num"),
           ("hours-per-week", "Hours per week", "num"), ("sex", "Sex", "cat"),
           ("income", "Annual income", "cat"), ("workclass", "Workclass", "cat"),
           ("occupation", "Occupation", "cat"), ("has-capital-gain", "Capital gain", "bin"),
           ("has-capital-loss", "Capital loss", "bin")]


# ----------------------------------------------------------------------------- helpers
def _label(name, value):
    return "Not specified" if value == "Unknown" else dict(OPTIONS[name]).get(value, value)


def _pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def _chart(c):
    st.altair_chart(c)


def _mode(s):
    return s.value_counts().index[0]


@st.cache_resource(show_spinner="Preparing the dashboard…")
def _ctx():
    """Loads the K-Means model and computes everything the sections display (once per server session)."""
    bundle = BY_SLUG["marital"].load()
    pipe = bundle["models"]["km"]["pipeline"]
    df = marital.prepared_frame(marital.load_frame(), bundle["bounds"])
    X = pipe["prep"].transform(df[marital.FEATURES])
    ids = pipe["model"].predict(X)
    key_of = {c: v["key"] for c, v in bundle["clusters"].items()}
    df["segment"] = pd.Series(ids).map(key_of).values
    id_of = {k: c for c, k in key_of.items()}
    pca = PCA(n_components=2, random_state=42).fit(X)
    Z = pca.transform(X)
    idx = np.random.RandomState(42).choice(len(df), 6000, replace=False)

    pop = dict(age=df["age"].mean(), edu=df["education-num"].mean(), hours=df["hours-per-week"].mean(),
               men=(df["sex"] == "Male").mean(), high=(df["income"] == ">50K").mean())
    prof = {}
    for k in ORDER:
        g = df[df["segment"] == k]
        prof[k] = dict(share=len(g) / len(df), age=g["age"].mean(), edu=g["education-num"].mean(),
                       hours=g["hours-per-week"].mean(), men=(g["sex"] == "Male").mean(),
                       high=(g["income"] == ">50K").mean(), gain=g["has-capital-gain"].mean(),
                       job=_mode(g["occupation"]), work=_mode(g["workclass"]),
                       marital=_mode(g["marital-status"]))
    return SimpleNamespace(bundle=bundle, pipe=pipe, df=df, X=X, Z=Z, idx=idx, pca=pca, id_of=id_of,
                           key_of=key_of, pop=pop, prof=prof, metrics=bundle["models"]["km"]["metrics"])


@st.cache_data(show_spinner=False)
def _k_table():
    return pd.read_csv(DATA / "choix_k.csv")


def _scatter(ctx, person=None):
    """PCA map (6,000 randomly drawn people); `person` = (x, y) to place the entered person on it."""
    v = ctx.pca.explained_variance_ratio_
    s = pd.DataFrame({"x": ctx.Z[ctx.idx, 0], "y": ctx.Z[ctx.idx, 1],
                      "Segment": ctx.df["segment"].iloc[ctx.idx].map(NAMES).values})
    c = alt.Chart(s).mark_circle(size=16, opacity=0.5).encode(
        x=alt.X("x:Q", title=f"Axis 1 ({v[0] * 100:.0f}% of variance)"),
        y=alt.Y("y:Q", title=f"Axis 2 ({v[1] * 100:.0f}% of variance)"),
        color=alt.Color("Segment:N", scale=SEG_SCALE, legend=alt.Legend(title=None, orient="top")))
    if person is not None:
        p = pd.DataFrame({"x": [person[0]], "y": [person[1]], "t": ["Entered person"]})
        base = alt.Chart(p).encode(x="x:Q", y="y:Q")
        c = c + base.mark_point(shape="diamond", size=260, filled=True, color=MARK, stroke="white",
                                strokeWidth=1.5) + base.mark_text(dy=-16, fontWeight="bold", color=MARK).encode(text="t:N")
    return c.properties(height=380, width="container")


# ----------------------------------------------------------------------------- 1. overview
def sec_overview(ctx):
    st.title("Overview")
    st.write("K-Means groups the people of the Adult census into **3 segments**, **without** using their marital "
             "status. We then look at who these segments are, whether they recover marital status, "
             "and what can be offered to each.")
    m = ctx.metrics
    c = st.columns(4)
    c[0].metric("People analysed", f"{len(ctx.df):,}")
    c[1].metric("Segments (k)", "3")
    c[2].metric("Silhouette", f"{m['silhouette']:.3f}", help="0 = overlapping segments, 1 = well separated.")
    c[3].metric("Purity vs marital status", _pct(m["purity"], 1))

    st.subheader("The three segments")
    cols = st.columns(3)
    for col, k in zip(cols, ORDER):
        with col, st.container(border=True):
            st.markdown(f"**{NAMES[k]}**")
            st.caption(f"{_pct(ctx.prof[k]['share'])} of the population")
            st.write(SUMMARY[k].split(". ")[0] + ".")

    st.subheader("Map of people")
    _chart(_scatter(ctx))
    st.caption("Each point is a person (6,000 drawn at random), projected into 2 dimensions (PCA). "
               "Segments overlap: they are tendencies, not clear-cut groups.")


# ----------------------------------------------------------------------------- 2. profiles
STATS = [("age", "Age", "{:.1f}"), ("edu", "Education", "{:.1f}"), ("hours", "Hours / week", "{:.1f}"),
         ("men", "% men", "{:.0%}"), ("high", "% income > $50,000", "{:.0%}")]


def _zscores(ctx):
    z = {}
    for k, _, _ in STATS:
        v = np.array([ctx.prof[s][k] for s in ORDER])
        z[k] = dict(zip(ORDER, (v - v.mean()) / (v.std() or 1)))
    return z


def _bullets(ctx, key, z):
    p, pop = ctx.prof[key], ctx.pop
    fmt = {"age": f"Average age of {p['age']:.1f} years (population: {pop['age']:.1f} years)",
           "high": f"{_pct(p['high'])} with income above $50,000 (population: {_pct(pop['high'])})",
           "hours": f"{p['hours']:.0f} hours per week (population: {pop['hours']:.0f} h)",
           "edu": f"Average education level of {p['edu']:.1f} (population: {pop['edu']:.1f}, on a scale of 1 to 16)",
           "men": f"{_pct(p['men'])} men (population: {_pct(pop['men'])})"}
    top = sorted(fmt, key=lambda k: -abs(z[k][key]))[:4]
    return [fmt[k] for k in top] + [f"Most common occupation: {_label('occupation', p['job'])}; "
                                    f"most common workclass: {_label('workclass', p['work'])}"]


def sec_profiles(ctx):
    st.title("Segment profiles and interpretation")
    st.write("For each segment: what sets it apart from the population, and what can be done with it.")
    z = _zscores(ctx)
    rows = [dict(Segment=NAMES[s], Variable=lab, z=float(z[k][s]), txt=f.format(ctx.prof[s][k]))
            for k, lab, f in STATS for s in ORDER]
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Variable:N", sort=[lab for _, lab, _ in STATS], title=None, axis=alt.Axis(labelAngle=0, orient="bottom")),
        y=alt.Y("Segment:N", sort=NAME_LIST, title=None))
    heat = base.mark_rect().encode(color=alt.Color("z:Q", scale=alt.Scale(scheme="blues", domain=[-1.5, 1.5]),
                                                   legend=alt.Legend(title="Relative gap")))
    txt = base.mark_text().encode(text="txt:N", color=alt.condition(alt.datum.z > 0.8, alt.value("white"),
                                                                   alt.value("#222")))
    _chart((heat + txt).properties(height=190, width="container"))
    st.caption("The number shown is the real value; the colour shows the gap from the average of the segments "
               "(dark = high value).")

    for k in ORDER:
        p = ctx.prof[k]
        with st.container(border=True):
            st.subheader(NAMES[k])
            st.caption(f"Segment {ctx.id_of[k]}: {_pct(p['share'])} of the population")
            left, right = st.columns([1, 2])
            with left:
                st.metric("Average age", f"{p['age']:.1f} years")
                st.metric("Income > $50,000", _pct(p["high"]))
                st.metric("Most common status", _label("marital-status", p["marital"]))
            with right:
                st.markdown("**What sets it apart**")
                st.markdown("\n".join(f"- {b}" for b in _bullets(ctx, k, z)))
                st.markdown(f"**In short:** {SUMMARY[k]}")


# ----------------------------------------------------------------------------- 3. marital status
def sec_marital(ctx):
    st.title("Do the segments recover marital status?")
    st.write("The segments were formed **without** marital status. We look at it now to check "
             "whether the groups recover it.")
    df = ctx.df
    ms, seg = df["marital-status"], df["segment"]
    ct = pd.crosstab(seg, ms)
    purity = ct.max(axis=1).sum() / len(df)
    pop_share = ms.value_counts(normalize=True)
    base = pop_share.max()
    nmi = normalized_mutual_info_score(ms, seg)
    ari = adjusted_rand_score(ms, seg)
    c = st.columns(3)
    c[0].metric("Segment purity", _pct(purity, 1), delta=f"+{(purity - base) * 100:.1f} pts vs no model",
                help="Share of people whose status is the most common one in their segment.")
    c[1].metric("NMI", f"{nmi:.3f}", help="0 = no link between segments and status, 1 = identical.")
    c[2].metric("ARI", f"{ari:.3f}", help="Same scale as NMI, corrected for chance.")

    shares = ct.div(ct.sum(axis=1), axis=0)
    cols = [m for m, _ in OPTIONS["marital-status"]]  # Married-civ-spouse, Never-married, Divorced, Separated, Widowed…
    rows = [dict(Group=NAMES[k], Status=_label("marital-status", m), part=float(shares.loc[k, m]))
            for k in ORDER for m in cols]
    rows += [dict(Group="Whole population", Status=_label("marital-status", m), part=float(pop_share.get(m, 0)))
             for m in cols]
    b = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Status:N", sort=[_label("marital-status", m) for m in cols], title=None,
                axis=alt.Axis(labelAngle=-30, orient="bottom")),
        y=alt.Y("Group:N", sort=NAME_LIST + ["Whole population"], title=None))
    heat = b.mark_rect().encode(color=alt.Color("part:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                                legend=alt.Legend(title="Share", format="%")))
    txt = b.mark_text(fontSize=11).encode(text=alt.Text("part:Q", format=".0%"),
                                          color=alt.condition(alt.datum.part > 0.45, alt.value("white"),
                                                              alt.value("#222")))
    _chart((heat + txt).properties(height=230, width="container"))

    st.subheader("What the segments recover")
    lines = []
    for k in ORDER:
        lift = (shares.loc[k] / pop_share).where(pop_share >= 0.02).dropna()
        m = lift.idxmax()
        lines.append(f"**{NAMES[k]}**: {_pct(shares.loc[k, m])} \"{_label('marital-status', m)}\" "
                     f"versus {_pct(pop_share[m])} in the population (x{lift[m]:.1f}).")
    st.markdown("\n".join(f"- {x}" for x in lines))

    st.subheader("What they do not recover")
    st.markdown(
        f"- Purity ({_pct(purity)}) is only {(purity - base) * 100:.0f} points above that of a single group "
        f"({_pct(base)}): about {round((1 - purity) * 10)} people out of 10 would be misclassified if we guessed "
        f"their status from their segment.\n"
        "- The segments look mostly like **life stages** (age, education, income) rather than marital "
        "statuses.\n"
        "- So a customer's status cannot be reliably guessed from their segment alone.")


# ----------------------------------------------------------------------------- 4. factors
def _eta(values, groups):
    v = pd.Series(np.asarray(values, float))
    g = v.groupby(np.asarray(groups))
    ssb = (g.size() * (g.mean() - v.mean()) ** 2).sum()
    return float(np.sqrt(ssb / ((v - v.mean()) ** 2).sum()))


def _cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))


@st.cache_data(show_spinner=False)
def _strengths(key):
    ctx = _ctx()
    out = []
    for col, label, kind in FACTORS:
        s = _eta(ctx.df[col], ctx.df["segment"]) if kind in ("num", "bin") else _cramers_v(ctx.df[col], ctx.df["segment"])
        out.append(dict(Variable=label, Strength=s))
    return pd.DataFrame(out).sort_values("Strength", ascending=False).reset_index(drop=True)


def sec_factors(ctx):
    st.title("Factor analysis")
    st.write("Which variables best separate the three segments? Each bar goes from 0 (the variable does not "
             "distinguish the segments) to 1 (it distinguishes them perfectly).")
    s = _strengths("marital")
    _chart(alt.Chart(s).mark_bar(color="#2a78d6").encode(
        x=alt.X("Strength:Q", scale=alt.Scale(domain=[0, 1]), title="Strength of the link with the segment"),
        y=alt.Y("Variable:N", sort=list(s["Variable"]), title=None),
        tooltip=["Variable", alt.Tooltip("Strength:Q", format=".2f")]).properties(height=300, width="container"))
    top = list(s["Variable"][:3])
    st.markdown(f"The variables that separate the segments most are **{top[0].lower()}**, **{top[1].lower()}** "
                f"and **{top[2].lower()}**. The measure is the correlation ratio (numeric variables) or Cramér's V "
                "(categorical variables).")

    st.subheader("Detail of one variable")
    label = st.selectbox("Variable", list(s["Variable"]), key="factor_var")
    col, _, kind = next(f for f in FACTORS if f[1] == label)
    d = ctx.df
    if kind == "num":
        samp = pd.DataFrame({"Segment": d["segment"].iloc[ctx.idx].map(NAMES).values, "Value": d[col].iloc[ctx.idx].values})
        _chart(alt.Chart(samp).mark_boxplot(extent="min-max", size=40).encode(
            x=alt.X("Segment:N", sort=NAME_LIST, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Value:Q", title=label),
            color=alt.Color("Segment:N", scale=SEG_SCALE, legend=None)).properties(height=300, width="container"))
    elif kind == "bin":
        m = d.groupby("segment")[col].mean().reindex(ORDER)
        t = pd.DataFrame({"Segment": NAME_LIST, "Share": m.values})
        _chart(alt.Chart(t).mark_bar().encode(
            x=alt.X("Segment:N", sort=NAME_LIST, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Share:Q", axis=alt.Axis(format="%"), title=f"Share with: {label.lower()}"),
            color=alt.Color("Segment:N", scale=SEG_SCALE, legend=None)).properties(height=300, width="container"))
    else:
        top_cats = d[col].value_counts().index[:6]
        cat = d[col].where(d[col].isin(top_cats), "Other")
        sh = pd.crosstab(d["segment"], cat, normalize="index").reindex(ORDER)
        t = sh.reset_index().melt(id_vars="segment", var_name="Value", value_name="Share")
        t["Segment"] = t["segment"].map(NAMES)
        t["Value"] = t["Value"].map(lambda v: "Other" if v == "Other" else _label(col, v))
        _chart(alt.Chart(t).mark_bar().encode(
            y=alt.Y("Segment:N", sort=NAME_LIST, title=None), x=alt.X("Share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Value:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(title=None)),
            tooltip=["Segment", "Value", alt.Tooltip("Share:Q", format=".0%")]).properties(height=220, width="container"))

    with st.expander("Variables left out of the clustering"):
        st.markdown(
            "- `marital-status`: the variable under study, excluded so the segments can be validated afterwards.\n"
            "- `relationship`: near-duplicate of `marital-status` (Husband / Wife = married), would make the result trivial.\n"
            "- `fnlwgt`: census statistical weight, unrelated to the person.\n"
            "- `education`: redundant with `education-num`.\n"
            "- `race`, `native-country`: very imbalanced and sensitive, useless for a socio-professional profile.")


# ----------------------------------------------------------------------------- 5. subgroup
def sec_explore(ctx):
    st.title("Explore a subgroup")
    st.write("Filter the population (age, sex, education, income, occupation) and see how the subgroup is split "
             "across segments and by marital status.")
    d = ctx.df
    a, b = st.columns(2)
    age = a.slider("Age", int(d["age"].min()), int(d["age"].max()),
                   (int(d["age"].min()), int(d["age"].max())), key="ex_age")
    edu = b.slider("Education level (1 to 16)", 1, 16, (1, 16), key="ex_edu")
    c1, c2 = st.columns(2)
    sex = c1.radio("Sex", ["All", "Male", "Female"], horizontal=True, key="ex_sex")
    inc = c2.radio("Annual income", ["All", "≤ $50,000", "> $50,000"], horizontal=True, key="ex_inc")
    jobs = [v for v, _ in OPTIONS["occupation"]] + ["Unknown"]
    pick = st.multiselect("Occupation (empty = all)", jobs, format_func=lambda v: _label("occupation", v), key="ex_job")

    mask = d["age"].between(*age) & d["education-num"].between(*edu)
    if sex != "All":
        mask &= d["sex"] == sex
    if inc != "All":
        mask &= d["income"] == {"≤ $50,000": "<=50K", "> $50,000": ">50K"}[inc]
    if pick:
        mask &= d["occupation"].isin(pick)
    sub = d[mask]
    n = len(sub)
    c = st.columns(3)
    c[0].metric("People", f"{n:,}")
    c[1].metric("Share of the population", _pct(n / len(d), 1))
    if n < 30:
        st.warning("Subgroup too small (fewer than 30 people) to be interpreted.")
        return
    shares = sub["segment"].value_counts(normalize=True).reindex(ORDER).fillna(0)
    c[2].metric("Dominant segment", NAMES[shares.idxmax()], delta=f"{_pct(shares.max())} of the subgroup", delta_color="off")

    pop = d["segment"].value_counts(normalize=True).reindex(ORDER)
    t = pd.DataFrame({"Segment": NAME_LIST * 2, "Group": ["Subgroup"] * 3 + ["Population"] * 3,
                      "Share": list(shares.values) + list(pop.values)})
    st.subheader("Split across segments")
    _chart(_grouped(t, "Segment", NAME_LIST))

    mcols = [m for m, _ in OPTIONS["marital-status"]]
    sm = sub["marital-status"].value_counts(normalize=True).reindex(mcols).fillna(0)
    pm = d["marital-status"].value_counts(normalize=True).reindex(mcols).fillna(0)
    labels = [_label("marital-status", m) for m in mcols]
    t = pd.DataFrame({"Status": labels * 2, "Group": ["Subgroup"] * len(mcols) + ["Population"] * len(mcols),
                      "Share": list(sm.values) + list(pm.values)})
    st.subheader("Marital status")
    _chart(_grouped(t, "Status", labels))


def _grouped(t, x, order):
    return alt.Chart(t).mark_bar().encode(
        x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=-20)), xOffset="Group:N",
        y=alt.Y("Share:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Group:N", scale=alt.Scale(domain=["Subgroup", "Population"],
                                                    range=["#2a78d6", "#b0b7c3"]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[x, "Group", alt.Tooltip("Share:Q", format=".1%")]).properties(height=260, width="container")


# ----------------------------------------------------------------------------- 6. simulator
def _apply_preset():
    p = PRESETS.get(st.session_state.get("sim_preset"))
    if p:
        for k, v in p.items():
            st.session_state[f"sim_{k}"] = v


def assign(ctx, v):
    """Segment of a person: returns (segment key, distances to the 3 centres, confidence, PCA position)."""
    row = {"age": v["age"], "workclass": v["work"], "education": v["edu"], "education-num": EDUCATION_NUM[v["edu"]],
           "occupation": v["job"], "sex": v["sex"], "hours-per-week": v["hours"], "income": v["income"],
           "capital-gain": int(v["gain"]), "capital-loss": int(v["loss"])}
    X = ctx.pipe["prep"].transform(marital._prepare(pd.DataFrame([row]), ctx.bundle["bounds"]))
    dist = ctx.pipe["model"].transform(X)[0]
    order = np.argsort(dist)
    ratio = dist[order[0]] / dist[order[1]]
    conf = "High" if ratio < 0.75 else "Medium" if ratio < 0.9 else "Low"
    return ctx.key_of[int(order[0])], {k: float(dist[ctx.id_of[k]]) for k in ORDER}, conf, ctx.pca.transform(X)[0]


def sec_simulator(ctx):
    st.title("Simulator and recommendations")
    st.write("Enter a person's profile: the dashboard finds their segment, their closest profiles "
             "and suggests offers.")
    st.selectbox("Starting example", list(PRESETS), key="sim_preset", on_change=_apply_preset)

    edu_opts = sorted(EDUCATION_NUM, key=EDUCATION_NUM.get)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Profile")
        v = {"age": st.slider("Age", 17, 90, 35, key="sim_age"),
             "sex": st.radio("Sex", ["Male", "Female"], horizontal=True, key="sim_sex"),
             "edu": st.selectbox("Education", edu_opts, index=edu_opts.index("HS-grad"),
                                 format_func=lambda e: f"{EDUCATION_NUM[e]} · {_label('education', e)}", key="sim_edu"),
             "work": st.selectbox("Workclass", [x for x, _ in OPTIONS["workclass"]] + ["Unknown"],
                                  format_func=lambda x: _label("workclass", x), key="sim_work"),
             "job": st.selectbox("Occupation", [x for x, _ in OPTIONS["occupation"]] + ["Unknown"],
                                 index=3, format_func=lambda x: _label("occupation", x), key="sim_job"),
             "hours": st.slider("Hours per week", 1, 99, 40, key="sim_hours"),
             "income": st.radio("Annual income", ["<=50K", ">50K"], horizontal=True, key="sim_income",
                                format_func={"<=50K": "$50,000 or less", ">50K": "More than $50,000"}.get),
             "gain": st.checkbox("Has a capital gain", key="sim_gain"),
             "loss": st.checkbox("Has a capital loss", key="sim_loss")}
        mar = st.selectbox("Marital status (if the customer declared it)", ["", *[m for m, _ in OPTIONS["marital-status"]]],
                           format_func=lambda m: "Unknown" if m == "" else _label("marital-status", m), key="sim_marital",
                           help="Used only for an extra offer, never to find the segment.")

    key, dist, conf, pos = assign(ctx, v)
    with right:
        st.subheader("Result")
        a, b = st.columns([1.6, 1])
        a.metric("Assigned segment", NAMES[key])
        b.metric("Assignment confidence", conf,
                 help="Compares the distance to the nearest segment with the second nearest: the bigger the gap, "
                      "the clearer the assignment.")
        st.write(SUMMARY[key])
        sc, bars = st.columns([1.6, 1])
        with sc:
            _chart(_scatter(ctx, pos).properties(height=300))
        with bars:
            t = pd.DataFrame({"Segment": NAME_LIST, "Distance": [dist[k] for k in ORDER]})
            b = alt.Chart(t).encode(
                x=alt.X("Distance:Q", title="Distance to segment (small = close)"),
                y=alt.Y("Segment:N", sort=NAME_LIST, title=None, axis=alt.Axis(labelLimit=120)),
                tooltip=["Segment", alt.Tooltip("Distance:Q", format=".2f")])
            _chart((b.mark_bar().encode(color=alt.Color("Segment:N", scale=SEG_SCALE, legend=None))
                    + b.mark_text(align="left", dx=4, color=LABEL).encode(text=alt.Text("Distance:Q", format=".2f")))
                   .properties(height=300, width="container"))

    st.subheader("Recommended offers")
    offers = [(o, f"{why} · Segment") for o, why in OFFERS[key]]
    extra = "married" if mar.startswith("Married") else "divorced" if mar in ("Divorced", "Separated") \
        else "widowed" if mar == "Widowed" else None
    if extra:
        o, why = MARITAL_OFFERS[extra]
        offers.append((o, f"{why} · Declared status"))
    for o, why in offers:
        st.markdown(f"**{o}**")
        st.caption(why)
    with st.expander("Fairness and limits of the system"):
        st.markdown(
            "- The data dates from 1994 (US census): the segments describe that population, not yours.\n"
            "- **Sex** is used to form the segments; `race` and `native-country` are deliberately left out.\n"
            "- The segment does **not** tell marital status: 4 people out of 10 would be misclassified.\n"
            "- Segments overlap (silhouette ≈ 0.16): a \"Low\" confidence means the person is "
            "on the border between two segments.\n"
            "- Offers are **commercial suggestions**, never a criterion for refusing credit, pricing "
            "or access to a service.")


# ----------------------------------------------------------------------------- 7. choice of k
def _k_line(t, col, title):
    d = t[["k", col]].dropna()
    line = alt.Chart(d).mark_line(point=True, color="#2a78d6").encode(
        x=alt.X("k:O", title="k", axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{col}:Q", scale=alt.Scale(zero=False), title=None))
    rule = alt.Chart(pd.DataFrame({"k": [3], "t": ["k = 3"]})).encode(x="k:O")
    r = rule.mark_rule(color="#eb6834", strokeDash=[5, 4]) + rule.mark_text(color="#eb6834", dy=-6, dx=14,
                                                                           align="left", baseline="top",
                                                                           y=0).encode(text="t:N")
    return (line + r).properties(title=title, height=210, width="container")


def sec_choose_k(ctx):
    st.title("Choosing the number of segments (k)")
    st.write("The same criteria as in the notebook: inertia (elbow), silhouette, Davies-Bouldin and stability.")
    t = _k_table()
    a, b, c = st.columns(3)
    with a:
        _chart(_k_line(t, "inertie", "Inertia (elbow)"))
    with b:
        _chart(_k_line(t, "silhouette", "Silhouette (maximise)"))
    with c:
        _chart(_k_line(t, "davies_bouldin", "Davies-Bouldin (minimise)"))

    st.subheader("Stability by k")
    s = t.dropna(subset=["ari_min"]).assign(Choice=lambda x: np.where(x["k"] == 3, "Chosen k", "Other k"))
    bars = alt.Chart(s).encode(
        x=alt.X("k:O", title="k", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ari_min:Q", scale=alt.Scale(domain=[0, 1.1]), title="Minimum ARI between seeds (1 = identical)"))
    _chart((bars.mark_bar().encode(color=alt.Color("Choice:N", scale=alt.Scale(
        domain=["Other k", "Chosen k"], range=["#2a78d6", "#eb6834"]), legend=alt.Legend(title=None, orient="top")))
            + bars.mark_text(dy=-8, color=LABEL).encode(text=alt.Text("ari_min:Q", format=".2f"))).properties(height=260, width="container"))

    best = t.loc[t["silhouette"].idxmax()]
    k3 = t[t["k"] == 3].iloc[0]
    st.markdown(f"**Choice: k = 3.** Silhouette is highest for k = {int(best['k'])} ({best['silhouette']:.3f}), the elbow "
                f"is not clear, Davies-Bouldin barely varies, and k = 3 is almost perfectly reproducible "
                f"(minimum ARI {k3['ari_min']:.3f}) whereas from k = 4 the split changes with the seed.")
    f3 = lambda x: "–" if pd.isna(x) else f"{x:.3f}"
    shown = pd.DataFrame({"k": t["k"], "Inertia": t["inertie"].map("{:.0f}".format), "Silhouette": t["silhouette"].map(f3),
                          "Davies-Bouldin": t["davies_bouldin"].map(f3), "ARI min": t["ari_min"].map(f3),
                          "ARI mean": t["ari_moyen"].map(f3)})
    st.dataframe(shown, hide_index=True)


# ----------------------------------------------------------------------------- 8. notebook
def sec_notebook(ctx):
    st.title("Link to the notebook")
    st.write("The dashboard reuses the notebook `projet_ml_clustering.ipynb` without changing it: same data, same "
             "variables, same model. Here is where to find each section.")
    st.dataframe(pd.DataFrame([
        ("Overview", "Parts 3 and 5", "Clustering matrix, K-Means with k = 3, silhouette"),
        ("Profiles and interpretation", "Part 7", "Summary of each cluster and profile heatmap"),
        ("Marital status", "Part 6", "Cross-tabulation, NMI, ARI, purity"),
        ("Factor analysis", "Parts 1 and 2", "Kept / dropped variables, cleaning"),
        ("Explore a subgroup", "Part 7", "Same logic as the profiles, on a filter of your choice"),
        ("Simulator and recommendations", "Parts 3 and 5", "Same preprocessing and same model, applied to one person"),
        ("Choosing k", "Part 4", "Inertia, silhouette, Davies-Bouldin, stability (ARI)"),
    ], columns=["Dashboard section", "Notebook part", "Content"]), hide_index=True)

    st.subheader("Why marital status?")
    st.write("The notebook tested several variables: for each one, it is removed from the clustering (k = 5) and we measure "
             "how well the groups recover it (NMI, average over 5 seeds).")
    scr = pd.DataFrame({"Variable": ["marital-status", "occupation", "education-num", "age", "workclass", "sex"],
                        "Mean NMI": [0.128, 0.103, 0.038, 0.036, 0.035, 0.020]})
    _chart(alt.Chart(scr).mark_bar(color="#2a78d6").encode(
        x=alt.X("NMI moyenne:Q", title="Mean NMI"), y=alt.Y("Variable:N", sort=list(scr["Variable"]), title=None),
        tooltip=["Variable", "Mean NMI"]).properties(height=200, width="container"))

    st.subheader("What the notebook does in addition")
    st.markdown("- **Hierarchical clustering (Ward method), part 8**: on 3,000 people, it gives a different split "
                "(ARI 0.121 with K-Means) but a comparable link with `marital-status` (NMI 0.136 versus 0.127).\n"
                "- **Limits**: 1994 data; K-Means on One-Hot variables = approximation (K-Prototypes would be an "
                "alternative); the gap between `marital-status` and `occupation` in the screening is modest.")


# ----------------------------------------------------------------------------- page
RENDERERS = {SECTIONS[0]: sec_overview, SECTIONS[1]: sec_profiles, SECTIONS[2]: sec_marital,
             SECTIONS[3]: sec_factors, SECTIONS[4]: sec_explore, SECTIONS[5]: sec_simulator,
             SECTIONS[6]: sec_choose_k, SECTIONS[7]: sec_notebook}


def render_dashboard(t):
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Customer profiles")
        st.caption("Adult / Census Income (UCI)")
        section = st.radio("Section", SECTIONS, key="dash_section", label_visibility="collapsed")
        st.divider()
        st.caption("K-Means clustering (k = 3), variable under study: marital status. "
                   "This dashboard reuses the notebook without changing it.")
    RENDERERS[section](_ctx())
    st.caption("Educational project on a 1994 US census. Segments are statistical tendencies "
               "and reflect the biases in the data: they must not be used to make real decisions.")
