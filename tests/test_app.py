"""Tests every page: the page opens, the form is submitted, results are consistent."""
import pytest
from streamlit.testing.v1 import AppTest

from targets import BY_SLUG, TARGETS

SCRIPT = """
import sys; sys.path.insert(0, {root!r})
from targets import BY_SLUG
from ui import render_target
render_target(BY_SLUG[{slug!r}])
"""
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


def run(slug):
    at = AppTest.from_string(SCRIPT.format(root=ROOT, slug=slug), default_timeout=300).run()
    assert not at.exception, at.exception
    return at


def submit(at):
    next(b for b in at.button if b.label == "Predict").click()
    at.run()
    assert not at.exception, at.exception
    return at


def results(at):
    """Result cards displayed (only one expected per prediction)."""
    return [m.value for m in at.markdown if "class=\"result-card\"" in m.value]


@pytest.mark.parametrize("slug", [t.slug for t in TARGETS])
def test_page_and_prediction(slug):
    at = submit(run(slug))
    assert len(results(at)) == 1, "exactly one result expected"
    assert not list(at.get("progress")), "no probability bars any more"


def test_income_one_result_whichever_model():
    at = run("income")
    for model in ("logreg", "rf"):
        at.selectbox(key="income_model").set_value(model)
        submit(at)
        assert len(results(at)) == 1


def test_relationship_husband_profile():
    at = run("relationship")  # default profile: married man, 30 years old
    submit(at)
    assert "Husband<" in results(at)[0]


def test_relationship_wife_profile():
    at = run("relationship")
    at.selectbox(key="relationship_sex").set_value("Female")
    submit(at)
    assert "Wife<" in results(at)[0]


def test_education_high_profile():
    at = run("education")
    at.selectbox(key="education_occupation").set_value("Prof-specialty")
    at.selectbox(key="education_income").set_value(">50K")
    submit(at)
    assert "Higher education<" in results(at)[0]


def test_unknown_option_only_where_trained():
    from core import options_for
    assert ("Unknown", "Not specified") in options_for("workclass", BY_SLUG["income"].unknown_ok)
    assert ("Unknown", "Not specified") not in options_for("workclass", BY_SLUG["relationship"].unknown_ok)


def test_marital_profiles():
    young = submit(run("marital"))  # default profile: 30 years old, Bachelors, Private
    assert len(results(young)) == 1
    at = run("marital")
    at.slider(key="marital_age").set_value(25)
    at.selectbox(key="marital_education").set_value("HS-grad")
    at.selectbox(key="marital_occupation").set_value("Adm-clerical")
    submit(at)
    assert "Young professionals<" in results(at)[0]
    at = run("marital")
    at.slider(key="marital_age").set_value(45)
    at.selectbox(key="marital_education").set_value("Masters")
    at.selectbox(key="marital_income").set_value(">50K")
    at.slider(key="marital_hours-per-week").set_value(50)
    submit(at)
    assert "Graduate executives<" in results(at)[0]
    at = run("marital")
    at.slider(key="marital_age").set_value(58)
    at.selectbox(key="marital_education").set_value("9th")
    at.selectbox(key="marital_occupation").set_value("Craft-repair")
    submit(at)
    assert "Experienced seniors<" in results(at)[0]


def test_workclass_has_all_classes_and_one_result():
    from core import OPTIONS
    at = run("workclass")
    assert not any(s.key == "workclass_workclass" for s in at.selectbox), "workclass is the target, not a field"
    submit(at)
    assert len(results(at)) == 1
    bundle = BY_SLUG["workclass"].load()
    assert set(bundle["models"]["rf"]["pipeline"].classes_) == {v for v, _ in OPTIONS["workclass"]}


def test_workclass_unknown_occupation_ok():
    at = run("workclass")
    at.selectbox(key="workclass_occupation").set_value("Unknown")
    at.selectbox(key="workclass_native-country").set_value("Unknown")
    submit(at)
    assert len(results(at)) == 1
