"""Tests the \"Customer profiles\" dashboard: every section opens without error, the simulator gives a segment."""
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import PRESETS, SECTIONS

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
SCRIPT = f"""
import sys; sys.path.insert(0, {ROOT!r})
from targets import BY_SLUG
from dashboard import render_dashboard
render_dashboard(BY_SLUG["marital"])
"""


@pytest.fixture(scope="module")
def at():
    return AppTest.from_string(SCRIPT, default_timeout=300).run()


@pytest.mark.parametrize("section", SECTIONS)
def test_section_opens(at, section):
    at.radio(key="dash_section").set_value(section).run()
    assert not at.exception, at.exception
    assert at.title


def test_simulator_presets_land_in_expected_segment(at):
    at.radio(key="dash_section").set_value("Simulator and recommendations").run()
    expected = {"Young clerical employee": "Young professionals", "Graduate executive": "Graduate executives",
                "Senior in a manual trade": "Experienced seniors"}
    for preset, segment in expected.items():
        at.selectbox(key="sim_preset").set_value(preset).run()
        assert not at.exception, at.exception
        assert next(m for m in at.metric if m.label == "Assigned segment").value == segment, preset
    assert set(expected) < set(PRESETS)


def test_simulator_single_result_and_offers(at):
    at.radio(key="dash_section").set_value("Simulator and recommendations").run()
    at.selectbox(key="sim_preset").set_value("Custom").run()
    assert len([m for m in at.metric if m.label == "Assigned segment"]) == 1
    assert not list(at.get("progress"))
    at.selectbox(key="sim_marital").set_value("").run()  # unknown status: only the 4 segment offers
    n = len(at.caption)
    at.selectbox(key="sim_marital").set_value("Widowed").run()
    assert len(at.caption) == n + 1  # extra offer tied to the declared status


def test_explore_small_subgroup_warns(at):
    at.radio(key="dash_section").set_value("Explore a subgroup").run()
    at.slider(key="ex_age").set_value((74, 74)).run()
    at.slider(key="ex_edu").set_value((16, 16)).run()
    assert not at.exception, at.exception
