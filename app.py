"""Streamlit platform: 6 pages on the Adult dataset (income, education, occupation, relationship, workclass, profile).

Run:  streamlit run app.py
Add a page: create targets/<name>.py (see targets/relationship.py) and add it to targets/__init__.py.
"""
import streamlit as st

from dashboard import render_dashboard
from targets import TARGETS
from ui import render_target

RENDERERS = {"marital": render_dashboard}  # pages with their own layout (otherwise: shared form)

st.set_page_config(page_title="Adult: predictions", page_icon="📊", layout="centered")


def home():
    st.title("Six models on the same census")
    st.write("All pages use the UCI Adult dataset. "
             "Choose what you want to estimate for a given profile.")
    for i in range(0, len(TARGETS), 2):
        for col, t in zip(st.columns(2), TARGETS[i:i + 2]):
            with col, st.container(border=True):
                st.subheader(f"{t.icon} {t.title}")
                st.write(t.question)
                st.caption(t.detail)
                st.page_link(PAGES[t.slug], label=f"Estimate: {t.title.lower()}")


def _make_page(t):
    def page():
        RENDERERS.get(t.slug, render_target)(t)
    page.__name__ = f"page_{t.slug}"
    return page


PAGES = {t.slug: st.Page(_make_page(t), title=t.title, icon=t.icon, url_path=t.slug) for t in TARGETS}
home_page = st.Page(home, title="Home", icon="🏠", default=True)

st.navigation([home_page, *PAGES.values()]).run()
