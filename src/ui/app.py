"""
Stage 7: Streamlit application entry point.
Run with: streamlit run src/ui/app.py

Deployment options (all free tiers available):
  - Streamlit Community Cloud  → push to GitHub, connect repo (RECOMMENDED — zero cost, no infra)
  - Render.com                 → Docker deploy, free tier (750 hrs/month)
  - Railway.app                → Docker deploy, $5 credit/month free
  - Hugging Face Spaces        → free Streamlit hosting, good for public demos
"""

import streamlit as st

st.set_page_config(
    page_title="UK Housing Market Model",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("UK Housing Market Model")
    st.caption("Buy / Hold / Don't Buy scoring for retail buyers and REITs")

    tab_consumer, tab_reit, tab_model = st.tabs(["Consumer View", "REIT View", "Model Explorer"])

    with tab_consumer:
        _consumer_view()

    with tab_reit:
        _reit_view()

    with tab_model:
        _model_explorer()


def _consumer_view():
    st.header("Personal Affordability & Timing Score")

    col1, col2 = st.columns(2)
    with col1:
        region  = st.selectbox("Region", options=["London", "South East", "North West"])
        income  = st.number_input("Annual household income (£)", min_value=10_000, max_value=500_000, value=55_000, step=1_000)
        deposit = st.number_input("Deposit available (£)", min_value=5_000, max_value=2_000_000, value=50_000, step=5_000)

    with col2:
        st.metric("Score", "— / 100", help="Runs once Stage 4–6 are complete")
        st.metric("Signal", "—")

    # TODO (Stage 7): connect to score_consumer(), display fan chart, map


def _reit_view():
    st.header("REIT Portfolio Score")

    col1, col2 = st.columns(2)
    with col1:
        region      = st.selectbox("Target region", options=["London", "South East", "North West"], key="reit_region")
        gross_yield = st.slider("Current gross yield (%)", min_value=1.0, max_value=12.0, value=5.0, step=0.1)

    with col2:
        st.metric("REIT Score", "— / 100")
        st.metric("Cycle Position", "—")

    # TODO (Stage 7): connect to score_reit(), display yield chart, regional heatmap


def _model_explorer():
    st.header("Model Parameter Explorer")
    st.info("Adjust SDE parameters below and see the impact on simulated price paths.")

    col1, col2, col3 = st.columns(3)
    with col1:
        theta = st.slider("Mean reversion speed (θ)", 0.01, 1.0, 0.15, 0.01)
    with col2:
        mu    = st.slider("Long-run mean price (£k)", 100, 800, 280, 10)
    with col3:
        sigma = st.slider("Volatility (σ)", 0.01, 0.5, 0.08, 0.01)

    # TODO (Stage 7): run live Monte Carlo with these params and plot fan chart


if __name__ == "__main__":
    main()
