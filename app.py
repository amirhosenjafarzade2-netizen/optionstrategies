# app.py
import streamlit as st
import pandas as pd
import numpy as np
from config import PREDEFINED_STRATEGIES, Strategy, Leg
from payoff import payoff_chart
from monte_carlo import monte_carlo_metrics
from genetic import genetic_optimize, FitnessKey

st.set_page_config(page_title="Option Strategy Lab", layout="wide")
st.title("Option Trading Strategy Visualiser & Optimiser")

# Sidebar
with st.sidebar:
    st.header("Asset Parameters")
    S0 = st.number_input("Current price ($)", min_value=1.0, value=150.0, step=5.0)
    vol = st.number_input("Annual volatility", min_value=0.01, value=0.20, step=0.01)
    mu = st.number_input("Daily drift (μ)", value=0.001, step=0.0001, format="%.5f")
    days = st.number_input("Days to expiration", min_value=1, value=30)
    T = days / 365.0
    asset = {"S0": S0, "vol": vol, "mu": mu, "T": T}

# Load strategies
all_strategies = PREDEFINED_STRATEGIES.copy()
if "custom" not in st.session_state:
    st.session_state.custom = []
all_strategies.extend(st.session_state.custom)

# Strategy selector
col1, col2 = st.columns([1, 3])
with col1:
    sel_name = st.selectbox("Select strategy", [s.name for s in all_strategies])
    selected = next(s for s in all_strategies if s.name == sel_name)
    if selected.is_custom and st.button("Delete"):
        st.session_state.custom = [s for s in st.session_state.custom if s.name != sel_name]
        st.experimental_rerun()

# Custom builder
with col2:
    with st.expander("Create Custom Strategy", expanded=False):
        cust_name = st.text_input("Name")
        leg_count = st.session_state.get("leg_count", 1)
        cust_legs = []
        for i in range(leg_count):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
            typ = c1.selectbox(f"Type {i}", ["call", "put", "stock"], key=f"t{i}")
            pos = c2.selectbox(f"Pos {i}", ["long", "short"], key=f"p{i}")
            strike = c3.number_input(f"Strike {i}", value=100.0, key=f"s{i}")
            prem = c4.number_input(f"Premium {i}", value=5.0, disabled=typ=="stock", key=f"pr{i}")
            cust_legs.append(Leg(type=typ, position=pos, strike=strike, premium=prem if typ!="stock" else 0))
            if c5.button("X", key=f"rm{i}"):
                st.session_state.leg_count = max(1, leg_count - 1)
                st.experimental_rerun()
        if st.button("Add Leg"):
            st.session_state.leg_count = leg_count + 1
            st.experimental_rerun()
        if st.button("Save"):
            if not cust_name.strip():
                st.error("Name required")
            else:
                st.session_state.custom.append(Strategy(
                    name=cust_name, description="Custom", legs=cust_legs,
                    max_profit="?", max_loss="?", good="", bad="", is_custom=True
                ))
                st.success("Saved!")
                st.experimental_rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Payoff", "Monte Carlo", "GA Optimizer", "Compare"])

with tab1:
    st.subheader("Adjust Legs")
    edit_legs = []
    for i, leg in enumerate(selected.legs):
        c1, c2, c3, c4 = st.columns(4)
        typ = c1.selectbox(f"T{i}", ["call","put","stock"], index=["call","put","stock"].index(leg.type), key=f"et{i}")
        pos = c2.selectbox(f"P{i}", ["long","short"], index=["long","short"].index(leg.position), key=f"ep{i}")
        strike = c3.number_input(f"S{i}", value=leg.strike, key=f"es{i}")
        prem = c4.number_input(f"Pr{i}", value=leg.premium, disabled=typ=="stock", key=f"epr{i}")
        edit_legs.append(Leg(type=typ, position=pos, strike=strike, premium=prem if typ!="stock" else 0))
    plot_strategy = Strategy(name=selected.name, description="", legs=edit_legs, max_profit="", max_loss="", good="", bad="")
    st.plotly_chart(payoff_chart(plot_strategy, S0), use_container_width=True)

with tab2:
    sims = st.slider("Simulations", 1000, 20000, 5000, 1000)
    if st.button("Run MC"):
        with st.spinner("Running..."):
            mc = monte_carlo_metrics(plot_strategy, sims=sims, **asset)
        c1, c2, c3 = st.columns(3)
        c1.metric("PoP", f"{mc['pop']:.1f}%")
        c2.metric("E[P/L]", f"${mc['expected_pl']:.2f}")
        c3.metric("95% VaR", f"${mc['var95']:.2f}")

with tab3:
    fitness = st.selectbox("Fitness", ["pop", "sharpe", "epl", "drawdown"],
                           format_func=lambda x: {"pop":"PoP", "sharpe":"Sharpe", "epl":"E[P/L]", "drawdown":"Min Loss"}[x])
    if st.button("Optimize"):
        with st.spinner("Optimizing..."):
            best = genetic_optimize(selected, asset, fitness)
        st.success("Top 3 Optimized Variants")
        for s in best:
            score = evaluate(s, asset, fitness, {})
            with st.expander(f"{s.name} – Score: {score:.2f}"):
                df = pd.DataFrame([{
                    "Type": l.type.capitalize(),
                    "Pos": l.position.capitalize(),
                    "Strike": f"${l.strike:.2f}",
                    "Premium": f"${l.premium:.2f}" if l.type != "stock" else "-"
                } for l in s.legs])
                st.dataframe(df)

with tab4:
    st.subheader("Strategy Comparison (2,000 sims)")
    rows = []
    for s in PREDEFINED_STRATEGIES:
        mc = monte_carlo_metrics(s, sims=2000, **asset)
        rows.append({"Strategy": s.name, "PoP (%)": mc["pop"], "E[P/L]": mc["expected_pl"], "95% VaR": mc["var95"]})
    df = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(df.style.format({"PoP (%)": "{:.1f}", "E[P/L]": "${:.2f}", "95% VaR": "${:.2f}"}))

# Footer
st.markdown("---")
st.caption("This tool assumes expiration payoff only. No time decay, IV, or early exercise.")
