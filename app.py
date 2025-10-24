# app.py
import streamlit as st
import pandas as pd
import numpy as np
from config import PREDEFINED_STRATEGIES, Strategy, Leg
from payoff import payoff_chart
from monte_carlo import monte_carlo_metrics
from genetic import genetic_optimize, evaluate, FitnessKey

st.set_page_config(page_title="Option Strategy Lab", layout="wide")
st.title("Option Trading Strategy Visualiser & Optimiser")

# Initialize session state
if "custom" not in st.session_state:
    st.session_state.custom = []
if "leg_count" not in st.session_state:
    st.session_state.leg_count = 1

# Sidebar
with st.sidebar:
    st.header("Asset Parameters")
    S0 = st.number_input("Current price ($)", min_value=1.0, value=150.0, step=5.0)
    vol = st.number_input("Annual volatility", min_value=0.01, max_value=2.0, value=0.20, step=0.01)
    mu = st.number_input("Daily drift (μ)", value=0.001, step=0.0001, format="%.5f")
    days = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)
    T = days / 365.0
    asset = {"S0": S0, "vol": vol, "mu": mu, "T": T}
    
    st.divider()
    st.caption(f"Time to expiration: {T:.4f} years")

# Load strategies
all_strategies = PREDEFINED_STRATEGIES.copy()
all_strategies.extend(st.session_state.custom)

# Strategy selector
col1, col2 = st.columns([1, 3])
with col1:
    sel_name = st.selectbox("Select strategy", [s.name for s in all_strategies])
    selected = next(s for s in all_strategies if s.name == sel_name)
    
    # Display strategy info
    with st.expander("ℹ️ Strategy Info", expanded=False):
        st.markdown(f"**{selected.short_desc if hasattr(selected, 'short_desc') and selected.short_desc else 'Custom Strategy'}**")
        st.write(selected.description)
        if selected.max_profit:
            st.metric("Max Profit", selected.max_profit)
        if selected.max_loss:
            st.metric("Max Loss", selected.max_loss)
    
    if selected.is_custom and st.button("🗑️ Delete Strategy", use_container_width=True):
        st.session_state.custom = [s for s in st.session_state.custom if s.name != sel_name]
        st.rerun()

# Custom builder
with col2:
    with st.expander("🔧 Create Custom Strategy", expanded=False):
        cust_name = st.text_input("Strategy Name", placeholder="e.g., My Custom Spread")
        cust_desc = st.text_input("Short Description", placeholder="e.g., Bullish with limited risk")
        
        leg_count = st.session_state.leg_count
        cust_legs = []
        
        st.subheader("Legs")
        for i in range(leg_count):
            col_container = st.container()
            with col_container:
                c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
                typ = c1.selectbox(f"Type", ["call", "put", "stock"], key=f"t{i}", label_visibility="collapsed")
                pos = c2.selectbox(f"Position", ["long", "short"], key=f"p{i}", label_visibility="collapsed")
                strike = c3.number_input(f"Strike", value=float(S0), min_value=1.0, key=f"s{i}", label_visibility="collapsed")
                prem = c4.number_input(f"Premium", value=5.0, min_value=0.0, disabled=typ=="stock", key=f"pr{i}", label_visibility="collapsed")
                
                if c5.button("✕", key=f"rm{i}", help="Remove leg", use_container_width=True):
                    if leg_count > 1:
                        st.session_state.leg_count = leg_count - 1
                        st.rerun()
                
                cust_legs.append(Leg(
                    type=typ, 
                    position=pos, 
                    strike=strike, 
                    premium=prem if typ != "stock" else 0
                ))
        
        col_add, col_save = st.columns(2)
        with col_add:
            if st.button("➕ Add Leg", use_container_width=True):
                st.session_state.leg_count = leg_count + 1
                st.rerun()
        
        with col_save:
            if st.button("💾 Save Strategy", use_container_width=True, type="primary"):
                if not cust_name.strip():
                    st.error("⚠️ Strategy name is required")
                elif any(s.name == cust_name for s in all_strategies):
                    st.error("⚠️ Strategy name already exists")
                elif len(cust_legs) == 0:
                    st.error("⚠️ At least one leg is required")
                else:
                    st.session_state.custom.append(Strategy(
                        name=cust_name,
                        short_desc=cust_desc or "Custom strategy",
                        description=cust_desc or "User-defined custom strategy",
                        legs=cust_legs,
                        max_profit="Custom",
                        max_loss="Custom",
                        good="",
                        bad="",
                        is_custom=True
                    ))
                    st.success(f"✅ Strategy '{cust_name}' saved successfully!")
                    st.session_state.leg_count = 1
                    st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Payoff", "🎲 Monte Carlo", "🧬 GA Optimizer", "⚖️ Compare"])

with tab1:
    st.subheader("Adjust Strategy Legs")
    
    if len(selected.legs) == 0:
        st.warning("⚠️ This strategy has no legs defined")
    else:
        edit_legs = []
        for i, leg in enumerate(selected.legs):
            with st.container():
                c1, c2, c3, c4 = st.columns(4)
                typ = c1.selectbox(
                    f"Type", 
                    ["call", "put", "stock"], 
                    index=["call", "put", "stock"].index(leg.type), 
                    key=f"et{i}"
                )
                pos = c2.selectbox(
                    f"Position", 
                    ["long", "short"], 
                    index=["long", "short"].index(leg.position), 
                    key=f"ep{i}"
                )
                strike = c3.number_input(
                    f"Strike ${i+1}", 
                    value=float(leg.strike), 
                    min_value=1.0,
                    key=f"es{i}"
                )
                prem = c4.number_input(
                    f"Premium ${i+1}", 
                    value=float(leg.premium), 
                    min_value=0.0,
                    disabled=typ=="stock", 
                    key=f"epr{i}"
                )
                edit_legs.append(Leg(
                    type=typ, 
                    position=pos, 
                    strike=strike, 
                    premium=prem if typ != "stock" else 0
                ))
        
        plot_strategy = Strategy(
            name=selected.name,
            short_desc=getattr(selected, 'short_desc', ''),
            description="",
            legs=edit_legs,
            max_profit="",
            max_loss="",
            good="",
            bad=""
        )
        
        st.plotly_chart(payoff_chart(plot_strategy, S0), use_container_width=True)
        
        # Display leg summary
        with st.expander("📋 Leg Summary"):
            leg_data = []
            for i, l in enumerate(edit_legs):
                leg_data.append({
                    "Leg": i + 1,
                    "Type": l.type.capitalize(),
                    "Position": l.position.capitalize(),
                    "Strike": f"${l.strike:.2f}",
                    "Premium": f"${l.premium:.2f}" if l.type != "stock" else "N/A"
                })
            st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Monte Carlo Simulation")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sims = st.slider("Number of simulations", 1000, 20000, 5000, 1000)
    with col_sim2:
        steps = st.slider("Time steps", 10, 100, 30, 10)
    
    if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        if len(plot_strategy.legs) == 0:
            st.error("⚠️ Cannot run simulation on strategy with no legs")
        else:
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    mc = monte_carlo_metrics(plot_strategy, sims=sims, steps=steps, **asset)
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Probability of Profit", f"{mc['pop']:.1f}%")
                    col2.metric("Expected P/L", f"${mc['expected_pl']:.2f}")
                    col3.metric("95% VaR", f"${mc['var95']:.2f}")
                    col4.metric("Max Drawdown", f"${mc['max_dd']:.2f}")
                    
                    # Additional metrics
                    st.divider()
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Probability of Loss", f"{mc['pol']:.1f}%")
                    col6.metric("Sharpe Ratio", f"{mc['sharpe']:.3f}")
                    col7.metric("Sortino Ratio", f"{mc['sortino']:.3f}")
                    col8.metric("Calmar Ratio", f"{mc['calmar']:.3f}")
                    
                    # Loss/Profit Ratio
                    if mc['lp_ratio'] > 0:
                        st.info(f"📊 Average Loss / Average Profit Ratio: {mc['lp_ratio']:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error running simulation: {str(e)}")

with tab3:
    st.subheader("Genetic Algorithm Optimization")
    
    col_ga1, col_ga2, col_ga3 = st.columns(3)
    with col_ga1:
        fitness = st.selectbox(
            "Optimization Target",
            ["pop", "sharpe", "epl", "drawdown"],
            format_func=lambda x: {
                "pop": "Probability of Profit",
                "sharpe": "Sharpe Ratio",
                "epl": "Expected P/L",
                "drawdown": "Minimize Max Loss"
            }[x]
        )
    with col_ga2:
        pop_size = st.number_input("Population Size", min_value=10, max_value=100, value=50, step=10)
    with col_ga3:
        generations = st.number_input("Generations", min_value=10, max_value=200, value=100, step=10)
    
    if st.button("🧬 Run Genetic Optimization", type="primary", use_container_width=True):
        if len(selected.legs) == 0:
            st.error("⚠️ Cannot optimize strategy with no legs")
        else:
            with st.spinner(f"Optimizing over {generations} generations..."):
                try:
                    best = genetic_optimize(selected, asset, fitness, pop_size, generations)
                    
                    st.success(f"✅ Top 3 Optimized Variants (Target: {fitness.upper()})")
                    
                    for rank, s in enumerate(best, 1):
                        score = evaluate(s, asset, fitness, {})
                        with st.expander(f"🏆 Rank #{rank}: {s.name} — Score: {score:.2f}", expanded=rank==1):
                            df = pd.DataFrame([{
                                "Type": l.type.capitalize(),
                                "Position": l.position.capitalize(),
                                "Strike": f"${l.strike:.2f}",
                                "Premium": f"${l.premium:.2f}" if l.type != "stock" else "N/A"
                            } for l in s.legs])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Quick metrics for optimized strategy
                            opt_mc = monte_carlo_metrics(s, sims=2000, **asset)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("PoP", f"{opt_mc['pop']:.1f}%")
                            c2.metric("E[P/L]", f"${opt_mc['expected_pl']:.2f}")
                            c3.metric("Sharpe", f"{opt_mc['sharpe']:.3f}")
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")

with tab4:
    st.subheader("Strategy Comparison Dashboard")
    
    compare_sims = st.slider("Simulations per strategy", 1000, 10000, 2000, 1000)
    
    if st.button("⚖️ Run Comparison", type="primary", use_container_width=True):
        with st.spinner(f"Comparing {len(PREDEFINED_STRATEGIES)} strategies..."):
            try:
                rows = []
                progress_bar = st.progress(0)
                
                for idx, s in enumerate(PREDEFINED_STRATEGIES):
                    mc = monte_carlo_metrics(s, sims=compare_sims, **asset)
                    rows.append({
                        "Strategy": s.name,
                        "PoP (%)": mc["pop"],
                        "E[P/L]": mc["expected_pl"],
                        "95% VaR": mc["var95"],
                        "Sharpe": mc["sharpe"],
                        "Max DD": mc["max_dd"]
                    })
                    progress_bar.progress((idx + 1) / len(PREDEFINED_STRATEGIES))
                
                progress_bar.empty()
                
                df = pd.DataFrame(rows).set_index("Strategy")
                
                # Style the dataframe
                styled_df = df.style.format({
                    "PoP (%)": "{:.1f}",
                    "E[P/L]": "${:.2f}",
                    "95% VaR": "${:.2f}",
                    "Sharpe": "{:.3f}",
                    "Max DD": "${:.2f}"
                }).background_gradient(subset=["PoP (%)"], cmap="RdYlGn")
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Download option
                csv = df.to_csv()
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="strategy_comparison.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool assumes expiration payoff only. It does not model time decay (theta), implied volatility changes, dividends, or early exercise. Results are for educational purposes only.")
st.caption("📚 Built with Streamlit | Monte Carlo simulation using Geometric Brownian Motion")
