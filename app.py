# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PREDEFINED_STRATEGIES, Strategy, Leg
from payoff import payoff_chart, calculate_breakeven_points, calculate_max_profit_loss
from monte_carlo import monte_carlo_metrics, monte_carlo_distribution
from genetic import genetic_optimize, evaluate, FitnessKey

st.set_page_config(
    page_title="Option Strategy Lab", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced Options Trading Strategy Visualizer & Optimizer"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Option Trading Strategy Visualizer & Optimizer")
st.markdown("**Analyze, optimize, and compare options strategies with advanced Monte Carlo simulation and genetic algorithms**")

# Initialize session state
if "custom" not in st.session_state:
    st.session_state.custom = []
if "leg_count" not in st.session_state:
    st.session_state.leg_count = 1
if "plot_strategy" not in st.session_state:
    st.session_state.plot_strategy = None

# Sidebar - Asset Parameters
with st.sidebar:
    st.header("📊 Asset Parameters")
    
    with st.expander("💰 Price & Volatility", expanded=True):
        S0 = st.number_input("Current price ($)", min_value=1.0, value=150.0, step=5.0)
        vol = st.number_input("Annual volatility", min_value=0.01, max_value=2.0, value=0.20, step=0.01, 
                              help="Historical volatility (e.g., 0.20 = 20%)")
    
    with st.expander("⏱️ Time Parameters", expanded=True):
        days = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)
        mu = st.number_input("Daily drift (μ)", value=0.001, step=0.0001, format="%.5f",
                            help="Expected daily return rate")
        T = days / 365.0
    
    asset = {"S0": S0, "vol": vol, "mu": mu, "T": T}
    
    st.divider()
    
    # Display calculated metrics
    col_a, col_b = st.columns(2)
    col_a.metric("Time (years)", f"{T:.4f}")
    col_b.metric("Annual Return", f"{mu*252:.2%}")
    
    st.caption(f"💡 Daily σ: {vol/np.sqrt(252):.4f}")

# Load all strategies
all_strategies = PREDEFINED_STRATEGIES.copy()
all_strategies.extend(st.session_state.custom)

# Main Strategy Selector
st.header("🎯 Strategy Selection")
col1, col2 = st.columns([1, 3])

with col1:
    sel_name = st.selectbox(
        "Choose a strategy",
        [s.name for s in all_strategies],
        help="Select from predefined strategies or create your own"
    )
    selected = next(s for s in all_strategies if s.name == sel_name)
    
    # Strategy information card
    with st.expander("ℹ️ Strategy Details", expanded=True):
        st.markdown(f"**{selected.short_desc}**")
        st.write(selected.description)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Max Profit", selected.max_profit)
        with col_info2:
            st.metric("Max Loss", selected.max_loss)
        
        if selected.good:
            st.success(f"✅ **Good for:** {selected.good}")
        if selected.bad:
            st.error(f"❌ **Bad for:** {selected.bad}")
    
    # Delete custom strategy button
    if selected.is_custom and st.button("🗑️ Delete Strategy", use_container_width=True, type="secondary"):
        st.session_state.custom = [s for s in st.session_state.custom if s.name != sel_name]
        st.rerun()

# Custom Strategy Builder
with col2:
    with st.expander("🔧 Custom Strategy Builder", expanded=False):
        st.markdown("### Create Your Own Strategy")
        
        cust_name = st.text_input("Strategy Name", placeholder="e.g., My Custom Iron Butterfly")
        cust_desc = st.text_input("Short Description", placeholder="e.g., Bullish spread with limited risk")
        
        leg_count = st.session_state.leg_count
        cust_legs = []
        
        st.markdown("### Strategy Legs")
        
        for i in range(leg_count):
            with st.container():
                st.markdown(f"**Leg {i+1}**")
                c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
                
                typ = c1.selectbox(
                    "Type", 
                    ["call", "put", "stock"], 
                    key=f"t{i}",
                    help="Option type or underlying stock"
                )
                pos = c2.selectbox(
                    "Position", 
                    ["long", "short"], 
                    key=f"p{i}",
                    help="Long = buy, Short = sell"
                )
                strike = c3.number_input(
                    "Strike", 
                    value=float(S0), 
                    min_value=1.0, 
                    key=f"s{i}",
                    help="Strike price for the option"
                )
                prem = c4.number_input(
                    "Premium", 
                    value=5.0, 
                    min_value=0.0, 
                    disabled=typ=="stock", 
                    key=f"pr{i}",
                    help="Option premium (cost)"
                )
                
                if c5.button("✕", key=f"rm{i}", help="Remove this leg", use_container_width=True):
                    if leg_count > 1:
                        st.session_state.leg_count = leg_count - 1
                        st.rerun()
                
                cust_legs.append(Leg(
                    type=typ, 
                    position=pos, 
                    strike=strike, 
                    premium=prem if typ != "stock" else 0
                ))
                
                st.divider()
        
        col_add, col_save = st.columns(2)
        
        with col_add:
            if st.button("➕ Add Another Leg", use_container_width=True):
                st.session_state.leg_count = leg_count + 1
                st.rerun()
        
        with col_save:
            if st.button("💾 Save Strategy", use_container_width=True, type="primary"):
                if not cust_name.strip():
                    st.error("⚠️ Please enter a strategy name")
                elif any(s.name == cust_name for s in all_strategies):
                    st.error("⚠️ A strategy with this name already exists")
                elif len(cust_legs) == 0:
                    st.error("⚠️ Add at least one leg to the strategy")
                else:
                    new_strategy = Strategy(
                        name=cust_name,
                        short_desc=cust_desc or "Custom strategy",
                        description=cust_desc or "User-defined custom strategy",
                        legs=cust_legs,
                        max_profit="Custom",
                        max_loss="Custom",
                        good="User-defined",
                        bad="User-defined",
                        is_custom=True
                    )
                    st.session_state.custom.append(new_strategy)
                    st.success(f"✅ Strategy '{cust_name}' saved successfully!")
                    st.session_state.leg_count = 1
                    st.rerun()

st.divider()

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Payoff Diagram", 
    "🎲 Monte Carlo Analysis", 
    "🧬 Genetic Optimizer", 
    "⚖️ Strategy Comparison"
])

# TAB 1: Payoff Diagram
with tab1:
    st.subheader("📈 Payoff Diagram & Leg Configuration")
    
    if len(selected.legs) == 0:
        st.warning("⚠️ This strategy has no legs defined. Please select a different strategy or create a custom one.")
    else:
        # Leg editor
        st.markdown("### Adjust Strategy Parameters")
        edit_legs = []
        
        for i, leg in enumerate(selected.legs):
            with st.container():
                col_leg = st.columns([3, 2, 2, 2, 2])
                
                with col_leg[0]:
                    st.markdown(f"**Leg {i+1}:** {leg.type.title()} {leg.position.title()}")
                
                with col_leg[1]:
                    typ = st.selectbox(
                        "Type", 
                        ["call", "put", "stock"], 
                        index=["call", "put", "stock"].index(leg.type), 
                        key=f"et{i}",
                        label_visibility="collapsed"
                    )
                
                with col_leg[2]:
                    pos = st.selectbox(
                        "Position", 
                        ["long", "short"], 
                        index=["long", "short"].index(leg.position), 
                        key=f"ep{i}",
                        label_visibility="collapsed"
                    )
                
                with col_leg[3]:
                    strike = st.number_input(
                        f"Strike", 
                        value=float(leg.strike), 
                        min_value=1.0,
                        key=f"es{i}",
                        label_visibility="collapsed"
                    )
                
                with col_leg[4]:
                    prem = st.number_input(
                        f"Premium", 
                        value=float(leg.premium), 
                        min_value=0.0,
                        disabled=typ=="stock", 
                        key=f"epr{i}",
                        label_visibility="collapsed"
                    )
                
                edit_legs.append(Leg(
                    type=typ, 
                    position=pos, 
                    strike=strike, 
                    premium=prem if typ != "stock" else 0
                ))
        
        # Create strategy with edited legs
        plot_strategy = Strategy(
            name=selected.name,
            short_desc=selected.short_desc,
            description=selected.description,
            legs=edit_legs,
            max_profit=selected.max_profit,
            max_loss=selected.max_loss,
            good=selected.good,
            bad=selected.bad
        )
        
        st.session_state.plot_strategy = plot_strategy
        
        # Display strategy summary
        st.markdown("### Strategy Summary")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        net_cost = plot_strategy.net_debit_credit()
        breakevens = calculate_breakeven_points(plot_strategy, S0)
        max_profit, max_loss = calculate_max_profit_loss(plot_strategy, S0)
        
        col_sum1.metric(
            "Net Cost/Credit", 
            f"${abs(net_cost):.2f}",
            delta="Debit" if net_cost > 0 else "Credit",
            delta_color="inverse" if net_cost > 0 else "normal"
        )
        col_sum2.metric("Breakeven Points", len(breakevens))
        col_sum3.metric(
            "Max Profit", 
            "Unlimited" if max_profit == float('inf') else f"${max_profit:.2f}"
        )
        col_sum4.metric(
            "Max Loss", 
            "Unlimited" if max_loss == float('-inf') else f"${abs(max_loss):.2f}"
        )
        
        # Payoff chart
        st.markdown("### Expiration Payoff")
        try:
            fig = payoff_chart(plot_strategy, S0)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating payoff chart: {str(e)}")
        
        # Leg details table
        with st.expander("📋 Detailed Leg Breakdown"):
            leg_data = []
            for i, l in enumerate(edit_legs):
                cost = l.net_cost()
                leg_data.append({
                    "Leg #": i + 1,
                    "Type": l.type.capitalize(),
                    "Position": l.position.capitalize(),
                    "Strike": f"${l.strike:.2f}",
                    "Premium": f"${l.premium:.2f}" if l.type != "stock" else "N/A",
                    "Net Cost": f"${cost:.2f}" if cost >= 0 else f"(${abs(cost):.2f})"
                })
            
            df = pd.DataFrame(leg_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show breakeven prices
            if breakevens:
                st.markdown("**Breakeven Prices:**")
                be_cols = st.columns(len(breakevens))
                for idx, be in enumerate(breakevens):
                    be_cols[idx].metric(f"BE {idx+1}", f"${be:.2f}")

# TAB 2: Monte Carlo Analysis
with tab2:
    st.subheader("🎲 Monte Carlo Simulation Analysis")
    st.markdown("Run probabilistic simulations to evaluate risk and return metrics")
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sims = st.slider("Number of simulations", 1000, 20000, 5000, 1000,
                        help="More simulations = more accuracy but slower")
    with col_sim2:
        steps = st.slider("Time steps", 10, 100, 30, 10,
                         help="Price path granularity")
    with col_sim3:
        show_dist = st.checkbox("Show distributions", value=True,
                               help="Display payoff distribution charts")
    
    if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        if st.session_state.plot_strategy is None or len(st.session_state.plot_strategy.legs) == 0:
            st.error("⚠️ Please configure a strategy in the Payoff Diagram tab first")
        else:
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    mc = monte_carlo_metrics(
                        st.session_state.plot_strategy, 
                        sims=sims, 
                        steps=steps, 
                        **asset
                    )
                    
                    st.success(f"✅ Completed {sims:,} simulations with {steps} time steps")
                    
                    # Main metrics
                    st.markdown("### 📊 Probability Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Probability of Profit", f"{mc['pop']:.1f}%",
                               help="Chance of making money")
                    col2.metric("Probability of Loss", f"{mc['pol']:.1f}%",
                               help="Chance of losing money")
                    col3.metric("Win Rate", f"{mc['win_rate']:.1%}",
                               help="Proportion of profitable outcomes")
                    col4.metric("Expected P/L", f"${mc['expected_pl']:.2f}",
                               delta=f"Median: ${mc['median_pl']:.2f}")
                    
                    # Risk metrics
                    st.markdown("### ⚠️ Risk Metrics")
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("95% VaR", f"${mc['var95']:.2f}",
                               help="5% chance of loss exceeding this")
                    col6.metric("99% VaR", f"${mc['var99']:.2f}",
                               help="1% chance of loss exceeding this")
                    col7.metric("CVaR (95%)", f"${mc['cvar95']:.2f}",
                               help="Expected loss in worst 5% of cases")
                    col8.metric("Max Drawdown", f"${mc['max_dd']:.2f}",
                               help="Worst possible outcome")
                    
                    # Risk-adjusted returns
                    st.markdown("### 📈 Risk-Adjusted Returns")
                    col9, col10, col11, col12 = st.columns(4)
                    col9.metric("Sharpe Ratio", f"{mc['sharpe']:.3f}",
                               help="Return per unit of risk")
                    col10.metric("Sortino Ratio", f"{mc['sortino']:.3f}",
                                help="Return per unit of downside risk")
                    col11.metric("Calmar Ratio", f"{mc['calmar']:.3f}",
                                help="Return per unit of max drawdown")
                    col12.metric("Return on Risk", f"{mc['ror']:.3f}",
                                help="Expected return / VaR")
                    
                    # Profit/Loss analysis
                    st.markdown("### 💰 Profit/Loss Analysis")
                    col13, col14, col15, col16 = st.columns(4)
                    col13.metric("Avg Win", f"${mc['avg_win']:.2f}")
                    col14.metric("Avg Loss", f"${abs(mc['avg_loss']):.2f}")
                    col15.metric("Loss/Profit Ratio", f"{mc['lp_ratio']:.2f}",
                                help="Average loss / average profit")
                    
                    pf = mc['profit_factor']
                    pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
                    col16.metric("Profit Factor", pf_display,
                                help="Gross profit / gross loss")
                    
                    # Distribution visualization
                    if show_dist:
                        st.markdown("### 📊 Payoff Distribution")
                        try:
                            dist_data = monte_carlo_distribution(
                                st.session_state.plot_strategy,
                                **asset,
                                sims=min(sims, 5000)  # Limit for performance
                            )
                            
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=("Terminal Price Distribution", "Payoff Distribution")
                            )
                            
                            # Price distribution
                            fig.add_trace(
                                go.Histogram(
                                    x=dist_data['terminal_prices'],
                                    name="Prices",
                                    marker_color='lightblue',
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
                            
                            # Payoff distribution
                            colors = ['red' if p < 0 else 'green' for p in dist_data['payoffs']]
                            fig.add_trace(
                                go.Histogram(
                                    x=dist_data['payoffs'],
                                    name="Payoffs",
                                    marker_color='lightgreen',
                                    showlegend=False
                                ),
                                row=1, col=2
                            )
                            
                            fig.update_xaxes(title_text="Price ($)", row=1, col=1)
                            fig.update_xaxes(title_text="Payoff ($)", row=1, col=2)
                            fig.update_yaxes(title_text="Frequency", row=1, col=1)
                            fig.update_yaxes(title_text="Frequency", row=1, col=2)
                            
                            fig.update_layout(height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"Could not generate distribution charts: {str(e)}")
                    
                except Exception as e:
                    st.error(f"❌ Error running simulation: {str(e)}")
                    st.exception(e)

# TAB 3: Genetic Algorithm Optimizer
with tab3:
    st.subheader("🧬 Genetic Algorithm Optimization")
    st.markdown("Evolve your strategy to optimize for specific metrics")
    
    col_ga1, col_ga2, col_ga3, col_ga4 = st.columns(4)
    
    with col_ga1:
        fitness = st.selectbox(
            "Optimization Target",
            ["pop", "sharpe", "epl", "drawdown", "sortino", "calmar"],
            format_func=lambda x: {
                "pop": "Probability of Profit",
                "sharpe": "Sharpe Ratio",
                "epl": "Expected P/L",
                "drawdown": "Minimize Max Loss",
                "sortino": "Sortino Ratio",
                "calmar": "Calmar Ratio"
            }[x],
            help="Metric to optimize"
        )
    
    with col_ga2:
        pop_size = st.number_input("Population Size", min_value=20, max_value=100, value=50, step=10,
                                   help="Number of strategies per generation")
    
    with col_ga3:
        generations = st.number_input("Generations", min_value=20, max_value=200, value=100, step=10,
                                     help="Number of evolution cycles")
    
    with col_ga4:
        mutation_rate = st.slider("Mutation Rate", 0.1, 0.5, 0.2, 0.05,
                                  help="Probability of random changes")
    
    if st.button("🧬 Run Genetic Optimization", type="primary", use_container_width=True):
        if len(selected.legs) == 0:
            st.error("⚠️ Cannot optimize a strategy with no legs")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"Evolving strategy over {generations} generations..."):
                try:
                    # Update progress (simulated - actual GA doesn't provide real-time updates)
                    for i in range(10):
                        progress_bar.progress((i + 1) * 10)
                        status_text.text(f"Generation {(i+1)*generations//10}/{generations}")
                    
                    best = genetic_optimize(
                        selected, 
                        asset, 
                        fitness, 
                        pop_size, 
                        generations,
                        mutation_rate=mutation_rate
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Optimization complete! Top 3 variants found.")
                    
                    st.markdown(f"### 🏆 Optimized Strategies (Target: {fitness.upper()})")
                    
                    for rank, s in enumerate(best, 1):
                        score = evaluate(s, asset, fitness, {})
                        
                        with st.expander(
                            f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} Rank #{rank}: Score = {score:.2f}", 
                            expanded=rank==1
                        ):
                            # Display legs
                            df = pd.DataFrame([{
                                "Type": l.type.capitalize(),
                                "Position": l.position.capitalize(),
                                "Strike": f"${l.strike:.2f}",
                                "Premium": f"${l.premium:.2f}" if l.type != "stock" else "N/A",
                                "Net Cost": f"${l.net_cost():.2f}"
                            } for l in s.legs])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Quick metrics
                            st.markdown("#### Quick Metrics")
                            opt_mc = monte_carlo_metrics(s, sims=2000, steps=30, **asset)
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("PoP", f"{opt_mc['pop']:.1f}%")
                            c2.metric("E[P/L]", f"${opt_mc['expected_pl']:.2f}")
                            c3.metric("Sharpe", f"{opt_mc['sharpe']:.3f}")
                            c4.metric("Max DD", f"${opt_mc['max_dd']:.2f}")
                            
                            # Payoff chart for optimized strategy
                            st.markdown("#### Payoff Diagram")
                            opt_fig = payoff_chart(s, S0)
                            st.plotly_chart(opt_fig, use_container_width=True)
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error during optimization: {str(e)}")
                    st.exception(e)

# TAB 4: Strategy Comparison
with tab4:
    st.subheader("⚖️ Strategy Comparison Dashboard")
    st.markdown("Compare all predefined strategies under current market conditions")
    
    col_comp1, col_comp2 = st.columns([3, 1])
    with col_comp1:
        compare_sims = st.slider("Simulations per strategy", 1000, 10000, 2000, 1000,
                                help="More = accurate but slower")
    with col_comp2:
        include_custom = st.checkbox("Include custom", value=False,
                                    help="Include your custom strategies")
    
    if st.button("⚖️ Run Full Comparison", type="primary", use_container_width=True):
        strategies_to_compare = PREDEFINED_STRATEGIES.copy()
        if include_custom:
            strategies_to_compare.extend(st.session_state.custom)
        
        with st.spinner(f"Comparing {len(strategies_to_compare)} strategies..."):
            try:
                rows = []
                progress_bar = st.progress(0)
                status = st.empty()
                
                for idx, s in enumerate(strategies_to_compare):
                    status.text(f"Evaluating: {s.name}")
                    mc = monte_carlo_metrics(s, sims=compare_sims, steps=30, **asset)
                    rows.append({
                        "Strategy": s.name,
                        "Type": "Custom" if s.is_custom else "Predefined",
                        "PoP (%)": mc["pop"],
                        "E[P/L]": mc["expected_pl"],
                        "Std Dev": mc["std_pl"],
                        "95% VaR": mc["var95"],
                        "Sharpe": mc["sharpe"],
                        "Sortino": mc["sortino"],
                        "Max DD": mc["max_dd"]
                    })
                    progress_bar.progress((idx + 1) / len(strategies_to_compare))
                
                progress_bar.empty()
                status.empty()
                
                df = pd.DataFrame(rows).set_index("Strategy")
                
                st.success(f"✅ Comparison complete for {len(strategies_to_compare)} strategies")
                
                # Display results table
                st.markdown("### 📊 Comparison Results")
                st.dataframe(
                    df.style.format({
                        "PoP (%)": "{:.1f}",
                        "E[P/L]": "${:.2f}",
                        "Std Dev": "${:.2f}",
                        "95% VaR": "${:.2f}",
                        "Sharpe": "{:.3f}",
                        "Sortino": "{:.3f}",
                        "Max DD": "${:.2f}"
                    }).background_gradient(subset=["PoP (%)", "E[P/L]", "Sharpe"], cmap="RdYlGn"),
                    use_container_width=True
                )
                
                # Best strategies
                st.markdown("### 🏆 Top Performers")
                col_best1, col_best2, col_best3, col_best4 = st.columns(4)
                
                with col_best1:
                    best_pop = df["PoP (%)"].idxmax()
                    st.metric(
                        "🎯 Best Probability",
                        best_pop,
                        f"{df.loc[best_pop, 'PoP (%)']:.1f}%"
                    )
                
                with col_best2:
                    best_epl = df["E[P/L]"].idxmax()
                    st.metric(
                        "💰 Best Expected P/L",
                        best_epl,
                        f"${df.loc[best_epl, 'E[P/L]']:.2f}"
                    )
                
                with col_best3:
                    best_sharpe = df["Sharpe"].idxmax()
                    st.metric(
                        "📈 Best Sharpe",
                        best_sharpe,
                        f"{df.loc[best_sharpe, 'Sharpe']:.3f}"
                    )
                
                with col_best4:
                    best_sortino = df["Sortino"].idxmax()
                    st.metric(
                        "🎲 Best Sortino",
                        best_sortino,
                        f"{df.loc[best_sortino, 'Sortino']:.3f}"
                    )
                
                # Visualization
                st.markdown("### 📊 Visual Comparison")
                
                # Create scatter plot
                fig = go.Figure()
                
                for idx, row in df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row["E[P/L]"]],
                        y=[row["PoP (%)"]],
                        mode="markers+text",
                        name=idx,
                        text=idx,
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=row["Sharpe"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Sharpe")
                        ),
                        hovertemplate=f"<b>{idx}</b><br>" +
                                    "E[P/L]: $%{x:.2f}<br>" +
                                    "PoP: %{y:.1f}%<br>" +
                                    f"Sharpe: {row['Sharpe']:.3f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title="Strategy Comparison: Expected P/L vs Probability of Profit",
                    xaxis_title="Expected P/L ($)",
                    yaxis_title="Probability of Profit (%)",
                    template="plotly_white",
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download option
                csv = df.reset_index().to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"strategy_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d;'>
    <p><strong>⚠️ Important Disclaimer</strong></p>
    <p>This tool models <strong>expiration payoff only</strong>. It does not account for:</p>
    <ul style='list-style-position: inside;'>
        <li>Time decay (Theta)</li>
        <li>Implied volatility changes (Vega)</li>
        <li>Dividends</li>
        <li>Early exercise</li>
        <li>Transaction costs and fees</li>
        <li>Liquidity and slippage</li>
    </ul>
    <p><strong>Results are for educational purposes only and should not be used as investment advice.</strong></p>
    <p style='margin-top: 20px;'>📚 Built with Streamlit | Monte Carlo (GBM) | Genetic Algorithm Optimization</p>
</div>
""", unsafe_allow_html=True)
