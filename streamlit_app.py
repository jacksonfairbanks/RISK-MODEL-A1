import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("BTC Treasury Monte Carlo Risk Explorer")

# --- CORE ASSUMPTIONS ---
BTC_INITIAL = 6000
BTC_PRICE_INITIAL = 111000

# --- NET ASSET VALUE (NAV) CALC ---
btc_total = BTC_INITIAL  # will add pref raise later
net_asset_value = BTC_INITIAL * BTC_PRICE_INITIAL

# --- PARAMETER SLIDERS ---
st.sidebar.header("Scenario Assumptions")
pref_equity_slider = st.sidebar.slider(
    "Notional Pref Equity Outstanding (% of NAV)",
    min_value=0, max_value=100, value=20, step=1)
pref_equity_raise = net_asset_value * pref_equity_slider / 100

dividend_rate = st.sidebar.slider(
    "Dividend Rate (%)", min_value=5, max_value=20, value=10, step=1) / 100

n_years = st.sidebar.slider("Time Horizon (years)", min_value=1, max_value=10, value=5, step=1)
btc_div_fraction = st.sidebar.slider("BTC Fraction Sold for Divs (%)", min_value=0, max_value=100, value=50, step=5) / 100

monthly_dividend = (pref_equity_raise * dividend_rate) / 12
additional_btc = pref_equity_raise / BTC_PRICE_INITIAL
btc_total = BTC_INITIAL + additional_btc
net_asset_value = btc_total * BTC_PRICE_INITIAL
t_months = n_years * 12
simulations = 3000  # quick for responsiveness

# --- Load Parameters from JSON ---
with open("btc_model_parameters.json") as f:
    params = json.load(f)
CAGR = params["Lognormal_parameters"]["annual_cagr"]
VOL = params["Lognormal_parameters"]["annual_vol"]
JUMP_FREQ = params["JumpDiffusion_parameters"]["jump_frequency_per_year"]
JUMP_SIZE_MEAN = params["JumpDiffusion_parameters"]["jump_mean"]
JUMP_SIZE_STD = params["JumpDiffusion_parameters"]["jump_std"]
GARCH_PARAMS = params["GARCH_parameters"]

def simulate_dividend_paths(prices_arr):
    final_collateral = []
    for prices in prices_arr:
        btc_holdings = btc_total
        for month in range(1, len(prices)):
            div_payment_btc = btc_div_fraction * monthly_dividend / prices[month]
            btc_holdings -= div_payment_btc
            if btc_holdings < 0:
                btc_holdings = 0
        ending_collateral = btc_holdings * prices[-1]
        final_collateral.append(ending_collateral)
    return np.array(final_collateral)

def summary_stats(collateral):
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "mean": np.mean(collateral),
        "std": np.std(collateral),
        "min": np.min(collateral),
        **{f"{p}%": np.percentile(collateral, p) for p in percentiles},
        "prob_deficit": np.mean(collateral < pref_equity_raise)
    }

def lognorm_sim(mu, sigma, simulations):
    all_paths = []
    for _ in range(simulations):
        m = np.log(1 + mu) / 12
        s = sigma / np.sqrt(12)
        shocks = np.random.normal(m, s, t_months)
        prices = [BTC_PRICE_INITIAL]
        for shock in shocks:
            prices.append(prices[-1] * np.exp(shock))
        all_paths.append(prices)
    return np.array(all_paths)

def jump_diffusion_sim(mu, sigma, lam, jump_mu, jump_sigma, simulations):
    dt = 1/12
    all_paths = []
    for _ in range(simulations):
        prices = [BTC_PRICE_INITIAL]
        for t in range(t_months):
            dW = np.random.normal(0, np.sqrt(dt))
            J = np.random.poisson(lam * dt)
            jump_sum = np.sum(np.random.normal(jump_mu, jump_sigma, J)) if J > 0 else 0
            dS = (mu - 0.5 * sigma**2)*dt + sigma*dW + jump_sum
            prices.append(prices[-1] * np.exp(dS))
        all_paths.append(prices)
    return np.array(all_paths)

def garch_simulate(garch_params, mu, sigma, simulations):
    dt = 1/12
    paths = []
    omega = garch_params.get("omega", 0.01)
    alpha = garch_params.get("alpha[1]", 0.05)
    beta = garch_params.get("beta[1]", 0.90)
    for _ in range(simulations):
        prices = [BTC_PRICE_INITIAL]
        vol_t = np.sqrt(omega / (1 - alpha - beta))
        ret_t = 0
        for _ in range(t_months):
            vol_t = np.sqrt(omega + alpha * ret_t**2 + beta * vol_t**2)
            ret_t = np.random.normal(mu / 12, vol_t * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(ret_t))
        paths.append(prices)
    return np.array(paths)

# RUN SIMULATIONS
with st.spinner("Running simulations..."):
    lognorm_paths = lognorm_sim(CAGR, VOL, simulations)
    final_lognorm = simulate_dividend_paths(lognorm_paths)
    jump_paths = jump_diffusion_sim(CAGR, VOL, JUMP_FREQ, JUMP_SIZE_MEAN, JUMP_SIZE_STD, simulations)
    final_jump = simulate_dividend_paths(jump_paths)
    garch_paths = garch_simulate(GARCH_PARAMS, CAGR, VOL, simulations)
    final_garch = simulate_dividend_paths(garch_paths)

outs = {
    "Lognormal": summary_stats(final_lognorm),
    "JumpDiffusion": summary_stats(final_jump),
    "GARCH": summary_stats(final_garch)
}

# OUTPUT CHART
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
colors = ['blue', 'red', 'green']
labels = ['Lognormal', 'Jump Diffusion', 'GARCH']

def billions_trillions(x, pos=None):
    if x >= 1e12:
        return f"${x*1e-12:.0f}T"
    elif x >= 1e9:
        return f"${x*1e-9:.1f}B"
    elif x >= 1e8:
        return f"${x*1e-9:.1f}B"
    elif x == 0:
        return "$0"
    else:
        return f"${x:.0f}"

fig, ax = plt.subplots(figsize=(12,7))
for arr, label, color in zip([final_lognorm, final_jump, final_garch], labels, colors):
    values = np.percentile(arr, percentiles)
    ax.plot(percentiles, values, marker='o', label=label, color=color, lw=2)
ax.axhline(pref_equity_raise, color="black", linestyle="dashed", linewidth=1, label="Pref Equity")
ax.set_xlabel("Percentile (%)", fontsize=13)
ax.set_ylabel("Terminal Collateral (USD, Billions/Trillions)", fontsize=13)
ax.set_title("OEP (Percentile) Curves: Terminal Collateral by Model", fontsize=17)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(billions_trillions))
ax.set_xticks(percentiles)
ax.set_xticklabels([f"{p}%" for p in percentiles])
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, which="both", ls="--", lw=0.7)
fig.tight_layout()

st.pyplot(fig)

# --- Scenario Summary Readout ---
st.subheader("Scenario Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**BTC Initial Holdings:** {BTC_INITIAL:,} BTC")
    st.write(f"**Initial BTC Price:** ${BTC_PRICE_INITIAL:,.0f}")
    st.write(f"**Additional BTC (Pref Raise):** {additional_btc:,.2f} BTC")
    st.write(f"**Total BTC at Start:** {btc_total:,.2f}")
    st.write(f"**Net Asset Value (NAV):** ${net_asset_value:,.0f}")
    st.write(f"**Notional Pref Equity Out:** ${pref_equity_raise:,.0f} ({pref_equity_slider}%)")
    st.write(f"**Dividend Rate:** {dividend_rate:.2%}/yr")
    st.write(f"**Monthly Dividend:** ${monthly_dividend:,.2f}")
    st.write(f"**BTC Fraction Sold for Divs:** {btc_div_fraction:.0%}")
    st.write(f"**Time Horizon:** {n_years} years")
with col2:
    st.write(f"**Probability Collateral < Obligations at End:**")
    st.write(f"- Lognormal: {100*outs['Lognormal']['prob_deficit']:.2f}%")
    st.write(f"- Jump Diffusion: {100*outs['JumpDiffusion']['prob_deficit']:.2f}%")
    st.write(f"- GARCH: {100*outs['GARCH']['prob_deficit']:.2f}%")

if st.checkbox("Show percentile table"):
    df_out = pd.DataFrame(outs).T.reset_index().rename(columns={"index": "Model"})
    cols = ["Model", "mean", "std", "min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "prob_deficit"]
    df_out = df_out[cols]
    st.dataframe(df_out.style.format(
        {c: "{:,.2f}" for c in df_out.columns if c != "Model"})
    )
# --- Descriptive Commentary ---
def generate_commentary(pref_equity_slider, dividend_rate, n_years, outs):
    scenario = f"With a {pref_equity_slider}% leveraged pref structure over a {n_years}-year horizon, " \
               f"paying a {dividend_rate:.1%} annual dividend, "
    base = "assuming you meet obligations with the sale of Bitcoin, the risk model projects the following probabilities of ending undercollateralized: "
    outcomes = (f"Lognormal: {100*outs['Lognormal']['prob_deficit']:.2f}%, "
                f"Jump Diffusion: {100*outs['JumpDiffusion']['prob_deficit']:.2f}%, "
                f"GARCH: {100*outs['GARCH']['prob_deficit']:.2f}%.")
    return scenario + base + outcomes


st.markdown("----")
st.subheader("Commentary")
st.write(generate_commentary(pref_equity_slider, dividend_rate, n_years, outs))

# --- Example Failed Path Display for Jump Diffusion ---

import matplotlib.pyplot as plt
import random

st.markdown("----")
st.subheader("Example Failed Path (Jump Diffusion)")

# Find failed Jump Diffusion paths (collateral < pref equity at end)
failed_idxs = [i for i, val in enumerate(final_jump) if val < pref_equity_raise]

if len(failed_idxs) == 0:
    st.info("No failed Jump Diffusion scenarios in current settings!")
else:
    show_path = st.button("Show random failed path")
    # Start with (or default to) first failed path unless button is pressed
    if show_path or 'failed_path_idx' not in st.session_state:
        st.session_state.failed_path_idx = random.choice(failed_idxs)
    idx = st.session_state.failed_path_idx
    prices = jump_paths[idx]
    holdings = [btc_total]
    for i in range(1, len(prices)):
        div_btc = btc_div_fraction * monthly_dividend / prices[i]
        holdings.append(max(holdings[-1] - div_btc, 0))

    months = np.arange(len(prices))
    years = months / 12

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(years, prices, color='blue', label='BTC Price')
    ax2.axhline(BTC_PRICE_INITIAL, color='gray', linestyle='dotted', linewidth=1, label='Initial BTC Price')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Simulated BTC Price (USD)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3 = ax2.twinx()
    ax3.plot(years, holdings, color='red', label='BTC Holdings')
    ax3.set_ylabel('BTC Holdings (BTC)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    fig2.tight_layout()
    st.pyplot(fig2)

    st.caption(
        f"Final BTC price: ${prices[-1]:,.0f}; "
        f"final BTC holdings: {holdings[-1]:,.2f}; "
        f"terminal collateral: ${holdings[-1]*prices[-1]:,.2f}; "
        f"pref obligation: ${pref_equity_raise:,.0f}"
    )
