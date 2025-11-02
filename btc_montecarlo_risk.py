import json
import numpy as np
import pandas as pd

# === LOAD PARAMETERS ===
param_path = r"C:\Users\jaxso\Documents\BTC RISK MODEL\btc_model_parameters.json"
with open(param_path, "r") as f:
    params = json.load(f)

# === SIMULATION SETTINGS ===
BTC_INITIAL = 6000
BTC_PRICE_INITIAL = 111000
PREF_EQUITY_RAISE = 150e6
DIVIDEND_RATE = 0.14
N_YEARS = 5
MONTHLY_DIVIDEND = (PREF_EQUITY_RAISE * DIVIDEND_RATE) / 12
BTC_DIV_FRACTION = 0.5
ADDITIONAL_BTC = PREF_EQUITY_RAISE / BTC_PRICE_INITIAL
BTC_TOTAL = BTC_INITIAL + ADDITIONAL_BTC
T_MONTHS = N_YEARS * 12
SIMULATIONS = 5000

CAGR = params["Lognormal_parameters"]["annual_cagr"]
VOL = params["Lognormal_parameters"]["annual_vol"]
JUMP_FREQ = params["JumpDiffusion_parameters"]["jump_frequency_per_year"]
JUMP_SIZE_MEAN = params["JumpDiffusion_parameters"]["jump_mean"]
JUMP_SIZE_STD = params["JumpDiffusion_parameters"]["jump_std"]
GARCH_PARAMS = params["GARCH_parameters"]

def simulate_dividend_paths(prices_arr):
    final_collateral = []
    for prices in prices_arr:
        BTC_holdings = BTC_TOTAL
        for month in range(1, len(prices)):
            div_payment_btc = BTC_DIV_FRACTION * MONTHLY_DIVIDEND / prices[month]
            BTC_holdings -= div_payment_btc
            if BTC_holdings < 0:
                BTC_holdings = 0
        ending_collateral = BTC_holdings * prices[-1]
        final_collateral.append(ending_collateral)
    return np.array(final_collateral)

def summary_stats(collateral):
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "mean": np.mean(collateral),
        "std": np.std(collateral),
        "min": np.min(collateral),
        **{f"{p}%": np.percentile(collateral, p) for p in percentiles},
        "prob_deficit": np.mean(collateral < PREF_EQUITY_RAISE)
    }

# === MODELS ===
def lognorm_sim(mu, sigma, simulations):
    all_paths = []
    for _ in range(simulations):
        m = np.log(1 + mu) / 12
        s = sigma / np.sqrt(12)
        shocks = np.random.normal(m, s, T_MONTHS)
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
        for t in range(T_MONTHS):
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
        for _ in range(T_MONTHS):
            vol_t = np.sqrt(omega + alpha * ret_t**2 + beta * vol_t**2)
            ret_t = np.random.normal(mu / 12, vol_t * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(ret_t))
        paths.append(prices)
    return np.array(paths)

# === RUN SIMULATIONS ===
print("Running lognormal simulation ...")
lognorm_paths = lognorm_sim(CAGR, VOL, SIMULATIONS)
final_lognorm = simulate_dividend_paths(lognorm_paths)

print("Running jump diffusion simulation ...")
jump_paths = jump_diffusion_sim(CAGR, VOL, JUMP_FREQ, JUMP_SIZE_MEAN, JUMP_SIZE_STD, SIMULATIONS)
final_jump = simulate_dividend_paths(jump_paths)

print("Running GARCH simulation ...")
garch_paths = garch_simulate(GARCH_PARAMS, CAGR, VOL, SIMULATIONS)
final_garch = simulate_dividend_paths(garch_paths)

outs = {
    "Lognormal": summary_stats(final_lognorm),
    "JumpDiffusion": summary_stats(final_jump),
    "GARCH": summary_stats(final_garch)
}

# PRINT EXCEL TABLE
df_out = pd.DataFrame(outs).T
df_out.reset_index(inplace=True)
df_out.rename(columns={"index": "Model"}, inplace=True)
cols = ["Model", "mean", "std", "min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "prob_deficit"]
df_out = df_out[cols]
print("\n=== Copy/Paste this table into Excel ===\n")
print("\t".join(df_out.columns))
for _, row in df_out.iterrows():
    print("\t".join([str(row[col]) for col in df_out.columns]))
print("\n=== End of table ===\n")

# --- BILLIONS/TRILLIONS PLOT ---
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def billions_trillions(x, pos):
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

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
percent_labels = [f"{p}%" for p in percentiles]

# --- Main OEP CHART --- #
fig1, ax1 = plt.subplots(figsize=(14, 8))

for arr, label, color in [
    (final_lognorm, "Lognormal", 'blue'),
    (final_jump, "Jump Diffusion", 'red'),
    (final_garch, "GARCH", 'green')
]:
    values = np.percentile(arr, percentiles)
    ax1.plot(percentiles, values, marker='o', label=label, color=color)

ax1.axhline(PREF_EQUITY_RAISE, color="black", linestyle="dashed", linewidth=1, label="Pref Equity")
ax1.set_xlabel("Percentile (%)", fontsize=13)
ax1.set_ylabel("Terminal Collateral (USD, Billions/Trillions)", fontsize=13)
ax1.set_title("OEP (Percentile) Curves: Terminal Collateral by Model", fontsize=17)
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(FuncFormatter(billions_trillions))
ax1.set_xticks(percentiles)
ax1.set_xticklabels(percent_labels)
ax1.legend(fontsize=13, loc='upper left')
ax1.grid(True, which="both", ls="--", lw=0.7)
plt.tight_layout(rect=[0, 0.21, 1, 1])  # Plenty of bottom space

# --- Nicely moved summary block, bottom of fig --- #
assumption_summary = (
    f"SCENARIO ASSUMPTIONS:\n"
    f"BTC Initial Holdings: {BTC_INITIAL:,} BTC    |    Initial BTC Price: ${BTC_PRICE_INITIAL:,.0f}\n"
    f"Additional BTC: {ADDITIONAL_BTC:,.2f} BTC (Pref raise)    |    Total BTC at Start: {BTC_TOTAL:,.2f} BTC\n"
    f"Notional Pref Equity Outstanding: ${PREF_EQUITY_RAISE:,.0f}    |    Div Rate: {DIVIDEND_RATE:.2%} per year\n"
    f"Time Horizon: {N_YEARS} years    |    Monthly Dividend: ${MONTHLY_DIVIDEND:,.2f}    |    BTC Fraction Sold for Divs: {BTC_DIV_FRACTION:.0%}\n"
)
prob_summary = (
    f"PROBABILITY COLLATERAL < OBLIGATIONS AT END:\n"
    f"Lognormal: {100*outs['Lognormal']['prob_deficit']:.2f}%    |    "
    f"Jump Diffusion: {100*outs['JumpDiffusion']['prob_deficit']:.2f}%    |    "
    f"GARCH: {100*outs['GARCH']['prob_deficit']:.2f}%"
)
full_summary = assumption_summary + prob_summary
plt.figtext(0.063, 0.00, full_summary,
    fontsize=12, ha="left", va="bottom",
    bbox=dict(facecolor='white', alpha=0.98, boxstyle="round,pad=1.0"))

plt.show(block=False)   # Immediately show OEP and keep running

# --- Interactive Failed Path Chart --- #
from matplotlib.widgets import Slider

failed_idxs = [i for i, val in enumerate(final_jump) if val < PREF_EQUITY_RAISE]
if not failed_idxs:
    print("No failed scenario found in Jump Diffusion simulation.")
else:
    # Set up first failed path
    start_idx = 0
    failed_prices = jump_paths[failed_idxs[start_idx]]
    btc_holdings = [BTC_TOTAL]
    for i in range(1, len(failed_prices)):
        div_payment_btc = BTC_DIV_FRACTION * MONTHLY_DIVIDEND / failed_prices[i]
        next_btc = btc_holdings[-1] - div_payment_btc
        btc_holdings.append(max(next_btc, 0))
    months = np.arange(len(failed_prices))

    fig2, ax21 = plt.subplots(figsize=(12,6))
    color1 = 'tab:blue'
    color2 = 'tab:red'
    l1, = ax21.plot(months, failed_prices, color=color1, label='BTC Price Path')
    l2 = ax21.axhline(BTC_PRICE_INITIAL, color='gray', linestyle='dotted', linewidth=1, label='Initial BTC Price')
    ax21.set_xlabel('Month')
    ax21.set_ylabel('Simulated BTC Price (USD)', color=color1)
    ax21.tick_params(axis='y', labelcolor=color1)
    ax22 = ax21.twinx()
    l3, = ax22.plot(months, btc_holdings, color=color2, label='BTC Holdings')
    ax22.set_ylabel('BTC Holdings (BTC)', color=color2)
    ax22.tick_params(axis='y', labelcolor=color2)
    plt.title("Jump Diffusion: Example Failed Paths (Use Slider Below Chart)")
    fig2.tight_layout(rect=[0, 0.05, 1, 1])

    ax_slider = plt.axes([0.14, 0.01, 0.72, 0.035])
    slider = Slider(ax_slider, 'Failed Path #', 1, len(failed_idxs), valinit=1, valstep=1)

    def update(val):
        idx = failed_idxs[int(slider.val)-1]
        prices = jump_paths[idx]
        holdings = [BTC_TOTAL]
        for i in range(1, len(prices)):
            div_btc = BTC_DIV_FRACTION * MONTHLY_DIVIDEND / prices[i]
            holdings.append(max(holdings[-1] - div_btc, 0))
        l1.set_ydata(prices)
        l3.set_ydata(holdings)
        fig2.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()