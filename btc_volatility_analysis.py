import pandas as pd
import numpy as np
from arch import arch_model
import json

file_path = r"C:\Users\jaxso\Documents\BTC RISK MODEL\BTCPRICEHISTORY.xlsx"

# Load the whole file, auto-find header with "Date"
raw = pd.read_excel(file_path, header=None)
header_row = None

# Find the header row containing "Date"
for i in range(10):
    if raw.iloc[i].astype(str).str.contains("Date").any():
        header_row = i
        break
assert header_row is not None, "Could not find a header row with 'Date'!"

df = pd.read_excel(file_path, header=header_row)
df.columns = [str(c).strip() for c in df.columns]

# Make sure to only take valid rows, drop extra unnamed columns
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
df = df[["Date", price_col]].dropna()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df = df.dropna(subset=[price_col])

df = df.sort_values("Date").reset_index(drop=True)
df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
df = df.dropna(subset=["log_return"])

# GARCH model
garch_mod = arch_model(df["log_return"] * 100, vol="Garch", p=1, q=1)
garch_fit = garch_mod.fit(disp="off")
garch_params = garch_fit.params.to_dict()

# Jump Diffusion
mean_ret = df["log_return"].mean()
std_ret = df["log_return"].std()
jump_threshold = 3 * std_ret
jump_days = df[np.abs(df["log_return"] - mean_ret) > jump_threshold]
jump_freq_per_year = (len(jump_days) / len(df)) * 365
jump_mean = jump_days["log_return"].mean() if len(jump_days) > 0 else 0.0
jump_std = jump_days["log_return"].std() if len(jump_days) > 0 else 0.0

annual_cagr = np.exp(mean_ret * 365) - 1
annual_vol = std_ret * np.sqrt(365)

output = {
    "GARCH_parameters": garch_params,
    "JumpDiffusion_parameters": {
        "jump_frequency_per_year": jump_freq_per_year,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        "threshold": float(jump_threshold)
    },
    "Lognormal_parameters": {
        "annual_cagr": annual_cagr,
        "annual_vol": annual_vol
    },
    "Sample": int(len(df)),
    "Period": {
        "Start": str(df["Date"].iloc[0].date()),
        "End": str(df["Date"].iloc[-1].date())
    }
}

output_path = r"C:\Users\jaxso\Documents\BTC RISK MODEL\btc_model_parameters.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Parameters saved to {output_path}")
print(json.dumps(output, indent=2))

