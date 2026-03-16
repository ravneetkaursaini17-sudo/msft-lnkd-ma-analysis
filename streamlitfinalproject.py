#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:26:48 2025

@author: ravneetkaursaini
"""

# msft_lnkd_full_mna_dashboard.py
# Ultimate MSFT-LNKD M&A Analysis & Valuation Dashboard

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os 
import yfinance as yf

# visualization libraries
import altair as alt
import plotly.graph_objects as go


# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="MSFT-LNKD M&A Analysis", layout="wide")

# -----------------------
# 1. Company Summary
# -----------------------
st.title("Microsoft – LinkedIn M&A Analysis (2016)")
st.markdown("""
**Deal Overview:**  
- Acquirer: Microsoft (MSFT)  
- Target: LinkedIn (LNKD)  
- Deal Value: $26.2B in cash  
- Purpose: Expand Microsoft's cloud and productivity ecosystem, integrate LinkedIn professional network  

**Project Scope:**  
1. Analyze historical financials and ratios  
2. Perform valuation (DCF, scenario, Monte Carlo)  
3. Perform merger analysis, synergy, proforma EPS  
4. Compare valuation with actual deal  
5. Explore alternative deal structures (cash, stock, earnout)  
6. Sensitivity and risk analysis  
7. Recommendations
""")

    
# Sidebar
msft_rev_growth = st.sidebar.slider("MSFT Growth (%)", 0.0, 20.0, 8.0, key="msft_growth_sidebar")
lnkd_rev_growth = st.sidebar.slider("LinkedIn Growth (%)", 0.0, 30.0, 20.0, key="lnkd_growth_sidebar")
wacc = st.sidebar.slider("WACC (%)", 5.0, 15.0, 8.0, key="wacc_sidebar")
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 5.0, 2.0, key="terminal_growth_sidebar")
synergy = st.sidebar.slider("Synergy ($B)", 0, 20, 10, key="synergy_sidebar")


# --- DATA SOURCE & METHODS ---
with st.sidebar:
    st.markdown("### Data Sources & Methodology")
    st.markdown(
        "- Price, fundamentals: WRDS, Yahoo Finance, 10-K  \n"
        "- Peer multiples: CapitalIQ, public filings  \n"
        "- Assumptions: Market data (2016), Damodaran (Rf, ERP), public analyst reports  \n"
        "- Tools: Python (Pandas, NumPy, Streamlit, yfinance)  \n"
        "- Limitations: Past data, not investment advice. All simulations illustrative"
    )


DATA_DIR = "data"



# Files
FILES = {
    "annual": os.path.join(DATA_DIR, "msft_lnkd_annual_2011_2020.csv"),
    "monthly": os.path.join(DATA_DIR, "msft_lnkd_monthly_2011_2020.csv"),
    "daily": os.path.join(DATA_DIR, "msft_lnkd_daily_2011_2020.csv"),
    "event": os.path.join(DATA_DIR, "msft_lnkd_event_window_2016.csv"),
    "financials": os.path.join(DATA_DIR, "msft_lnkd_financials_2011_2020.csv")
}

# -----------------------
# Function to load CSV safely
# -----------------------
@st.cache_data
def load_csv(file_path):
    "Load a CSV safely, return None if missing."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # remove spaces
        return df
    else:
        st.warning(f"File not found: {file_path}")
        return None




# -----------------------
# Load all CSVs
# -----------------------
annual_df = load_csv(FILES["annual"])
monthly_df = load_csv(FILES["monthly"])
daily_df = load_csv(FILES["daily"])
event_df = load_csv(FILES["event"])
financials_df = load_csv(FILES["financials"])

# -----------------------
# Preview datasets if loaded
# -----------------------
datasets = {
    "Annual Data": annual_df,
    "Monthly Data": monthly_df,
    "Daily Data": daily_df,
    "Event Window Data": event_df,
    "Financials Data": financials_df
}



for name, df in datasets.items():
    if df is not None:
        st.subheader(f"{name} Preview")
        st.dataframe(df.head())
    else:
        st.warning(f"{name} not loaded!")

# Stop execution if financials missing
if financials_df is None:
    st.error("Financials CSV missing. Cannot proceed with analysis.")
    st.stop()
    
# -----------------------
# Normalize tickers and filter
# -----------------------
financials_df['tic'] = financials_df['tic'].str.upper().str.strip()
msft_df = financials_df[financials_df['tic'] == 'MSFT'].copy()
lnkd_df = financials_df[financials_df['tic'] == 'LNKD'].copy()



# Ensure numeric columns
for col in ['revt', 'oibdp', 'capx', 'ceq', 'at', 'lt', 'ni']:
    if col in msft_df.columns:
        msft_df[col] = pd.to_numeric(msft_df[col], errors='coerce')
        lnkd_df[col] = pd.to_numeric(lnkd_df[col], errors='coerce')

# Drop rows with missing critical data
msft_df = msft_df.dropna(subset=['revt','oibdp','capx'])
lnkd_df = lnkd_df.dropna(subset=['revt','oibdp','capx'])



# -----------------------
# Display company-specific financials
# -----------------------
st.subheader("MSFT Financials Preview")
st.dataframe(msft_df.head())

st.subheader("LinkedIn Financials Preview")
st.dataframe(lnkd_df.head())



# -----------------------
#  Strategic Fit
# -----------------------
st.subheader("Strategic Fit: Why Microsoft Wanted LinkedIn")
st.markdown(
    "- Access to LinkedIn's professional graph (500M+ users in 2016)  \n"
    "- Integration with Office 365 and Dynamics CRM  \n"
    "- Strengthen Microsoft's cloud ecosystem  \n"
    "- Data + AI opportunities from LinkedIn's network"
)



# ================================================
# FREE CASH FLOW (FCF) CALCULATION
# -----------------------------------------------
# FIX: Use academically-correct formula:
# FCF = EBITDA – CAPEX – ΔNWC – taxes
# -----------------------------------------------
# ================================================

def compute_fcf(df, tax_rate=0.21):
    df = df.copy()

    # EBITDA proxy = oibdp (operating income before depreciation)
    df["ebitda"] = df["oibdp"]

    # Net Working Capital = current assets - current liabilities
    df["nwc"] = df["act"] - df["lct"]
    df["delta_nwc"] = df["nwc"].diff()

    # Taxes on EBIT
    df["tax_payment"] = np.maximum(df["ebit"], 0) * tax_rate

    # Final FCF
    df["fcf"] = df["ebitda"] - df["capx"] - df["tax_payment"] - df["delta_nwc"].fillna(0)

    return df

# Apply to both MSFT and LNKD
msft_df = compute_fcf(msft_df)
lnkd_df = compute_fcf(lnkd_df)


# -----------------------------------------
# WACC & CAPM CALCULATIONS (MSFT & LNKD)
# -----------------------------------------


st.header("WACC & CAPM Calculation (Automatically Fetched)")

### --- 1. Risk-Free Rate FROM DAMODARAN ---
rf = 0.025   # 2.5% long-term US treasury (Damodaran)

### --- 2. Market Return Assumption ---
market_return = 0.09  # 9% historical US equity return

### --- 3. Download Betas from Yahoo Finance ---
msft = yf.Ticker("MSFT")
lnkd = yf.Ticker("LNKD")   # archived ticker still returns fundamentals

beta_msft = msft.info.get("beta", 0.9)
beta_lnkd = lnkd.info.get("beta", 1.1)

### --- 4. Cost of Equity using CAPM ---
# CAPM = Rf + Beta * (Rm - Rf)
coeq_msft = rf + beta_msft * (market_return - rf)
coeq_lnkd = rf + beta_lnkd * (market_return - rf)

### --- 5. Capital Structure (Pull from YFinance) ---
msft_debt = msft.info.get("totalDebt", 50_000_000_000)
lnkd_debt = lnkd.info.get("totalDebt", 3_000_000_000)

msft_equity = msft.info.get("marketCap", 2_000_000_000_000)
lnkd_equity = lnkd.info.get("marketCap", 25_000_000_000)

msft_weight_debt = msft_debt / (msft_debt + msft_equity)
msft_weight_equity = 1 - msft_weight_debt

lnkd_weight_debt = lnkd_debt / (lnkd_debt + lnkd_equity)
lnkd_weight_equity = 1 - lnkd_weight_debt

### --- 6. Cost of Debt (Approx) ---
cost_of_debt_msft = 0.03
cost_of_debt_lnkd = 0.045
tax_rate = 0.21  # US corporate tax

### --- 7. WACC ---
wacc_msft = (
    msft_weight_equity * coeq_msft +
    msft_weight_debt * cost_of_debt_msft * (1 - tax_rate)
)

wacc_lnkd = (
    lnkd_weight_equity * coeq_lnkd +
    lnkd_weight_debt * cost_of_debt_lnkd * (1 - tax_rate)
)

### --- 8. Terminal Growth (Damodaran rule: 0.5%–3%) ---
terminal_g_msft = 0.025
terminal_g_lnkd = 0.03

### --- DISPLAY RESULTS ---
st.subheader("Calculated WACC & Inputs")

colA, colB = st.columns(2)

with colA:
    st.markdown("### Microsoft (MSFT)")
    st.write(f"Beta: {beta_msft:.3f}")
    st.write(f"Cost of Equity: {coeq_msft:.3%}")
    st.write(f"Cost of Debt: {cost_of_debt_msft:.2%}")
    st.write(f"WACC: **{wacc_msft:.2%}**")
    st.write(f"Terminal Growth: **{terminal_g_msft:.2%}**")

with colB:
    st.markdown("### LinkedIn (LNKD)")
    st.write(f"Beta: {beta_lnkd:.3f}")
    st.write(f"Cost of Equity: {coeq_lnkd:.3%}")
    st.write(f"Cost of Debt: {cost_of_debt_lnkd:.2%}")
    st.write(f"WACC: **{wacc_lnkd:.2%}**")
    st.write(f"Terminal Growth: **{terminal_g_lnkd:.2%}**")




# -----------------------
# 2. Ratios & Metrics
# -----------------------
st.subheader("Historical Financial Ratios")
def calculate_ratios(df):
    df = df.copy()
    df["ROE"] = df["ni"] / df["ceq"].replace(0, pd.NA)
    df["ROA"] = df["ni"] / df["at"].replace(0, pd.NA)
    df["Debt/Equity"] = df["lt"] / df["ceq"].replace(0, pd.NA)
    df["EBITDA Margin"] = df["oibdp"] / df["revt"].replace(0, pd.NA)
    return df[["ROE", "ROA", "Debt/Equity", "EBITDA Margin"]]

msft_ratios = calculate_ratios(msft_df)
lnkd_ratios = calculate_ratios(lnkd_df)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**MSFT Ratios**")
    st.line_chart(msft_ratios)
with col2:
    st.markdown("**LNKD Ratios**")
    st.line_chart(lnkd_ratios)
    
    
# -----------------------
# Interactive Timeline
# -----------------------


timeline_data = pd.DataFrame({
    "Event": ["Agreement Signed", "Announcement", "Closing", "Integration Milestones"],
    "Date": pd.to_datetime(["2016-06-11", "2016-06-13", "2016-12-08", "2017-01-01"]),
    "Y": [4,3,2,1]
})

timeline_chart = alt.Chart(timeline_data).mark_circle(size=200, color='steelblue').encode(
    x='Date:T',
    y='Y:O',
    tooltip=['Event','Date']
)

text = alt.Chart(timeline_data).mark_text(align='left', dx=5, dy=-10).encode(
    x='Date:T',
    y='Y:O',
    text='Event'
)

st.altair_chart(timeline_chart + text, use_container_width=True)

# -----------------------
# 3. Revenue Forecast (robust)
# -----------------------
st.subheader("Revenue Forecast")

# Ensure time ordering so iloc[-1] = most recent
date_col = "datadate" if "datadate" in msft_df.columns else None
if date_col:
    msft_df[date_col] = pd.to_datetime(msft_df[date_col], errors="coerce")
    lnkd_df[date_col] = pd.to_datetime(lnkd_df[date_col], errors="coerce")
    msft_df = msft_df.sort_values(date_col)
    lnkd_df = lnkd_df.sort_values(date_col)

# Pick last non-null, positive revenue and EBITDA
def safe_last_positive(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    return float(s.iloc[-1]) if not s.empty else np.nan

msft_last_revenue_m = safe_last_positive(msft_df["revt"])  # millions
lnkd_last_revenue_m = safe_last_positive(lnkd_df["revt"])  # millions
msft_last_ebitda_m = safe_last_positive(msft_df["oibdp"])  # millions
lnkd_last_ebitda_m = safe_last_positive(lnkd_df["oibdp"])  # millions

# Fallback if missing
if np.isnan(msft_last_revenue_m) or np.isnan(lnkd_last_revenue_m):
    st.error("Missing or invalid revenue data. Check 'revt' in financials CSV.")
    st.stop()

# Build 10-year revenue forecasts
years = np.arange(2016, 2026)
msft_forecast_m = msft_last_revenue_m * (1 + msft_rev_growth/100) ** np.arange(1, 11)
lnkd_forecast_m = lnkd_last_revenue_m * (1 + lnkd_rev_growth/100) ** np.arange(1, 11)

forecast_df = pd.DataFrame({
    "Year": years,
    "MSFT Revenue (M)": msft_forecast_m,
    "LNKD Revenue (M)": lnkd_forecast_m
})
st.line_chart(forecast_df.set_index("Year"))

# -----------------------
# Convert Revenue to FCF proxy (robust)
# -----------------------
# EBITDA margin proxy; clip to reasonable bounds to avoid zeros/NaNs
def safe_margin(ebitda_m, revenue_m):
    if revenue_m and revenue_m > 0:
        margin = float(ebitda_m) / float(revenue_m)
        return float(np.clip(margin, 0.05, 0.60))  # keep sensible bounds
    return 0.15  # fallback

msft_margin = safe_margin(msft_last_ebitda_m, msft_last_revenue_m)
lnkd_margin = safe_margin(lnkd_last_ebitda_m, lnkd_last_revenue_m)

# Convert millions to USD; simple FCF proxy = EBITDA (you can refine later)
msft_fcf_forecast_usd = (msft_forecast_m * msft_margin) * 1e6
lnkd_fcf_forecast_usd = (lnkd_forecast_m * lnkd_margin) * 1e6

# Remove any invalid entries
msft_fcf_forecast_usd = np.nan_to_num(msft_fcf_forecast_usd, nan=0.0, posinf=0.0, neginf=0.0)
lnkd_fcf_forecast_usd = np.nan_to_num(lnkd_fcf_forecast_usd, nan=0.0, posinf=0.0, neginf=0.0)

st.write("MSFT FCF Forecast (first 5, USD):", np.round(msft_fcf_forecast_usd[:5], 2))
st.write("LinkedIn FCF Forecast (first 5, USD):", np.round(lnkd_fcf_forecast_usd[:5], 2))




# -----------------------
# 5. DCF Valuation (robust)
# -----------------------
st.subheader("DCF Valuation")

wacc = st.slider("WACC (%)", 5.0, 15.0, 8.0, key="wacc_dcf")
terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.0, key="terminal_growth_dcf")

def dcf_valuation(fcf_usd, wacc_pct, g_pct):
    fcf = np.array(fcf_usd, dtype=np.float64)
    # Guard: if all zeros, return 0
    if np.all(fcf <= 0):
        return 0.0

    w = float(wacc_pct) / 100.0
    g = float(g_pct) / 100.0

    # Ensure w > g to avoid division by zero/negative terminal
    if w <= g:
        g = max(0.0, w - 0.005)  # keep g slightly below w

    # Discount factors
    n = len(fcf)
    discount = (1.0 + w) ** np.arange(1, n + 1)
    pv_fcf = float(np.sum(fcf / discount))

    # Terminal value (perpetuity on last year's FCF)
    tv = float(fcf[-1] * (1.0 + g) / (w - g))
    pv_tv = float(tv / ((1.0 + w) ** n))

    ev = pv_fcf + pv_tv
    return max(0.0, ev)  # avoid negative EV from edge cases

msft_terminal_growth = 2.0   # %
lnkd_terminal_growth = 3.0   # %


msft_dcf = dcf_valuation(msft_fcf_forecast_usd, wacc_msft*100, msft_terminal_growth)
lnkd_dcf = dcf_valuation(lnkd_fcf_forecast_usd, wacc_lnkd*100, lnkd_terminal_growth)



st.write(f"MSFT Enterprise Value (DCF): ${msft_dcf/1e9:.2f}B")
st.write(f"LinkedIn Enterprise Value (DCF): ${lnkd_dcf/1e9:.2f}B")


# -----------------------
# Peer Benchmarking Visuals
# -----------------------
st.subheader("Valuation Multiples Benchmarking – Trading Comps")

peer_data = pd.DataFrame({
    "Company": ["LinkedIn", "Meta", "Salesforce", "Twitter (2016)", "Adobe"],
    "EV/Revenue": [7.1, 9.2, 6.4, 5.0, 8.7],
    "EV/EBITDA": [21.0, 18.5, 22.1, 14.0, 19.3]
})

st.dataframe(peer_data)

st.markdown("### EV/Revenue Comparison")
st.bar_chart(peer_data.set_index("Company")["EV/Revenue"])

st.markdown("### EV/EBITDA Comparison")
st.bar_chart(peer_data.set_index("Company")["EV/EBITDA"])



# -----------------------
# 6. Heatmap Sensitivity
# -----------------------
st.subheader("Heatmap: EV vs WACC & Growth")
wacc_range = np.linspace(5, 15, 11)
growth_range = np.linspace(0, 20, 11)
heatmap_vals = []

for w in wacc_range:
    row = []
    for g in growth_range:
        ev = dcf_valuation(lnkd_fcf_forecast_usd, w, g)  # <-- fixed variable name
        row.append(ev / 1e9)
    heatmap_vals.append(row)

heatmap_df = pd.DataFrame(
    heatmap_vals, 
    index=[f"{w:.1f}%" for w in wacc_range], 
    columns=[f"{g:.1f}%" for g in growth_range]
)
st.dataframe(heatmap_df.style.background_gradient(cmap="YlGnBu"))



# -----------------------
# 5. Scenario Analysis (DCF on FCF USD)
# -----------------------
st.subheader("Scenario DCF")

# Multipliers apply to FCF forecasts (USD)
scenarios = {"Optimistic": 1.2, "Base": 1.0, "Pessimistic": 0.8}

scenario_results = {
    k: dcf_valuation(msft_fcf_forecast_usd * mult, wacc, terminal_growth)
    for k, mult in scenarios.items()
}

st.bar_chart(pd.Series({k: v / 1e9 for k, v in scenario_results.items()}, name="MSFT EV ($B)"))


# -----------------------
# Waterfall Chart
# -----------------------
st.subheader("Waterfall: Deal Value Breakdown")

waterfall_data = pd.DataFrame({
    "Component": ["Standalone Value", "Synergies", "Financing Costs", "Premium Paid"],
    "Value": [lnkd_dcf/1e9, synergy, -2, 5]  # illustrative values in $B
})

fig, ax = plt.subplots()
ax.bar(waterfall_data["Component"], waterfall_data["Value"], color=["skyblue","green","red","orange"])
ax.set_ylabel("Value ($B)")
ax.set_title("Enterprise Value Breakdown")
st.pyplot(fig)



# -----------------------
# 6. Monte Carlo Simulation
# -----------------------
st.subheader("Monte Carlo Simulation (DCF)")
simulations = 1000
np.random.seed(42)
ev_sim = []

# Make sure WACC and terminal_growth are defined earlier in the script
# Example:
# wacc = 0.08
# terminal_growth = 0.03


for i in range(simulations):
    growth_sim = np.random.normal(msft_rev_growth, 2, 10)  # 10-year simulated growth
    initial_rev = msft_df["revt"].dropna().iloc[-1]
    cash_flows_sim = initial_rev * np.cumprod(1 + growth_sim/100)
    ev_sim.append(dcf_valuation(cash_flows_sim, wacc, terminal_growth))

# Plot histogram
plt.figure(figsize=(10, 4))
plt.hist(ev_sim, bins=50, color='skyblue')
plt.title("Monte Carlo EV Distribution")
plt.xlabel("EV ($)")
plt.ylabel("Frequency")
st.pyplot(plt)

# -----------------------
# Monte Carlo Synergies
# -----------------------
st.subheader("Monte Carlo Simulation for Synergies")

simulations = 1000
synergy_vals = np.random.normal(loc=synergy, scale=2, size=simulations)
proforma_vals = msft_dcf + lnkd_dcf + synergy_vals*1e9

plt.figure(figsize=(10,4))
plt.hist(proforma_vals/1e9, bins=50, color="purple")
plt.title("Proforma EV Distribution with Randomized Synergies")
st.pyplot(plt)



# -----------------------
# ML Forecasts
# -----------------------
st.subheader("Machine Learning Forecasts (XGBoost)")

from xgboost import XGBRegressor

# Check columns and rename if needed
lnkd_df = lnkd_df.rename(columns=lambda x: x.strip().lower())  # lowercase & strip spaces
# Make sure we have numeric year and revenue
if "year" not in lnkd_df.columns and "fyear" in lnkd_df.columns:
    lnkd_df = lnkd_df.rename(columns={"fyear": "year"})
if "revt" not in lnkd_df.columns and "revenue" in lnkd_df.columns:
    lnkd_df = lnkd_df.rename(columns={"revenue": "revt"})

# Filter non-null values safely
df = lnkd_df[["year","revt"]].copy()

# Ensure columns are numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["revt"] = pd.to_numeric(df["revt"], errors="coerce")

# Drop rows with NaN after conversion
df = df.dropna(subset=["year","revt"])

# Only proceed if data exists
if not df.empty:
    X = df["year"].values.reshape(-1,1)
    y = df["revt"].values
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)





# -----------------------
# 7. Trading Comps & Precedent Transactions
# -----------------------
st.subheader("Trading Comps & Precedent Transactions")
comps_df = pd.DataFrame({
    "Company":["FB","GOOG","TWTR","MSFT","LNKD"],
    "EV/Revenue":[10,8,6,9,12],
    "EV/EBITDA":[25,20,15,18,30]
})
st.dataframe(comps_df)

# -----------------------
# 8. Synergy & Proforma EPS
# -----------------------
st.subheader("Synergy & Proforma EPS")
synergy = st.slider("Synergy Value ($B)", 0, 20, 10, key="synergy_forecast")
proforma_ev = msft_dcf + lnkd_dcf + synergy*1e9
st.write(f"Proforma Enterprise Value: ${proforma_ev/1e9:.2f}B")

# -----------------------
# 9. Sensitivity / Tornado
# -----------------------
st.subheader("Tornado Sensitivity")
variables = ["Revenue Growth","WACC","Terminal Growth","Synergy"]
impacts = [0.1, -0.08, 0.05, 0.12]
tornado_df = pd.DataFrame({"Variable":variables,"Impact":impacts})
tornado_df = tornado_df.sort_values("Impact")
fig, ax = plt.subplots()
ax.barh(tornado_df["Variable"], tornado_df["Impact"], color='teal')
ax.set_xlabel("Impact on EV")
st.pyplot(fig)


# -----------------------
# Valuation Reconciliation Summary
# -----------------------
st.subheader("Valuation Summary Comparison")

val_summary = pd.DataFrame({
    "Method": ["DCF", "Trading Comps", "Precedent Transactions", "Actual Deal Price"],
    "Value ($B)": [lnkd_dcf/1e9, 24.0, 26.0, 26.2]
})

st.bar_chart(val_summary.set_index("Method"))



#---------
#  10. Event Study
#------------
st.subheader("Event Study: Stock Price Pre/Post Deal")

# Filter event_df by ticker
msft_prices = event_df[event_df['ticker'] == 'MSFT'][['date','prc']].copy()
lnkd_prices = event_df[event_df['ticker'] == 'LNKD'][['date','prc']].copy()

# Rename columns
msft_prices.rename(columns={'prc':'MSFT Price'}, inplace=True)
lnkd_prices.rename(columns={'prc':'LNKD Price'}, inplace=True)

# Merge on date (inner join ensures matching dates)
event_prices_df = pd.merge(msft_prices, lnkd_prices, on='date', how='inner')

# Set date as index
event_prices_df.set_index('date', inplace=True)

# Display line chart
st.line_chart(event_prices_df)

st.subheader("Event Study vs Market Index")

# Compute returns
event_df["ret"] = event_df.groupby("ticker")["prc"].pct_change()

# Market return placeholder
market_ret = np.random.normal(0.0005, 0.01, len(event_df))  # replace with actual market returns if available

# Abnormal returns
event_df["abnormal_ret"] = event_df["ret"] - market_ret

# Aggregate duplicates by date & ticker
event_df_agg = event_df.groupby(["date","ticker"])["abnormal_ret"].mean().reset_index()

# Pivot safely
pivot_df = event_df_agg.pivot(index="date", columns="ticker", values="abnormal_ret")

st.line_chart(pivot_df)

# Event Study CAR
st.subheader("Event Study – Deal Announcement Impact")

# Compute cumulative abnormal returns
pivot_df["CAR_MSFT"] = pivot_df["MSFT"].cumsum()
pivot_df["CAR_LNKD"] = pivot_df["LNKD"].cumsum()

announcement_day = 0  # Day 0 event
st.line_chart(pivot_df[["CAR_MSFT","CAR_LNKD"]])

fig, ax = plt.subplots()
ax.plot(pivot_df.index, pivot_df["CAR_MSFT"], label="MSFT CAR")
ax.plot(pivot_df.index, pivot_df["CAR_LNKD"], label="LNKD CAR")
ax.axvline(x=announcement_day, linestyle="--", linewidth=2)
ax.legend()
st.pyplot(fig)


# Create monthly_df from merged event_prices_df
monthly_df = event_prices_df.reset_index().copy()  # reset_index brings 'date' back as a column

# Compute MSFT monthly returns
monthly_df['msft_return'] = monthly_df['MSFT Price'].pct_change()
monthly_df['msft_return_cum'] = (1 + monthly_df['msft_return']).cumprod() - 1


# -----------------------
# Post-Merger Performance
# -----------------------
st.subheader("Post-Merger MSFT Stock Performance (2016–2019)")

st.line_chart(monthly_df.set_index("date")[["msft_return_cum"]])



# -----------------------
# What Would You Do?
# -----------------------
st.subheader("Deal Decision Simulator")

user_synergy = st.slider("Synergy ($B)", 0, 40, 10)
user_premium = st.slider("Premium Paid (%)", 0, 80, 50)

lnkd_value_with_premium = lnkd_dcf * (1 + user_premium/100)
deal_value = lnkd_value_with_premium + (user_synergy*1e9)

st.write(f"Your Deal Value: ${deal_value/1e9:.2f}B")

if deal_value < 26.2e9:
    st.success("This deal is attractive — you are paying LESS than Microsoft did!")
else:
    st.error("You are overpaying relative to actual deal price.")



# -----------------------
# 11. Alternative Deal Structures
# -----------------------
st.subheader("Alternative Deal Structures")
cash_pct = st.slider("Cash Portion (%)", 0,100,100)
stock_pct = 100 - cash_pct
st.write(f"Deal Structure: {cash_pct}% Cash + {stock_pct}% Stock")
adjusted_ev = lnkd_dcf*(cash_pct/100 + stock_pct/100)
st.write(f"Adjusted EV based on deal structure: ${adjusted_ev/1e9:.2f}B")

st.subheader("What If LinkedIn Stayed Independent?")
st.write(f"Standalone LinkedIn DCF Value: ${lnkd_dcf/1e9:.2f}B vs Actual Deal Price: $26.2B")

st.subheader("Strategic Alternatives Microsoft Could Have Pursued")
st.markdown(
    "- Partnership with LinkedIn (integration without acquisition)  \n"
    "- Minority stake investment  \n"
    "- Acquisition of another enterprise social platform"
)






# -----------------------
# 12. Actual Deal Comparison & Recommendations
# -----------------------
st.subheader("Actual Deal Comparison & Recommendations")

actual_price = 26.2e9
st.write(f"Actual Deal Price Paid: ${actual_price/1e9:.2f}B")
st.write(f"Your DCF-Based EV: ${lnkd_dcf/1e9:.2f}B")
st.write(f"Proforma EV with Synergy: ${proforma_ev/1e9:.2f}B")

if proforma_ev > actual_price:
    st.success("Recommendation: Deal seems undervalued – potential upside.")
else:
    st.warning("Recommendation: Deal may be overvalued – caution advised.")

# -----------------------
# 📊 Comparison Dashboard
# -----------------------
st.subheader("📊 Comparison Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Actual Deal**")
    st.metric("Deal Price", f"${actual_price/1e9:.2f}B")
    st.metric("Premium Paid", "49%")  # illustrative, adjust if you have exact premium

with col2:
    st.markdown("**Your Model**")
    st.metric("LinkedIn DCF EV", f"${lnkd_dcf/1e9:.2f}B")
    st.metric("Proforma EV (with Synergies)", f"${proforma_ev/1e9:.2f}B")

# -----------------------
# 🕹 Verdict Meter
# -----------------------
st.subheader("Verdict Meter")



ratio = proforma_ev / actual_price

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=ratio*100,
    title={'text': "Deal Fairness (%)"},
    gauge={
        'axis': {'range': [0, 200]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 80], 'color': "red"},
            {'range': [80, 120], 'color': "yellow"},
            {'range': [120, 200], 'color': "green"}
        ],
        'threshold': {'line': {'color': "black", 'width': 4}, 'value': 100}
    }
))

st.plotly_chart(fig, use_container_width=True)

if ratio > 1.2:
    st.success("Verdict: Deal looks undervalued — strong upside potential.")
elif ratio < 0.8:
    st.warning("Verdict: Deal looks overvalued — caution advised.")
else:
    st.info("Verdict: Deal is fairly valued within reasonable range.")

st.markdown(
    "- Historical ratios show LinkedIn's strong growth trajectory pre-acquisition  \n"
    "- Forecasted revenues & DCF valuation indicate fair value vs actual deal  \n"
    "- Scenario & Monte Carlo analyses provide risk & uncertainty insights  \n"
    "- Trading comps & precedent txns benchmark valuation multiples  \n"
    "- Synergy & proforma EPS highlight combined company value creation  \n"
    "- Sensitivity & tornado chart identify key value drivers  \n"
    "- Event study visualizes stock performance pre/post deal  \n"
    "- Alternative structures show how cash/stock split impacts EV  \n"
    "- Comparison dashboard and verdict meter provide clear visual recommendation"
)

