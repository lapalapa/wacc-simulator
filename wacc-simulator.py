# ==============================================================================
# Strategic WACC Simulator
# Version: 1.4.3
# Last Updated: 2026-02-10
# 
# Changelog:
# v1.4.3 (2026-02-10)
# - [FIX] DataFrame Style Formatting Error: Fixed ValueError caused by applying 
#   float formatting "{:.2f}" to string columns (Ticker, Name, Period).
#   Now applies formatting only to numeric columns via dictionary.
#
# v1.4.2 (2026-02-10)
# - [UI] Beta Label Update: Labeled "Beta (Adjusted)" for Financial Firms.
#
# v1.4.1 (2026-02-10)
# - [MOD] Financial Firms Beta Logic: Direct usage of Peer Adjusted Beta.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
import time
import random
import re
import urllib3
import warnings
import logging

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(page_title="Strategic WACC Simulator", layout="wide")

# Version display
VERSION = "1.4.3"
BUILD_DATE = "2026-02-10"

# ==============================================================================
# [MODULE] Helper: Safe Fetcher with Retry
# ==============================================================================
def safe_yf_info(ticker_obj, max_retries=3):
    """안전한 Yahoo Finance 정보 조회 (재시도 로직 포함)"""
    for i in range(max_retries):
        try:
            info = ticker_obj.info
            if info and len(info) > 5:
                return info
        except Exception as e:
            if i == max_retries - 1:
                logger.warning(f"Failed to fetch info after {max_retries} attempts: {str(e)}")
        time.sleep(random.uniform(0.5, 1.5))
    return {}

# ==============================================================================
# [MODULE] Helper: Yahoo Finance Timeseries API for Financial TTM Data
# ==============================================================================
_financial_ttm_cache = {}

def fetch_financial_ttm_from_api(ticker):
    """Fetch TTM financial data from Yahoo Finance's fundamentals-timeseries API."""
    cache_key = ticker.upper()
    if cache_key in _financial_ttm_cache:
        return _financial_ttm_cache[cache_key]
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        keys = ["trailingCreditLossesProvision", "annualCreditLossesProvision", "trailingPretaxIncome", "annualPretaxIncome"]
        end_ts = int(time.time())
        start_ts = int((datetime.now() - timedelta(days=365*6)).timestamp())
        url = f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{cache_key}?symbol={cache_key}&type={','.join(keys)}&period1={start_ts}&period2={end_ts}"
        
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        results = data.get("timeseries", {}).get("result", [])
        
        if not results: return None
        
        api_data = {"provision_ttm": 0, "provision_annual": [], "pretax_ttm": 0, "pretax_annual": []}
        for item in results:
            type_name = item.get("meta", {}).get("type", [None])[0]
            values = item.get(type_name, [])
            if not values: continue
            if "trailing" in type_name.lower():
                val = values[-1].get("reportedValue", {}).get("raw", 0)
                if "provision" in type_name.lower(): api_data["provision_ttm"] = val
                else: api_data["pretax_ttm"] = val
            elif "annual" in type_name.lower():
                for v in values:
                    api_data["provision_annual" if "provision" in type_name.lower() else "pretax_annual"].append((v.get("asOfDate"), v.get("reportedValue", {}).get("raw", 0)))
        
        _financial_ttm_cache[cache_key] = api_data
        return api_data
    except: return None

# ==============================================================================
# [MODULE] Helper: Deep Search with Normalization
# ==============================================================================
def get_value_max_fuzzy_with_priority(df, col_idx, keyword_priority_list, exclusion_keywords=None):
    matches = []
    exclusions = [e.lower().replace(" ", "") for e in exclusion_keywords] if exclusion_keywords else []
    for idx in df.index:
        norm_idx = str(idx).lower().replace(" ", "").replace("-", "").replace("_", "")
        if any(ex in norm_idx for ex in exclusions): continue
        for kw, priority in keyword_priority_list:
            if kw.lower().replace(" ", "") in norm_idx:
                val = df.loc[idx].iloc[col_idx]
                if pd.notna(val) and val != 0: matches.append((priority, abs(val)))
                break
    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]
    return 0

def get_value_max_fuzzy(df, col_idx, search_keywords, exclusion_keywords=None):
    candidates = []
    exclusions = [e.lower().replace(" ", "") for e in exclusion_keywords] if exclusion_keywords else []
    for idx in df.index:
        norm_idx = str(idx).lower().replace(" ", "").replace("-", "").replace("_", "")
        if any(ex in norm_idx for ex in exclusions): continue
        for kw in search_keywords:
            if kw.lower().replace(" ", "") in norm_idx:
                val = df.loc[idx].iloc[col_idx]
                if pd.notna(val) and val != 0: candidates.append(abs(val))
                break
    return max(candidates) if candidates else 0

# ==============================================================================
# [MODULE] Data Fetcher: FRED, NYU, KPMG
# ==============================================================================
@st.cache_data(ttl=86400)
def fetch_all_fred_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    targets = [("GDP", "A191RP1A027NBEA"), ("RF", "DGS10"), ("AAA", "BAMLC0A1CAAA"), ("AA", "BAMLC0A2CAA"), ("A", "BAMLC0A3CA"), ("BBB", "BAMLC0A4CBBB"), ("BB", "BAMLH0A1HYBB"), ("B", "BAMLH0A2HYB"), ("CCC", "BAMLH0A3HYC")]
    results = {}
    for key, series_id in targets:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10, verify=False)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            results[key] = df.dropna()
        except: pass

    latest_gdp = results["GDP"]["VALUE"].iloc[-1] if "GDP" in results else 2.5
    latest_rf = results["RF"]["VALUE"].iloc[-1] if "RF" in results else 4.2
    
    df_gdp_disp = None
    if "GDP" in results:
        df_gdp_disp = results["GDP"].sort_values(by="DATE", ascending=False).head(10)
        df_gdp_disp.columns = ["Date", "GDP Growth %"]
        
    df_rf_trend = None
    if "RF" in results:
        cutoff = results["RF"]["DATE"].iloc[-1] - timedelta(days=365*5)
        df_rf_trend = results["RF"][results["RF"]["DATE"] >= cutoff].copy()
        df_rf_trend.columns = ["Date", "Rate"]

    oas_map = {"AAA": "AAA US Corporate", "AA": "AA US Corporate", "A": "Single-A US Corporate", "BBB": "BBB US Corporate", "BB": "BB US High Yield", "B": "Single-B US High Yield", "CCC": "CCC & Lower US High Yield"}
    fallback_map = {"AAA": 0.45, "AA": 0.55, "A": 0.75, "BBB": 1.05, "BB": 1.95, "B": 3.10, "CCC": 8.50}
    oas_rows = []
    
    for k, name in oas_map.items():
        val = fallback_map[k]
        date_str = "Fallback"
        sid = next((t[1] for t in targets if t[0] == k), "")
        link = f"https://fred.stlouisfed.org/series/{sid}"
        if k in results:
            val = float(results[k]["VALUE"].iloc[-1])
            date_str = results[k]["DATE"].iloc[-1].strftime('%Y-%m-%d')
        oas_rows.append({"OAS Name": name, "Latest Spread (%)": val, "Date": date_str, "Link": link})
    
    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, pd.DataFrame(oas_rows)

@st.cache_data(ttl=86400)
def get_sp_buyback_data():
    try:
        dfs = pd.read_html("https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html", header=0)
        return 2.0, 1.5, dfs[0].dropna(subset=[dfs[0].columns[0]]), []
    except: return 2.0, 1.5, None, ["Fetch Error"]

@st.cache_data(ttl=86400)
def get_kpmg_tax_rates():
    try:
        dfs = pd.read_html("https://kpmg.com/dk/en/services/tax/corporate-tax/corporate-tax-rates-table.html")
        df = dfs[0]
        df.rename(columns={df.columns[0]: "Country"}, inplace=True)
        col_name = df.columns[-1]
        df = df[["Country", col_name]].copy()
        df.columns = ["Country", "Rate"]
        df["Rate"] = pd.to_numeric(df["Rate"], errors='coerce')
        tax_dict = dict(zip(df["Country"].str.upper().str.strip(), df["Rate"]))
        tax_dict.update({"UNITED STATES": 25.57, "USA": 25.57, "KOREA": 26.40})
        return df, tax_dict, 2025
    except: return None, {"UNITED STATES": 25.57, "USA": 25.57, "KOREA": 26.40}, 2025

@st.cache_data(ttl=86400)
def get_damodaran_spreads():
    fallback_fin = pd.DataFrame([{"greater than": "3.0", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"}, {"greater than": "-100000", "≤ to": "0.04", "Rating": "D2/D", "Spread": "19.00%"}])
    return {"Financial Firms": (fallback_fin, "Source: Fallback"), "Large Firms": (fallback_fin, "Source: Fallback"), "Small/Risky Firms": (fallback_fin, "Source: Fallback")}

# ==============================================================================
# [MODULE] Financial Data Extraction Logic
# ==============================================================================
def get_financial_data_with_priority(ticker_obj, info_dict, ticker_symbol=None):
    sector = str(info_dict.get('sector', '')).lower()
    is_financial = 'financial' in sector or 'bank' in sector
    api_data = fetch_financial_ttm_from_api(ticker_symbol) if is_financial and ticker_symbol else None
    
    rev = info_dict.get('totalRevenue', 0)
    ebitda = info_dict.get('ebitda', 0)
    int_exp = info_dict.get('interestExpense', 0) or 0
    raw_pt, raw_pp = 0, 0
    
    if is_financial and api_data:
        raw_pt = api_data.get('pretax_ttm', 0)
        raw_pp = abs(api_data.get('provision_ttm', 0))
        return rev, (raw_pt + raw_pp), ebitda, int_exp, "TTM (Yahoo API)", "TTM (Yahoo Info)", raw_pt, raw_pp
    
    op_margin = info_dict.get('operatingMargins', 0)
    ebit = rev * op_margin if op_margin else ebitda
    return rev, ebit, ebitda, int_exp, "TTM (Yahoo Info)", "TTM (Yahoo Info)", 0, 0

# ==============================================================================
# [MODULE] DetailWACCModel
# ==============================================================================
class DetailWACCModel:
    def __init__(self, target, peers, rf_rate, crp, size_prem, buyback, div_yield, growth, tax, rf_trend_df, gdp_df):
        self.target = target
        self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.rf = rf_rate / 100
        self.mrp = (buyback + div_yield + growth) / 100 - self.rf
        self.crp = crp / 100
        self.size_prem = size_prem / 100
        self.tax = tax / 100
        self.rf_trend_df = rf_trend_df
        self.gdp_df = gdp_df
        _, self.kpmg_map, _ = get_kpmg_tax_rates()

    def get_exchange_rate_to_usd(self, currency): return 1.0, "USD"

    def get_financials_latest(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            if not info: return None, "No data"
            rev, ebit, ebitda, int_exp, label_ebit, label_int, pt, pp = get_financial_data_with_priority(t, info, ticker)
            tax_rate = self.kpmg_map.get(str(info.get('country')).upper().strip(), 25.0)
            
            return {
                "name": info.get('longName', ticker), "country": info.get('country', 'Unknown'), "tax_rate": tax_rate,
                "vals": {"Revenue": rev, "EBIT": ebit, "EBITDA": ebitda, "Total Debt": info.get('totalDebt', 0), "Market Cap": info.get('marketCap', 0)},
                "raw_beta": info.get('beta', 1.0), "period": label_ebit
            }, None
        except Exception as e: return None, str(e)

    def run(self, category_in, sens_method):
        peer_data = []
        for p in self.peers:
            fin, err = self.get_financials_latest(p)
            if fin:
                d = fin['vals']
                adj_beta = fin['raw_beta'] * 0.67 + 0.33
                peer_data.append({
                    "Ticker": p, "Tax Rate": fin['tax_rate'], "Adj Beta": adj_beta, 
                    "D/E Ratio": d['Total Debt'] / d['Market Cap'] if d['Market Cap'] > 0 else 0,
                    "Debt/TIC Ratio": d['Total Debt'] / (d['Total Debt'] + d['Market Cap']) if (d['Total Debt'] + d['Market Cap']) > 0 else 0,
                    "Total Debt": d['Total Debt'], "Market Cap": d['Market Cap'], "Revenue": d['Revenue'], "Company Name": fin['name'], "Period": fin['period'],
                    "Raw Beta": fin['raw_beta'], "Currency": "USD", "FX Rate": 1.0
                })
        
        df = pd.DataFrame(peer_data)
        if df.empty: return {"full_df": df, "errors": ["No valid peer data"]}

        is_financial = (category_in == "Financial Firms")
        
        if is_financial:
            if sens_method == "Average": target_relevered_beta = df["Adj Beta"].mean(); sel_dtic = df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median": target_relevered_beta = df["Adj Beta"].median(); sel_dtic = df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum": target_relevered_beta = df["Adj Beta"].max(); sel_dtic = df["Debt/TIC Ratio"].max()
            else: target_relevered_beta = df["Adj Beta"].min(); sel_dtic = df["Debt/TIC Ratio"].min()
            
            df["Unlevered Beta"] = df["Adj Beta"]
            df["Re-levered Beta"] = df["Adj Beta"]
        else:
            df["Unlevered Beta"] = df["Adj Beta"] / (1 + (1 - df["Tax Rate"]/100) * df["D/E Ratio"])
            if sens_method == "Average": sel_unlev = df["Unlevered Beta"].mean(); sel_dtic = df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median": sel_unlev = df["Unlevered Beta"].median(); sel_dtic = df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum": sel_unlev = df["Unlevered Beta"].max(); sel_dtic = df["Debt/TIC Ratio"].max()
            else: sel_unlev = df["Unlevered Beta"].min(); sel_dtic = df["Debt/TIC Ratio"].min()
            
            target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
            target_relevered_beta = sel_unlev * (1 + (1 - self.tax) * target_de)
            df["Re-levered Beta"] = df["Unlevered Beta"] * (1 + (1 - self.tax) * target_de)

        ke = self.rf + (target_relevered_beta * self.mrp) + self.crp + self.size_prem
        return {"full_df": df, "target_beta": target_relevered_beta, "ke": ke, "wd": sel_dtic, "mrp": self.mrp}

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas = fetch_all_fred_data()

with st.sidebar:
    st.title(f"WACC Simulator v{VERSION}")
    target_ticker = st.text_input("Target Ticker", "JPM")
    peers_input = st.text_area("Peer Tickers", "GS, MS, BAC, C, WFC")
    category_in = st.selectbox("Firm Category", ["Financial Firms", "Large Firms", "Small/Risky Firms"])
    
    st.divider()
    rf_in = st.number_input("Risk Free Rate (%)", value=latest_rf)
    tax_in = st.number_input("Target Tax Rate (%)", value=21.0)
    growth_in = st.number_input("Terminal Growth (%)", value=latest_gdp)
    
    st.session_state['sens_method'] = st.radio("Aggregation Method", ["Median", "Average", "Maximum", "Minimum"], horizontal=True, index=0)
    calc_btn = st.button("Calculate WACC", type="primary", use_container_width=True)

if calc_btn:
    model = DetailWACCModel(target_ticker, peers_input, rf_in, 0, 0, 2.0, 1.5, growth_in, tax_in, df_rf_trend, df_gdp_disp)
    res = model.run(category_in, st.session_state['sens_method'])
    
    if "full_df" in res and not res["full_df"].empty:
        is_fin_calc = (category_in == "Financial Firms")
        beta_label = "Beta (Adjusted)" if is_fin_calc else "Beta (Re-levered)"

        st.header(f"WACC Analysis: {target_ticker}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cost of Equity (Ke)", f"{res['ke']:.2%}")
        c2.metric(beta_label, f"{res['target_beta']:.2f}")
        c3.metric("Debt Weight (Wd)", f"{res['wd']:.1%}")
        c4.metric("Market Risk Prem", f"{res['mrp']:.2%}")

        if is_fin_calc:
            st.success("ℹ️ **Financial Firms Logic Applied**: Bypassed unlevering/re-levering. Used Peer Adjusted Beta directly.")

        # [FIX v1.4.3] Correct dataframe formatting to avoid ValueError on string columns
        st.subheader("Peer Beta & Structure Analysis")
        st.dataframe(
            res['full_df'][["Ticker", "Company Name", "Adj Beta", "Unlevered Beta", "Re-levered Beta", "D/E Ratio", "Period"]].style.format({
                "Adj Beta": "{:.2f}",
                "Unlevered Beta": "{:.2f}",
                "Re-levered Beta": "{:.2f}",
                "D/E Ratio": "{:.2f}"
            }), 
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Cost of Equity")
        st.latex(r"K_e = R_f + \beta \times (R_m - R_f) + CRP + SP")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Risk Free Rate", f"{rf_in:.2f}%")
        k2.metric(beta_label, f"{res['target_beta']:.2f}")
        k3.metric("Market Risk Prem", f"{res['mrp']*100:.2f}%")
        k4.metric("Country Risk Prem", "0.00%")
        k5.metric("Size Premium", "0.00%")
        
        st.info(f"Calculation: {rf_in:.2f}% + {res['target_beta']:.2f} * {res['mrp']*100:.2f}% = {res['ke']*100:.2f}%")
