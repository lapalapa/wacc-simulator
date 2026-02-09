# ==============================================================================
# Strategic WACC Simulator
# Version: 1.4.1
# Last Updated: 2025-02-10
# 
# Changelog:
# v1.4.1 (2025-02-10)
# - [MOD] Financial Firms Beta Logic: If the category is "Financial Firms", 
#   the simulator uses the Adjusted Beta from peers directly (Mean/Median/etc.) 
#   without the Unlevering/Re-levering process, as debt in financial firms 
#   is operational rather than just capital structure.
#
# v1.4.0 (2025-02-09)
# - [NEW] fetch_financial_ttm_from_api(): single API call fetches both
#   CreditLossesProvision AND PretaxIncome TTM from Yahoo Timeseries API
# - [REFACTOR] Priority logic now strictly follows:
#   P1: Annual (Year-1) from yfinance income_stmt + API annual provision fallback
#   P2: TTM from info_dict (rev/ebitda/int_exp) + API TTM (pretax/provision)
#   P3: Sum of 4 most recent quarters from yfinance + API TTM fallback
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

# Version display in sidebar
VERSION = "1.4.1"
BUILD_DATE = "2025-02-10"

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
    """
    Fetch TTM financial data from Yahoo Finance's fundamentals-timeseries API.
    """
    cache_key = ticker.upper()
    if cache_key in _financial_ttm_cache:
        return _financial_ttm_cache[cache_key]
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        keys = [
            "trailingCreditLossesProvision",
            "annualCreditLossesProvision",
            "trailingPretaxIncome",
            "annualPretaxIncome",
        ]
        
        end_ts = int(time.time())
        start_ts = int((datetime.now() - timedelta(days=365*6)).timestamp())
        
        url = (
            f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{cache_key}"
            f"?symbol={cache_key}"
            f"&type={','.join(keys)}"
            f"&period1={start_ts}&period2={end_ts}"
        )
        
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        results = data.get("timeseries", {}).get("result", [])
        
        if not results:
            _financial_ttm_cache[cache_key] = None
            return None
        
        api_data = {
            "provision_ttm": 0,
            "provision_annual": [],
            "pretax_ttm": 0,
            "pretax_annual": [],
            "source": f"Yahoo Timeseries API ({cache_key})"
        }
        
        for item in results:
            meta = item.get("meta", {})
            type_list = meta.get("type", [])
            type_name = type_list[0] if isinstance(type_list, list) and type_list else str(type_list)
            values = item.get(type_name, [])
            if not values: continue
            
            type_lower = type_name.lower()
            if "trailing" in type_lower:
                for v in reversed(values):
                    if isinstance(v, dict) and "reportedValue" in v:
                        raw_val = v["reportedValue"].get("raw", 0)
                        if raw_val != 0:
                            if "creditloss" in type_lower or "provision" in type_lower:
                                api_data["provision_ttm"] = raw_val
                            elif "pretax" in type_lower:
                                api_data["pretax_ttm"] = raw_val
                            break
            elif "annual" in type_lower:
                for v in values:
                    if isinstance(v, dict) and "reportedValue" in v:
                        date_str = v.get("asOfDate", "Unknown")
                        raw_val = v["reportedValue"].get("raw", 0)
                        if "creditloss" in type_lower or "provision" in type_lower:
                            api_data["provision_annual"].append((date_str, raw_val))
                        elif "pretax" in type_lower:
                            api_data["pretax_annual"].append((date_str, raw_val))
        
        _financial_ttm_cache[cache_key] = api_data
        return api_data
    except Exception as e:
        logger.error(f"[Financial API] Error for {cache_key}: {str(e)}")
    return None

# ==============================================================================
# [MODULE] Helper: Deep Search with Normalization
# ==============================================================================
def get_value_max_fuzzy_with_priority(df, col_idx, keyword_priority_list, exclusion_keywords=None):
    matches = []
    try:
        exclusions = [e.lower().replace(" ", "").replace("-", "").replace("_", "") for e in exclusion_keywords] if exclusion_keywords else []
        for idx in df.index:
            norm_idx_str = str(idx).lower().replace(" ", "").replace("-", "").replace("_", "")
            if any(ex in norm_idx_str for ex in exclusions): continue
            for kw, priority in keyword_priority_list:
                norm_kw = kw.lower().replace(" ", "").replace("-", "").replace("_", "")
                if norm_kw in norm_idx_str:
                    try:
                        val = df.loc[idx].iloc[col_idx]
                        if pd.notna(val) and val != 0: matches.append((priority, abs(val), str(idx)))
                    except: pass
                    break
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]
    except: pass
    return 0

def get_value_max_fuzzy(df, col_idx, search_keywords, exclusion_keywords=None):
    candidates = []
    try:
        exclusions = [e.lower().replace(" ", "") for e in exclusion_keywords] if exclusion_keywords else []
        for idx in df.index:
            norm_idx_str = str(idx).lower().replace(" ", "").replace("-", "").replace("_", "")
            if any(ex in norm_idx_str for ex in exclusions): continue
            for kw in search_keywords:
                norm_kw = kw.lower().replace(" ", "").replace("-", "").replace("_", "")
                if norm_kw in norm_idx_str:
                    try:
                        val = df.loc[idx].iloc[col_idx]
                        if pd.notna(val) and val != 0: candidates.append(abs(val))
                    except: pass
                    break
        if candidates: return max(candidates)
    except: pass
    return 0

# ==============================================================================
# [MODULE] Data Fetcher: FRED, NYU, KPMG, Spreads (Maintained v1.4.0)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_all_fred_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    targets = [("GDP", "A191RP1A027NBEA", False), ("RF", "DGS10", False), ("AAA", "BAMLC0A1CAAA", True), ("AA", "BAMLC0A2CAA", True), ("A", "BAMLC0A3CA", True), ("BBB", "BAMLC0A4CBBB", True), ("BB", "BAMLH0A1HYBB", True), ("B", "BAMLH0A2HYB", True), ("CCC", "BAMLH0A3HYC", True)]
    results = {}
    for key, series_id, is_oas in targets:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10, verify=False)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            results[key] = df.dropna()
        except: pass
    latest_gdp = results["GDP"]["VALUE"].iloc[-1] if "GDP" in results else 2.5
    latest_rf = results["RF"]["VALUE"].iloc[-1] if "RF" in results else 4.2
    oas_rows = []
    oas_map = {"AAA": "AAA US Corporate", "AA": "AA US Corporate", "A": "Single-A US Corporate", "BBB": "BBB US Corporate", "BB": "BB US High Yield", "B": "Single-B US High Yield", "CCC": "CCC & Lower US High Yield"}
    for k, name in oas_map.items():
        val = results[k]["VALUE"].iloc[-1] if k in results else 1.0
        oas_rows.append({"OAS Name": name, "Latest Spread (%)": val, "Date": results[k]["DATE"].iloc[-1].strftime('%Y-%m-%d') if k in results else "N/A"})
    return latest_gdp, results.get("GDP"), latest_rf, results.get("RF"), pd.DataFrame(oas_rows)

@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    return 2.0, 1.5, None, []

@st.cache_data(ttl=3600*24)
def get_kpmg_tax_rates():
    return None, {"UNITED STATES": 25.57, "USA": 25.57, "KOREA": 26.40}, 2025

@st.cache_data(ttl=3600*24)
def get_damodaran_spreads():
    # Fallback tables defined in v1.4.0
    fallback_fin = pd.DataFrame([{"greater than": "3.0", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"}, {"greater than": "-100000", "≤ to": "0.04", "Rating": "D2/D", "Spread": "19.00%"}])
    return {"Financial Firms": (fallback_fin, "Source: Fallback"), "Large Firms": (fallback_fin, "Source: Fallback"), "Small/Risky Firms": (fallback_fin, "Source: Fallback")}

# ==============================================================================
# [MODULE] Financial Data Extraction Logic (Maintained v1.4.0)
# ==============================================================================
def get_financial_data_with_priority(ticker_obj, info_dict, ticker_symbol=None):
    sector = str(info_dict.get('sector', '')).lower()
    is_financial = 'financial' in sector or 'bank' in sector
    api_data = fetch_financial_ttm_from_api(ticker_symbol) if is_financial and ticker_symbol else None
    
    rev = info_dict.get('totalRevenue', 0)
    ebitda = info_dict.get('ebitda', 0)
    int_exp = info_dict.get('interestExpense', 0) or info_dict.get('totalInterestExpense', 0) or 0
    
    if is_financial and api_data:
        raw_pt = api_data.get('pretax_ttm', 0)
        raw_pp = abs(api_data.get('provision_ttm', 0))
        return rev, (raw_pt + raw_pp), ebitda, int_exp, "TTM (Yahoo API)", "TTM (Yahoo Info)", raw_pt, raw_pp
    
    op_margin = info_dict.get('operatingMargins', 0)
    ebit = rev * op_margin if op_margin else ebitda
    return rev, ebit, ebitda, int_exp, "TTM (Yahoo Info)", "TTM (Yahoo Info)", 0, 0

# ==============================================================================
# [MODULE] DetailWACCModel (Updated v1.4.1: Financial Beta Logic)
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

    def get_financials_latest(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            if not info: return None, f"⚠️ {ticker}: No data"
            rev, ebit, ebitda, int_exp, label_ebit, label_int, pt, pp = get_financial_data_with_priority(t, info, ticker)
            
            return {
                "name": info.get('longName', ticker),
                "country": info.get('country', 'Unknown'),
                "tax_rate": self.kpmg_map.get(str(info.get('country')).upper(), 25.0),
                "vals": {"Revenue": rev, "EBIT": ebit, "EBITDA": ebitda, "Total Debt": info.get('totalDebt', 0), "Market Cap": info.get('marketCap', 0)},
                "raw_beta": info.get('beta', 1.0),
                "period": label_ebit
            }, None
        except Exception as e: return None, str(e)

    def run(self, category_in):
        peer_data = []
        for p in self.peers:
            fin, err = self.get_financials_latest(p)
            if fin:
                d = fin['vals']
                adj_beta = fin['raw_beta'] * 0.67 + 0.33
                peer_data.append({
                    "Ticker": p, "Company Name": fin['name'], "Tax Rate": fin['tax_rate'],
                    "D/E Ratio": d['Total Debt'] / d['Market Cap'] if d['Market Cap'] > 0 else 0,
                    "Debt/TIC Ratio": d['Total Debt'] / (d['Total Debt'] + d['Market Cap']) if (d['Total Debt'] + d['Market Cap']) > 0 else 0,
                    "Adj Beta": adj_beta, "Market Cap": d['Market Cap'], "Total Debt": d['Total Debt'],
                    "Revenue": d['Revenue'], "EBIT": d['EBIT'], "Period": fin['period']
                })
        
        df = pd.DataFrame(peer_data)
        if df.empty: return {"full_df": df, "errors": ["No valid peer data"]}

        # [v1.4.1] Logic Change for Financial Firms
        is_financial = (category_in == "Financial Firms")
        sens_method = st.session_state.get('sens_method', "Median")

        if is_financial:
            # 금융사의 경우 Unlevering/Re-levering 없이 Peer의 Adjusted Beta 통계값을 직접 사용
            if sens_method == "Average": target_relevered_beta = df["Adj Beta"].mean()
            elif sens_method == "Median": target_relevered_beta = df["Adj Beta"].median()
            elif sens_method == "Maximum": target_relevered_beta = df["Adj Beta"].max()
            else: target_relevered_beta = df["Adj Beta"].min()
            
            df["Unlevered Beta"] = df["Adj Beta"] # UI Display consistency
            df["Re-levered Beta"] = df["Adj Beta"]
            sel_dtic = df["Debt/TIC Ratio"].median()
        else:
            # 일반 기업은 기존 Unlevering -> Re-levering 로직 수행
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
# [UI] Execution Logic
# ==============================================================================
latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas = fetch_all_fred_data()

with st.sidebar:
    st.title(f"Strategic WACC Simulator v{VERSION}")
    target_ticker = st.text_input("Target Ticker", "JPM")
    peers_input = st.text_area("Peer Tickers", "GS, MS, BAC, C, WFC")
    category_in = st.selectbox("Firm Category", ["Financial Firms", "Large Firms", "Small/Risky Firms"])
    
    st.divider()
    rf_in = st.number_input("Risk Free Rate (%)", value=latest_rf)
    tax_in = st.number_input("Target Tax Rate (%)", value=21.0)
    growth_in = st.number_input("Terminal Growth (%)", value=latest_gdp)
    
    st.session_state['sens_method'] = st.radio("Aggregation Method", ["Median", "Average", "Maximum", "Minimum"], horizontal=True)
    calc_btn = st.button("Calculate WACC", type="primary", use_container_width=True)

if calc_btn:
    model = DetailWACCModel(target_ticker, peers_input, rf_in, 0, 0, 2.0, 1.5, growth_in, tax_in, df_rf_trend, df_gdp_disp)
    res = model.run(category_in)
    
    if "full_df" in res and not res["full_df"].empty:
        st.header(f"WACC Analysis: {target_ticker}")
        
        # Results Metric
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cost of Equity (Ke)", f"{res['ke']:.2%}")
        c2.metric("Applied Beta", f"{res['target_beta']:.2f}")
        c3.metric("Debt Weight (Wd)", f"{res['wd']:.1%}")
        c4.metric("Market Risk Prem", f"{res['mrp']:.2%}")

        if category_in == "Financial Firms":
            st.success("ℹ️ **Financial Firms Beta Logic Applied**: Bypassed unlevering/re-levering. Used Peer Adjusted Beta directly.")

        st.subheader("Peer Beta & Structure Analysis")
        st.dataframe(res['full_df'][["Ticker", "Company Name", "Adj Beta", "Unlevered Beta", "Re-levered Beta", "D/E Ratio", "Period"]].style.format("{:.2f}"))
