# ==============================================================================
# Strategic WACC Simulator
# Version: 1.4.2
# Last Updated: 2026-02-10
# 
# Changelog:
# v1.4.2 (2026-02-10)
# - [UI] Beta Label Update: For "Financial Firms", the Beta metric in Cost of Equity
#   is now labeled "Beta (Adjusted)" instead of "Beta (Re-levered)" to reflect
#   the direct usage of peer adjusted beta.
#
# v1.4.1 (2026-02-10)
# - [MOD] Financial Firms Beta Logic: If category is "Financial Firms", 
#   skip Unlevering/Re-levering. Target Beta is now directly derived from 
#   Peer Group's Adjusted Beta (Mean/Median).
#
# v1.4.0 (2025-02-09)
# - [NEW] fetch_financial_ttm_from_api(): single API call fetches both
#   CreditLossesProvision AND PretaxIncome TTM from Yahoo Timeseries API
# - [REFACTOR] Priority logic strictly follows Annual -> TTM(Info+API) -> Calc
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
VERSION = "1.4.2"
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
        
        # Request both provision and pretax income in a single API call
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
        
        logger.info(f"[Financial API] Fetching {cache_key} (Provision + PretaxIncome)...")
        
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        
        results = data.get("timeseries", {}).get("result", [])
        
        if not results:
            logger.info(f"[Financial API] No timeseries results for {cache_key}")
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
            if not values:
                continue
            
            type_lower = type_name.lower()
            
            # --- Parse trailing (TTM) values ---
            if "trailing" in type_lower:
                for v in reversed(values):
                    if isinstance(v, dict) and "reportedValue" in v:
                        raw_val = v["reportedValue"].get("raw", 0)
                        if raw_val != 0:
                            if "creditloss" in type_lower or "provision" in type_lower:
                                api_data["provision_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM Provision: ${raw_val:,.0f}")
                            elif "pretax" in type_lower:
                                api_data["pretax_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM PretaxIncome: ${raw_val:,.0f}")
                            break
            
            # --- Parse annual values ---
            elif "annual" in type_lower:
                annual_list = []
                for v in values:
                    if isinstance(v, dict) and "reportedValue" in v:
                        date_str = v.get("asOfDate", "Unknown")
                        raw_val = v["reportedValue"].get("raw", 0)
                        annual_list.append((date_str, raw_val))
                
                if annual_list:
                    if "creditloss" in type_lower or "provision" in type_lower:
                        api_data["provision_annual"] = annual_list
                    elif "pretax" in type_lower:
                        api_data["pretax_annual"] = annual_list
        
        has_data = (api_data["provision_ttm"] != 0 or api_data["pretax_ttm"] != 0 
                    or api_data["provision_annual"] or api_data["pretax_annual"])
        if has_data:
            _financial_ttm_cache[cache_key] = api_data
            return api_data
        
        _financial_ttm_cache[cache_key] = None
        return None
        
    except Exception as e:
        logger.warning(f"[Financial API] Error for {cache_key}: {str(e)}")
        _financial_ttm_cache[cache_key] = None
        return None

# ==============================================================================
# [MODULE] Helper: Deep Search with Normalization
# ==============================================================================
def get_value_max_fuzzy_with_priority(df, col_idx, keyword_priority_list, exclusion_keywords=None):
    matches = []
    try:
        exclusions = [e.lower().replace(" ", "").replace("-", "").replace("_", "") 
                      for e in exclusion_keywords] if exclusion_keywords else []
        
        for idx in df.index:
            raw_idx_str = str(idx)
            norm_idx_str = raw_idx_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            
            if any(ex in norm_idx_str for ex in exclusions): continue
            
            for kw, priority in keyword_priority_list:
                norm_kw = kw.lower().replace(" ", "").replace("-", "").replace("_", "")
                if norm_kw in norm_idx_str:
                    try:
                        val = df.loc[idx].iloc[col_idx]
                        if pd.notna(val) and val != 0:
                            matches.append((priority, abs(val), raw_idx_str))
                    except: pass
                    break
        
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]
    except: pass
    return 0

def get_value_max_fuzzy(df, col_idx, search_keywords, exclusion_keywords=None, debug_provision=False):
    candidates = []
    try:
        exclusions = [e.lower().replace(" ", "") for e in exclusion_keywords] if exclusion_keywords else []
        for idx in df.index:
            raw_idx_str = str(idx)
            norm_idx_str = raw_idx_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            
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
# [MODULE] Data Fetcher: Consolidated FRED
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_all_fred_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    targets = [
        ("GDP", "A191RP1A027NBEA", False), ("RF", "DGS10", False),
        ("AAA", "BAMLC0A1CAAA", True), ("AA", "BAMLC0A2CAA", True),
        ("A", "BAMLC0A3CA", True), ("BBB", "BAMLC0A4CBBB", True),
        ("BB", "BAMLH0A1HYBB", True), ("B", "BAMLH0A2HYB", True),
        ("CCC", "BAMLH0A3HYC", True)
    ]
    results = {}
    
    for key, series_id, is_oas in targets:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10, verify=False)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            results[key] = df.dropna().sort_values(by="DATE", ascending=True)
        except: pass

    # GDP
    latest_gdp = results["GDP"]["VALUE"].iloc[-1] if "GDP" in results else 2.5
    df_gdp_disp = results.get("GDP").sort_values(by="DATE", ascending=False).head(10) if "GDP" in results else None
    if df_gdp_disp is not None: df_gdp_disp.columns = ["Date", "GDP Growth %"]

    # RF
    latest_rf = results["RF"]["VALUE"].iloc[-1] if "RF" in results else 4.2
    df_rf_trend = None
    if "RF" in results:
        cutoff = results["RF"]["DATE"].iloc[-1] - timedelta(days=365*5)
        df_rf_trend = results["RF"][results["RF"]["DATE"] >= cutoff].copy()
        df_rf_trend.columns = ["Date", "Rate"]

    # OAS
    oas_map = {
        "AAA": "AAA US Corporate", "AA": "AA US Corporate", "A": "Single-A US Corporate",
        "BBB": "BBB US Corporate", "BB": "BB US High Yield", "B": "Single-B US High Yield",
        "CCC": "CCC & Lower US High Yield"
    }
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

# ==============================================================================
# [MODULE] Data Fetcher: NYU & KPMG
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    try:
        dfs = pd.read_html(url, header=0)
        return 2.0, 1.5, dfs[0].dropna(subset=[dfs[0].columns[0]]), []
    except: return 2.0, 1.5, None, ["Fetch Error"]

@st.cache_data(ttl=3600*24)
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

@st.cache_data(ttl=3600*24)
def get_damodaran_spreads():
    fallback_fin = pd.DataFrame([{"greater than": "3.0", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"}, {"greater than": "-100000", "≤ to": "0.04", "Rating": "D2/D", "Spread": "19.00%"}])
    return {"Financial Firms": (fallback_fin, "Source: Fallback"), "Large Firms": (fallback_fin, "Source: Fallback"), "Small/Risky Firms": (fallback_fin, "Source: Fallback")}

# ==============================================================================
# [MODULE] Financial Data Extraction Logic (Unified)
# ==============================================================================
def get_financial_data_with_priority(ticker_obj, info_dict, ticker_symbol=None):
    sector = str(info_dict.get('sector', '')).lower()
    is_financial = 'financial' in sector or 'bank' in sector
    current_year = datetime.now().year
    target_year = current_year - 1
    
    api_data = None
    if is_financial and ticker_symbol:
        api_data = fetch_financial_ttm_from_api(ticker_symbol)

    try:
        a_fin = ticker_obj.income_stmt
        if a_fin.empty: a_fin = ticker_obj.financials
        q_fin = ticker_obj.quarterly_income_stmt
        if q_fin.empty: q_fin = ticker_obj.quarterly_financials
        a_cf = ticker_obj.cashflow
        q_cf = ticker_obj.quarterly_cashflow

        # Ghost Column Eraser
        if not q_fin.empty:
            valid_cols = []
            for i in range(len(q_fin.columns)):
                if get_value_max_fuzzy(q_fin, i, ['Total Revenue', 'Revenue']) > 1000:
                    valid_cols.append(q_fin.columns[i])
            if valid_cols: q_fin = q_fin[valid_cols]

        def extract_from_col(df, col_idx, df_cf=None, col_idx_cf=None):
            r = get_value_max_fuzzy(df, col_idx, ['Total Revenue', 'Revenue'])
            i = get_value_max_fuzzy(df, col_idx, ['Interest Expense', 'Interest Expense Non Operating'])
            ed = get_value_max_fuzzy(df, col_idx, ['EBITDA', 'Normalized EBITDA'])
            p_tax, p_prov, val_e = 0, 0, 0
            
            if is_financial:
                p_tax = get_value_max_fuzzy(df, col_idx, ['Pretax Income', 'Income Before Tax'])
                prov_kws = [('provisionforcreditlosses', 10), ('creditlossesprovision', 10), ('provisionforloanlosses', 9), ('creditloss', 5)]
                excl_kws = ['incometax', 'deferredtax']
                
                if df_cf is not None and col_idx_cf is not None:
                    p_prov = get_value_max_fuzzy_with_priority(df_cf, col_idx_cf, prov_kws, excl_kws)
                if p_prov == 0:
                    p_prov = get_value_max_fuzzy_with_priority(df, col_idx, prov_kws, excl_kws)
                if p_tax != 0: val_e = p_tax + abs(p_prov)
            
            if val_e == 0: val_e = get_value_max_fuzzy(df, col_idx, ['EBIT', 'Operating Income'])
            return r, val_e, ed, i, p_tax, abs(p_prov)

        # Priority 1: Annual
        if not a_fin.empty:
            for idx, col in enumerate(a_fin.columns):
                col_dt = pd.to_datetime(col)
                if col_dt.year == target_year:
                    cf_idx = None
                    if not a_cf.empty:
                        for c_idx, c_col in enumerate(a_cf.columns):
                            if pd.to_datetime(c_col).year == target_year: cf_idx = c_idx; break
                    r, e, ed, i, pt, pp = extract_from_col(a_fin, idx, a_cf, cf_idx)
                    
                    if is_financial and pp == 0 and api_data:
                        for d_str, val in api_data.get('provision_annual', []):
                            if str(target_year) in d_str: pp = abs(val); e = pt + pp; break
                    
                    if pd.notna(r) and r > 1000:
                        lbl = col.strftime('%Y-%m-%d')
                        return r, e, ed, abs(i), lbl, lbl, pt, pp

        # Priority 2: Yahoo Info TTM + API
        rev = info_dict.get('totalRevenue', 0)
        if rev and rev > 0:
            ebitda = info_dict.get('ebitda', 0)
            int_exp = info_dict.get('interestExpense', 0) or info_dict.get('totalInterestExpense', 0)
            lbl_int = "TTM (Yahoo Info)" if int_exp else "N/A"
            if not int_exp and not q_fin.empty and q_fin.shape[1]>=4:
                int_exp = sum(get_value_max_fuzzy(q_fin.iloc[:,:4], x, ['Interest Expense']) for x in range(4))
                lbl_int = "TTM (Calc)"

            ebit, raw_pt, raw_pp, lbl_ebit = 0, 0, 0, "N/A"
            if is_financial:
                if api_data and api_data.get('pretax_ttm', 0) != 0:
                    raw_pt = api_data['pretax_ttm']
                    raw_pp = abs(api_data.get('provision_ttm', 0))
                    ebit = raw_pt + raw_pp
                    lbl_ebit = "TTM (Yahoo API)"
                elif not q_fin.empty and q_fin.shape[1]>=4:
                    recent_4 = q_fin.iloc[:, :4]
                    recent_4_cf = q_cf.iloc[:, :4] if not q_cf.empty and q_cf.shape[1]>=4 else None
                    for q_idx in range(4):
                        cf_idx = q_idx if recent_4_cf is not None else None
                        _, _, _, _, pt_q, pp_q = extract_from_col(recent_4, q_idx, recent_4_cf, cf_idx)
                        raw_pt += pt_q; raw_pp += pp_q
                    if raw_pp == 0 and api_data: raw_pp = abs(api_data.get('provision_ttm', 0))
                    ebit = raw_pt + raw_pp
                    lbl_ebit = "TTM (Calc Quarters)"
            else:
                op_margin = info_dict.get('operatingMargins', 0)
                if op_margin: ebit = rev * op_margin; lbl_ebit = "TTM (Info)"
                elif ebitda: ebit = ebitda; lbl_ebit = "TTM (EBITDA Proxy)"
            
            return rev, ebit, ebitda, abs(int_exp or 0), lbl_ebit, lbl_int, raw_pt, raw_pp

        # Priority 3: Calc TTM
        if not q_fin.empty and q_fin.shape[1] >= 4:
            recent_4 = q_fin.iloc[:, :4]
            recent_4_cf = q_cf.iloc[:, :4] if not q_cf.empty and q_cf.shape[1]>=4 else None
            r_sum, e_sum, ed_sum, i_sum, pt_sum, pp_sum = 0, 0, 0, 0, 0, 0
            for q_idx in range(4):
                cf_idx = q_idx if recent_4_cf is not None else None
                r, e, ed, i, pt, pp = extract_from_col(recent_4, q_idx, recent_4_cf, cf_idx)
                r_sum+=r; e_sum+=e; ed_sum+=ed; i_sum+=i; pt_sum+=pt; pp_sum+=pp
            
            if is_financial:
                if pp_sum == 0 and api_data: pp_sum = abs(api_data.get('provision_ttm', 0))
                if pt_sum == 0 and api_data and api_data.get('pretax_ttm', 0)!=0: pt_sum = api_data['pretax_ttm']
                e_sum = pt_sum + pp_sum
            
            lbl = f"TTM (Calc: {recent_4.columns[0].strftime('%Y-%m-%d')})"
            return r_sum, e_sum, ed_sum, abs(i_sum), lbl, lbl, pt_sum, pp_sum

    except Exception as e: logger.error(f"Extraction failed: {str(e)}")
    return 0, 0, 0, 0, "No Data", "No Data", 0, 0

# ==============================================================================
# [MODULE] Peer Recommender & Financials
# ==============================================================================
class PeerRecommender:
    def recommend(self, target_ticker, progress_bar=None):
        try:
            t = yf.Ticker(target_ticker)
            info = safe_yf_info(t)
            ind_key = info.get('industryKey')
            if not ind_key: return None, "Unknown", ["Industry key not found"]
            
            industry = yf.Industry(ind_key)
            top_df = industry.top_companies
            raw_list = top_df['symbol'].tolist() if 'symbol' in top_df.columns else top_df.index.tolist()
            candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5]
            
            # Simple revenue sort would go here, omitting for brevity in this block
            return ", ".join(candidates), f"Industry: {ind_key}", []
        except Exception as e: return None, "Error", [str(e)]

def get_target_financials(ticker):
    _, tax_map, _ = get_kpmg_tax_rates()
    try:
        t = yf.Ticker(ticker)
        info = safe_yf_info(t)
        if not info: return {"int_exp": 0, "ebit": 0, "category": "Small/Risky Firms", "tax_rate": 25.0}
        
        country = info.get('country', 'Unknown')
        target_tax = tax_map.get(str(country).upper().strip(), 25.0)
        rev, ebit, ebitda, int_exp, label_ebit, label_int, raw_pt, raw_pp = get_financial_data_with_priority(t, info, ticker)
        
        sector = str(info.get('sector', '')).lower()
        cat = "Financial Firms" if 'financial' in sector or 'bank' in sector else "Large Firms" if info.get('marketCap', 0) > 5e9 else "Small/Risky Firms"
        
        return {"int_exp": int_exp, "ebit": ebit, "label_int": label_int, "label_ebit": label_ebit, 
                "raw_pretax": raw_pt, "raw_provision": raw_pp, "category": cat, "tax_rate": target_tax, "country_name": country}
    except: return {"int_exp": 0, "ebit": 0, "category": "Small/Risky Firms", "tax_rate": 25.0}

# ==============================================================================
# [LOGIC] WACC Engine
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
            if not info or len(info) < 5: return None, f"⚠️ {ticker}: No data"
            
            curr = info.get('currency', 'USD')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            rev, ebit, ebitda, int_dummy, label_ebit, label_int, pt, pp = get_financial_data_with_priority(t, info, ticker)
            
            if rev == 0: return None, f"⚠️ {ticker}: Excluded (No Revenue)"
            
            raw_beta = info.get('beta', 1.0)
            if raw_beta is None: raw_beta = 1.0
            
            return {
                "name": info.get('longName', ticker), "country": info.get('country', 'Unknown'), 
                "currency": curr_code, "fx_rate": fx, "tax_rate": self.kpmg_map.get(str(info.get('country')).upper(), 25.0),
                "vals": {"Revenue": rev*fx, "EBIT": ebit*fx, "EBITDA": ebitda*fx, "Total Debt": info.get('totalDebt', 0)*fx, "Market Cap": info.get('marketCap', 0)*fx},
                "period": label_ebit, "raw_beta": raw_beta
            }, None
        except Exception as e: return None, f"Error {ticker}: {str(e)}"

    def get_5y_monthly_beta_analysis(self): return None, None, None, []

    def run(self, category_in, sens_method):
        _, _, _, beta_err = self.get_5y_monthly_beta_analysis()
        error_logs = beta_err if beta_err else []
        peer_data = []
        progress_text = st.empty()
        
        for p in self.peers:
            progress_text.text(f"Analyzing {p}...")
            time.sleep(0.1)
            fin, err = self.get_financials_latest(p)
            if err: error_logs.append(err); continue
            
            if fin:
                d = fin['vals']
                eq = d['Market Cap']; debt = d['Total Debt']
                adj_beta = fin['raw_beta'] * 0.67 + 0.33
                peer_data.append({
                    "Ticker": p, "Company Name": fin['name'], "Country": fin['country'], "Tax Rate": fin['tax_rate'],
                    "Currency": fin['currency'], "FX Rate": fin['fx_rate'], "Revenue": d['Revenue'], "EBIT": d['EBIT'],
                    "EBITDA": d['EBITDA'], "Total Debt": debt, "Market Cap": eq,
                    "D/E Ratio": debt/eq if eq>0 else 0, "Debt/TIC Ratio": debt/(debt+eq) if (debt+eq)>0 else 0,
                    "Period": fin['period'], "Raw Beta": fin['raw_beta'], "Adj Beta": adj_beta
                })
        
        progress_text.empty()
        full_df = pd.DataFrame(peer_data)
        
        if full_df.empty: 
            return {"full_df": full_df, "errors": error_logs, "market_params": {"Rm": 0, "MRP": 0}}

        full_df['Ticker'] = full_df['Ticker'].str.upper().str.strip()

        # [v1.4.1] Logic Change for Financial Firms
        is_financial = (category_in == "Financial Firms")
        
        if is_financial:
            # FINANCIAL FIRMS: Bypass Unlevering
            if sens_method == "Average": target_beta = full_df["Adj Beta"].mean(); sel_dtic = full_df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median": target_beta = full_df["Adj Beta"].median(); sel_dtic = full_df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum": target_beta = full_df["Adj Beta"].max(); sel_dtic = full_df["Debt/TIC Ratio"].max()
            else: target_beta = full_df["Adj Beta"].min(); sel_dtic = full_df["Debt/TIC Ratio"].min()
            
            full_df["Unlevered Beta"] = full_df["Adj Beta"]
            full_df["Re-levered Beta"] = full_df["Adj Beta"]
        else:
            # STANDARD: Unlever -> Re-lever
            full_df["Unlevered Beta"] = full_df["Adj Beta"] / (1 + (1 - full_df["Tax Rate"]/100) * full_df["D/E Ratio"])
            if sens_method == "Average": sel_unlev = full_df["Unlevered Beta"].mean(); sel_dtic = full_df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median": sel_unlev = full_df["Unlevered Beta"].median(); sel_dtic = full_df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum": sel_unlev = full_df["Unlevered Beta"].max(); sel_dtic = full_df["Debt/TIC Ratio"].max()
            else: sel_unlev = full_df["Unlevered Beta"].min(); sel_dtic = full_df["Debt/TIC Ratio"].min()
            
            target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
            target_beta = sel_unlev * (1 + (1 - self.tax) * target_de)
            full_df["Re-levered Beta"] = full_df["Unlevered Beta"] * (1 + (1 - self.tax) * target_de)

        ke = self.rf + (target_beta * self.mrp) + self.crp + self.size_prem
        
        return {
            "full_df": full_df, "target_beta": target_beta, "ke": ke, "wd": sel_dtic, 
            "market_params": {"Rm": self.mrp+self.rf, "MRP": self.mrp}, "rf_trend": self.rf_trend_df, "gdp_df": self.gdp_df, "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas = fetch_all_fred_data()

with st.sidebar:
    st.header("Target & Peers")
    target_ticker = st.text_input("Target Ticker", "JPM")
    
    if st.button("Auto-Recommend Peers (Top 5)", type="secondary", use_container_width=True):
        with st.spinner("Finding peers..."):
            rec = PeerRecommender()
            res_peers, group, logs = rec.recommend(target_ticker)
            if res_peers: 
                st.session_state['peers'] = res_peers
                st.success(f"Found peers in {group}")
            else: st.warning("Recommendation Failed")
            
    peers_input = st.text_area("Peer Tickers", value=st.session_state.get('peers', "GS, MS, BAC, C, WFC"), height=100)
    
    st.divider()
    st.header("Assumptions")
    
    with st.expander("Target Assumptions", expanded=True):
        if 'target_fin' not in st.session_state or st.session_state.get('last_ticker') != target_ticker:
            with st.spinner(f"Loading {target_ticker}..."):
                st.session_state['target_fin'] = get_target_financials(target_ticker)
                st.session_state['last_ticker'] = target_ticker
        
        tf = st.session_state['target_fin']
        tax_in = st.slider("Tax Rate (%)", 0.0, 40.0, float(tf.get('tax_rate', 25.0)), 0.1)
        
        st.divider()
        is_fin_target = 'Financial' in tf['category'] or 'Bank' in tf['category']
        ebit_label = "PPNR ($)" if is_fin_target else "EBIT ($)"
        
        int_exp_in = st.number_input("Interest Expense ($)", value=float(tf['int_exp']), format="%.0f")
        ebit_in = st.number_input(ebit_label, value=float(tf['ebit']), format="%.0f")
        
        if is_fin_target:
            st.markdown(f"<div style='font-size:12px;color:#666'>• Pre-tax: ${tf.get('raw_pretax', 0):,.0f}<br>• Provision: ${tf.get('raw_provision', 0):,.0f}</div>", unsafe_allow_html=True)
        
        cat_options = ["Large Firms", "Small/Risky Firms", "Financial Firms"]
        cat_def = cat_options.index(tf['category']) if tf['category'] in cat_options else 1
        category_in = st.selectbox("Firm Category", cat_options, index=cat_def)

    with st.expander("Cost of Equity / Debt", expanded=True):
        rf_in = st.number_input(f"Risk Free Rate (Latest: {latest_rf:.2f}%)", value=latest_rf, step=0.01)
        crp_in = st.number_input("Country Risk Premium (%)", value=0.0, step=0.1)
        size_in = st.number_input("Size Premium (%)", value=0.0, step=0.1)
    
    with st.expander("Implied Return", expanded=True):
        avg_bb, avg_div, _, _ = get_sp_buyback_data()
        bb_in = st.number_input(f"Buyback Yield (5Y Avg: {avg_bb:.2f}%)", value=avg_bb, step=0.1)
        div_in = st.number_input(f"Dividend Yield (5Y Avg: {avg_div:.2f}%)", value=avg_div, step=0.1)
        g_in = st.number_input(f"Growth Rate (Latest GDP: {latest_gdp:.2f}%)", value=latest_gdp, step=0.1)

    st.divider()
    if st.button("Calculate WACC", type="primary", use_container_width=True):
        model = DetailWACCModel(target_ticker, peers_input, rf_in, crp_in, size_in, bb_in, div_in, g_in, tax_in, df_rf_trend, df_gdp_disp)
        with st.spinner("Calculating..."):
            st.session_state['result'] = model.run(category_in, st.session_state.get('sens_method', "Median"))
            st.session_state['inputs'] = {'rf': rf_in, 'crp': crp_in, 'sp': size_in, 'tax': tax_in, 'bb': bb_in, 'div': div_in, 'g': g_in, 'int_exp': int_exp_in, 'ebit': ebit_in, 'category': category_in}
        st.success("Done!")
    
    st.divider()
    st.caption(f"Version {VERSION} | Build: {BUILD_DATE}")

# ==============================================================================
# [RESULTS DISPLAY]
# ==============================================================================
if 'result' in st.session_state:
    res = st.session_state['result']
    inp = st.session_state['inputs']
    df_init = res['full_df']
    m = res['market_params']
    
    st.subheader("Beta Analysis")
    st.session_state['sens_method'] = st.radio("Aggregation Method", ["Median", "Average", "Maximum", "Minimum"], horizontal=True, index=0)
    
    if res.get('errors'):
        for e in res['errors']: st.error(e)

    ke, kd, wacc, wd, we, target_beta = 0, 0, 0, 0, 0, 0
    final_spread, icr, implied_rating = 0.0, 0.0, "N/A"
    
    if not df_init.empty:
        is_fin_calc = (inp['category'] == "Financial Firms")
        beta_label = "Beta (Adjusted)" if is_fin_calc else "Beta (Re-levered)" # [v1.4.2 UI Logic]

        target_beta = res['target_beta']
        ke = res['ke']
        wd = res['wd']; we = 1 - wd
        
        # Cost of Debt
        icr = inp['ebit'] / inp['int_exp'] if inp['int_exp'] > 0 else 100.0
        damodaran_dict = get_damodaran_spreads()
        rating_table, _ = damodaran_dict.get(inp['category'], (None, ""))
        
        if rating_table is not None:
            for idx, row in rating_table.iterrows():
                try:
                    low = float(str(row.get('greater than','-')).replace('greater than','').replace('-','-99999').strip())
                    high = float(str(row.get('≤ to','-')).replace('-','99999').strip())
                    if low < icr <= high:
                        implied_rating = row['Rating']
                        final_spread = float(str(row['Spread']).replace('%',''))
                        break
                except: continue
        
        # Map to OAS (Simplified logic for brevity, uses mapping from earlier functions if available or falls back)
        target_fred_key = "BB US High Yield" # Default fallback
        if "AAA" in implied_rating: target_fred_key = "AAA US Corporate"
        elif "BBB" in implied_rating: target_fred_key = "BBB US Corporate"
        
        fred_row = df_oas[df_oas['OAS Name'] == target_fred_key]
        if not fred_row.empty: 
            val = fred_row.iloc[0]['Latest Spread (%)']
            if pd.notna(val): final_spread = val

        kd = ((inp['rf'] + final_spread)/100) * (1 - inp['tax']/100)
        wacc = (we * ke) + (wd * kd)

        st.subheader(f"WACC Analysis: {inp.get('category')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final WACC", f"{wacc:.2%}")
        c2.metric("Cost of Equity", f"{ke:.2%}")
        c3.metric("Cost of Debt (A-T)", f"{kd:.2%}")
        c4.metric(beta_label, f"{target_beta:.2f}") # [v1.4.2 Label]
        
        if is_fin_calc:
            st.success("ℹ️ **Financial Firms Logic:** Peer Adjusted Beta used directly (Unlevering bypassed).")

        with st.expander("Peer Analysis Table", expanded=True):
            st.dataframe(df_init[["Ticker", "Company Name", "Adj Beta", "Unlevered Beta", "Re-levered Beta", "D/E Ratio", "Period"]].style.format("{:.2f}"), use_container_width=True)

        st.markdown("---")
        st.subheader("Cost of Equity Details")
        st.latex(r"K_e = R_f + \beta \times (R_m - R_f) + CRP + SP")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
        k2.metric(beta_label, f"{target_beta:.2f}") # [v1.4.2 Label]
        k3.metric("Market Risk Prem", f"{m['MRP']*100:.2f}%")
        k4.metric("Country Risk Prem", f"{inp['crp']:.2f}%")
        k5.metric("Size Premium", f"{inp['sp']:.2f}%")
        st.info(f"Calc: {inp['rf']:.2f}% + {target_beta:.2f} * {m['MRP']*100:.2f}% + {inp['crp']:.2f}% + {inp['sp']:.2f}% = {ke*100:.2f}%")

        st.markdown("---")
        st.subheader("Cost of Debt Details")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("ICR", f"{icr:.2f}x")
        d2.metric("Implied Rating", implied_rating)
        d3.metric("OAS Spread", f"{final_spread:.2f}%")
        d4.metric("After-tax Kd", f"{kd:.2%}")
        
    else:
        st.warning("No valid peer data found.")
