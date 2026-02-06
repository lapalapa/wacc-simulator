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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Page Config
st.set_page_config(page_title="Strategic WACC Simulator", layout="wide")

# ==============================================================================
# [MODULE] Helper: Safe Fetcher with Retry
# ==============================================================================
def safe_yf_info(ticker_obj, max_retries=3):
    for i in range(max_retries):
        try:
            info = ticker_obj.info
            if info and len(info) > 5:
                return info
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 1.5))
    return {}

# ==============================================================================
# [MODULE] Helper: Deep Search with MAX Value Strategy
# ==============================================================================
def get_value_max_fuzzy(df, col_idx, search_keywords):
    """
    Scans ALL rows containing any of the keywords.
    Returns the absolute largest value found (to catch 'Total' fields).
    """
    candidates = []
    try:
        for idx in df.index:
            idx_str = str(idx).lower()
            for kw in search_keywords:
                if kw.lower() in idx_str:
                    val = df.loc[idx].iloc[col_idx]
                    if pd.notna(val) and val != 0:
                        candidates.append(abs(val))
                    break 
        if candidates:
            return max(candidates)
    except:
        pass
    return 0

# ==============================================================================
# [MODULE] Helper: Common Financial Data Extraction Logic (Unified)
# ==============================================================================
def get_financial_data_with_priority(ticker_obj, info_dict):
    """
    Priority Logic v118.0 (Expanded Provision Keywords):
    1. Annual (Year-1) -> Label: YYYY-MM-DD
    2. Yahoo Info TTM -> Label: TTM (Yahoo Info) OR TTM (Yahoo Info + Calc Interest)
    3. Calc TTM (Manual Sum) -> Label: TTM (Calculated: YYYY-MM-DD)
    
    * Ghost Column Eraser applied.
    * PPNR = Pretax + abs(Provision)
    * Provision Keywords: Added 'Credit Losses Provision', 'Loan and Lease Losses' etc.
    """
    rev = 0; ebit = 0; ebitda = 0; int_exp = 0
    label_ebit = "N/A"
    label_int = "N/A"
    
    sector = info_dict.get('sector', '').lower()
    is_financial = 'financial' in sector or 'bank' in sector
    
    current_year = datetime.now().year
    target_year = current_year - 1 
    
    try:
        # Load Statements
        a_fin = ticker_obj.income_stmt
        if a_fin.empty: a_fin = ticker_obj.financials
        
        q_fin = ticker_obj.quarterly_income_stmt
        if q_fin.empty: q_fin = ticker_obj.quarterly_financials

        # [STEP 0] GHOST COLUMN ERASER
        if not q_fin.empty:
            valid_cols = []
            for i in range(len(q_fin.columns)):
                r_check = get_value_max_fuzzy(q_fin, i, ['Total Revenue', 'Revenue'])
                if r_check > 1000:
                    valid_cols.append(q_fin.columns[i])
            if valid_cols:
                q_fin = q_fin[valid_cols]

        # Helper to extract from a specific column
        def extract_from_col(df, col_idx):
            # 1. Revenue
            r = get_value_max_fuzzy(df, col_idx, ['Total Revenue', 'Revenue'])
            
            # 2. Interest Expense
            i = get_value_max_fuzzy(df, col_idx, ['Interest Expense', 'Interest Expense Non Operating'])
            
            # 3. EBITDA
            ed = get_value_max_fuzzy(df, col_idx, ['EBITDA', 'Normalized EBITDA'])
            
            # 4. EBIT / PPNR
            val_e = 0
            if is_financial:
                pretax = get_value_max_fuzzy(df, col_idx, ['Pretax Income', 'Income Before Tax'])
                
                # [FIX v118] Expanded Keywords
                provision = get_value_max_fuzzy(df, col_idx, [
                    'Provision For Credit Losses',
                    'Credit Losses Provision',          # User Request
                    'Provision For Loan Losses',
                    'Provision For Loan And Lease Losses', # Added
                    'Provision for losses on loans',       # Added
                    'Credit Loss Provision',
                    'Loan Loss Provision',
                    'Credit Loss'                       # Broad Catch
                ])
                
                # [FORMULA: Pretax + abs(Provision)]
                if pretax != 0: 
                    val_e = pretax + abs(provision)
            
            if val_e == 0:
                val_e = get_value_max_fuzzy(df, col_idx, ['EBIT', 'Operating Income', 'Operating Profit'])
            
            return r, val_e, ed, i

        # --- Priority 1: Annual (Year-1) with TRIPLE LOCK ---
        if not a_fin.empty:
            for idx, col in enumerate(a_fin.columns):
                col_dt = pd.to_datetime(col)
                if col_dt.year == target_year:
                    r_annual, e, ed, i = extract_from_col(a_fin, idx)
                    
                    if pd.notna(r_annual) and r_annual > 1000:
                        # Validation
                        is_valid = False
                        if not q_fin.empty:
                            cutoff_date = col_dt - timedelta(days=360)
                            valid_quarters = []
                            q_rev_sum = 0
                            for q_idx, q_col in enumerate(q_fin.columns):
                                q_dt = pd.to_datetime(q_col)
                                if cutoff_date < q_dt <= col_dt:
                                    valid_quarters.append(q_idx)
                                    q_rev_sum += get_value_max_fuzzy(q_fin, q_idx, ['Total Revenue', 'Revenue'])
                            
                            if len(valid_quarters) >= 4:
                                if 0.9 <= (q_rev_sum / r_annual) <= 1.1:
                                    is_valid = True
                        
                        if is_valid:
                            lbl = col.strftime('%Y-%m-%d')
                            return r_annual, e, ed, abs(i), lbl, lbl
        
        # --- Priority 2: Yahoo Info TTM ---
        rev_ttm = info_dict.get('totalRevenue', 0)
        
        if rev_ttm is not None and rev_ttm > 0:
            rev = rev_ttm
            ebitda = info_dict.get('ebitda', 0)
            
            # Interest Expense Source
            int_exp = info_dict.get('interestExpense', 0)
            if int_exp is None or int_exp == 0:
                 int_exp = info_dict.get('totalInterestExpense', 0)
            
            if int_exp is not None and int_exp > 0:
                label_int = "TTM (Yahoo Info)"
            else:
                if not q_fin.empty and q_fin.shape[1] >= 4:
                    recent_4 = q_fin.iloc[:, :4]
                    q_int = 0
                    for q_idx in range(4):
                        q_int += get_value_max_fuzzy(recent_4, q_idx, ['Interest Expense'])
                    int_exp = q_int
                    label_int = "TTM (Yahoo Info + Calc Interest)"
                else:
                    int_exp = 0
                    label_int = "N/A"

            # EBIT / PPNR Source
            ebit = 0
            if is_financial:
                if not q_fin.empty and q_fin.shape[1] >= 4:
                    recent_4 = q_fin.iloc[:, :4]
                    q_pretax = 0; q_prov = 0
                    for q_idx in range(4):
                        q_pretax += get_value_max_fuzzy(recent_4, q_idx, ['Pretax Income', 'Income Before Tax'])
                        # [FIX v118] Expanded Keywords (Quarterly)
                        q_prov += get_value_max_fuzzy(recent_4, q_idx, [
                            'Provision For Credit Losses', 'Credit Losses Provision',
                            'Provision For Loan Losses', 'Provision For Loan And Lease Losses',
                            'Provision for losses on loans', 'Credit Loss'
                        ])
                    
                    if q_pretax != 0: ebit = q_pretax + abs(q_prov)
                    label_ebit = "TTM (Calculated)"
                else:
                    label_ebit = "N/A"
            else:
                op_margin = info_dict.get('operatingMargins', 0)
                if op_margin: 
                    ebit = rev * op_margin
                    label_ebit = "TTM (Yahoo Info)"
                elif ebitda: 
                    ebit = ebitda
                    label_ebit = "TTM (Yahoo Info Proxy)"
                else:
                    label_ebit = "N/A"
            
            if int_exp is None: int_exp = 0
            return rev, ebit, ebitda, abs(int_exp), label_ebit, label_int

        # --- Priority 3: Calc TTM (Manual Sum) ---
        if not q_fin.empty and q_fin.shape[1] >= 4:
            recent_4 = q_fin.iloc[:, :4]
            last_date = recent_4.columns[0].strftime('%Y-%m-%d')
            common_label = f"TTM (Calculated: {last_date})"
            
            rev = 0; ebitda = 0; int_exp = 0; ebit = 0
            sum_pretax = 0; sum_prov = 0; sum_ebit_std = 0
            
            for q_idx in range(4):
                rev += get_value_max_fuzzy(recent_4, q_idx, ['Total Revenue', 'Revenue'])
                ebitda += get_value_max_fuzzy(recent_4, q_idx, ['EBITDA', 'Normalized EBITDA'])
                int_exp += get_value_max_fuzzy(recent_4, q_idx, ['Interest Expense'])
                
                if is_financial:
                    sum_pretax += get_value_max_fuzzy(recent_4, q_idx, ['Pretax Income', 'Income Before Tax'])
                    # [FIX v118] Expanded Keywords (Manual Sum)
                    sum_prov += get_value_max_fuzzy(recent_4, q_idx, [
                        'Provision For Credit Losses', 'Credit Losses Provision',
                        'Provision For Loan Losses', 'Provision For Loan And Lease Losses',
                        'Provision for losses on loans', 'Credit Loss'
                    ])
                else:
                    sum_ebit_std += get_value_max_fuzzy(recent_4, q_idx, ['EBIT', 'Operating Income'])
            
            if is_financial: ebit = sum_pretax + abs(sum_prov)
            else: ebit = sum_ebit_std
            
            return rev, ebit, ebitda, abs(int_exp), common_label, common_label

    except Exception:
        pass
    
    return 0, 0, 0, 0, "No Data", "No Data"

# ==============================================================================
# [MODULE] Data Fetcher: Consolidated FRED
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_all_fred_data():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    targets = [
        ("GDP", "A191RP1A027NBEA", False),
        ("RF", "DGS10", False),
        ("AAA", "BAMLC0A1CAAA", True),
        ("AA", "BAMLC0A2CAA", True),
        ("A", "BAMLC0A3CA", True),
        ("BBB", "BAMLC0A4CBBB", True),
        ("BB", "BAMLH0A1HYBB", True),
        ("B", "BAMLH0A2HYB", True),
        ("CCC", "BAMLH0A3HYC", True)
    ]
    
    results = {}
    
    for key, series_id, is_oas in targets:
        time.sleep(random.uniform(1.2, 2.5)) 
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10, verify=False)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            df = df.dropna().sort_values(by="DATE", ascending=True)
            if not df.empty: results[key] = df
        except: pass

    # GDP
    latest_gdp = 2.5; df_gdp_disp = None
    if "GDP" in results:
        df = results["GDP"]
        latest_gdp = float(df["VALUE"].iloc[-1])
        df_gdp_disp = df.sort_values(by="DATE", ascending=False).head(10)
        df_gdp_disp.columns = ["Date", "GDP Growth %"]

    # RF
    latest_rf = 4.2; df_rf_trend = None
    if "RF" in results:
        df = results["RF"]
        latest_rf = float(df["VALUE"].iloc[-1])
        cutoff = df["DATE"].iloc[-1] - timedelta(days=365*5)
        df_rf_trend = df[df["DATE"] >= cutoff].copy()
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
            df = results[k]
            val = float(df["VALUE"].iloc[-1])
            date_str = df["DATE"].iloc[-1].strftime('%Y-%m-%d')
            
        oas_rows.append({"OAS Name": name, "Latest Spread (%)": val, "Date": date_str, "Link": link})
    
    df_oas = pd.DataFrame(oas_rows)
    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas

# ==============================================================================
# [MODULE] Data Fetcher: NYU & KPMG
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        dfs = pd.read_html(io.StringIO(response.text), header=0)
        clean_df = dfs[0].dropna(subset=[dfs[0].columns[0]])
        return 2.0, 1.5, clean_df, []
    except: return 2.0, 1.5, None, ["Error"]

@st.cache_data(ttl=3600*24)
def get_kpmg_tax_rates():
    url = "https://kpmg.com/dk/en/services/tax/corporate-tax/corporate-tax-rates-table.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        dfs = pd.read_html(io.StringIO(r.text))
        target_df = dfs[0]
        target_df.rename(columns={target_df.columns[0]: "Country"}, inplace=True)
        col_name = target_df.columns[-1]
        result_df = target_df[["Country", col_name]].copy()
        result_df.columns = ["Country", f"Rate"]
        result_df["Rate"] = pd.to_numeric(result_df["Rate"], errors='coerce')
        tax_dict = dict(zip(result_df["Country"].str.upper().str.strip(), result_df["Rate"]))
        tax_dict["UNITED STATES"] = 25.57; tax_dict["USA"] = 25.57; tax_dict["KOREA"] = 26.40
        return result_df, tax_dict, 2025
    except: return None, {}, 2025

@st.cache_data(ttl=3600*24)
def get_damodaran_spreads():
    # 2026 Updated Fallback Data
    fallback_large = pd.DataFrame([
        {"greater than": "8.5", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"},
        {"greater than": "6.5", "≤ to": "8.49", "Rating": "Aa2/AA", "Spread": "0.55%"},
        {"greater than": "5.5", "≤ to": "6.49", "Rating": "A1/A+", "Spread": "0.70%"},
        {"greater than": "4.25", "≤ to": "5.49", "Rating": "A2/A", "Spread": "0.78%"},
        {"greater than": "3.0", "≤ to": "4.24", "Rating": "A3/A-", "Spread": "0.89%"},
        {"greater than": "2.5", "≤ to": "2.99", "Rating": "Baa2/BBB", "Spread": "1.11%"},
        {"greater than": "2.25", "≤ to": "2.49", "Rating": "Ba1/BB+", "Spread": "1.38%"},
        {"greater than": "2.0", "≤ to": "2.24", "Rating": "Ba2/BB", "Spread": "1.84%"},
        {"greater than": "1.75", "≤ to": "1.99", "Rating": "B1/B+", "Spread": "2.75%"},
        {"greater than": "1.5", "≤ to": "1.74", "Rating": "B2/B", "Spread": "3.21%"},
        {"greater than": "1.25", "≤ to": "1.49", "Rating": "B3/B-", "Spread": "5.09%"},
        {"greater than": "0.8", "≤ to": "1.24", "Rating": "Caa/CCC", "Spread": "8.85%"},
        {"greater than": "0.65", "≤ to": "0.79", "Rating": "Ca2/CC", "Spread": "12.61%"},
        {"greater than": "0.2", "≤ to": "0.64", "Rating": "C2/C", "Spread": "16.00%"},
        {"greater than": "-100000", "≤ to": "0.19", "Rating": "D2/D", "Spread": "19.00%"}
    ])
    
    fallback_small = pd.DataFrame([
        {"greater than": "12.5", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"},
        {"greater than": "9.5", "≤ to": "12.49", "Rating": "Aa2/AA", "Spread": "0.55%"},
        {"greater than": "7.5", "≤ to": "9.49", "Rating": "A1/A+", "Spread": "0.70%"},
        {"greater than": "6.0", "≤ to": "7.49", "Rating": "A2/A", "Spread": "0.78%"},
        {"greater than": "4.5", "≤ to": "5.99", "Rating": "A3/A-", "Spread": "0.89%"},
        {"greater than": "4.0", "≤ to": "4.49", "Rating": "Baa2/BBB", "Spread": "1.11%"},
        {"greater than": "3.5", "≤ to": "3.99", "Rating": "Ba1/BB+", "Spread": "1.38%"},
        {"greater than": "3.0", "≤ to": "3.49", "Rating": "Ba2/BB", "Spread": "1.84%"},
        {"greater than": "2.5", "≤ to": "2.99", "Rating": "B1/B+", "Spread": "2.75%"},
        {"greater than": "2.0", "≤ to": "2.49", "Rating": "B2/B", "Spread": "3.21%"},
        {"greater than": "1.5", "≤ to": "1.99", "Rating": "B3/B-", "Spread": "5.09%"},
        {"greater than": "1.25", "≤ to": "1.49", "Rating": "Caa/CCC", "Spread": "8.85%"},
        {"greater than": "0.8", "≤ to": "1.24", "Rating": "Ca2/CC", "Spread": "12.61%"},
        {"greater than": "0.5", "≤ to": "0.79", "Rating": "C2/C", "Spread": "16.00%"},
        {"greater than": "-100000", "≤ to": "0.49", "Rating": "D2/D", "Spread": "19.00%"}
    ])
    
    fallback_fin = pd.DataFrame([
        {"greater than": "3.0", "≤ to": "100000", "Rating": "Aaa/AAA", "Spread": "0.40%"},
        {"greater than": "2.5", "≤ to": "2.99", "Rating": "Aa2/AA", "Spread": "0.55%"},
        {"greater than": "2.0", "≤ to": "2.49", "Rating": "A1/A+", "Spread": "0.70%"},
        {"greater than": "1.5", "≤ to": "1.99", "Rating": "A2/A", "Spread": "0.78%"},
        {"greater than": "1.2", "≤ to": "1.49", "Rating": "A3/A-", "Spread": "0.89%"},
        {"greater than": "0.9", "≤ to": "1.19", "Rating": "Baa2/BBB", "Spread": "1.11%"},
        {"greater than": "0.75", "≤ to": "0.89", "Rating": "Ba1/BB+", "Spread": "1.38%"},
        {"greater than": "0.6", "≤ to": "0.74", "Rating": "Ba2/BB", "Spread": "1.84%"},
        {"greater than": "0.5", "≤ to": "0.59", "Rating": "B1/B+", "Spread": "2.75%"},
        {"greater than": "0.4", "≤ to": "0.49", "Rating": "B2/B", "Spread": "3.21%"},
        {"greater than": "0.3", "≤ to": "0.39", "Rating": "B3/B-", "Spread": "5.09%"},
        {"greater than": "0.2", "≤ to": "0.29", "Rating": "Caa/CCC", "Spread": "8.85%"},
        {"greater than": "0.1", "≤ to": "0.19", "Rating": "Ca2/CC", "Spread": "12.61%"},
        {"greater than": "0.05", "≤ to": "0.09", "Rating": "C2/C", "Spread": "16.00%"},
        {"greater than": "-100000", "≤ to": "0.04", "Rating": "D2/D", "Spread": "19.00%"}
    ])

    result_dict = {
        "Large Firms": (fallback_large, "Source: Fallback (Offline, Jan 2026 Data)"),
        "Small/Risky Firms": (fallback_small, "Source: Fallback (Offline, Jan 2026 Data)"), 
        "Financial Firms": (fallback_fin, "Source: Fallback (Offline, Jan 2026 Data)")
    }

    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        
        try:
            df = pd.read_excel(io.BytesIO(response.content), sheet_name="Start here Ratings sheet", header=None)
        except:
            df = pd.read_excel(io.BytesIO(response.content), sheet_name=0, header=None)

        def extract_block(keyword):
            start_row = -1; start_col = -1
            for idx, row in df.iterrows():
                row_str = " ".join([str(x) for x in row.values if pd.notna(x)]).lower()
                if keyword.lower() in row_str:
                    start_row = idx
                    for c_idx, cell in enumerate(row):
                        if keyword.lower() in str(cell).lower():
                            start_col = c_idx; break
                    break
            
            if start_row == -1: return None
            
            data_start_row = -1; marker_col_idx = -1
            for i in range(start_row + 1, start_row + 20):
                if i >= len(df): break
                row = df.iloc[i]
                search_cols = range(max(0, start_col - 2), min(len(row), start_col + 10))
                for c_idx in search_cols:
                    cell_str = str(row[c_idx]).strip().lower()
                    if cell_str.startswith(">") or "greater" in cell_str:
                        data_start_row = i; marker_col_idx = c_idx; break
                if data_start_row != -1: break
            
            if data_start_row == -1: return None
            
            data_rows = []
            for i in range(data_start_row, data_start_row + 20):
                if i >= len(df): break
                row = df.iloc[i]
                try:
                    val_start = row[marker_col_idx]
                    val_end = row[marker_col_idx + 1]
                    val_rating = row[marker_col_idx + 2]
                    val_spread = row[marker_col_idx + 3]
                    
                    if pd.isna(val_rating) or pd.isna(val_spread): break
                    
                    if isinstance(val_spread, (int, float)):
                        spread_fmt = f"{val_spread * 100:.2f}%"
                    else:
                        spread_fmt = str(val_spread)
                        
                    start_str = str(val_start).strip()
                    if start_str == ">": start_str = "greater than"
                    
                    entry = {
                        "greater than": start_str, 
                        "≤ to": val_end,
                        "Rating": str(val_rating), 
                        "Spread": spread_fmt
                    }
                    data_rows.append(entry)
                except: continue
            return pd.DataFrame(data_rows) if data_rows else None

        # Execute Extraction
        df_large = extract_block("for large")
        if df_large is not None and not df_large.empty: 
            result_dict["Large Firms"] = (df_large, "Source: NYU Stern (Live Extract)")

        df_small = extract_block("for smaller")
        if df_small is not None and not df_small.empty: 
            result_dict["Small/Risky Firms"] = (df_small, "Source: NYU Stern (Live Extract)")
        
        df_fin = extract_block("for financial")
        if df_fin is not None and not df_fin.empty:
             result_dict["Financial Firms"] = (df_fin, "Source: NYU Stern (Live Extract)")

    except Exception as e:
        pass 

    return result_dict

# ==============================================================================
# [MODULE] Peer Recommender & Financials
# ==============================================================================
class PeerRecommender:
    def get_revenue(self, ticker):
        t = yf.Ticker(ticker); info = safe_yf_info(t)
        return info.get('totalRevenue', 0)

    def recommend(self, target_ticker, progress_bar=None):
        try:
            t = yf.Ticker(target_ticker)
            info = safe_yf_info(t)
            ind_key = info.get('industryKey')
            if ind_key: industry = yf.Industry(ind_key); top_df = industry.top_companies
            else: return None, "Unknown", []
            
            raw_list = top_df['symbol'].tolist() if 'symbol' in top_df.columns else top_df.index.tolist()
            candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5]
            
            revenue_map = []
            for idx, ticker in enumerate(candidates):
                time.sleep(0.5)
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                if progress_bar: progress_bar.progress(0.2 + (0.8 * (idx/len(candidates))), text=f"Analyzing {ticker}...")
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_5 = [item[0] for item in revenue_map][:5]
            return ", ".join(top_5), f"Industry: {ind_key}", []
        except: return None, "Error", []

def get_target_financials(ticker):
    _, tax_map, _ = get_kpmg_tax_rates()
    try:
        t = yf.Ticker(ticker)
        info = safe_yf_info(t, max_retries=5)
        
        country = info.get('country', 'Unknown')
        country_norm = str(country).upper().strip()
        target_tax = tax_map.get(country_norm)
        if target_tax is None:
            if "UNITED STATES" in country_norm or "USA" in country_norm: target_tax = 25.57
            elif "KOREA" in country_norm: target_tax = 26.40
            else: target_tax = 25.0
        
        rev, ebit, ebitda, int_exp, label_ebit, label_int = get_financial_data_with_priority(t, info)
        
        mkt_cap = info.get('marketCap', 0)
        sector = info.get('sector', '')
        if 'Financial' in sector or 'Bank' in sector: category = "Financial Firms"
        elif mkt_cap > 5e9: category = "Large Firms" 
        else: category = "Small/Risky Firms"
        
        return {
            "int_exp": int_exp, "ebit": ebit, 
            "label_int": label_int, "label_ebit": label_ebit,
            "category": category, "tax_rate": target_tax, "country_name": country
        }
    except: pass
    return {"int_exp": 0.0, "ebit": 0.0, "label_int": "N/A", "label_ebit": "N/A", "category": "Small/Risky Firms", "tax_rate": 25.0, "country_name": "Unknown"}

# ==============================================================================
# [LOGIC] WACC Engine
# ==============================================================================
class DetailWACCModel:
    def __init__(self, target, peers, rf_rate, crp, size_prem, buyback, div_yield, growth, tax, rf_trend_df, gdp_df):
        self.target = target; self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.rf = rf_rate / 100; self.crp = crp / 100; self.size_prem = size_prem / 100
        self.buyback_yield = buyback / 100; self.div_yield = div_yield / 100
        self.growth_rate = growth / 100; self.tax = tax / 100
        self.rf_trend_df = rf_trend_df; self.gdp_df = gdp_df
        self.market_index = "^GSPC"; self.fx_cache = {}
        _, self.kpmg_map, _ = get_kpmg_tax_rates()

    def get_exchange_rate_to_usd(self, currency):
        return 1.0, "USD" 

    def get_financials_latest(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            if not info: return None, f"⚠️ {ticker}: No data."
            curr = info.get('currency', 'USD')
            country = info.get('country', 'Unknown')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            mkt_cap = info.get('marketCap', 0)
            debt = info.get('totalDebt', 0)
            if mkt_cap == 0: 
                try: mkt_cap = t.fast_info['market_cap']
                except: return None, f"⚠️ {ticker}: Excluded (Missing Market Cap)."

            rev, ebit, ebitda, int_exp_dummy, label_ebit, label_int = get_financial_data_with_priority(t, info)
            
            if rev == 0: return None, f"⚠️ {ticker}: Excluded (Missing Revenue)."

            period_display = label_ebit if "Calculated" in label_ebit else label_int

            tax_rate = self.kpmg_map.get(str(country).upper(), 25.0) 
            data = {
                "name": info.get('longName', ticker), "country": country, "currency": curr_code, "fx_rate": fx, "tax_rate": tax_rate,
                "vals": { "Revenue": rev * fx, "EBIT": ebit * fx, "EBITDA": ebitda * fx, "Total Debt": debt * fx, "Market Cap": mkt_cap * fx },
                "period": period_display
            }
            return data, None
        except Exception as e: return None, f"⚠️ {ticker}: Error {str(e)}"

    def get_5y_monthly_beta_analysis(self):
        beta_list = []
        for t in self.peers:
            beta_list.append({"Ticker": t, "Raw Beta": 1.2, "Adj Beta": 1.13})
        return pd.DataFrame(beta_list), None, None, []

    def run(self):
        beta_df, _, _, beta_err = self.get_5y_monthly_beta_analysis()
        error_logs = beta_err if beta_err else []
        peer_data = []
        progress_text = st.empty()
        
        for idx, p in enumerate(self.peers):
            progress_text.text(f"Analyzing {p}...")
            time.sleep(0.5)
            fin, err = self.get_financials_latest(p)
            if err: error_logs.append(err); continue 
            if fin:
                d = fin['vals']; equity = d['Market Cap']; debt = d['Total Debt']; tic = equity + debt
                de_ratio = debt / equity if equity > 0 else 0.0; dtic_ratio = debt / tic if tic > 0 else 0.0
                peer_data.append({
                    "Ticker": p, "Company Name": fin['name'], "Company": fin['name'], "Company": fin['name'], "Country": fin['country'],
                    "Tax Rate": fin['tax_rate'], "Currency": fin['currency'], "FX Rate": fin['fx_rate'],
                    "Revenue": d['Revenue'], "EBIT": d['EBIT'], "EBITDA": d['EBITDA'], "Total Debt": d['Total Debt'],
                    "Market Cap": d['Market Cap'], "D/E Ratio": de_ratio, "Debt/TIC Ratio": dtic_ratio,
                    "Period": fin['period']
                })
        progress_text.empty()
        df_peers = pd.DataFrame(peer_data)
        if beta_df is not None and not beta_df.empty and not df_peers.empty:
            beta_df['Ticker'] = beta_df['Ticker'].str.upper().str.strip(); df_peers['Ticker'] = df_peers['Ticker'].str.upper().str.strip()
            full_df = pd.merge(df_peers, beta_df, on="Ticker", how="left")
        else: full_df = pd.DataFrame()

        rm = self.div_yield + self.buyback_yield + self.growth_rate; mrp = rm - self.rf
        return {
            "full_df": full_df, "prices": None, "market_params": {"Rm": rm, "MRP": mrp},
            "rf_trend": self.rf_trend_df, "gdp_df": self.gdp_df, "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
# FETCH FRED DATA (GLOBAL)
latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas = fetch_all_fred_data()

with st.sidebar:
    st.header("Target & Peers")
    target_ticker = st.text_input("Target Ticker", "WOLF")
    
    if st.button("🤖 Auto-Recommend Peers (Top 5)", type="secondary", use_container_width=True):
        with st.spinner("Finding..."):
            rec = PeerRecommender()
            res_peers, group, logs = rec.recommend(target_ticker)
            if res_peers: st.session_state['peers'] = res_peers
            else: st.warning("Recommendation Failed")
            
    peers_input = st.text_area("Peer Tickers", value=st.session_state.get('peers', "ON, STM, IFX.DE"), height=100)
    st.caption("※ Top 5 revenue companies in the industry\n(Source: Yahoo Finance Industry/Sector Data)")
    
    st.divider()
    st.header("Assumptions")
    
    # [SECTION] Target Assumptions
    with st.expander("Target Assumptions", expanded=True):
        if 'target_fin' not in st.session_state or st.session_state.get('last_ticker') != target_ticker:
            st.session_state['target_fin'] = get_target_financials(target_ticker)
            st.session_state['last_ticker'] = target_ticker
        
        tf = st.session_state['target_fin']
        
        tax_in = st.slider("Tax Rate (%)", 0.0, 40.0, float(tf.get('tax_rate', 25.0)), 0.1)
        st.caption(f"📍 Corporate Tax based on HQ: **{tf.get('country_name', 'Unknown/Default')}**")
        
        st.divider()
        is_fin_target = 'Financial' in tf['category'] or 'Bank' in tf['category']
        ebit_label = "PPNR ($)" if is_fin_target else "EBIT ($)"
        
        st.markdown(f"**Target Financials** (for Credit Spread)")
        
        # [SEPARATE LABELS for Target]
        int_exp_in = st.number_input("Interest Expense ($)", value=float(tf['int_exp']), format="%.0f")
        st.caption(f"Source: **{tf.get('label_int', 'N/A')}**")
        
        ebit_in = st.number_input(ebit_label, value=float(tf['ebit']), format="%.0f", help="For Financial Firms, PPNR is calculated as: Pre-tax Income + |Provision|")
        st.caption(f"Source: **{tf.get('label_ebit', 'N/A')}**")
        
        cat_options = ["Large Firms", "Small/Risky Firms", "Financial Firms"]
        cat_default_idx = cat_options.index(tf['category']) if tf['category'] in cat_options else 1
        category_in = st.selectbox("Firm Category", cat_options, index=cat_default_idx)

    # [SECTION] Cost of Equity / Debt
    with st.expander("Cost of Equity / Debt", expanded=True):
        rf_in = st.number_input(f"Risk Free Rate (Latest: {latest_rf:.2f}%)", value=latest_rf, step=0.01)
        crp_in = st.number_input("Country Risk Premium (%)", value=0.0, step=0.1)
        size_in = st.number_input("Size Premium (%)", value=0.0, step=0.1)
    
    # [SECTION] Implied Return
    with st.expander("Implied Return", expanded=True):
        avg_bb, avg_div, _, _ = get_sp_buyback_data()
        bb_in = st.number_input(f"Buyback Yield (5Y Avg: {avg_bb:.2f}%)", value=avg_bb, step=0.1)
        div_in = st.number_input(f"Dividend Yield (5Y Avg: {avg_div:.2f}%)", value=avg_div, step=0.1)
        g_in = st.number_input(f"Growth Rate (Latest GDP: {latest_gdp:.2f}%)", value=latest_gdp, step=0.1)

    st.divider()
    if st.button("Calculate WACC", type="primary", use_container_width=True):
        model = DetailWACCModel(
            target_ticker, peers_input, rf_in, crp_in, size_in, 
            bb_in, div_in, g_in, tax_in, df_rf_trend, df_gdp_disp
        )
        with st.spinner("Calculating..."):
            st.session_state['result'] = model.run()
            st.session_state['inputs'] = {
                'rf': rf_in, 'crp': crp_in, 'sp': size_in, 'tax': tax_in,
                'bb': bb_in, 'div': div_in, 'g': g_in,
                'int_exp': int_exp_in, 'ebit': ebit_in, 'category': category_in
            }

if 'result' in st.session_state:
    res = st.session_state['result']
    inp = st.session_state['inputs']
    df_init = res['full_df']
    m = res['market_params']
    
    # 1. Beta & Structure
    results_container = st.container()
    st.subheader("Beta Analysis")
    sens_method = st.radio("Sensitivity Selection (Aggregation Method)", 
                           ["Average", "Median", "Maximum", "Minimum"], horizontal=True, index=1)

    target_relevered_beta=0; ke=0; kd=0; wacc=0; wd=0; we=0; target_de=0; sel_dtic=0

    if res.get('errors'):
        st.error("⚠️ The following peers were excluded due to missing critical data (Strict Validation):")
        for e in res['errors']: st.write(f"- {e}")

    if not df_init.empty:
        user_tax_rates = {}
        for idx, row in df_init.iterrows():
            key = f"tax_{row['Ticker']}"
            if key in st.session_state: user_tax_rates[row['Ticker']] = st.session_state[key]
            else: user_tax_rates[row['Ticker']] = float(row['Tax Rate'])

        calc_df = df_init.copy()
        calc_df["Tax Rate"] = calc_df["Ticker"].map(user_tax_rates)
        calc_df["Unlevered Beta"] = calc_df["Adj Beta"] / (1 + (1 - calc_df["Tax Rate"]/100) * calc_df["D/E Ratio"])
        
        if sens_method == "Average":
            sel_unlev = calc_df["Unlevered Beta"].mean(); sel_dtic = calc_df["Debt/TIC Ratio"].mean()
        elif sens_method == "Median":
            sel_unlev = calc_df["Unlevered Beta"].median(); sel_dtic = calc_df["Debt/TIC Ratio"].median()
        elif sens_method == "Maximum":
            sel_unlev = calc_df["Unlevered Beta"].max(); sel_dtic = calc_df["Debt/TIC Ratio"].max()
        else:
            sel_unlev = calc_df["Unlevered Beta"].min(); sel_dtic = calc_df["Debt/TIC Ratio"].min()
            
        target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
        target_relevered_beta = sel_unlev * (1 + (1 - inp['tax']/100) * target_de)
        calc_df["Re-levered Beta"] = calc_df["Unlevered Beta"] * (1 + (1 - inp['tax']/100) * target_de)
        
        ke = (inp['rf']/100) + (target_relevered_beta * m['MRP']) + (inp['crp']/100) + (inp['sp']/100)
        
        # [LOGIC] Determine Spread from ICR
        int_exp = inp.get('int_exp', 0.0)
        ebit = inp.get('ebit', 0.0)
        category = inp.get('category', "Small/Risky Firms")
        
        icr = ebit / int_exp if int_exp > 0 else 100.0 # High ICR if no interest
        
        # 1. Get Table
        damodaran_dict = get_damodaran_spreads()
        rating_table, _ = damodaran_dict.get(category, (None, ""))
        
        implied_rating = "N/A"
        implied_spread_val = 2.00 # Default fallback
        
        if rating_table is not None:
            for idx, row in rating_table.iterrows():
                try:
                    low_v = float(str(row.get('greater than','-')).replace('greater than','').replace('-','-99999').strip())
                    high_v = float(str(row.get('≤ to','-')).replace('-','99999').strip())
                    if low_v < icr <= high_v:
                        implied_rating = row['Rating']
                        spread_str = str(row['Spread']).replace('%','')
                        implied_spread_val = float(spread_str)
                        break
                except: continue
        
        # 2. Map Rating to OAS
        target_fred_key = "BB US High Yield" # Default
        if "AAA" in implied_rating: target_fred_key = "AAA US Corporate"
        elif "AA" in implied_rating: target_fred_key = "AA US Corporate"
        elif "A" in implied_rating: target_fred_key = "Single-A US Corporate"
        elif "BBB" in implied_rating: target_fred_key = "BBB US Corporate"
        elif "BB" in implied_rating: target_fred_key = "BB US High Yield"
        elif "B" in implied_rating: target_fred_key = "Single-B US High Yield"
        elif "C" in implied_rating: target_fred_key = "CCC & Lower US High Yield"
        
        final_spread = implied_spread_val 
        fred_row = df_oas[df_oas['OAS Name'] == target_fred_key]
        if not fred_row.empty:
            val = fred_row.iloc[0]['Latest Spread (%)']
            if val is not None and not pd.isna(val): final_spread = val
            
        kd = ((inp['rf'] + final_spread)/100) * (1 - inp['tax']/100)
        wd = sel_dtic
        we = 1 - sel_dtic
        wacc = (we * ke) + (wd * kd)

        with st.expander("5-Year Monthly Beta Analysis Table", expanded=True):
            cols_show = ["Ticker", "Company Name", "Country", "Period", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", "Tax Rate", "Raw Beta", "Adj Beta", "Unlevered Beta", "Re-levered Beta"]
            disp_df = calc_df.copy()
            disp_df["Total Debt"] = disp_df.apply(lambda x: f"{x['Currency']} {x['Total Debt']/1e9:,.2f}B", axis=1)
            disp_df["Market Cap"] = disp_df.apply(lambda x: f"{x['Currency']} {x['Market Cap']/1e9:,.2f}B", axis=1)
            
            st.dataframe(disp_df[cols_show], use_container_width=True, hide_index=True,
                column_config={
                    "Tax Rate": st.column_config.NumberColumn("Tax Rate (%)", format="%.2f"),
                    "D/E Ratio": st.column_config.NumberColumn(format="%.3f"),
                    "Debt/TIC Ratio": st.column_config.NumberColumn(format="%.3f"),
                    "Raw Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Adj Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Unlevered Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Re-levered Beta": st.column_config.NumberColumn(format="%.2f"),
                }
            )
            
            st.divider()
            st.markdown("##### Beta Calculation Methodologies")
            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.markdown("**1. Adjusted Beta**"); st.latex(r"\beta_{adj} = 0.67 \cdot \beta_{raw} + 0.33")
            with mc2: st.markdown("**2. Unlevered Beta**"); st.latex(r"\beta_U = \frac{\beta_{adj}}{1 + (1 - T_{peer}) \frac{D}{E}}")
            with mc3: st.markdown("**3. Re-levered Beta**"); st.latex(r"\beta_{re} = \beta_U [1 + (1 - T_{target}) (\frac{D}{E})_{target}]")

            st.divider()
            st.markdown("##### Adjust Peer Tax Rates")
            cols = st.columns(len(df_init))
            for idx, row in df_init.iterrows():
                with cols[idx % len(cols)]:
                    st.number_input(f"{row['Ticker']}", value=user_tax_rates[row['Ticker']], step=0.01, format="%.2f", key=f"tax_{row['Ticker']}")
            st.caption("※ Note: If the headquarter location is not available in the KPMG tax table, a default rate of 25.00% is applied.")
    else:
        st.warning("No valid peer data available for calculation.")

    with results_container:
        st.subheader("WACC Calculation & Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final WACC", f"{wacc:.2%}")
        c2.metric("Cost of Equity", f"{ke:.2%}")
        c3.metric("Cost of Debt (A-T)", f"{kd:.2%}")
        c4.metric("Re-levered Beta", f"{target_relevered_beta:.2f}")
        st.caption(f"**Target Structure ({sens_method}):** Debt {wd:.1%} | Equity {we:.1%} (Implied D/E: {target_de:.2%})")
        
        st.divider()
        with st.expander("👉 WACC Calculation Details (Methodology)", expanded=False):
            ce, cd, cw = st.columns(3)
            with ce:
                st.markdown("**Cost of Equity ($K_e$)**")
                st.latex(r"K_e = R_f + \beta \times (R_m - R_f) + CRP + SP")
                st.info(f"{inp['rf']:.2f}% + {target_relevered_beta:.2f} × {(m['MRP']*100):.2f}% + {inp['crp']:.2f}% + {inp['sp']:.2f}% = **{ke*100:.2f}%**")
            with cd:
                st.markdown("**Cost of Debt ($K_d$)**")
                st.latex(r"K_d = (R_f + \text{Spread}) \times (1 - T_{target})")
                st.info(f"({inp['rf']:.2f}% + {final_spread:.2f}%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**")
            with cw:
                st.markdown("**WACC Weighting**")
                st.latex(r"WACC = K_e \cdot W_e + K_d \cdot W_d")
                st.info(f"{ke*100:.2f}% × {we:.1%} + {kd*100:.2f}% × {wd:.1%} = **{wacc*100:.2f}%**")
        st.markdown("---")

    st.markdown("---")
    st.subheader("Cost of Equity")
    st.latex(r"K_e = R_f + \beta_{L} \times (R_m - R_f) + CRP + SP")
    st.info(f"**Calculation:** {inp['rf']:.2f}% + ({target_relevered_beta:.2f} × {(m['MRP']*100):.2f}%) + {inp['crp']:.2f}% + {inp['sp']:.2f}% = **{ke*100:.2f}%**")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    k2.metric("Beta (Re-levered)", f"{target_relevered_beta:.2f}")
    k3.metric("Market Risk Prem", f"{m['MRP']*100:.2f}%")
    k4.metric("Country Risk Prem", f"{inp['crp']:.2f}%")
    k5.metric("Size Premium", f"{inp['sp']:.2f}%")
    with st.expander("Implied Market Return Details"):
        st.write(f"**Implied Market Return ($R_m$): {m['Rm']:.2%}**")
        st.write(f"= Buyback Yield ({inp['bb']:.2f}%) + Dividend Yield ({inp['div']:.2f}%) + Growth Rate ({inp['g']:.2f}%)")

    st.markdown("---")
    st.subheader("Cost of Debt")
    
    with st.expander("🎯 Target Credit Spread Calculation", expanded=True):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Interest Coverage Ratio", f"{icr:.2f}x")
        sc2.metric("Firm Category", category)
        sc3.metric("Implied Rating", implied_rating)
        sc4.metric("Implied OAS Spread", f"{final_spread:.2f}%", help=f"Mapped to FRED: {target_fred_key}")
        st.caption(f"Based on {category} Table from Damodaran. ICR = EBIT / Interest Exp = {ebit:,.0f} / {int_exp:,.0f}")

    st.latex(r"K_d = (R_f + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
    st.info(f"**Calculation:** ({inp['rf']:.2f}% + {final_spread:.2f}%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    d2.metric("Credit Spread (OAS)", f"{final_spread:.2f}%")
    d3.metric("Pre-tax Cost of Debt", f"{(inp['rf'] + final_spread):.2f}%")
    d4.metric("Tax Rate", f"{inp['tax']:.1f}%")
    d5.metric("After-tax Cost of Debt", f"{kd:.2%}")

    st.markdown("---")
    st.subheader("Peer Group Analysis (Financials)")
    if not df_init.empty:
        fin_cols = ["Ticker", "Company Name", "Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", "Period"]
        fin_df = df_init.copy()
        for c in ["Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap"]: fin_df[c] = fin_df[c] / 1e9 
        st.dataframe(fin_df[fin_cols], use_container_width=True, hide_index=True,
            column_config={
                "Revenue": st.column_config.NumberColumn("Revenue ($B)", format="%.2f"),
                "EBIT": st.column_config.NumberColumn("EBIT ($B)", format="%.2f"),
                "EBITDA": st.column_config.NumberColumn("EBITDA ($B)", format="%.2f"),
                "Total Debt": st.column_config.NumberColumn("Total Debt ($B)", format="%.2f"),
                "Market Cap": st.column_config.NumberColumn("Market Cap ($B)", format="%.2f"),
                "D/E Ratio": st.column_config.NumberColumn(format="%.3f"),
                "Debt/TIC Ratio": st.column_config.NumberColumn(format="%.3f"),
            }
        )
        st.caption("Note: Converted to USD Billions.")
        
        with st.expander("Applied FX Rates Details"):
            st.dataframe(df_init[["Ticker", "Currency", "FX Rate"]].T, use_container_width=True)

    st.markdown("---")
    st.subheader("Market Data Reference")
    t1, t2, t3, t4, t5, t6 = st.tabs(["📉 Risk Free Rate", "📈 US GDP Growth", "📊 S&P 500 Yields", "🏛️ KPMG Corp Tax", "📉 US Corp Spreads", "📊 Damodaran Ratings"])
    with t1:
        st.caption("Source: FRED (St. Louis Fed) - Series DGS10")
        if df_rf_trend is not None: st.line_chart(df_rf_trend.set_index("Date")["Rate"], color="#FF4B4B")
    with t2:
        st.caption("Source: FRED (St. Louis Fed) - Series A191RP1A027NBEA")
        if df_gdp_disp is not None:
            st.dataframe(df_gdp_disp, use_container_width=True, hide_index=True,
                column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"), "GDP Growth %": st.column_config.NumberColumn("GDP Growth (%)", format="%.2f%%")})
    with t3:
        st.caption("Source: Aswath Damodaran (NYU Stern)")
        _, _, sp_table, _ = get_sp_buyback_data()
        if sp_table is not None: st.dataframe(sp_table, use_container_width=True)
    with t4:
        kpmg_df, _, yr = get_kpmg_tax_rates()
        st.caption(f"Source: KPMG (Live Data, {yr} Rates)")
        if kpmg_df is not None: 
            st.dataframe(kpmg_df, use_container_width=True, hide_index=True, column_config={kpmg_df.columns[1]: st.column_config.NumberColumn(format="%.2f%%")})
    with t5:
        st.caption("Source: FRED (St. Louis Fed) - ICE BofA US Corporate Option-Adjusted Spread Data")
        if df_oas is not None and not df_oas.empty:
            st.dataframe(df_oas, use_container_width=True, hide_index=True, 
                         column_config={
                             "Latest Spread (%)": st.column_config.NumberColumn(format="%.2f%%"),
                             "Link": st.column_config.LinkColumn(display_text="View on FRED")
                         })
    with t6:
        damodaran_dict = get_damodaran_spreads()
        source_note = damodaran_dict["Large Firms"][1]
        st.caption(f"{source_note}")
        
        dt1, dt2, dt3 = st.tabs(["🏭 Large Firms", "🚀 Smaller/Risky Firms", "🏦 Financial Firms"])
        
        with dt1:
            df1, _ = damodaran_dict["Large Firms"]
            if df1 is not None: st.dataframe(df1, use_container_width=True, hide_index=True)
            else: st.info("Data not found.")
            
        with dt2:
            df2, _ = damodaran_dict["Small/Risky Firms"]
            if df2 is not None: st.dataframe(df2, use_container_width=True, hide_index=True)
            else: st.info("Data not found.")
            
        with dt3:
            df3, note = damodaran_dict["Financial Firms"]
            if df3 is not None: st.dataframe(df3, use_container_width=True, hide_index=True)
            else: st.info(note)
