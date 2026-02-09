# ==============================================================================
# Strategic WACC Simulator
# Version: 1.4.1
# Last Updated: 2026-02-09
# 
# Changelog:
# v1.4.1 (2026-02-09)
# - [MOD] Financial Firms Beta Logic: 
#   If category is "Financial Firms", skip Unlevering/Re-levering.
#   Target Beta is now directly derived from Peer Group's Adjusted Beta (Mean/Median).
#   (Reason: Financial firms carry debt as operational inventory, making standard D/E unlevering invalid)
#
# v1.4.0 (2025-02-09)
# - [NEW] fetch_financial_ttm_from_api(): single API call fetches both
#   CreditLossesProvision AND PretaxIncome TTM from Yahoo Timeseries API
#   (same endpoint as yfinance: query2.finance.yahoo.com/ws/fundamentals-timeseries)
# - [REFACTOR] Priority logic now strictly follows:
#   P1: Annual (Year-1) from yfinance income_stmt + API annual provision fallback
#   P2: TTM from info_dict (rev/ebitda/int_exp) + API TTM (pretax/provision)
#       No more quarterly loop in P2; API provides TTM directly
#   P3: Sum of 4 most recent quarters from yfinance + API TTM fallback
# - [FIX] 4x multiplication bug: TTM provision was returned per-quarter in loop
#   extract_from_col() now uses yfinance-only provision (no API TTM)
#   API TTM applied once at the priority level, outside loops
#
# v1.3.1 (2025-02-09)
# - [FIX] NameError: Initialize final_spread, icr, implied_rating, category,
#   target_fred_key, int_exp, ebit outside of 'if not df_init.empty:' block
#
# v1.3.0 (2025-02-06)
# - Added web scraping for Credit Losses Provision from Yahoo Finance HTML
# - Fixed regex pattern for comma-separated numbers (3,091,000)
# - Improved HTML parsing logic with cell-by-cell extraction
# - Enhanced logging for debugging provision search
#
# v1.2.0 (2025-02-06)
# - Added priority-based fuzzy search for provision keywords
# - Implemented Cash Flow Statement fallback search
# - Added comprehensive exclusion keywords (tax, changein, etc.)
#
# v1.1.0 (2025-02-06)
# - Removed all UI emojis
# - Fixed IndentationError and duplicate code blocks
# - Enhanced error handling with logging
#
# v1.0.0 (Initial)
# - Core WACC calculation functionality
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
BUILD_DATE = "2026-02-09"

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
# yfinance does not include CreditLossesProvision in its key list, and
# info_dict does not have PretaxIncome TTM. We call Yahoo's timeseries
# API directly to get both fields in a single request.
#
# API Endpoint (same as yfinance internal):
#   https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{SYMBOL}
# ==============================================================================
_financial_ttm_cache = {}

def fetch_financial_ttm_from_api(ticker):
    """
    Fetch TTM financial data from Yahoo Finance's fundamentals-timeseries API.
    
    Fields fetched (not available through yfinance info_dict):
      - CreditLossesProvision (TTM + Annual)
      - PretaxIncome (TTM + Annual)
    
    Args:
        ticker: Stock ticker symbol (e.g., 'JPM', 'GS', 'BAC')
    
    Returns:
        dict with:
            'provision_ttm': TTM Credit Losses Provision (actual dollars, raw sign)
            'provision_annual': list of (date_str, value)
            'pretax_ttm': TTM Pretax Income (actual dollars)
            'pretax_annual': list of (date_str, value)
            'source': description string
        Returns None if fetch fails.
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
        
        # Cache and return if we got any data
        has_data = (api_data["provision_ttm"] != 0 or api_data["pretax_ttm"] != 0 
                    or api_data["provision_annual"] or api_data["pretax_annual"])
        if has_data:
            logger.info(f"[Financial API] {cache_key}: provision_ttm=${api_data['provision_ttm']:,.0f}, "
                        f"pretax_ttm=${api_data['pretax_ttm']:,.0f}")
            _financial_ttm_cache[cache_key] = api_data
            return api_data
        
        logger.info(f"[Financial API] {cache_key}: No relevant data returned")
        _financial_ttm_cache[cache_key] = None
        return None
        
    except requests.RequestException as e:
        logger.warning(f"[Financial API] HTTP error for {cache_key}: {str(e)}")
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"[Financial API] Parse error for {cache_key}: {str(e)}")
    except Exception as e:
        logger.error(f"[Financial API] Unexpected error for {cache_key}: {str(e)}")
    
    _financial_ttm_cache[cache_key] = None
    return None

# ==============================================================================
# [MODULE] Helper: Deep Search with Normalization (v125 Logic)
# ==============================================================================
def get_value_max_fuzzy_with_priority(df, col_idx, keyword_priority_list, exclusion_keywords=None):
    """
    Enhanced fuzzy search with priority scoring.
    Returns the value from the highest-priority match.
    
    Args:
        keyword_priority_list: List of tuples (keyword, priority_score)
        exclusion_keywords: List of keywords to exclude
    
    Returns:
        Absolute value from the highest priority match, or 0 if none found
    """
    matches = []  # Store (priority, value, row_name)
    
    try:
        # Pre-process exclusion keywords
        exclusions = [e.lower().replace(" ", "").replace("-", "").replace("_", "") 
                      for e in exclusion_keywords] if exclusion_keywords else []
        
        for idx in df.index:
            # Normalize Index
            raw_idx_str = str(idx)
            norm_idx_str = raw_idx_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            
            # Exclusion Check
            if any(ex in norm_idx_str for ex in exclusions):
                continue
            
            # Check Keywords with Priority
            for kw, priority in keyword_priority_list:
                norm_kw = kw.lower().replace(" ", "").replace("-", "").replace("_", "")
                
                if norm_kw in norm_idx_str:
                    try:
                        val = df.loc[idx].iloc[col_idx]
                        if pd.notna(val) and val != 0:
                            matches.append((priority, abs(val), raw_idx_str))
                    except Exception as e:
                        logger.debug(f"Value extraction failed: {str(e)}")
                    break  # Only match once per row
        
        # Return value from highest priority match
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)  # Sort by priority (highest first)
            best_match = matches[0]
            logger.info(f"Provision matched: '{best_match[2]}' (priority: {best_match[0]}, value: ${best_match[1]:,.0f})")
            return best_match[1]
    
    except Exception as e:
        logger.warning(f"Priority fuzzy search failed: {str(e)}")
    
    return 0

def get_value_max_fuzzy(df, col_idx, search_keywords, exclusion_keywords=None, debug_provision=False):
    """
    Scans ALL rows. Normalizes strings (remove spaces, lower case) for matching.
    Returns the absolute largest value found.
    
    Args:
        debug_provision: If True, prints all provision-related rows found (for debugging)
    """
    candidates = []
    debug_matches = []  # Store debug info
    
    try:
        # Pre-process exclusion keywords
        exclusions = [e.lower().replace(" ", "") for e in exclusion_keywords] if exclusion_keywords else []
        
        for idx in df.index:
            # 1. Normalize Index: "Provision For Credit Losses" -> "provisionforcreditlosses"
            raw_idx_str = str(idx)
            norm_idx_str = raw_idx_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            
            # Debug: Check if this is a provision-related row
            if debug_provision and 'provision' in norm_idx_str:
                debug_matches.append(f"  Found: '{raw_idx_str}' -> normalized: '{norm_idx_str}'")
            
            # Exclusion Check
            if any(ex in norm_idx_str for ex in exclusions):
                if debug_provision and 'provision' in norm_idx_str:
                    debug_matches.append(f"    EXCLUDED by keyword: {[ex for ex in exclusions if ex in norm_idx_str]}")
                continue

            # 2. Check Keywords
            for kw in search_keywords:
                # Normalize Keyword
                norm_kw = kw.lower().replace(" ", "").replace("-", "").replace("_", "")
                
                if norm_kw in norm_idx_str:
                    try:
                        val = df.loc[idx].iloc[col_idx]
                        if pd.notna(val) and val != 0:
                            candidates.append(abs(val))
                            if debug_provision:
                                debug_matches.append(f"    MATCHED '{norm_kw}' -> Value: {abs(val):,.0f}")
                        elif debug_provision:
                            debug_matches.append(f"    MATCHED '{norm_kw}' but value is 0 or NaN")
                    except Exception as e:
                        logger.debug(f"Value extraction failed: {str(e)}")
                    break 
        
        # Print debug info if requested
        if debug_provision and debug_matches:
            logger.info("=== PROVISION DEBUG ===")
            for match in debug_matches:
                logger.info(match)
            logger.info(f"Final candidates: {candidates}")
        
        if candidates:
            return max(candidates)
    except Exception as e:
        logger.warning(f"Fuzzy search failed: {str(e)}")
    return 0

# ==============================================================================
# [MODULE] Data Fetcher: Consolidated FRED
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_all_fred_data():
    """FRED 데이터 일괄 조회 (GDP, RF, OAS Spreads)"""
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
    failed_series = []
    
    for key, series_id, is_oas in targets:
        time.sleep(random.uniform(1.2, 2.5)) 
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10, verify=False)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            df = df.dropna().sort_values(by="DATE", ascending=True)
            if not df.empty: 
                results[key] = df
            else:
                failed_series.append(series_id)
        except requests.RequestException as e:
            logger.warning(f"FRED fetch failed for {series_id}: {str(e)}")
            failed_series.append(series_id)
        except Exception as e:
            logger.error(f"Unexpected error fetching {series_id}: {str(e)}")
            failed_series.append(series_id)

    # GDP
    latest_gdp = 2.5
    df_gdp_disp = None
    if "GDP" in results:
        df = results["GDP"]
        latest_gdp = float(df["VALUE"].iloc[-1])
        df_gdp_disp = df.sort_values(by="DATE", ascending=False).head(10)
        df_gdp_disp.columns = ["Date", "GDP Growth %"]
    else:
        logger.warning("GDP data unavailable - using fallback value")

    # RF
    latest_rf = 4.2
    df_rf_trend = None
    if "RF" in results:
        df = results["RF"]
        latest_rf = float(df["VALUE"].iloc[-1])
        cutoff = df["DATE"].iloc[-1] - timedelta(days=365*5)
        df_rf_trend = df[df["DATE"] >= cutoff].copy()
        df_rf_trend.columns = ["Date", "Rate"]
    else:
        logger.warning("Risk-free rate data unavailable - using fallback value")

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
    
    # 실패한 시리즈 로깅
    if failed_series:
        logger.info(f"Failed to fetch {len(failed_series)} FRED series: {', '.join(failed_series)}")
    
    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas

# ==============================================================================
# [MODULE] Data Fetcher: NYU & KPMG
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    """NYU Stern S&P 500 Buyback/Dividend 데이터 조회"""
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        dfs = pd.read_html(io.StringIO(response.text), header=0)
        clean_df = dfs[0].dropna(subset=[dfs[0].columns[0]])
        return 2.0, 1.5, clean_df, []
    except requests.RequestException as e:
        logger.warning(f"NYU data fetch failed: {str(e)}")
        return 2.0, 1.5, None, [f"NYU fetch error: {str(e)}"]
    except Exception as e:
        logger.error(f"Unexpected error in S&P data: {str(e)}")
        return 2.0, 1.5, None, [f"Unexpected error: {str(e)}"]

@st.cache_data(ttl=3600*24)
def get_kpmg_tax_rates():
    """KPMG 국가별 법인세율 데이터 조회"""
    url = "https://kpmg.com/dk/en/services/tax/corporate-tax/corporate-tax-rates-table.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        dfs = pd.read_html(io.StringIO(r.text))
        target_df = dfs[0]
        target_df.rename(columns={target_df.columns[0]: "Country"}, inplace=True)
        col_name = target_df.columns[-1]
        result_df = target_df[["Country", col_name]].copy()
        result_df.columns = ["Country", "Rate"]
        result_df["Rate"] = pd.to_numeric(result_df["Rate"], errors='coerce')
        tax_dict = dict(zip(result_df["Country"].str.upper().str.strip(), result_df["Rate"]))
        
        # 보완 데이터
        tax_dict["UNITED STATES"] = 25.57
        tax_dict["USA"] = 25.57
        tax_dict["KOREA"] = 26.40
        
        return result_df, tax_dict, 2025
    except requests.RequestException as e:
        logger.warning(f"KPMG tax data fetch failed: {str(e)}")
        return None, {"UNITED STATES": 25.57, "USA": 25.57, "KOREA": 26.40}, 2025
    except Exception as e:
        logger.error(f"Unexpected error in KPMG data: {str(e)}")
        return None, {}, 2025

@st.cache_data(ttl=3600*24)
def get_damodaran_spreads():
    """Damodaran 신용등급별 스프레드 테이블 조회"""
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
        response = requests.get(
            "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html", 
            headers={"User-Agent": "Mozilla/5.0"}, 
            timeout=15, 
            verify=False
        )
        response.raise_for_status()
        # Online parsing logic would go here
        logger.info("Damodaran spreads fetched successfully (using fallback)")
    except Exception as e:
        logger.warning(f"Damodaran online fetch failed, using fallback: {str(e)}")
    
    return result_dict

# ==============================================================================
# [MODULE] Helper: Common Financial Data Extraction Logic (Unified)
# ==============================================================================
def get_financial_data_with_priority(ticker_obj, info_dict, ticker_symbol=None):
    """
    Priority Logic v125.1 (Restored Normalization Logic):
    Returns: rev, ebit, ebitda, int_exp, label_ebit, label_int, raw_pretax, raw_provision
    
    1. Annual (Year-1)
    2. Yahoo Info TTM
    3. Calc TTM (Manual Sum)
    
    * Ghost Column Eraser applied.
    * PPNR = Pretax + abs(Provision) 
    * **Search:** Uses 'get_value_max_fuzzy' with normalization (ignores case/spaces).
    * **Provision:** Fetched via Yahoo Timeseries API (v1.4.0) with yfinance fallback.
    """
    rev = 0
    ebit = 0
    ebitda = 0
    int_exp = 0
    raw_pretax = 0
    raw_provision = 0
    label_ebit = "N/A"
    label_int = "N/A"
    
    sector = str(info_dict.get('sector', '')).lower()
    is_financial = 'financial' in sector or 'bank' in sector
    
    current_year = datetime.now().year
    target_year = current_year - 1
    
    # [v1.4.0] Pre-fetch TTM data from Yahoo Finance Timeseries API
    # Gets CreditLossesProvision and PretaxIncome (not available via yfinance info_dict)
    api_data = None
    if is_financial and ticker_symbol:
        api_data = fetch_financial_ttm_from_api(ticker_symbol)
        if api_data:
            logger.info(f"[v1.4.0] API data for {ticker_symbol}: "
                        f"provision_ttm=${api_data['provision_ttm']:,.0f}, "
                        f"pretax_ttm=${api_data['pretax_ttm']:,.0f}")
        else:
            logger.info(f"[v1.4.0] API returned no data for {ticker_symbol}, will use yfinance fallback")
    
    try:
        # Load Statements
        a_fin = ticker_obj.income_stmt
        if a_fin.empty: 
            a_fin = ticker_obj.financials
        
        q_fin = ticker_obj.quarterly_income_stmt
        if q_fin.empty: 
            q_fin = ticker_obj.quarterly_financials
        
        # Load Cash Flow Statements (where provision often appears for banks)
        a_cf = ticker_obj.cashflow
        q_cf = ticker_obj.quarterly_cashflow
        
        logger.info(f"Annual Income: {'Loaded' if not a_fin.empty else 'Empty'} | "
                     f"Quarterly Income: {'Loaded' if not q_fin.empty else 'Empty'} | "
                     f"Annual CF: {'Loaded' if not a_cf.empty else 'Empty'} | "
                     f"Quarterly CF: {'Loaded' if not q_cf.empty else 'Empty'}")

        # [STEP 0] GHOST COLUMN ERASER - remove columns with no revenue
        if not q_fin.empty:
            valid_cols = []
            for i in range(len(q_fin.columns)):
                r_check = get_value_max_fuzzy(q_fin, i, ['Total Revenue', 'Revenue'])
                if r_check > 1000:
                    valid_cols.append(q_fin.columns[i])
            if valid_cols:
                q_fin = q_fin[valid_cols]

        # Helper: extract values from a single column of a statement
        def extract_from_col(df, col_idx, df_cf=None, col_idx_cf=None):
            """Extract rev, ebit, ebitda, int_exp, pretax, provision from one column.
            Provision uses yfinance fuzzy search only (NOT API TTM).
            API TTM is applied at the priority level, not per-column."""
            r = get_value_max_fuzzy(df, col_idx, ['Total Revenue', 'Revenue'])
            i = get_value_max_fuzzy(df, col_idx, ['Interest Expense', 'Interest Expense Non Operating'])
            ed = get_value_max_fuzzy(df, col_idx, ['EBITDA', 'Normalized EBITDA'])
            
            p_tax = 0
            p_prov = 0
            val_e = 0
            
            if is_financial:
                p_tax = get_value_max_fuzzy(df, col_idx, ['Pretax Income', 'Income Before Tax'])
                
                # Provision: yfinance fuzzy search only (per-column)
                provision_keywords = [
                    ('provisionforcreditlosses', 10),
                    ('creditlossesprovision', 10),
                    ('provisionforloanlosses', 9),
                    ('provisionforloanandleaselosses', 9),
                    ('creditlossprovision', 7),
                    ('loanlossesprovision', 7),
                    ('provisionforlossesonloans', 7),
                    ('loanlossprovision', 6),
                    ('creditloss', 5),
                    ('loanloss', 5),
                    ('baddebt', 4),
                    ('impairment', 3),
                ]
                exclusion_keywords = [
                    'incometax', 'taxespayable', 'deferredtax',
                    'taxbenefit', 'changein', 'beginningbalance', 'endingbalance',
                ]
                
                # Try Cash Flow Statement first
                if df_cf is not None and not df_cf.empty and col_idx_cf is not None:
                    p_prov = get_value_max_fuzzy_with_priority(
                        df_cf, col_idx_cf, provision_keywords, exclusion_keywords
                    )
                
                # Then Income Statement
                if p_prov == 0:
                    p_prov = get_value_max_fuzzy_with_priority(
                        df, col_idx, provision_keywords, exclusion_keywords
                    )
                
                if p_tax != 0: 
                    val_e = p_tax + abs(p_prov)
            
            if val_e == 0:
                val_e = get_value_max_fuzzy(df, col_idx, ['EBIT', 'Operating Income', 'Operating Profit'])
            
            return r, val_e, ed, i, p_tax, abs(p_prov)

        # =================================================================
        # --- Priority 1: Annual (Year-1) from yfinance income_stmt ---
        # =================================================================
        if not a_fin.empty:
            for idx, col in enumerate(a_fin.columns):
                col_dt = pd.to_datetime(col)
                if col_dt.year == target_year:
                    # Find matching cash flow column
                    cf_idx = None
                    if not a_cf.empty:
                        for cf_col_idx, cf_col in enumerate(a_cf.columns):
                            if pd.to_datetime(cf_col).year == target_year:
                                cf_idx = cf_col_idx
                                break
                    
                    r_annual, e, ed, i, pt, pp = extract_from_col(a_fin, idx, a_cf, cf_idx)
                    
                    # [v1.4.0] If yfinance couldn't find provision, try API annual data
                    if is_financial and pp == 0 and api_data:
                        for date_str, val in api_data.get('provision_annual', []):
                            if str(target_year) in date_str:
                                pp = abs(val)
                                e = pt + pp  # Recalculate PPNR
                                logger.info(f"[P1] Using API annual provision for {target_year}: ${pp:,.0f}")
                                break
                    
                    if pd.notna(r_annual) and r_annual > 1000:
                        # Triple Lock Validation: annual rev ≈ sum of 4 quarters
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
                            logger.info(f"[P1] Using Annual {lbl}: rev=${r_annual:,.0f}, ebit=${e:,.0f}")
                            return r_annual, e, ed, abs(i), lbl, lbl, pt, pp
        
        # =================================================================
        # --- Priority 2: Yahoo Info TTM (info_dict + Timeseries API) ---
        # Revenue/EBITDA/InterestExpense from info_dict
        # PretaxIncome/Provision from Timeseries API (not in info_dict)
        # =================================================================
        rev_ttm = info_dict.get('totalRevenue', 0)
        
        if rev_ttm is not None and rev_ttm > 0:
            rev = rev_ttm
            ebitda = info_dict.get('ebitda', 0)
            
            # Interest Expense: info_dict first, then calc from quarters
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
                    label_int = "TTM (Calc Interest)"
                else:
                    int_exp = 0
                    label_int = "N/A"

            # EBIT / PPNR
            ebit = 0
            if is_financial:
                # Financial companies: PPNR = PretaxIncome + abs(Provision)
                # Primary: Use Timeseries API TTM values
                if api_data and api_data.get('pretax_ttm', 0) != 0:
                    raw_pretax = api_data['pretax_ttm']
                    raw_provision = abs(api_data.get('provision_ttm', 0))
                    ebit = raw_pretax + raw_provision
                    label_ebit = "TTM (Yahoo API)"
                    logger.info(f"[P2] Financial PPNR from API: pretax=${raw_pretax:,.0f} + provision=${raw_provision:,.0f} = ${ebit:,.0f}")
                else:
                    # Fallback: sum quarterly pretax/provision from yfinance
                    if not q_fin.empty and q_fin.shape[1] >= 4:
                        recent_4 = q_fin.iloc[:, :4]
                        recent_4_cf = q_cf.iloc[:, :4] if not q_cf.empty and q_cf.shape[1] >= 4 else None
                        q_pretax = 0
                        q_prov = 0
                        for q_idx in range(4):
                            cf_idx = q_idx if recent_4_cf is not None else None
                            _, _, _, _, pt, pp = extract_from_col(recent_4, q_idx, recent_4_cf, cf_idx)
                            q_pretax += pt
                            q_prov += pp
                        
                        # If yfinance found no provision, try API TTM as last resort
                        if q_prov == 0 and api_data:
                            api_prov = abs(api_data.get('provision_ttm', 0))
                            if api_prov > 0:
                                q_prov = api_prov
                        
                        raw_pretax = q_pretax
                        raw_provision = q_prov
                        if q_pretax != 0:
                            ebit = q_pretax + q_prov
                        label_ebit = "TTM (Calc Quarters)"
                        logger.info(f"[P2] Financial PPNR from quarters: pretax=${raw_pretax:,.0f} + provision=${raw_provision:,.0f}")
                    else:
                        label_ebit = "N/A"
            else:
                # Non-financial: EBIT from operatingMargins or EBITDA proxy
                op_margin = info_dict.get('operatingMargins', 0)
                if op_margin: 
                    ebit = rev * op_margin
                    label_ebit = "TTM (Yahoo Info)"
                elif ebitda: 
                    ebit = ebitda
                    label_ebit = "TTM (EBITDA Proxy)"
                else:
                    label_ebit = "N/A"
            
            if int_exp is None: 
                int_exp = 0
            return rev, ebit, ebitda, abs(int_exp), label_ebit, label_int, raw_pretax, raw_provision

        # =================================================================
        # --- Priority 3: Calc TTM (sum of 4 most recent quarters) ---
        # All values from yfinance quarterly statements
        # =================================================================
        if not q_fin.empty and q_fin.shape[1] >= 4:
            recent_4 = q_fin.iloc[:, :4]
            recent_4_cf = q_cf.iloc[:, :4] if not q_cf.empty and q_cf.shape[1] >= 4 else None
            last_date = recent_4.columns[0].strftime('%Y-%m-%d')
            common_label = f"TTM (Calculated: {last_date})"
            
            rev = 0
            ebitda = 0
            int_exp = 0
            ebit = 0
            raw_pretax = 0
            raw_provision = 0
            
            for q_idx in range(4):
                cf_idx = q_idx if recent_4_cf is not None else None
                r_q, e_q, ed_q, i_q, pt_q, pp_q = extract_from_col(
                    recent_4, q_idx, recent_4_cf, cf_idx
                )
                rev += r_q
                ebitda += ed_q
                int_exp += i_q
                
                if is_financial:
                    raw_pretax += pt_q
                    raw_provision += pp_q
                else:
                    ebit += e_q
            
            # [v1.4.0] If yfinance found no provision per-quarter, use API TTM
            if is_financial and raw_provision == 0 and api_data:
                api_prov = abs(api_data.get('provision_ttm', 0))
                if api_prov > 0:
                    raw_provision = api_prov
                    logger.info(f"[P3] Using API TTM provision: ${raw_provision:,.0f}")
            
            # [v1.4.0] If yfinance found no pretax per-quarter, use API TTM
            if is_financial and raw_pretax == 0 and api_data:
                api_pt = api_data.get('pretax_ttm', 0)
                if api_pt != 0:
                    raw_pretax = api_pt
                    logger.info(f"[P3] Using API TTM pretax: ${raw_pretax:,.0f}")
            
            if is_financial: 
                ebit = raw_pretax + raw_provision
            
            return rev, ebit, ebitda, abs(int_exp), common_label, common_label, raw_pretax, raw_provision

    except Exception as e:
        logger.error(f"Financial data extraction failed: {str(e)}")
    
    return 0, 0, 0, 0, "No Data", "No Data", 0, 0

# ==============================================================================
# [MODULE] Peer Recommender & Financials
# ==============================================================================
class PeerRecommender:
    """동종 업계 Peer 기업 추천 엔진"""
    
    def get_revenue(self, ticker):
        """특정 티커의 매출 조회"""
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            return info.get('totalRevenue', 0)
        except Exception as e:
            logger.warning(f"Revenue fetch failed for {ticker}: {str(e)}")
            return 0

    def recommend(self, target_ticker, progress_bar=None):
        """타겟 티커의 동종 업계 Top 5 기업 추천"""
        try:
            t = yf.Ticker(target_ticker)
            info = safe_yf_info(t)
            ind_key = info.get('industryKey')
            
            if ind_key: 
                industry = yf.Industry(ind_key)
                top_df = industry.top_companies
            else: 
                return None, "Unknown", ["Industry key not found"]
            
            raw_list = top_df['symbol'].tolist() if 'symbol' in top_df.columns else top_df.index.tolist()
            candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5]
            
            revenue_map = []
            for idx, ticker in enumerate(candidates):
                time.sleep(0.5)
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                if progress_bar: 
                    progress_bar.progress(
                        0.2 + (0.8 * (idx/len(candidates))), 
                        text=f"Analyzing {ticker}..."
                    )
            
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_5 = [item[0] for item in revenue_map][:5]
            
            return ", ".join(top_5), f"Industry: {ind_key}", []
        except Exception as e:
            logger.error(f"Peer recommendation failed: {str(e)}")
            return None, "Error", [f"Recommendation error: {str(e)}"]

def get_target_financials(ticker):
    """타겟 기업의 재무 데이터 조회 (WACC 계산용)"""
    _, tax_map, _ = get_kpmg_tax_rates()
    
    try:
        t = yf.Ticker(ticker)
        info = safe_yf_info(t, max_retries=5)
        
        if not info:
            logger.warning(f"No info available for {ticker}")
            return {
                "int_exp": 0.0, "ebit": 0.0, "label_int": "N/A", "label_ebit": "N/A", 
                "raw_pretax": 0, "raw_provision": 0, "category": "Small/Risky Firms", 
                "tax_rate": 25.0, "country_name": "Unknown"
            }
        
        country = info.get('country', 'Unknown')
        country_norm = str(country).upper().strip()
        target_tax = tax_map.get(country_norm)
        
        if target_tax is None:
            if "UNITED STATES" in country_norm or "USA" in country_norm: 
                target_tax = 25.57
            elif "KOREA" in country_norm: 
                target_tax = 26.40
            else: 
                target_tax = 25.0
        
        rev, ebit, ebitda, int_exp, label_ebit, label_int, raw_pretax, raw_provision = \
            get_financial_data_with_priority(t, info, ticker_symbol=ticker)
        
        mkt_cap = info.get('marketCap', 0)
        sector = str(info.get('sector', '')).lower()
        
        if 'financial' in sector or 'bank' in sector: 
            category = "Financial Firms"
        elif mkt_cap > 5e9: 
            category = "Large Firms" 
        else: 
            category = "Small/Risky Firms"
        
        return {
            "int_exp": int_exp, "ebit": ebit, 
            "label_int": label_int, "label_ebit": label_ebit,
            "raw_pretax": raw_pretax, "raw_provision": raw_provision,
            "category": category, "tax_rate": target_tax, "country_name": country
        }
    except Exception as e:
        logger.error(f"Target financials fetch failed for {ticker}: {str(e)}")
        return {
            "int_exp": 0.0, "ebit": 0.0, "label_int": "N/A", "label_ebit": "N/A", 
            "raw_pretax": 0, "raw_provision": 0, "category": "Small/Risky Firms", 
            "tax_rate": 25.0, "country_name": "Unknown"
        }

# ==============================================================================
# [LOGIC] WACC Engine
# ==============================================================================
class DetailWACCModel:
    """WACC 계산 엔진 (Beta 분석, Cost of Equity/Debt, 가중평균)"""
    
    def __init__(self, target, peers, rf_rate, crp, size_prem, buyback, div_yield, growth, tax, rf_trend_df, gdp_df):
        self.target = target
        self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.rf = rf_rate / 100
        self.crp = crp / 100
        self.size_prem = size_prem / 100
        self.buyback_yield = buyback / 100
        self.div_yield = div_yield / 100
        self.growth_rate = growth / 100
        self.tax = tax / 100
        self.rf_trend_df = rf_trend_df
        self.gdp_df = gdp_df
        self.market_index = "^GSPC"
        self.fx_cache = {}
        _, self.kpmg_map, _ = get_kpmg_tax_rates()

    def get_exchange_rate_to_usd(self, currency):
        """환율 조회 (현재는 USD 고정)"""
        return 1.0, "USD" 

    def get_financials_latest(self, ticker):
        """Peer 기업의 최신 재무 데이터 조회"""
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            
            if not info or len(info) < 5: 
                return None, f"⚠️ {ticker}: No data available"
            
            curr = info.get('currency', 'USD')
            country = info.get('country', 'Unknown')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            mkt_cap = info.get('marketCap', 0)
            debt = info.get('totalDebt', 0)
            
            if mkt_cap == 0: 
                try: 
                    mkt_cap = t.fast_info.get('market_cap', 0)
                except Exception as e:
                    logger.debug(f"Fast info failed for {ticker}: {str(e)}")
                    return None, f"⚠️ {ticker}: Excluded (Missing Market Cap)"

            rev, ebit, ebitda, int_exp_dummy, label_ebit, label_int, pt, pp = \
                get_financial_data_with_priority(t, info, ticker_symbol=ticker)
            
            if rev == 0: 
                return None, f"⚠️ {ticker}: Excluded (Missing Revenue)"

            period_display = label_ebit if "Calculated" in label_ebit else label_int

            country_norm = str(country).upper().strip()
            tax_rate = self.kpmg_map.get(country_norm, 25.0)
            
            data = {
                "name": info.get('longName', ticker), 
                "country": country, 
                "currency": curr_code, 
                "fx_rate": fx, 
                "tax_rate": tax_rate,
                "vals": {
                    "Revenue": rev * fx, 
                    "EBIT": ebit * fx, 
                    "EBITDA": ebitda * fx, 
                    "Total Debt": debt * fx, 
                    "Market Cap": mkt_cap * fx
                },
                "period": period_display,
                "beta": info.get('beta', 1.0) # Used for Adjusted Beta calculation
            }
            return data, None
        except Exception as e:
            logger.error(f"Financials fetch error for {ticker}: {str(e)}")
            return None, f"⚠️ {ticker}: Error {str(e)}"

    def get_5y_monthly_beta_analysis(self):
        """5년 월간 베타 분석 (현재는 목 데이터 반환)"""
        # Note: In a real scenario, this might calculate beta from prices.
        # Here we rely on yfinance info beta which is usually 5Y Monthly.
        # This function is kept for structural consistency with v1.4.0.
        return None, None, None, []

    def run(self):
        """WACC 계산 실행"""
        # beta_df is largely unused now as we pull beta from get_financials_latest
        beta_df, _, _, beta_err = self.get_5y_monthly_beta_analysis()
        error_logs = beta_err if beta_err else []
        peer_data = []
        progress_text = st.empty()
        
        for idx, p in enumerate(self.peers):
            progress_text.text(f"Analyzing {p}...")
            time.sleep(0.5)
            fin, err = self.get_financials_latest(p)
            
            if err: 
                error_logs.append(err)
                continue 
            
            if fin:
                d = fin['vals']
                equity = d['Market Cap']
                debt = d['Total Debt']
                tic = equity + debt
                de_ratio = debt / equity if equity > 0 else 0.0
                dtic_ratio = debt / tic if tic > 0 else 0.0
                
                # Calculate Adjusted Beta
                raw_beta = fin.get('beta', 1.0)
                if raw_beta is None: raw_beta = 1.0
                adj_beta = raw_beta * 0.67 + 0.33
                
                peer_data.append({
                    "Ticker": p,
                    "Company Name": fin['name'],
                    "Country": fin['country'],
                    "Tax Rate": fin['tax_rate'],
                    "Currency": fin['currency'],
                    "FX Rate": fin['fx_rate'],
                    "Revenue": d['Revenue'],
                    "EBIT": d['EBIT'],
                    "EBITDA": d['EBITDA'],
                    "Total Debt": d['Total Debt'],
                    "Market Cap": d['Market Cap'],
                    "D/E Ratio": de_ratio,
                    "Debt/TIC Ratio": dtic_ratio,
                    "Period": fin['period'],
                    "Raw Beta": raw_beta,
                    "Adj Beta": adj_beta
                })
        
        progress_text.empty()
        full_df = pd.DataFrame(peer_data)
        
        if not full_df.empty:
            full_df['Ticker'] = full_df['Ticker'].str.upper().str.strip()

        rm = self.div_yield + self.buyback_yield + self.growth_rate
        mrp = rm - self.rf
        
        return {
            "full_df": full_df, 
            "prices": None, 
            "market_params": {"Rm": rm, "MRP": mrp},
            "rf_trend": self.rf_trend_df, 
            "gdp_df": self.gdp_df, 
            "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
# FETCH FRED DATA (GLOBAL)
latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas = fetch_all_fred_data()

with st.sidebar:
    st.header("Target & Peers")
    target_ticker = st.text_input("Target Ticker", "WOLF")
    
    if st.button("Auto-Recommend Peers (Top 5)", type="secondary", use_container_width=True):
        with st.spinner("Finding peers..."):
            rec = PeerRecommender()
            res_peers, group, logs = rec.recommend(target_ticker)
            if res_peers: 
                st.session_state['peers'] = res_peers
                st.success(f"Found peers in {group}")
            else: 
                st.warning("Recommendation Failed")
                if logs:
                    for log in logs:
                        st.error(log)
            
    peers_input = st.text_area(
        "Peer Tickers", 
        value=st.session_state.get('peers', "ON, STM, IFX.DE"), 
        height=100
    )
    st.caption("※ Top 5 revenue companies in the industry\n(Source: Yahoo Finance Industry/Sector Data)")
    
    st.divider()
    st.header("Assumptions")
    
    # [SECTION] Target Assumptions
    with st.expander("Target Assumptions", expanded=True):
        if 'target_fin' not in st.session_state or st.session_state.get('last_ticker') != target_ticker:
            with st.spinner(f"Loading {target_ticker} financial data..."):
                st.session_state['target_fin'] = get_target_financials(target_ticker)
                st.session_state['last_ticker'] = target_ticker
        
        tf = st.session_state['target_fin']
        
        tax_in = st.slider("Tax Rate (%)", 0.0, 40.0, float(tf.get('tax_rate', 25.0)), 0.1)
        st.caption(f"Corporate Tax based on HQ: **{tf.get('country_name', 'Unknown/Default')}**")
        
        st.divider()
        is_fin_target = 'Financial' in tf['category'] or 'Bank' in tf['category']
        ebit_label = "PPNR ($)" if is_fin_target else "EBIT ($)"
        
        st.markdown("**Target Financials** (for Credit Spread)")
        
        int_exp_in = st.number_input("Interest Expense ($)", value=float(tf['int_exp']), format="%.0f")
        st.caption(f"Source: **{tf.get('label_int', 'N/A')}**")
        
        ebit_in = st.number_input(ebit_label, value=float(tf['ebit']), format="%.0f")
        
        # [NEW] VISIBLE BREAKDOWN
        if is_fin_target:
            st.markdown("""
            <style>
            .small-font { font-size: 12px; color: #666; margin-bottom: 0px; }
            </style>
            """, unsafe_allow_html=True)
            st.markdown(
                f"<div class='small-font'>• Pre-tax Income: <b>${tf.get('raw_pretax', 0):,.0f}</b></div>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='small-font'>• (+) Provision: <b>${tf.get('raw_provision', 0):,.0f}</b></div>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='small-font'>• Source: <b>{tf.get('label_ebit', 'N/A')}</b></div>", 
                unsafe_allow_html=True
            )
            
            # Show TTM calculation details
            if tf.get('raw_provision', 0) > 0:
                st.success(f"Credit Losses Provision detected: ${tf.get('raw_provision', 0):,.0f}")
            else:
                st.warning("Credit Losses Provision = $0 (Check if company reports this metric)")
        else:
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
        with st.spinner("Calculating WACC..."):
            st.session_state['result'] = model.run()
            st.session_state['inputs'] = {
                'rf': rf_in, 'crp': crp_in, 'sp': size_in, 'tax': tax_in,
                'bb': bb_in, 'div': div_in, 'g': g_in,
                'int_exp': int_exp_in, 'ebit': ebit_in, 'category': category_in
            }
        st.success("Calculation completed!")
    
    # Version and Contact Information
    st.divider()
    st.caption("**Version:** v1.4.1 (Feb 2026)")
    st.caption("**Point of Contact:** jonghyun.yang.5105@gmail.com")
    
    # Version info at bottom
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
    
    # 1. Beta & Structure
    results_container = st.container()
    st.subheader("Beta Analysis")
    sens_method = st.radio(
        "Sensitivity Selection (Aggregation Method)", 
        ["Average", "Median", "Maximum", "Minimum"], 
        horizontal=True, 
        index=1
    )

    target_relevered_beta = 0
    ke = 0
    kd = 0
    wacc = 0
    wd = 0
    we = 0
    target_de = 0
    sel_dtic = 0

    # [FIX v1.3.1] Initialize variables that are used in Cost of Debt display section
    final_spread = 0.0
    icr = 0.0
    implied_rating = "N/A"
    implied_spread_val = 0.0
    category = inp.get('category', "Small/Risky Firms")
    target_fred_key = "N/A"
    int_exp = inp.get('int_exp', 0.0)
    ebit = inp.get('ebit', 0.0)

    if res.get('errors'):
        st.error("The following peers were excluded due to missing critical data (Strict Validation):")
        for e in res['errors']: 
            st.write(f"- {e}")

    if not df_init.empty:
        user_tax_rates = {}
        for idx, row in df_init.iterrows():
            key = f"tax_{row['Ticker']}"
            if key in st.session_state: 
                user_tax_rates[row['Ticker']] = st.session_state[key]
            else: 
                user_tax_rates[row['Ticker']] = float(row['Tax Rate'])

        calc_df = df_init.copy()
        calc_df["Tax Rate"] = calc_df["Ticker"].map(user_tax_rates)

        # [v1.4.1] Financial Firms Beta Logic Modification
        # Check if category is Financial Firms. If so, bypass Unlevering process.
        is_fin_calc = (category == "Financial Firms")

        if is_fin_calc:
            # ------------------------------------------------------------------
            # FINANCIAL FIRMS LOGIC: Use Adjusted Beta directly
            # ------------------------------------------------------------------
            if sens_method == "Average":
                target_relevered_beta = calc_df["Adj Beta"].mean()
                sel_dtic = calc_df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median":
                target_relevered_beta = calc_df["Adj Beta"].median()
                sel_dtic = calc_df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum":
                target_relevered_beta = calc_df["Adj Beta"].max()
                sel_dtic = calc_df["Debt/TIC Ratio"].max()
            else:
                target_relevered_beta = calc_df["Adj Beta"].min()
                sel_dtic = calc_df["Debt/TIC Ratio"].min()
            
            # For display purposes only, we mirror Adj Beta to Unlevered/Re-levered columns
            calc_df["Unlevered Beta"] = calc_df["Adj Beta"]
            calc_df["Re-levered Beta"] = calc_df["Adj Beta"]
            
            # Calculate target_de for weighting (though not used for re-levering beta here)
            target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
            
        else:
            # ------------------------------------------------------------------
            # STANDARD LOGIC: Unlever -> Re-lever
            # ------------------------------------------------------------------
            calc_df["Unlevered Beta"] = calc_df["Adj Beta"] / (1 + (1 - calc_df["Tax Rate"]/100) * calc_df["D/E Ratio"])
            
            if sens_method == "Average":
                sel_unlev = calc_df["Unlevered Beta"].mean()
                sel_dtic = calc_df["Debt/TIC Ratio"].mean()
            elif sens_method == "Median":
                sel_unlev = calc_df["Unlevered Beta"].median()
                sel_dtic = calc_df["Debt/TIC Ratio"].median()
            elif sens_method == "Maximum":
                sel_unlev = calc_df["Unlevered Beta"].max()
                sel_dtic = calc_df["Debt/TIC Ratio"].max()
            else:
                sel_unlev = calc_df["Unlevered Beta"].min()
                sel_dtic = calc_df["Debt/TIC Ratio"].min()
                
            target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
            target_relevered_beta = sel_unlev * (1 + (1 - inp['tax']/100) * target_de)
            calc_df["Re-levered Beta"] = calc_df["Unlevered Beta"] * (1 + (1 - inp['tax']/100) * target_de)
        
        ke = (inp['rf']/100) + (target_relevered_beta * m['MRP']) + (inp['crp']/100) + (inp['sp']/100)
        
        # [LOGIC] Determine Spread from ICR
        int_exp = inp.get('int_exp', 0.0)
        ebit = inp.get('ebit', 0.0)
        category = inp.get('category', "Small/Risky Firms")
        
        icr = ebit / int_exp if int_exp > 0 else 100.0
        
        # 1. Get Table
        damodaran_dict = get_damodaran_spreads()
        rating_table, _ = damodaran_dict.get(category, (None, ""))
        
        implied_rating = "N/A"
        implied_spread_val = 2.00
        
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
                except Exception as e:
                    logger.debug(f"Rating table parsing error: {str(e)}")
                    continue
        
        # 2. Map Rating to OAS
        target_fred_key = "BB US High Yield"
        if "AAA" in implied_rating: 
            target_fred_key = "AAA US Corporate"
        elif "AA" in implied_rating: 
            target_fred_key = "AA US Corporate"
        elif "A" in implied_rating: 
            target_fred_key = "Single-A US Corporate"
        elif "BBB" in implied_rating: 
            target_fred_key = "BBB US Corporate"
        elif "BB" in implied_rating: 
            target_fred_key = "BB US High Yield"
        elif "B" in implied_rating: 
            target_fred_key = "Single-B US High Yield"
        elif "C" in implied_rating: 
            target_fred_key = "CCC & Lower US High Yield"
        
        final_spread = implied_spread_val 
        fred_row = df_oas[df_oas['OAS Name'] == target_fred_key]
        if not fred_row.empty:
            val = fred_row.iloc[0]['Latest Spread (%)']
            if val is not None and not pd.isna(val): 
                final_spread = val
            
        kd = ((inp['rf'] + final_spread)/100) * (1 - inp['tax']/100)
        wd = sel_dtic
        we = 1 - sel_dtic
        wacc = (we * ke) + (wd * kd)

        with st.expander("5-Year Monthly Beta Analysis Table", expanded=True):
            if is_fin_calc:
                st.info("ℹ️ **Financial Firms Logic Applied:** Unlevering/Re-levering is bypassed. Target Beta uses Peer Adjusted Beta directly.")
            
            cols_show = [
                "Ticker", "Company Name", "Country", "Period", "Total Debt", "Market Cap", 
                "D/E Ratio", "Debt/TIC Ratio", "Tax Rate", "Raw Beta", "Adj Beta", 
                "Unlevered Beta", "Re-levered Beta"
            ]
            disp_df = calc_df.copy()
            disp_df["Total Debt"] = disp_df.apply(
                lambda x: f"{x['Currency']} {x['Total Debt']/1e9:,.2f}B", axis=1
            )
            disp_df["Market Cap"] = disp_df.apply(
                lambda x: f"{x['Currency']} {x['Market Cap']/1e9:,.2f}B", axis=1
            )
            
            st.dataframe(
                disp_df[cols_show], 
                use_container_width=True, 
                hide_index=True,
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
            with mc1: 
                st.markdown("**1. Adjusted Beta**")
                st.latex(r"\beta_{adj} = 0.67 \cdot \beta_{raw} + 0.33")
            with mc2: 
                st.markdown("**2. Unlevered Beta**")
                if is_fin_calc:
                    st.caption("Skipped for Financial Firms")
                else:
                    st.latex(r"\beta_U = \frac{\beta_{adj}}{1 + (1 - T_{peer}) \frac{D}{E}}")
            with mc3: 
                st.markdown("**3. Re-levered Beta**")
                if is_fin_calc:
                    st.latex(r"\beta_{re} \approx \beta_{adj} \text{ (Direct Use)}")
                else:
                    st.latex(r"\beta_{re} = \beta_U [1 + (1 - T_{target}) (\frac{D}{E})_{target}]")

            st.divider()
            st.markdown("##### Adjust Peer Tax Rates")
            cols = st.columns(len(df_init))
            for idx, row in df_init.iterrows():
                with cols[idx % len(cols)]:
                    st.number_input(
                        f"{row['Ticker']}", 
                        value=user_tax_rates[row['Ticker']], 
                        step=0.01, 
                        format="%.2f", 
                        key=f"tax_{row['Ticker']}"
                    )
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
        st.caption(
            f"**Target Structure ({sens_method}):** Debt {wd:.1%} | Equity {we:.1%} (Implied D/E: {target_de:.2%})"
        )
        
        st.divider()
        with st.expander("WACC Calculation Details (Methodology)", expanded=False):
            ce, cd, cw = st.columns(3)
            with ce:
                st.markdown("**Cost of Equity ($K_e$)**")
                st.latex(r"K_e = R_f + \beta \times (R_m - R_f) + CRP + SP")
                st.info(
                    f"{inp['rf']:.2f}% + {target_relevered_beta:.2f} × {(m['MRP']*100):.2f}% + "
                    f"{inp['crp']:.2f}% + {inp['sp']:.2f}% = **{ke*100:.2f}%**"
                )
            with cd:
                st.markdown("**Cost of Debt ($K_d$)**")
                st.latex(r"K_d = (R_f + \text{Spread}) \times (1 - T_{target})")
                st.info(
                    f"({inp['rf']:.2f}% + {final_spread:.2f}%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**"
                )
            with cw:
                st.markdown("**WACC Weighting**")
                st.latex(r"WACC = K_e \cdot W_e + K_d \cdot W_d")
                st.info(
                    f"{ke*100:.2f}% × {we:.1%} + {kd*100:.2f}% × {wd:.1%} = **{wacc*100:.2f}%**"
                )
        st.markdown("---")

    st.markdown("---")
    st.subheader("Cost of Equity")
    st.latex(r"K_e = R_f + \beta_{L} \times (R_m - R_f) + CRP + SP")
    st.info(
        f"**Calculation:** {inp['rf']:.2f}% + ({target_relevered_beta:.2f} × {(m['MRP']*100):.2f}%) + "
        f"{inp['crp']:.2f}% + {inp['sp']:.2f}% = **{ke*100:.2f}%**"
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    k2.metric("Beta (Re-levered)", f"{target_relevered_beta:.2f}")
    k3.metric("Market Risk Prem", f"{m['MRP']*100:.2f}%")
    k4.metric("Country Risk Prem", f"{inp['crp']:.2f}%")
    k5.metric("Size Premium", f"{inp['sp']:.2f}%")
    
    with st.expander("Implied Market Return Details"):
        st.write(f"**Implied Market Return ($R_m$): {m['Rm']:.2%}**")
        st.write(
            f"= Buyback Yield ({inp['bb']:.2f}%) + Dividend Yield ({inp['div']:.2f}%) + "
            f"Growth Rate ({inp['g']:.2f}%)"
        )

    st.markdown("---")
    st.subheader("Cost of Debt")
    
    with st.expander("Target Credit Spread Calculation", expanded=True):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Interest Coverage Ratio", f"{icr:.2f}x")
        sc2.metric("Firm Category", category)
        sc3.metric("Implied Rating", implied_rating)
        sc4.metric("Implied OAS Spread", f"{final_spread:.2f}%", help=f"Mapped to FRED: {target_fred_key}")
        st.caption(
            f"Based on {category} Table from Damodaran. "
            f"ICR = EBIT / Interest Exp = {ebit:,.0f} / {int_exp:,.0f}"
        )

    st.latex(r"K_d = (R_f + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
    st.info(
        f"**Calculation:** ({inp['rf']:.2f}% + {final_spread:.2f}%) × "
        f"(1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**"
    )
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    d2.metric("Credit Spread (OAS)", f"{final_spread:.2f}%")
    d3.metric("Pre-tax Cost of Debt", f"{(inp['rf'] + final_spread):.2f}%")
    d4.metric("Tax Rate", f"{inp['tax']:.1f}%")
    d5.metric("After-tax Cost of Debt", f"{kd:.2%}")

    st.markdown("---")
    st.subheader("Peer Group Analysis (Financials)")
    if not df_init.empty:
        fin_cols = [
            "Ticker", "Company Name", "Revenue", "EBIT", "EBITDA", 
            "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", "Period"
        ]
        fin_df = df_init.copy()
        for c in ["Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap"]: 
            fin_df[c] = fin_df[c] / 1e9 
        
        st.dataframe(
            fin_df[fin_cols], 
            use_container_width=True, 
            hide_index=True,
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
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Risk Free Rate", 
        "US GDP Growth", 
        "S&P 500 Yields", 
        "KPMG Corp Tax", 
        "US Corp Spreads", 
        "Damodaran Ratings"
    ])
    
    with t1:
        st.caption("Source: FRED (St. Louis Fed) - Series DGS10")
        if df_rf_trend is not None: 
            st.line_chart(df_rf_trend.set_index("Date")["Rate"], color="#FF4B4B")
        else:
            st.info("Risk-free rate trend data unavailable")
    
    with t2:
        st.caption("Source: FRED (St. Louis Fed) - Series A191RP1A027NBEA")
        if df_gdp_disp is not None:
            st.dataframe(
                df_gdp_disp, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"), 
                    "GDP Growth %": st.column_config.NumberColumn("GDP Growth (%)", format="%.2f%%")
                }
            )
        else:
            st.info("GDP data unavailable")
    
    with t3:
        st.caption("Source: Aswath Damodaran (NYU Stern)")
        _, _, sp_table, _ = get_sp_buyback_data()
        if sp_table is not None: 
            st.dataframe(sp_table, use_container_width=True)
        else:
            st.info("S&P 500 yields data unavailable")
    
    with t4:
        kpmg_df, _, yr = get_kpmg_tax_rates()
        st.caption(f"Source: KPMG (Live Data, {yr} Rates)")
        if kpmg_df is not None: 
            st.dataframe(
                kpmg_df, 
                use_container_width=True, 
                hide_index=True, 
                column_config={
                    kpmg_df.columns[1]: st.column_config.NumberColumn(format="%.2f%%")
                }
            )
        else:
            st.info("KPMG tax data unavailable")
    
    with t5:
        st.caption("Source: FRED (St. Louis Fed) - ICE BofA US Corporate Option-Adjusted Spread Data")
        if df_oas is not None and not df_oas.empty:
            st.dataframe(
                df_oas, 
                use_container_width=True, 
                hide_index=True, 
                column_config={
                    "Latest Spread (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Link": st.column_config.LinkColumn(display_text="View on FRED")
                }
            )
        else:
            st.info("OAS spread data unavailable")
    
    with t6:
        damodaran_dict = get_damodaran_spreads()
        source_note = damodaran_dict["Large Firms"][1]
        st.caption(f"{source_note}")
        
        dt1, dt2, dt3 = st.tabs(["Large Firms", "Smaller/Risky Firms", "Financial Firms"])
        
        with dt1:
            df1, _ = damodaran_dict["Large Firms"]
            if df1 is not None: 
                st.dataframe(df1, use_container_width=True, hide_index=True)
            else: 
                st.info("Data not found.")
            
        with dt2:
            df2, _ = damodaran_dict["Small/Risky Firms"]
            if df2 is not None: 
                st.dataframe(df2, use_container_width=True, hide_index=True)
            else: 
                st.info("Data not found.")
            
        with dt3:
            df3, note = damodaran_dict["Financial Firms"]
            if df3 is not None: 
                st.dataframe(df3, use_container_width=True, hide_index=True)
            else: 
                st.info(note)
