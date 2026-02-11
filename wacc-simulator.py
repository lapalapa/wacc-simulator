# ==============================================================================
# Strategic WACC Simulator
# Version: 1.5.0
# Last Updated: 2025-02-09
# 
# Changelog:
# v1.5.0 (2025-02-09)
# - [NEW] 3-tier company profile fallback (fetch_company_profile_from_api):
#   Phase 1: quoteSummary API (v10 JSON)
#   Phase 2: Profile page HTML scraping (embedded JSON + text parsing)
#   Phase 3: Quote page HTML scraping (last resort)
# - [NEW] _extract_json_from_html(): root.App.main JSON + regex fragment extraction
# - [NEW] _scrape_profile_from_text(): "headquartered in..." pattern with US state detection
# - [FIX] safe_yf_info() now calls fetch_company_profile_from_api() when .info fails
# - [FIX] Peer get_financials_latest(): no longer early-returns on empty info;
#   tries fast_info + balance_sheet fallback for mkt_cap and debt
# - [FIX] Target get_target_financials(): info={} fallback instead of early return
#
# v1.4.0 (2025-02-09)
# - [NEW] fetch_financial_ttm_from_api(): single API call fetches both
#   CreditLossesProvision AND PretaxIncome TTM from Yahoo Timeseries API
# - [REFACTOR] Priority logic: P1 Annual -> P2 TTM (API) -> P3 Quarterly Sum
# - [FIX] 4x multiplication bug in TTM provision
#
# v1.3.1 (2025-02-09)
# - [FIX] NameError: Initialize variables outside 'if not df_init.empty:' block
#
# v1.3.0 (2025-02-06)
# - Added web scraping for Credit Losses Provision from Yahoo Finance HTML
#
# v1.2.0 (2025-02-06)
# - Added priority-based fuzzy search for provision keywords
#
# v1.1.0 (2025-02-06)
# - Removed all UI emojis, Fixed IndentationError
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
VERSION = "1.5.0"
BUILD_DATE = "2025-02-09"

# ==============================================================================
# [MODULE] Helper: Safe Fetcher with Retry
# ==============================================================================
def safe_yf_info(ticker_obj, max_retries=3):
    """안전한 Yahoo Finance 정보 조회 (재시도 + API fallback + 필드 보강)
    
    Strategy:
      1. Try yfinance .info (standard path) up to max_retries
      2. If .info fails entirely, call Yahoo API fallback
      3. If .info succeeds but CRITICAL fields (country, sector) are missing,
         call API fallback and MERGE results (API fills gaps, .info takes priority)
    """
    CRITICAL_FIELDS = ['country', 'sector']
    
    info = {}
    
    # Phase 1: Try yfinance .info
    for i in range(max_retries):
        try:
            raw_info = ticker_obj.info
            if raw_info and len(raw_info) > 5:
                info = raw_info
                break
        except Exception as e:
            if i == max_retries - 1:
                logger.warning(f"Failed to fetch .info after {max_retries} attempts: {str(e)}")
        time.sleep(random.uniform(0.5, 1.5))
    
    # Check if critical fields are present
    has_critical = all(info.get(f) for f in CRITICAL_FIELDS)
    
    if has_critical:
        return info
    
    # Phase 2: API fallback (either .info failed entirely OR critical fields missing)
    ticker_str = getattr(ticker_obj, 'ticker', '')
    if ticker_str:
        reason = "no .info data" if not info else f"missing {[f for f in CRITICAL_FIELDS if not info.get(f)]}"
        logger.info(f"[safe_yf_info] {ticker_str}: {reason} -> trying API fallback...")
        
        api_info = fetch_company_profile_from_api(ticker_str)
        
        if api_info:
            # Filter out invalid country from API (e.g., "Q1FY26")
            if api_info.get("country") and not _is_valid_country(api_info["country"]):
                logger.warning(f"[safe_yf_info] {ticker_str}: Rejected invalid API country: '{api_info['country']}'")
                api_info["country"] = ""
            
            if info:
                # Merge: API fills gaps, existing .info values take priority
                merged = {**api_info, **{k: v for k, v in info.items() if v is not None and v != '' and v != 0}}
                logger.info(f"[safe_yf_info] {ticker_str}: Merged .info ({len(info)} keys) + API ({len(api_info)} keys) -> {len(merged)} keys")
                logger.info(f"[safe_yf_info] {ticker_str}: country={merged.get('country','N/A')}, sector={merged.get('sector','N/A')}")
                return merged
            elif len(api_info) > 3:
                logger.info(f"[safe_yf_info] {ticker_str}: API fallback only ({len(api_info)} keys)")
                return api_info
    
    return info if info else {}

# ==============================================================================
# [MODULE] Helper: Yahoo Finance Company Profile Fetcher (3-tier fallback)
# ==============================================================================
# When yfinance .info fails (403, timeout, empty), we try multiple methods:
#   1. quoteSummary API (v10) - direct JSON API
#   2. Profile page HTML scraping with lxml XPath (user-provided XPath)  
#   3. Profile page text/regex parsing (fallback)
# ==============================================================================
_profile_cache = {}
_financial_ttm_cache = {}

# Known countries for validation (used by all profile parsers)
VALID_COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Germany", "France", "Japan",
    "China", "South Korea", "Taiwan", "Netherlands", "Switzerland", "Sweden",
    "Ireland", "Israel", "India", "Australia", "Brazil", "Mexico", "Singapore",
    "Hong Kong", "Italy", "Spain", "Belgium", "Denmark", "Norway", "Finland",
    "Austria", "Luxembourg", "Portugal", "South Africa", "New Zealand",
    "Thailand", "Indonesia", "Malaysia", "Philippines", "Vietnam",
    "Saudi Arabia", "United Arab Emirates", "Turkey", "Russia", "Poland",
    "Czech Republic", "Hungary", "Greece", "Argentina", "Chile", "Colombia",
    "Peru", "Egypt", "Nigeria", "Kenya", "Pakistan", "Bangladesh",
    "Bermuda", "Cayman Islands", "British Virgin Islands", "Jersey", "Guernsey",
    "Isle of Man", "Puerto Rico", "Curacao", "Monaco", "Liechtenstein",
}

def _is_valid_country(text):
    """Validate that text looks like a country name, not garbage like 'Q1FY26'."""
    if not text or len(text) > 50 or len(text) < 3:
        return False
    if text.isdigit():
        return False
    # Reject fiscal period patterns (Q1FY26, FY2025, H1, CY2025)
    if re.match(r'^(Q\d|FY|H\d|CY)', text, re.IGNORECASE):
        return False
    # Reject if contains digits mixed with short text (like "Q1FY26", "2025-06-30")
    if any(c.isdigit() for c in text):
        return False
    # Check against known countries (case-insensitive)
    text_lower = text.lower().strip()
    for c in VALID_COUNTRIES:
        if c.lower() == text_lower:
            return True
    # Allow if all alpha/spaces and title-like (potential unknown country)
    if all(c.isalpha() or c.isspace() for c in text) and text[0].isupper() and len(text) > 3:
        return True
    return False

def _parse_profile_with_xpath(html_text, ticker):
    """
    Yahoo Finance profile 페이지 HTML을 lxml XPath로 파싱하여 company 정보 추출.
    
    Yahoo Finance profile page 구조 (2025-2026):
      /html/body/div[1]/div[4]/main/section/section/section/section/section[2]/section[2]/div/div/div/
        div[1] = 주소 1줄 (e.g., "4600 Silicon Drive")
        div[2] = 시/주/ZIP (e.g., "Durham, NC 27703")  
        div[3] = 국가 (e.g., "United States")  <-- user-provided XPath
        
      Sector/Industry는 같은 section 내 또는 nearby elements에 텍스트로 존재.
    
    Returns: dict with country, sector, industry, etc. or {} if parsing fails.
    """
    try:
        from lxml import etree
    except ImportError:
        logger.warning("[XPath] lxml not installed - skipping XPath parsing")
        return {}
    
    result = {}
    
    try:
        # Parse HTML (lxml.html supports text_content())
        from lxml.html import fromstring as html_fromstring
        tree = html_fromstring(html_text)
        
        # ==================================================================
        # XPath Strategy 1: Exact user-provided path for Country
        # /html/body/div[1]/div[4]/main/section/section/section/section/section[2]/section[2]/div/div/div/div[3]
        # ==================================================================
        country_xpaths = [
            # User-provided exact XPath
            '/html/body/div[1]/div[4]/main/section/section/section/section/section[2]/section[2]/div/div/div/div[3]',
            # Variations (Yahoo may adjust nesting)
            '/html/body/div[1]/div[3]/main/section/section/section/section/section[2]/section[2]/div/div/div/div[3]',
            '/html/body/div[1]/div[5]/main/section/section/section/section/section[2]/section[2]/div/div/div/div[3]',
            # More flexible: find div[3] inside address-like block under main
            '//main//section//section[2]//div/div/div/div[3]',
        ]
        
        for xpath in country_xpaths:
            try:
                elements = tree.xpath(xpath)
                if elements:
                    country_text = elements[0].text_content().strip()
                    # Validate it looks like a country name (not "Q1FY26" etc.)
                    if _is_valid_country(country_text):
                        result["country"] = country_text
                        logger.info(f"[XPath] {ticker}: Country='{country_text}' via XPath: {xpath}")
                        break
            except Exception:
                continue
        
        # ==================================================================
        # XPath Strategy 2: Address block - also get city/state from div[2]
        # ==================================================================
        addr_xpaths = [
            '/html/body/div[1]/div[4]/main/section/section/section/section/section[2]/section[2]/div/div/div/div[2]',
            '//main//section//section[2]//div/div/div/div[2]',
        ]
        for xpath in addr_xpaths:
            try:
                elements = tree.xpath(xpath)
                if elements:
                    addr_text = elements[0].text_content().strip()
                    if addr_text:
                        logger.info(f"[XPath] {ticker}: Address line='{addr_text}'")
                    break
            except Exception:
                continue
        
        # ==================================================================
        # XPath Strategy 3: Sector & Industry
        # Usually in nearby elements: look for text containing "Sector" and "Industry"
        # These appear as labels in the profile sidebar
        # ==================================================================
        
        # Method A: Search all text nodes for "Sector :" / "Industry :" patterns
        all_text = tree.xpath('//text()')
        for i, t in enumerate(all_text):
            t_stripped = t.strip()
            if t_stripped == 'Sector':
                # Next non-empty text should be the sector value
                for j in range(i+1, min(i+5, len(all_text))):
                    val = all_text[j].strip()
                    if val and val != ':' and val != 'Sector':
                        result["sector"] = val
                        logger.info(f"[XPath] {ticker}: Sector='{val}'")
                        break
            elif t_stripped == 'Industry':
                for j in range(i+1, min(i+5, len(all_text))):
                    val = all_text[j].strip()
                    if val and val != ':' and val != 'Industry':
                        result["industry"] = val
                        logger.info(f"[XPath] {ticker}: Industry='{val}'")
                        break
        
        # Method B: XPath for common sector/industry containers
        if not result.get("sector"):
            sector_xpaths = [
                '//a[contains(@href, "/sectors/")]//text()',
                '//span[contains(text(),"Sector")]/following-sibling::*//text()',
                '//dt[contains(text(),"Sector")]/following-sibling::dd//text()',
            ]
            for xpath in sector_xpaths:
                try:
                    vals = tree.xpath(xpath)
                    for v in vals:
                        v_s = v.strip()
                        if v_s and v_s != 'Sector' and len(v_s) < 40:
                            result["sector"] = v_s
                            logger.info(f"[XPath] {ticker}: Sector='{v_s}' (href/sibling)")
                            break
                except Exception:
                    continue
                if result.get("sector"):
                    break
        
        if not result.get("industry"):
            industry_xpaths = [
                '//a[contains(@href, "/industries/")]//text()',
                '//span[contains(text(),"Industry")]/following-sibling::*//text()',
                '//dt[contains(text(),"Industry")]/following-sibling::dd//text()',
            ]
            for xpath in industry_xpaths:
                try:
                    vals = tree.xpath(xpath)
                    for v in vals:
                        v_s = v.strip()
                        if v_s and v_s != 'Industry' and len(v_s) < 60:
                            result["industry"] = v_s
                            logger.info(f"[XPath] {ticker}: Industry='{v_s}' (href/sibling)")
                            break
                except Exception:
                    continue
                if result.get("industry"):
                    break
        
        # ==================================================================
        # XPath Strategy 4: Company name from <h1> or title
        # ==================================================================
        if not result.get("longName"):
            name_xpaths = [
                '//h1//text()',
                '//title//text()',
            ]
            for xpath in name_xpaths:
                try:
                    vals = tree.xpath(xpath)
                    for v in vals:
                        v_s = v.strip()
                        if v_s and len(v_s) > 3 and ticker.upper() in v_s.upper():
                            # Clean up: "Wolfspeed, Inc. (WOLF) Company Profile" -> "Wolfspeed, Inc."
                            name = re.sub(r'\s*\(' + re.escape(ticker.upper()) + r'\).*', '', v_s).strip()
                            if name:
                                result["longName"] = name
                                break
                except Exception:
                    continue
                if result.get("longName"):
                    break
        
        n_found = len([v for v in result.values() if v])
        if n_found > 0:
            logger.info(f"[XPath] {ticker}: Total {n_found} fields extracted")
        
    except Exception as e:
        logger.warning(f"[XPath] {ticker}: Parse failed - {str(e)}")
    
    return result


def _parse_profile_with_regex(html_text, ticker):
    """
    Regex/text-based fallback for extracting profile info from Yahoo Finance HTML.
    Used when lxml XPath fails or returns incomplete data.
    """
    result = {}
    
    # --- Embedded JSON extraction ---
    import json as _json
    
    # Pattern 1: root.App.main (legacy)
    pat1 = re.compile(r'root\.App\.main\s*=\s*(\{.*?\})\s*;\s*\n', re.DOTALL)
    m = pat1.search(html_text)
    if m:
        try:
            data = _json.loads(m.group(1))
            qss = data.get("context", {}).get("dispatcher", {}).get("stores", {}).get("QuoteSummaryStore", {})
            if qss:
                profile = qss.get("assetProfile", {})
                price = qss.get("price", {})
                if profile.get("country"):
                    result["country"] = profile["country"]
                if profile.get("sector"):
                    result["sector"] = profile["sector"]
                if profile.get("industry"):
                    result["industry"] = profile["industry"]
                if price.get("currency"):
                    result["currency"] = price["currency"]
                if price.get("longName"):
                    result["longName"] = price["longName"]
                mc = price.get("marketCap", {})
                if isinstance(mc, dict) and mc.get("raw"):
                    result["marketCap"] = mc["raw"]
                
                if result.get("country"):
                    logger.info(f"[Regex] {ticker}: Got {len(result)} fields via root.App.main JSON")
                    return result
        except Exception:
            pass
    
    # Pattern 2: Direct JSON fragments
    json_patterns = {
        "country": re.compile(r'"country"\s*:\s*"([^"]+)"'),
        "sector": re.compile(r'"sector"\s*:\s*"([^"]+)"'),
        "industry": re.compile(r'"industry"\s*:\s*"([^"]+)"'),
        "currency": re.compile(r'"currency"\s*:\s*"([A-Z]{3})"'),
        "longName": re.compile(r'"longName"\s*:\s*"([^"]+)"'),
    }
    
    for field, pat in json_patterns.items():
        if not result.get(field):
            m = pat.search(html_text)
            if m:
                val = m.group(1)
                # Validate country values
                if field == "country" and not _is_valid_country(val):
                    logger.debug(f"[Regex] Rejected invalid country: '{val}'")
                    continue
                result[field] = val
    
    # Pattern 3: "headquartered in" text
    if not result.get("country"):
        hq_match = re.search(r'(?:headquartered|based)\s+in\s+([^.]+?)\.', html_text, re.IGNORECASE)
        if hq_match:
            hq_str = hq_match.group(1).strip()
            parts = [p.strip() for p in hq_str.split(",")]
            if len(parts) >= 2:
                last = parts[-1].strip()
                us_states = {
                    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
                    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
                    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
                    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire",
                    "New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio",
                    "Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota",
                    "Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
                    "Wisconsin","Wyoming","District of Columbia"
                }
                result["country"] = "United States" if last in us_states else last
                logger.info(f"[Regex] {ticker}: HQ='{hq_str}' -> country='{result['country']}'")
    
    if result:
        logger.info(f"[Regex] {ticker}: Got {len([v for v in result.values() if v])} fields via regex")
    
    return result


def fetch_company_profile_from_api(ticker):
    """
    Yahoo Finance에서 회사 프로필 (country, sector, marketCap 등) 조회.
    
    3단계 fallback:
      1. quoteSummary API (v10) - 가장 빠르고 정확한 JSON API
      2. Profile page XPath parsing (lxml) - user-provided XPath로 DOM 파싱
      3. Profile page regex/text parsing - embedded JSON + "headquartered in" 패턴
    
    Returns: dict matching yfinance .info format, or {} if all fail.
    """
    cache_key = ticker.upper()
    if cache_key in _profile_cache:
        return _profile_cache[cache_key]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    result = {}
    
    # =========================================================================
    # Phase 1: quoteSummary API (v10 JSON endpoint)
    # =========================================================================
    try:
        modules = "assetProfile,defaultKeyStatistics,financialData,price"
        api_url = (
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{cache_key}"
            f"?modules={modules}"
        )
        
        logger.info(f"[Profile] Phase 1: quoteSummary API for {cache_key}...")
        r = requests.get(api_url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        
        summary = data.get("quoteSummary", {}).get("result", [])
        if summary:
            item = summary[0]
            
            profile = item.get("assetProfile", {})
            if profile:
                _c = profile.get("country", "")
                result["country"] = _c if _is_valid_country(_c) else ""
                result["sector"] = profile.get("sector", "")
                result["industry"] = profile.get("industry", "")
                result["industryKey"] = profile.get("industryKey", "")
            
            price_data = item.get("price", {})
            if price_data:
                result["currency"] = price_data.get("currency", "USD")
                result["longName"] = price_data.get("longName", cache_key)
                mc = price_data.get("marketCap", {})
                result["marketCap"] = mc.get("raw", 0) if isinstance(mc, dict) else (mc or 0)
            
            fin_data = item.get("financialData", {})
            if fin_data:
                for key in ["totalRevenue","ebitda","totalDebt","operatingMargins",
                            "interestExpense","totalCash","currentRatio"]:
                    val = fin_data.get(key, {})
                    result[key] = val.get("raw", 0) if isinstance(val, dict) else (val or 0)
            
            key_stats = item.get("defaultKeyStatistics", {})
            if key_stats:
                if not result.get("marketCap"):
                    mc2 = key_stats.get("enterpriseValue", {})
                    result["marketCap"] = mc2.get("raw", 0) if isinstance(mc2, dict) else (mc2 or 0)
                shares = key_stats.get("sharesOutstanding", {})
                result["sharesOutstanding"] = shares.get("raw", 0) if isinstance(shares, dict) else (shares or 0)
            
            if result.get("country"):
                logger.info(f"[Profile] Phase 1 SUCCESS: {cache_key} country={result['country']}")
                _profile_cache[cache_key] = result
                return result
    except Exception as e:
        logger.warning(f"[Profile] Phase 1 failed for {cache_key}: {str(e)}")
    
    time.sleep(random.uniform(0.5, 1.0))
    
    # =========================================================================
    # Phase 2 & 3: Profile page HTML scraping (XPath + Regex fallback)
    # =========================================================================
    profile_urls = [
        f"https://finance.yahoo.com/quote/{cache_key}/profile/",
        f"https://finance.yahoo.com/quote/{cache_key}/",
    ]
    
    for url_idx, profile_url in enumerate(profile_urls):
        phase = url_idx + 2
        try:
            logger.info(f"[Profile] Phase {phase}: Scraping {profile_url}...")
            r = requests.get(profile_url, headers=headers, timeout=15, verify=False)
            
            if r.status_code != 200 or len(r.text) < 1000:
                logger.warning(f"[Profile] Phase {phase}: HTTP {r.status_code}, len={len(r.text)}")
                continue
            
            html_text = r.text
            
            # Try XPath parsing first (most precise with user-provided path)
            xpath_result = _parse_profile_with_xpath(html_text, cache_key)
            if xpath_result:
                result.update({k: v for k, v in xpath_result.items() if v})
            
            # Supplement with regex parsing (fills gaps)
            regex_result = _parse_profile_with_regex(html_text, cache_key)
            if regex_result:
                for k, v in regex_result.items():
                    if v and not result.get(k):
                        result[k] = v
            
            if result.get("country"):
                logger.info(f"[Profile] Phase {phase} SUCCESS: {cache_key} country={result['country']}, sector={result.get('sector','N/A')}")
                _profile_cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.warning(f"[Profile] Phase {phase} failed for {cache_key}: {str(e)}")
        
        time.sleep(random.uniform(0.5, 1.0))
    
    # Final: cache whatever we got
    n_keys = len([v for v in result.values() if v])
    if n_keys > 0:
        logger.info(f"[Profile] {cache_key}: Partial result ({n_keys} fields)")
    else:
        logger.warning(f"[Profile] {cache_key}: All phases failed - no profile data")
    
    _profile_cache[cache_key] = result
    return result

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
        
        # Request key financial metrics via Yahoo Timeseries API
        # Works for both financial and non-financial companies
        keys = [
            "trailingCreditLossesProvision",
            "annualCreditLossesProvision",
            "trailingPretaxIncome",
            "annualPretaxIncome",
            "trailingInterestExpense",
            "trailingInterestIncome",
            "trailingEBIT",
            "trailingInterestExpenseNonOperating",
            # Deposit interest - try every possible naming convention
            "trailingInterestExpenseOnDeposits",
            "annualInterestExpenseOnDeposits",
            "trailingInterestExpenseForDeposit",
            "annualInterestExpenseForDeposit",
            "trailingInterestExpenseForDeposits",
            "annualInterestExpenseForDeposits",
            "trailingInterestPaidOnDeposits",
            "trailingDepositInterestExpense",
            "trailingInterestOnDeposits",
            "trailingInterestExpenseDeposit",
            "annualInterestExpenseForDeposit",
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
            "int_exp_on_deposits_ttm": 0,
            "int_exp_ttm": 0,
            "int_income_ttm": 0,
            "ebit_ttm": 0,
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
                            elif "deposit" in type_lower and "interest" in type_lower:
                                api_data["int_exp_on_deposits_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM IntExpOnDeposits: ${raw_val:,.0f} (key: {type_name})")
                            elif type_lower == "trailingebit":
                                api_data["ebit_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM EBIT: ${raw_val:,.0f}")
                            elif "interestexpense" in type_lower and "deposit" not in type_lower and "nonoperating" not in type_lower:
                                api_data["int_exp_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM IntExp: ${raw_val:,.0f}")
                            elif "interestexpensenonoperating" in type_lower:
                                if api_data["int_exp_ttm"] == 0:
                                    api_data["int_exp_ttm"] = raw_val
                                    logger.info(f"[Financial API] {cache_key} TTM IntExpNonOp: ${raw_val:,.0f}")
                            elif "interestincome" in type_lower:
                                api_data["int_income_ttm"] = raw_val
                                logger.info(f"[Financial API] {cache_key} TTM IntIncome: ${raw_val:,.0f}")
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
                    or api_data["int_exp_on_deposits_ttm"] != 0
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

def scrape_yahoo_deposit_interest(ticker):
    """
    Scrape Yahoo Finance HTML page for Interest Expense on Deposits.
    Yahoo Finance embeds financial data as JSON in the page source.
    Falls back to multiple URL patterns.
    Returns dollar amount or 0 if not found.
    """
    cache_key = f"_deposit_{ticker.upper()}"
    if cache_key in _financial_ttm_cache:
        return _financial_ttm_cache[cache_key] or 0
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        # Try financials page - data is embedded as JSON in script tags
        urls = [
            f"https://finance.yahoo.com/quote/{ticker}/financials/",
            f"https://finance.yahoo.com/quote/{ticker}/financials",
        ]
        
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=15, verify=False)
                if r.status_code != 200:
                    continue
                
                html = r.text
                
                # Method 1: Search for JSON data embedded in page
                # Yahoo Finance embeds data in window.__PRELOADED_STATE__ or similar
                import re, json
                
                # Look for deposit interest in raw HTML/JSON
                deposit_patterns = [
                    r'"InterestExpenseForDeposit[s]?"[^}]*?"raw"\s*:\s*(-?\d+)',
                    r'"InterestExpenseOnDeposit[s]?"[^}]*?"raw"\s*:\s*(-?\d+)',
                    r'"InterestOnDeposit[s]?"[^}]*?"raw"\s*:\s*(-?\d+)',
                    r'"DepositInterestExpense"[^}]*?"raw"\s*:\s*(-?\d+)',
                    r'"InterestPaidOnDeposit[s]?"[^}]*?"raw"\s*:\s*(-?\d+)',
                    r'"interestExpenseForDeposit[s]?"[^}]*?"raw"\s*:\s*(-?\d+)',
                    # Also try searching in text content
                    r'[Ii]nterest\s+[Ee]xpense\s+(?:for|on)\s+[Dd]eposit[^"]*?(\d[\d,]+)',
                ]
                
                for pattern in deposit_patterns:
                    match = re.search(pattern, html)
                    if match:
                        val_str = match.group(1).replace(',', '')
                        val = abs(int(val_str))
                        if val > 1_000_000:  # Sanity check: > $1M
                            logger.info(f"[Scrape] {ticker} Interest on Deposits from HTML: ${val:,.0f} (pattern: {pattern[:40]})")
                            _financial_ttm_cache[cache_key] = val
                            return val
                
                # Method 2: Parse embedded JSON from __PRELOADED_STATE__
                json_match = re.search(r'root\.App\.main\s*=\s*({.*?});\s*\n', html, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'"financialData"\s*:\s*({[^}]+})', html)
                
                if json_match:
                    try:
                        json_str = json_match.group(1)
                        # Search for deposit keywords in this JSON blob
                        deposit_search = re.findall(r'"([^"]*[Dd]eposit[^"]*)":\s*\{[^}]*"raw"\s*:\s*(-?\d+)', json_str)
                        for key_found, val_str in deposit_search:
                            val = abs(int(val_str))
                            if val > 1_000_000 and 'interest' in key_found.lower():
                                logger.info(f"[Scrape] {ticker} Found via JSON key '{key_found}': ${val:,.0f}")
                                _financial_ttm_cache[cache_key] = val
                                return val
                    except:
                        pass
                        
            except requests.RequestException:
                continue
        
        logger.info(f"[Scrape] {ticker}: Interest on Deposits not found in HTML")
        _financial_ttm_cache[cache_key] = 0
        return 0
        
    except Exception as e:
        logger.debug(f"[Scrape] {ticker} failed: {e}")
        _financial_ttm_cache[cache_key] = 0
        return 0

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
                            matches.append((priority, val, raw_idx_str))
                    except Exception as e:
                        logger.debug(f"Value extraction failed: {str(e)}")
                    break  # Only match once per row
        
        # Return value from highest priority match (by abs magnitude for ranking, but keep sign)
        if matches:
            matches.sort(key=lambda x: (x[0], abs(x[1])), reverse=True)  # Sort by priority then abs magnitude
            best_match = matches[0]
            logger.info(f"Provision matched: '{best_match[2]}' (priority: {best_match[0]}, value: ${best_match[1]:,.0f})")
            return best_match[1]
    
    except Exception as e:
        logger.warning(f"Priority fuzzy search failed: {str(e)}")
    
    return 0

def get_value_max_fuzzy(df, col_idx, search_keywords, exclusion_keywords=None, debug_provision=False, keep_sign=False):
    """
    Scans ALL rows. Normalizes strings (remove spaces, lower case) for matching.
    Returns the value with the largest absolute magnitude found.
    
    Args:
        keep_sign: If True, return the original signed value (for EBIT, PretaxIncome etc.)
                   If False (default), return abs(value) (for Revenue, Interest Expense etc.)
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
                            candidates.append(val)  # Keep original signed value
                            if debug_provision:
                                debug_matches.append(f"    MATCHED '{norm_kw}' -> Value: {val:,.0f}")
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
            # Pick the candidate with the largest absolute magnitude
            best = max(candidates, key=abs)
            return best if keep_sign else abs(best)
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
    """Damodaran 신용등급별 스프레드 테이블 조회 (FRED OAS 매핑 포함)"""
    # 2026 Updated Fallback Data - FRED column maps rating to FRED OAS category
    fallback_large = pd.DataFrame([
        {"greater than": "8.5",      "≤ to": "100000", "Rating": "Aaa/AAA",  "FRED": "AAA US Corporate",           "Spread": "0.40%"},
        {"greater than": "6.5",      "≤ to": "8.5",    "Rating": "Aa2/AA",   "FRED": "AA US Corporate",            "Spread": "0.55%"},
        {"greater than": "5.5",      "≤ to": "6.5",    "Rating": "A1/A+",    "FRED": "Single-A US Corporate",      "Spread": "0.70%"},
        {"greater than": "4.25",     "≤ to": "5.5",    "Rating": "A2/A",     "FRED": "Single-A US Corporate",      "Spread": "0.78%"},
        {"greater than": "3.0",      "≤ to": "4.25",   "Rating": "A3/A-",    "FRED": "Single-A US Corporate",      "Spread": "0.89%"},
        {"greater than": "2.5",      "≤ to": "3.0",    "Rating": "Baa2/BBB", "FRED": "BBB US Corporate",           "Spread": "1.11%"},
        {"greater than": "2.25",     "≤ to": "2.5",    "Rating": "Ba1/BB+",  "FRED": "BB US High Yield",           "Spread": "1.38%"},
        {"greater than": "2.0",      "≤ to": "2.25",   "Rating": "Ba2/BB",   "FRED": "BB US High Yield",           "Spread": "1.84%"},
        {"greater than": "1.75",     "≤ to": "2.0",    "Rating": "B1/B+",    "FRED": "Single-B US High Yield",     "Spread": "2.75%"},
        {"greater than": "1.5",      "≤ to": "1.75",   "Rating": "B2/B",     "FRED": "Single-B US High Yield",     "Spread": "3.21%"},
        {"greater than": "1.25",     "≤ to": "1.5",    "Rating": "B3/B-",    "FRED": "Single-B US High Yield",     "Spread": "5.09%"},
        {"greater than": "0.8",      "≤ to": "1.25",   "Rating": "Caa/CCC",  "FRED": "CCC & Lower US High Yield",  "Spread": "8.85%"},
        {"greater than": "0.65",     "≤ to": "0.8",    "Rating": "Ca2/CC",   "FRED": "CCC & Lower US High Yield",  "Spread": "12.61%"},
        {"greater than": "0.2",      "≤ to": "0.65",   "Rating": "C2/C",     "FRED": "CCC & Lower US High Yield",  "Spread": "16.00%"},
        {"greater than": "-100000",  "≤ to": "0.2",    "Rating": "D2/D",     "FRED": "CCC & Lower US High Yield",  "Spread": "19.00%"},
    ])
    
    fallback_small = pd.DataFrame([
        {"greater than": "12.5",     "≤ to": "100000", "Rating": "Aaa/AAA",  "FRED": "AAA US Corporate",           "Spread": "0.40%"},
        {"greater than": "9.5",      "≤ to": "12.5",   "Rating": "Aa2/AA",   "FRED": "AA US Corporate",            "Spread": "0.55%"},
        {"greater than": "7.5",      "≤ to": "9.5",    "Rating": "A1/A+",    "FRED": "Single-A US Corporate",      "Spread": "0.70%"},
        {"greater than": "6.0",      "≤ to": "7.5",    "Rating": "A2/A",     "FRED": "Single-A US Corporate",      "Spread": "0.78%"},
        {"greater than": "4.5",      "≤ to": "6.0",    "Rating": "A3/A-",    "FRED": "Single-A US Corporate",      "Spread": "0.89%"},
        {"greater than": "4.0",      "≤ to": "4.5",    "Rating": "Baa2/BBB", "FRED": "BBB US Corporate",           "Spread": "1.11%"},
        {"greater than": "3.5",      "≤ to": "4.0",    "Rating": "Ba1/BB+",  "FRED": "BB US High Yield",           "Spread": "1.38%"},
        {"greater than": "3.0",      "≤ to": "3.5",    "Rating": "Ba2/BB",   "FRED": "BB US High Yield",           "Spread": "1.84%"},
        {"greater than": "2.5",      "≤ to": "3.0",    "Rating": "B1/B+",    "FRED": "Single-B US High Yield",     "Spread": "2.75%"},
        {"greater than": "2.0",      "≤ to": "2.5",    "Rating": "B2/B",     "FRED": "Single-B US High Yield",     "Spread": "3.21%"},
        {"greater than": "1.5",      "≤ to": "2.0",    "Rating": "B3/B-",    "FRED": "Single-B US High Yield",     "Spread": "5.09%"},
        {"greater than": "1.25",     "≤ to": "1.5",    "Rating": "Caa/CCC",  "FRED": "CCC & Lower US High Yield",  "Spread": "8.85%"},
        {"greater than": "0.8",      "≤ to": "1.25",   "Rating": "Ca2/CC",   "FRED": "CCC & Lower US High Yield",  "Spread": "12.61%"},
        {"greater than": "0.5",      "≤ to": "0.8",    "Rating": "C2/C",     "FRED": "CCC & Lower US High Yield",  "Spread": "16.00%"},
        {"greater than": "-100000",  "≤ to": "0.5",    "Rating": "D2/D",     "FRED": "CCC & Lower US High Yield",  "Spread": "19.00%"},
    ])
    
    fallback_fin = pd.DataFrame([
        {"greater than": "3.0",      "≤ to": "100000", "Rating": "Aaa/AAA",  "FRED": "AAA US Corporate",           "Spread": "0.40%"},
        {"greater than": "2.5",      "≤ to": "3.0",    "Rating": "Aa2/AA",   "FRED": "AA US Corporate",            "Spread": "0.55%"},
        {"greater than": "2.0",      "≤ to": "2.5",    "Rating": "A1/A+",    "FRED": "Single-A US Corporate",      "Spread": "0.70%"},
        {"greater than": "1.5",      "≤ to": "2.0",    "Rating": "A2/A",     "FRED": "Single-A US Corporate",      "Spread": "0.78%"},
        {"greater than": "1.2",      "≤ to": "1.5",    "Rating": "A3/A-",    "FRED": "Single-A US Corporate",      "Spread": "0.89%"},
        {"greater than": "0.9",      "≤ to": "1.2",    "Rating": "Baa2/BBB", "FRED": "BBB US Corporate",           "Spread": "1.11%"},
        {"greater than": "0.75",     "≤ to": "0.9",    "Rating": "Ba1/BB+",  "FRED": "BB US High Yield",           "Spread": "1.38%"},
        {"greater than": "0.6",      "≤ to": "0.75",   "Rating": "Ba2/BB",   "FRED": "BB US High Yield",           "Spread": "1.84%"},
        {"greater than": "0.5",      "≤ to": "0.6",    "Rating": "B1/B+",    "FRED": "Single-B US High Yield",     "Spread": "2.75%"},
        {"greater than": "0.4",      "≤ to": "0.5",    "Rating": "B2/B",     "FRED": "Single-B US High Yield",     "Spread": "3.21%"},
        {"greater than": "0.3",      "≤ to": "0.4",    "Rating": "B3/B-",    "FRED": "Single-B US High Yield",     "Spread": "5.09%"},
        {"greater than": "0.2",      "≤ to": "0.3",    "Rating": "Caa/CCC",  "FRED": "CCC & Lower US High Yield",  "Spread": "8.85%"},
        {"greater than": "0.1",      "≤ to": "0.2",    "Rating": "Ca2/CC",   "FRED": "CCC & Lower US High Yield",  "Spread": "12.61%"},
        {"greater than": "0.05",     "≤ to": "0.1",    "Rating": "C2/C",     "FRED": "CCC & Lower US High Yield",  "Spread": "16.00%"},
        {"greater than": "-100000",  "≤ to": "0.05",   "Rating": "D2/D",     "FRED": "CCC & Lower US High Yield",  "Spread": "19.00%"},
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
    Returns: rev, ebit, ebitda, int_exp, label_ebit, label_int, raw_pretax, raw_provision, int_on_deposits
    
    - ebit: Yahoo-style EBIT (includes unusual items) - used for both sidebar and ICR
    - For financial companies: PPNR = Pretax + abs(Provision)
    
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
    industry = str(info_dict.get('industry', '')).lower()
    is_financial = 'financial' in sector or 'bank' in sector
    
    # Financial industry subtype for ICR numerator formula
    # Group 1: Banks/Credit → PPNR = Pretax + Provision (add-back credit losses)
    # Group 2: Insurance → Pretax − Realized Gain/Loss on Investments + Interest Expense
    # Group 3: Others (Asset Mgmt, Brokers, Exchanges, etc.) → Non-financial EBIT logic
    BANK_INDUSTRIES = [
        'banks - diversified', 'banks - regional', 'mortgage finance', 'credit services',
        'banks—diversified', 'banks—regional',
    ]
    INSURANCE_INDUSTRIES = [
        'insurance - life', 'insurance - property & casualty', 'insurance - diversified',
        'insurance - reinsurance', 'insurance - specialty',
        'insurance—life', 'insurance—property & casualty', 'insurance—diversified',
        'insurance—reinsurance', 'insurance—specialty',
    ]
    NONFINANCIAL_LOGIC_INDUSTRIES = [
        'financial data & stock exchanges', 'insurance brokers', 'asset management',
        'capital markets', 'financial conglomerates', 'shell companies',
    ]
    
    fin_subtype = 'non_financial'  # default
    if is_financial:
        if any(ind in industry for ind in BANK_INDUSTRIES):
            fin_subtype = 'bank'
        elif any(ind in industry for ind in INSURANCE_INDUSTRIES):
            fin_subtype = 'insurance'
        elif any(ind in industry for ind in NONFINANCIAL_LOGIC_INDUSTRIES):
            fin_subtype = 'fin_nonbank'  # use non-financial EBIT logic
        else:
            fin_subtype = 'bank'  # default for unrecognized financial industry
        logger.info(f"[Financial Subtype] industry='{industry}' → subtype='{fin_subtype}'")
    
    current_year = datetime.now().year
    
    # [v1.5.1] Pre-fetch TTM data from Yahoo Finance Timeseries API for ALL tickers
    # Gets EBIT, InterestExpense, PretaxIncome, Provision TTM (consistent with Yahoo Finance web)
    api_data = None
    if ticker_symbol:
        api_data = fetch_financial_ttm_from_api(ticker_symbol)
        if api_data:
            logger.info(f"[v1.5.1] API data for {ticker_symbol}: "
                        f"ebit_ttm=${api_data.get('ebit_ttm', 0):,.0f}, "
                        f"int_exp_ttm=${api_data.get('int_exp_ttm', 0):,.0f}, "
                        f"provision_ttm=${api_data['provision_ttm']:,.0f}, "
                        f"pretax_ttm=${api_data['pretax_ttm']:,.0f}")
        else:
            logger.info(f"[v1.5.1] API returned no data for {ticker_symbol}, will use yfinance fallback")
    
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
            """Extract rev, ebit, ebitda, int_exp, pretax, provision, int_on_deposits, realized_gl from one column.
            
            Financial subtypes:
            - bank: PPNR = Pretax - Provision (add-back credit losses)
            - insurance: Adj Pretax = Pretax - Net Realized Gain/Loss on Investments
            - fin_nonbank: same as non-financial (use EBIT)
            - non_financial: EBIT from Yahoo's EBIT row
            """
            r = get_value_max_fuzzy(df, col_idx, ['Total Revenue', 'Revenue'])
            i = get_value_max_fuzzy(df, col_idx, ['Interest Expense', 'Interest Expense Non Operating'])
            ed = get_value_max_fuzzy(df, col_idx, ['EBITDA', 'Normalized EBITDA'], keep_sign=True)
            
            # Interest on Deposits (for bank subtype)
            iod = 0
            if fin_subtype == 'bank':
                iod = get_value_max_fuzzy(df, col_idx, 
                    ['Interest Expense For Deposit', 'Interest Expense For Deposits',
                     'Interest Expense On Deposits', 'Interest Paid On Deposits',
                     'Interest On Deposits', 'Deposit Interest Expense'])
            
            p_tax = 0
            p_prov = 0
            realized_gl = 0
            val_e = 0
            
            if fin_subtype == 'bank':
                # Banks/Credit: PPNR = Pretax - Provision
                p_tax = get_value_max_fuzzy(df, col_idx, ['Pretax Income', 'Income Before Tax'], keep_sign=True)
                
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
                
                if df_cf is not None and not df_cf.empty and col_idx_cf is not None:
                    p_prov = get_value_max_fuzzy_with_priority(
                        df_cf, col_idx_cf, provision_keywords, exclusion_keywords
                    )
                if p_prov == 0:
                    p_prov = get_value_max_fuzzy_with_priority(
                        df, col_idx, provision_keywords, exclusion_keywords
                    )
                
                if p_tax != 0: 
                    val_e = p_tax - p_prov
                    
            elif fin_subtype == 'insurance':
                # Insurance: Adj Pretax = Pretax - Net Realized Gain/Loss on Investments
                p_tax = get_value_max_fuzzy(df, col_idx, ['Pretax Income', 'Income Before Tax'], keep_sign=True)
                realized_gl = get_value_max_fuzzy(df, col_idx, 
                    ['Gain On Sale Of Security', 'Net Realized Gain Loss On Investments',
                     'Realized Gain Loss On Investments', 'Gain Loss On Investment Securities'],
                    keep_sign=True)
                
                if p_tax != 0:
                    val_e = p_tax - realized_gl  # Remove realized gains to get operating pretax
                    
            # fin_nonbank and non_financial: fall through to EBIT
            
            if val_e == 0:
                val_e = get_value_max_fuzzy(df, col_idx, 
                    ['EBIT'], 
                    exclusion_keywords=['EBITDA', 'Normalized EBITDA'],
                    keep_sign=True)
                
                logger.info(f"[extract_from_col] col={col_idx}: EBIT={val_e:,.0f}, EBITDA={ed:,.0f}, Rev={r:,.0f}, IntExp={i:,.0f}")
            
            return r, val_e, ed, i, p_tax, p_prov, iod, realized_gl

        # =================================================================
        # --- Priority 1: Annual from yfinance income_stmt ---
        # Only accept columns from current_year or current_year - 1.
        # Skip ghost columns (NaN revenue). Triple Lock is warning only.
        # =================================================================
        target_year = current_year - 1  # 2025 as of Feb 2026
        if not a_fin.empty:
            for col_idx_p1, col in enumerate(a_fin.columns):
                col_dt = pd.to_datetime(col)
                
                # Only accept current_year or current_year - 1
                if col_dt.year < target_year:
                    logger.info(f"[P1] {col.strftime('%Y-%m-%d')}: Too old (need >={target_year}), skipping P1")
                    break  # Columns are sorted newest first, so stop
                
                # Find matching cash flow column (same year)
                cf_idx = None
                if not a_cf.empty:
                    for cf_col_idx, cf_col in enumerate(a_cf.columns):
                        if pd.to_datetime(cf_col).year == col_dt.year:
                            cf_idx = cf_col_idx
                            break
                
                r_annual, e, ed, i, pt, pp, iod_annual, rgl_annual = extract_from_col(a_fin, col_idx_p1, a_cf, cf_idx)
                
                # Skip ghost columns (all NaN / no revenue)
                if not (pd.notna(r_annual) and r_annual > 1000):
                    logger.info(f"[P1] {col.strftime('%Y-%m-%d')}: Ghost column (rev={r_annual}), trying next...")
                    continue
                
                # [v1.4.0] If yfinance couldn't find provision, try API annual data
                if is_financial and pp == 0 and api_data:
                    for date_str, val in api_data.get('provision_annual', []):
                        if str(col_dt.year) in date_str:
                            pp = val  # keep original sign
                            e = pt - pp  # Recalculate PPNR
                            logger.info(f"[P1] Using API annual provision for {col_dt.year}: ${pp:,.0f}")
                            break
                
                # Triple Lock Validation (warning only)
                if not q_fin.empty:
                    cutoff_date = col_dt - timedelta(days=360)
                    valid_quarters = []
                    q_rev_sum = 0
                    for q_idx, q_col in enumerate(q_fin.columns):
                        q_dt = pd.to_datetime(q_col)
                        if cutoff_date < q_dt <= col_dt:
                            valid_quarters.append(q_idx)
                            q_rev_sum += get_value_max_fuzzy(q_fin, q_idx, ['Total Revenue', 'Revenue'])
                    
                    if len(valid_quarters) >= 4 and r_annual > 0:
                        ratio = q_rev_sum / r_annual
                        if not (0.9 <= ratio <= 1.1):
                            logger.warning(f"[P1] Triple Lock WARNING: ratio={ratio:.2f}")
                    elif len(valid_quarters) < 4:
                        logger.info(f"[P1] Triple Lock: only {len(valid_quarters)} quarters for {col.strftime('%Y-%m-%d')}")
                
                lbl = col.strftime('%Y-%m-%d')
                logger.info(f"[P1] Using Annual {lbl}: rev=${r_annual:,.0f}, ebit=${e:,.0f}, ie=${i:,.0f}, iod=${iod_annual:,.0f}")
                return r_annual, e, ed, abs(i), lbl, lbl, pt, pp, iod_annual, fin_subtype
        
        # =================================================================
        # --- Priority 2: Yahoo Info TTM (info_dict + Timeseries API) ---
        # Revenue/EBITDA/InterestExpense from info_dict
        # PretaxIncome/Provision from Timeseries API (not in info_dict)
        # =================================================================
        rev_ttm = info_dict.get('totalRevenue', 0)
        
        if rev_ttm is not None and rev_ttm > 0:
            rev = rev_ttm
            ebitda = info_dict.get('ebitda', 0)
            
            # Interest Expense: API timeseries TTM first (matches Yahoo Finance web),
            # then info_dict, then quarterly sum
            int_exp = 0
            if api_data and api_data.get('int_exp_ttm', 0) != 0:
                int_exp = abs(api_data['int_exp_ttm'])
                label_int = "TTM (Yahoo API)"
                logger.info(f"[P2] InterestExp from API TTM: ${int_exp:,.0f}")
            
            if int_exp == 0:
                int_exp = info_dict.get('interestExpense', 0)
                if int_exp is None or int_exp == 0:
                    int_exp = info_dict.get('totalInterestExpense', 0)
                if int_exp is not None and int_exp > 0:
                    label_int = "TTM (Yahoo Info)"
            
            if int_exp is None or int_exp == 0:
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

            # EBIT / PPNR by financial subtype
            ebit = 0
            if fin_subtype == 'bank':
                # Banks: PPNR = PretaxIncome - Provision
                if api_data and api_data.get('pretax_ttm', 0) != 0:
                    raw_pretax = api_data['pretax_ttm']
                    raw_provision = api_data.get('provision_ttm', 0)
                    ebit = raw_pretax - raw_provision
                    label_ebit = "TTM (Yahoo API)"
                    logger.info(f"[P2] Bank PPNR from API: pretax=${raw_pretax:,.0f} - provision=${raw_provision:,.0f} = ${ebit:,.0f}")
                else:
                    if not q_fin.empty and q_fin.shape[1] >= 4:
                        recent_4 = q_fin.iloc[:, :4]
                        recent_4_cf = q_cf.iloc[:, :4] if not q_cf.empty and q_cf.shape[1] >= 4 else None
                        q_pretax = 0
                        q_prov = 0
                        for q_idx in range(4):
                            cf_idx = q_idx if recent_4_cf is not None else None
                            _, _, _, _, pt, pp, _, _ = extract_from_col(recent_4, q_idx, recent_4_cf, cf_idx)
                            q_pretax += pt
                            q_prov += pp
                        if q_prov == 0 and api_data:
                            api_prov = api_data.get('provision_ttm', 0)
                            if api_prov != 0:
                                q_prov = api_prov
                        raw_pretax = q_pretax
                        raw_provision = q_prov
                        if q_pretax != 0:
                            ebit = q_pretax - q_prov
                        label_ebit = "TTM (Calc Quarters)"
                    else:
                        label_ebit = "N/A"
                        
            elif fin_subtype == 'insurance':
                # Insurance: Adj Pretax = Pretax - Net Realized Gain/Loss on Investments
                if api_data and api_data.get('pretax_ttm', 0) != 0:
                    raw_pretax = api_data['pretax_ttm']
                    # realized_gl from API not available, try quarterly sum
                    rgl_sum = 0
                    if not q_fin.empty and q_fin.shape[1] >= 4:
                        for q_idx in range(4):
                            _, _, _, _, _, _, _, rgl_q = extract_from_col(q_fin.iloc[:, :4], q_idx)
                            rgl_sum += rgl_q
                    ebit = raw_pretax - rgl_sum
                    label_ebit = "TTM (Yahoo API)"
                    logger.info(f"[P2] Insurance Adj Pretax from API: pretax=${raw_pretax:,.0f} - realized_gl=${rgl_sum:,.0f} = ${ebit:,.0f}")
                else:
                    if not q_fin.empty and q_fin.shape[1] >= 4:
                        recent_4 = q_fin.iloc[:, :4]
                        q_pretax = 0
                        q_rgl = 0
                        for q_idx in range(4):
                            _, _, _, _, pt, _, _, rgl_q = extract_from_col(recent_4, q_idx)
                            q_pretax += pt
                            q_rgl += rgl_q
                        raw_pretax = q_pretax
                        if q_pretax != 0:
                            ebit = q_pretax - q_rgl
                        label_ebit = "TTM (Calc Quarters)"
                    else:
                        label_ebit = "N/A"
            else:
                # Non-financial & fin_nonbank: EBIT from API TTM first, then 4Q sum
                if api_data and api_data.get('ebit_ttm', 0) != 0:
                    ebit = api_data['ebit_ttm']
                    label_ebit = "TTM (Yahoo API)"
                    logger.info(f"[P2] Non-Financial EBIT from API TTM: ${ebit:,.0f}")
                else:
                    if not q_fin.empty and q_fin.shape[1] >= 4:
                        recent_4 = q_fin.iloc[:, :4]
                        q_ebit_sum = 0
                        for q_idx in range(4):
                            _, e_q, _, _, _, _, _, _ = extract_from_col(recent_4, q_idx)
                            q_ebit_sum += e_q
                        if q_ebit_sum != 0:
                            ebit = q_ebit_sum
                            label_ebit = "TTM (Calc Quarters)"
                            logger.info(f"[P2] Non-Financial EBIT from 4Q sum: ${ebit:,.0f}")
                        elif ebitda:
                            ebit = ebitda
                            label_ebit = "TTM (EBITDA Proxy)"
                        else:
                            label_ebit = "N/A"
                    elif ebitda:
                        ebit = ebitda
                        label_ebit = "TTM (EBITDA Proxy)"
                    else:
                        label_ebit = "N/A"
            
            if int_exp is None: 
                int_exp = 0
            # P2: get int_on_deposits from API TTM (same period as other TTM data)
            iod_p2 = 0
            if is_financial and api_data:
                iod_p2 = abs(api_data.get('int_exp_on_deposits_ttm', 0))
            return rev, ebit, ebitda, abs(int_exp), label_ebit, label_int, raw_pretax, raw_provision, iod_p2, fin_subtype

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
            iod_p3 = 0
            rgl_p3 = 0
            
            for q_idx in range(4):
                cf_idx = q_idx if recent_4_cf is not None else None
                r_q, e_q, ed_q, i_q, pt_q, pp_q, iod_q, rgl_q = extract_from_col(
                    recent_4, q_idx, recent_4_cf, cf_idx
                )
                rev += r_q
                ebitda += ed_q
                int_exp += i_q
                iod_p3 += iod_q
                rgl_p3 += rgl_q
                
                if fin_subtype in ('bank', 'insurance'):
                    raw_pretax += pt_q
                    raw_provision += pp_q
                else:
                    ebit += e_q
            
            # [v1.4.0] If yfinance found no provision per-quarter, use API TTM
            if fin_subtype == 'bank' and raw_provision == 0 and api_data:
                api_prov = api_data.get('provision_ttm', 0)
                if api_prov != 0:
                    raw_provision = api_prov
                    logger.info(f"[P3] Using API TTM provision: ${raw_provision:,.0f}")
            
            # [v1.4.0] If yfinance found no pretax per-quarter, use API TTM
            if fin_subtype in ('bank', 'insurance') and raw_pretax == 0 and api_data:
                api_pt = api_data.get('pretax_ttm', 0)
                if api_pt != 0:
                    raw_pretax = api_pt
                    logger.info(f"[P3] Using API TTM pretax: ${raw_pretax:,.0f}")
            
            if fin_subtype == 'bank':
                ebit = raw_pretax - raw_provision
            elif fin_subtype == 'insurance':
                ebit = raw_pretax - rgl_p3
            
            return rev, ebit, ebitda, abs(int_exp), common_label, common_label, raw_pretax, raw_provision, iod_p3, fin_subtype

    except Exception as e:
        logger.error(f"Financial data extraction failed: {str(e)}")
    
    return 0, 0, 0, 0, "No Data", "No Data", 0, 0, 0, "non_financial"

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
            logger.warning(f"No info available for {ticker} - will try financial statements directly")
            info = {}  # Use empty dict instead of early return
        
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
        
        rev, ebit, ebitda, int_exp, label_ebit, label_int, raw_pretax, raw_provision, int_on_deposits, fin_subtype = \
            get_financial_data_with_priority(t, info, ticker_symbol=ticker)
        
        # For bank subtype, if int_on_deposits still 0, try HTML scraping as last resort
        if fin_subtype == 'bank' and int_on_deposits == 0:
            int_on_deposits = scrape_yahoo_deposit_interest(ticker)
            if int_on_deposits > 0:
                logger.info(f"[Target] Interest on Deposits from HTML scrape: ${int_on_deposits:,.0f}")
        
        mkt_cap = info.get('marketCap', 0)
        sector = str(info.get('sector', '')).lower()
        local_currency = info.get('currency', 'USD')
        
        # Get FX rate to USD (fiscal year average based on financial statement date)
        fx_rate = 1.0
        fx_basis = ""
        if local_currency and local_currency != 'USD':
            try:
                fx_ticker = f"{local_currency}USD=X"
                fx_data = yf.Ticker(fx_ticker)
                
                # Determine fiscal year period from label_ebit (e.g., "2025-06-30")
                fy_end = None
                try:
                    fy_end = pd.to_datetime(label_ebit)
                except:
                    pass
                
                if fy_end:
                    # Fiscal year: 12 months ending at label_ebit date
                    fy_start = fy_end - pd.DateOffset(years=1) + pd.DateOffset(days=1)
                    fx_hist = fx_data.history(start=fy_start.strftime('%Y-%m-%d'), 
                                              end=(fy_end + pd.DateOffset(days=1)).strftime('%Y-%m-%d'))
                    if not fx_hist.empty and len(fx_hist) > 20:
                        fx_rate = float(fx_hist['Close'].mean())
                        fx_basis = f"FY avg ({fy_start.strftime('%Y.%m')}–{fy_end.strftime('%Y.%m')}, {len(fx_hist)} days)"
                        logger.info(f"[FX] {local_currency}->USD: {fx_rate:.4f} ({fx_basis})")
                    else:
                        # Fallback to 1Y average
                        fx_hist = fx_data.history(period="1y")
                        if not fx_hist.empty:
                            fx_rate = float(fx_hist['Close'].mean())
                            fx_basis = f"1Y avg ({len(fx_hist)} days)"
                            logger.info(f"[FX] {local_currency}->USD: {fx_rate:.4f} (1Y fallback)")
                else:
                    # No date info — use 1Y average
                    fx_hist = fx_data.history(period="1y")
                    if not fx_hist.empty:
                        fx_rate = float(fx_hist['Close'].mean())
                        fx_basis = f"1Y avg ({len(fx_hist)} days)"
                
                if fx_rate == 1.0:
                    # Last resort: spot rate
                    fx_hist = fx_data.history(period="5d")
                    if not fx_hist.empty:
                        fx_rate = float(fx_hist['Close'].iloc[-1])
                        fx_basis = "Spot (latest)"
                        
            except Exception as e:
                logger.warning(f"[FX] Failed to get rate for {local_currency}: {e}")
        
        # Convert to USD
        int_exp_usd = int_exp * fx_rate
        ebit_usd = ebit * fx_rate
        raw_pretax_usd = raw_pretax * fx_rate
        raw_provision_usd = raw_provision * fx_rate
        int_on_deposits_usd = int_on_deposits * fx_rate
        # Non-deposit interest = Total IE - Deposit Interest
        non_deposit_int_exp = max(int_exp - int_on_deposits, 0)
        non_deposit_int_exp_usd = non_deposit_int_exp * fx_rate
        
        if 'financial' in sector or 'bank' in sector: 
            category = "Financial Firms"
        elif mkt_cap > 5e9: 
            category = "Large Firms" 
        else: 
            category = "Small/Risky Firms"
        
        return {
            "int_exp": int_exp_usd, "ebit": ebit_usd, 
            "int_exp_local": int_exp, "ebit_local": ebit,
            "int_on_deposits": int_on_deposits_usd, "int_on_deposits_local": int_on_deposits,
            "non_deposit_int_exp": non_deposit_int_exp_usd, "non_deposit_int_exp_local": non_deposit_int_exp,
            "label_int": label_int, "label_ebit": label_ebit,
            "raw_pretax": raw_pretax_usd, "raw_provision": raw_provision_usd,
            "raw_pretax_local": raw_pretax, "raw_provision_local": raw_provision,
            "category": category, "tax_rate": target_tax, "country_name": country,
            "currency": local_currency, "fx_rate": fx_rate, "fx_basis": fx_basis,
            "fin_subtype": fin_subtype,
        }
    except Exception as e:
        import traceback
        logger.error(f"Target financials fetch failed for {ticker}: {str(e)}\n{traceback.format_exc()}")
        return {
            "int_exp": 0.0, "ebit": 0.0,
            "int_exp_local": 0.0, "ebit_local": 0.0,
            "int_on_deposits": 0.0, "int_on_deposits_local": 0.0,
            "non_deposit_int_exp": 0.0, "non_deposit_int_exp_local": 0.0,
            "label_int": "N/A", "label_ebit": "N/A", 
            "raw_pretax": 0, "raw_provision": 0, 
            "raw_pretax_local": 0, "raw_provision_local": 0,
            "category": "Small/Risky Firms", 
            "tax_rate": 25.0, "country_name": "Unknown",
            "currency": "USD", "fx_rate": 1.0, "fx_basis": "",
            "fin_subtype": "non_financial",
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
                logger.warning(f"No info for {ticker} - trying alternative sources")
                info = {}
            
            curr = info.get('currency', 'USD')
            country = info.get('country', 'Unknown')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            mkt_cap = info.get('marketCap', 0)
            debt = info.get('totalDebt', 0)
            
            # Fallback: try fast_info for market cap
            if mkt_cap == 0: 
                try: 
                    fi = t.fast_info
                    mkt_cap = getattr(fi, 'market_cap', 0) or 0
                    if curr == 'USD' and mkt_cap == 0:
                        # Try shares * price
                        shares = getattr(fi, 'shares', 0) or 0
                        price = getattr(fi, 'last_price', 0) or 0
                        if shares > 0 and price > 0:
                            mkt_cap = shares * price
                    if mkt_cap > 0:
                        logger.info(f"[Peer] {ticker}: Got mkt_cap from fast_info: ${mkt_cap:,.0f}")
                except Exception as e:
                    logger.debug(f"Fast info failed for {ticker}: {str(e)}")
            
            if mkt_cap == 0:
                return None, f"⚠️ {ticker}: Excluded (Missing Market Cap)"
            
            # Fallback: try balance_sheet for total debt
            if debt == 0:
                try:
                    bs = t.balance_sheet
                    if not bs.empty:
                        debt_val = get_value_max_fuzzy(bs, 0, ['Total Debt', 'Long Term Debt', 'Total Non Current Liabilities Net Minority Interest'])
                        if debt_val > 0:
                            debt = debt_val
                            logger.info(f"[Peer] {ticker}: Got debt from balance_sheet: ${debt:,.0f}")
                except Exception as e:
                    logger.debug(f"Balance sheet failed for {ticker}: {str(e)}")

            rev, ebit, ebitda, int_exp_dummy, label_ebit, label_int, pt, pp, _, _ = \
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
                "period": period_display
            }
            return data, None
        except Exception as e:
            logger.error(f"Financials fetch error for {ticker}: {str(e)}")
            return None, f"⚠️ {ticker}: Error {str(e)}"

    def get_5y_monthly_beta_analysis(self):
        """5년 월간 베타 분석 (현재는 목 데이터 반환)"""
        beta_list = []
        for t in self.peers:
            beta_list.append({"Ticker": t, "Raw Beta": 1.2, "Adj Beta": 1.13})
        return pd.DataFrame(beta_list), None, None, []

    def run(self):
        """WACC 계산 실행"""
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
                    "Period": fin['period']
                })
        
        progress_text.empty()
        df_peers = pd.DataFrame(peer_data)
        
        if beta_df is not None and not beta_df.empty and not df_peers.empty:
            beta_df['Ticker'] = beta_df['Ticker'].str.upper().str.strip()
            df_peers['Ticker'] = df_peers['Ticker'].str.upper().str.strip()
            full_df = pd.merge(df_peers, beta_df, on="Ticker", how="left")
        else: 
            full_df = pd.DataFrame()

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
        _fin_subtype = tf.get('fin_subtype', 'non_financial')
        is_fin_target = _fin_subtype in ('bank', 'insurance')
        
        if _fin_subtype == 'bank':
            ebit_label = "PPNR"
        elif _fin_subtype == 'insurance':
            ebit_label = "Adj. Pretax"
        else:
            ebit_label = "EBIT"
        
        st.markdown("**Target Financials** (for Credit Spread)")
        
        # Currency & FX info
        local_curr = tf.get('currency', 'USD')
        fx = tf.get('fx_rate', 1.0)
        is_foreign = local_curr != 'USD' and fx != 1.0
        
        if is_foreign:
            _fx_basis = tf.get('fx_basis', '')
            st.caption(f"💱 {local_curr} → USD | Rate: **{fx:.4f}** | {_fx_basis}")
        
        # Helper: format dollar value with commas (full dollars, not thousands)
        def _fmt_dollar(val):
            """Format value as $-1,303,700,000 style."""
            if val < 0:
                return f"-${abs(val):,.0f}"
            return f"${val:,.0f}"
        
        def _parse_dollar(text, fallback=0.0):
            """Parse formatted dollar string back to float."""
            try:
                cleaned = text.replace('$', '').replace(',', '').replace(' ', '')
                return float(cleaned)
            except:
                return fallback
        
        # Helper: format in thousands with $ sign (for captions)
        def _fmt_k(val):
            """Format value in $thousands with commas, preserving sign."""
            v_k = val / 1000
            if v_k < 0:
                return f"-${abs(v_k):,.0f}k"
            return f"${v_k:,.0f}k"
        
        # Interest Expense section by financial subtype
        if _fin_subtype == 'bank':
            # === BANKS: ICR = (PPNR + Financial Interest) / Financial Interest ===
            st.caption("🏦 *Bank: ICR = (PPNR + Fin.Interest) / Fin.Interest*")
            
            # Total Interest Expense
            _ie_default = _fmt_dollar(tf['int_exp'])
            _ie_text = st.text_input(
                "Total Interest Expense (USD)",
                value=_ie_default,
                help="Total interest expense (includes deposit interest + debt interest)"
            )
            int_exp_in = _parse_dollar(_ie_text, tf['int_exp'])
            st.caption(f"{_fmt_k(int_exp_in)} · Source: {tf.get('label_int', 'N/A')}")
            
            # Interest on Deposits
            _iod_val = tf.get('int_on_deposits', 0)
            _iod_default = _fmt_dollar(_iod_val)
            _iod_text = st.text_input(
                "(−) Interest on Deposits (USD)",
                value=_iod_default,
                help="Interest paid to depositors. Often not available via API — enter from 10-K."
            )
            int_on_deposits_in = _parse_dollar(_iod_text, _iod_val)
            if int_on_deposits_in == 0 and int_exp_in > 0:
                st.warning("⚠️ Interest on Deposits = $0. Enter manually from 10-K.")
            else:
                st.caption(f"{_fmt_k(int_on_deposits_in)}")
            
            # Financial Interest
            _ndie_calc = max(int_exp_in - int_on_deposits_in, 0)
            _ndie_default = _fmt_dollar(_ndie_calc)
            _ndie_text = st.text_input(
                "Financial Interest (USD) — *used for ICR*",
                value=_ndie_default,
                help="Total IE − Interest on Deposits = debt/borrowing interest only"
            )
            non_deposit_ie_in = _parse_dollar(_ndie_text, _ndie_calc)
            st.caption(f"{_fmt_k(non_deposit_ie_in)} = Total IE − Deposits")
            
            # PPNR
            _ebit_default = _fmt_dollar(tf['ebit'])
            _ebit_text = st.text_input(
                f"{ebit_label} (USD)",
                value=_ebit_default,
                help="Pre-Provision Net Revenue = Pretax Income − Provision"
            )
            ebit_in = _parse_dollar(_ebit_text, tf['ebit'])
            
            # PPNR breakdown
            st.markdown("""<style>.small-font { font-size: 12px; color: #666; margin-bottom: 0px; }</style>""", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• Pre-tax Income: <b>{_fmt_k(tf.get('raw_pretax', 0))}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• (−) Provision: <b>{_fmt_k(tf.get('raw_provision', 0))}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• Source: <b>{tf.get('label_ebit', 'N/A')}</b></div>", unsafe_allow_html=True)
            
            if tf.get('raw_provision', 0) != 0:
                st.success(f"Credit Losses Provision detected: {_fmt_k(tf.get('raw_provision', 0))}")
            else:
                st.warning("Credit Losses Provision = $0 (Check 10-K)")
            
            int_exp_for_icr = non_deposit_ie_in
            
        elif _fin_subtype == 'insurance':
            # === INSURANCE: ICR = (Adj.Pretax + IE) / IE ===
            st.caption("🛡️ *Insurance: ICR = (Adj.Pretax + IE) / IE*")
            st.caption("*Adj.Pretax = Pretax − Net Realized Gain/Loss on Investments*")
            
            # Interest Expense (standard for insurance)
            _ie_default = _fmt_dollar(tf['int_exp'])
            _ie_text = st.text_input(
                "Interest Expense (USD)",
                value=_ie_default,
                help="Interest expense on borrowings/debt"
            )
            int_exp_in = _parse_dollar(_ie_text, tf['int_exp'])
            st.caption(f"{_fmt_k(int_exp_in)} · Source: {tf.get('label_int', 'N/A')}")
            
            # Adj. Pretax Income
            _ebit_default = _fmt_dollar(tf['ebit'])
            _ebit_text = st.text_input(
                f"{ebit_label} (USD)",
                value=_ebit_default,
                help="Pretax Income − Net Realized Gain/Loss on Investments"
            )
            ebit_in = _parse_dollar(_ebit_text, tf['ebit'])
            
            # Breakdown
            st.markdown("""<style>.small-font { font-size: 12px; color: #666; margin-bottom: 0px; }</style>""", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• Pre-tax Income: <b>{_fmt_k(tf.get('raw_pretax', 0))}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• (−) Realized Gain/Loss: <em>embedded in Adj.Pretax</em></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-font'>• Source: <b>{tf.get('label_ebit', 'N/A')}</b></div>", unsafe_allow_html=True)
            
            int_exp_for_icr = int_exp_in
            
        else:
            # === NON-FINANCIAL FIRMS: Standard IE and EBIT ===
            _ie_default = _fmt_dollar(tf['int_exp'])
            _ie_text = st.text_input(
                "Interest Expense (USD)",
                value=_ie_default,
                help="Enter with $ and commas (e.g. $315,200,000)" + (f" | Local: {local_curr} {_fmt_dollar(tf.get('int_exp_local',0))}" if is_foreign else "")
            )
            int_exp_in = _parse_dollar(_ie_text, tf['int_exp'])
            if is_foreign:
                _ie_local = tf.get('int_exp_local', 0)
                st.caption(f"{_fmt_k(int_exp_in)} · Source: {tf.get('label_int', 'N/A')} · Local: {local_curr} {_fmt_k(_ie_local)}")
            else:
                st.caption(f"{_fmt_k(int_exp_in)} · Source: {tf.get('label_int', 'N/A')}")
            
            # EBIT
            _ebit_default = _fmt_dollar(tf['ebit'])
            _ebit_text = st.text_input(
                f"{ebit_label} (USD)",
                value=_ebit_default,
                help="Enter with $ and commas" + (f" | Local: {local_curr} {_fmt_dollar(tf.get('ebit_local',0))}" if is_foreign else "")
            )
            ebit_in = _parse_dollar(_ebit_text, tf['ebit'])
            
            if is_foreign:
                _ebit_local = tf.get('ebit_local', 0)
                st.caption(f"{_fmt_k(ebit_in)} · Source: {tf.get('label_ebit', 'N/A')} · Local: {local_curr} {_fmt_k(_ebit_local)}")
            else:
                st.caption(f"{_fmt_k(ebit_in)} · Source: {tf.get('label_ebit', 'N/A')}")
            
            # For ICR calc: use gross int_exp as-is
            int_exp_for_icr = int_exp_in
        
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
                'int_exp': int_exp_for_icr, 'ebit': ebit_in, 'category': category_in
            }
        st.success("Calculation completed!")
    
    # Version and Contact Information
    st.divider()
    st.caption("**Version:** v1.5.0 (Feb 2026)")
    st.caption("**Point of Contact:** jonghyun.yang.5105@gmail.com")
    
    # Compact Debug Panel
    with st.expander("🔧 Debug", expanded=False):
        try:
            _d = st.session_state.get('target_fin', {})
            st.caption(f"**{target_ticker}** | {_d.get('country_name','?')} | Tax {_d.get('tax_rate',0):.1f}%")
            st.caption(f"EBIT ${_d.get('ebit',0):,.0f} | IntExp ${_d.get('int_exp',0):,.0f}")
            if _d.get('int_on_deposits', 0) > 0:
                st.caption(f"DepositInt ${_d.get('int_on_deposits',0):,.0f} | NonDepIE ${_d.get('non_deposit_int_exp',0):,.0f}")
            st.caption(f"Src: {_d.get('label_ebit','?')} | Cat: {_d.get('category','?')}")
            
            _di = {}
            try: _di = yf.Ticker(target_ticker).info or {}
            except: pass
            st.caption(f"info: {len(_di)} keys | country={_di.get('country','?')} | sector={_di.get('sector','?')}")
            st.caption(f"opMargin={_di.get('operatingMargins','?')} | mktCap={_di.get('marketCap','?')}")
            
            _api = fetch_company_profile_from_api(target_ticker)
            st.caption(f"API: country={_api.get('country','?')} | sector={_api.get('sector','?')}")
            
            # Financial data priority path diagnostic
            st.divider()
            st.caption("**Data Priority Path:**")
            try:
                _dbg_t = yf.Ticker(target_ticker)
                _dbg_info = _dbg_t.info or {}
                _dbg_sector = str(_dbg_info.get('sector','')).lower()
                _dbg_is_fin = 'financial' in _dbg_sector or 'bank' in _dbg_sector
                st.caption(f"is_financial: **{_dbg_is_fin}** (sector='{_dbg_sector}')")
                
                _dbg_inc = _dbg_t.income_stmt
                if not _dbg_inc.empty:
                    _c0 = _dbg_inc.columns[0]
                    st.caption(f"Annual cols: {[c.strftime('%Y-%m-%d') for c in _dbg_inc.columns[:3]]}")
                    # Show key rows for col 0
                    _key_rows = ['EBIT', 'Pretax Income', 'Operating Income', 'Total Revenue',
                                 'Interest Expense', 'Interest Income', 'Net Interest Income']
                    for _kr in _key_rows:
                        if _kr in _dbg_inc.index:
                            _v = _dbg_inc.loc[_kr].iloc[0]
                            st.caption(f"  {_kr}: {'${:,.0f}'.format(_v) if pd.notna(_v) else 'NaN'}")
                        else:
                            st.caption(f"  {_kr}: ❌ NOT IN INDEX")
                    
                    # Dump ALL interest/deposit related rows
                    _int_rows = [r for r in _dbg_inc.index 
                                 if any(kw in str(r).lower() for kw in ['interest', 'deposit', 'net income'])]
                    if _int_rows:
                        st.caption("**All interest/deposit rows (Annual):**")
                        for _ir in _int_rows:
                            _v = _dbg_inc.loc[_ir].iloc[0]
                            st.caption(f"  → {_ir}: {'${:,.0f}'.format(_v) if pd.notna(_v) else 'NaN'}")
                    
                    # Also check quarterly for deposit-specific rows
                    _dbg_q = _dbg_t.quarterly_income_stmt
                    if not _dbg_q.empty:
                        _q_dep_rows = [r for r in _dbg_q.index 
                                       if 'deposit' in str(r).lower()]
                        if _q_dep_rows:
                            st.caption("**Quarterly deposit rows found:**")
                            for _qr in _q_dep_rows:
                                _v = _dbg_q.loc[_qr].iloc[0]
                                st.caption(f"  → {_qr}: {'${:,.0f}'.format(_v) if pd.notna(_v) else 'NaN'}")
                else:
                    st.caption("income_stmt: **EMPTY**")
                
                # Test get_financial_data_with_priority directly
                _dbg_rev, _dbg_ebit, _dbg_ebitda, _dbg_ie, _dbg_le, _dbg_li, _dbg_pt, _dbg_pp, _dbg_iod, _dbg_subtype = \
                    get_financial_data_with_priority(_dbg_t, _dbg_info, ticker_symbol=target_ticker)
                st.caption(f"Priority result: rev=${_dbg_rev:,.0f} ebit=${_dbg_ebit:,.0f} ie=${_dbg_ie:,.0f} iod=${_dbg_iod:,.0f}")
                st.caption(f"  pretax=${_dbg_pt:,.0f} provision=${_dbg_pp:,.0f} label={_dbg_le} subtype={_dbg_subtype}")
                
                # Show API data for all firms
                _api_cache = _financial_ttm_cache.get(target_ticker.upper())
                if _api_cache:
                    _iod = _api_cache.get('int_exp_on_deposits_ttm', 0)
                    _ie_api = _api_cache.get('int_exp_ttm', 0)
                    _ii_api = _api_cache.get('int_income_ttm', 0)
                    _ebit_api = _api_cache.get('ebit_ttm', 0)
                    st.caption(f"  API: EBIT=${_ebit_api:,.0f} IntExp=${_ie_api:,.0f} DepositInt=${_iod:,.0f} IntIncome=${_ii_api:,.0f}")
                
                # Show scraping result
                _scrape_key = f"_deposit_{target_ticker.upper()}"
                _scrape_val = _financial_ttm_cache.get(_scrape_key, 'not tried')
                st.caption(f"  HTML Scrape DepositInt: {('${:,.0f}'.format(_scrape_val) if isinstance(_scrape_val, (int, float)) else str(_scrape_val))}")
            except Exception as _pe:
                st.caption(f"Priority path error: {_pe}")
            
            # ICR → Rating → Spread pipeline debug
            if 'inputs' in st.session_state:
                _inp = st.session_state['inputs']
                _ebit_dbg = _inp.get('ebit', 0)
                _int_dbg = _inp.get('int_exp', 0)
                _cat_dbg = _inp.get('category', '?')
                if _cat_dbg == "Financial Firms" and _fin_subtype in ('bank', 'insurance'):
                    _icr_dbg = (_ebit_dbg + _int_dbg) / _int_dbg if _int_dbg > 0 else 100.0
                else:
                    _icr_dbg = _ebit_dbg / _int_dbg if _int_dbg > 0 else 100.0
                
                st.divider()
                st.caption("**ICR → Spread Pipeline:**")
                if _cat_dbg == "Financial Firms" and _fin_subtype in ('bank', 'insurance'):
                    st.caption(f"① ICR = ({_ebit_dbg:,.0f} + {_int_dbg:,.0f}) / {_int_dbg:,.0f} = **{_icr_dbg:.2f}x** [{_fin_subtype}]")
                else:
                    st.caption(f"① ICR = {_ebit_dbg:,.0f} / {_int_dbg:,.0f} = **{_icr_dbg:.2f}x**")
                
                # Damodaran lookup
                _dam_dict = get_damodaran_spreads()
                _dam_tbl, _dam_src = _dam_dict.get(_cat_dbg, (None, ""))
                _dam_rating = "N/A"
                _dam_spread = 0.0
                _fred_key = "BB US High Yield"
                if _dam_tbl is not None:
                    for _, _row in _dam_tbl.iterrows():
                        try:
                            _lo = float(str(_row.get('greater than','-')).replace('greater than','').replace('-','-99999').strip())
                            _hi = float(str(_row.get('≤ to','-')).replace('-','99999').strip())
                            if _lo < _icr_dbg <= _hi:
                                _dam_rating = _row['Rating']
                                _dam_spread = float(str(_row['Spread']).replace('%',''))
                                if 'FRED' in _row and pd.notna(_row['FRED']):
                                    _fred_key = _row['FRED']
                                break
                        except:
                            continue
                st.caption(f"② Damodaran ({_cat_dbg}): Rating=**{_dam_rating}** | Spread=**{_dam_spread:.2f}%**")
                st.caption(f"③ FRED map: {_dam_rating} → **{_fred_key}** (from table)")
                
                _fred_row = df_oas[df_oas['OAS Name'] == _fred_key]
                _fred_val = None
                if not _fred_row.empty:
                    _fred_val = _fred_row.iloc[0]['Latest Spread (%)']
                st.caption(f"④ FRED OAS ({_fred_key}): **{_fred_val}%**" if _fred_val else f"④ FRED OAS: NOT FOUND")
                st.caption(f"⑤ Final Spread used: **{_fred_val if _fred_val else _dam_spread:.2f}%**")
        except Exception as _e:
            st.caption(f"Error: {_e}")
    
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
    # These are computed inside 'if not df_init.empty:' but referenced outside it.
    # Without this, NameError occurs when df_init is empty (no valid peer data).
    final_spread = 0.0
    icr = 0.0
    implied_rating = "N/A"
    implied_spread_val = 0.0
    category = inp.get('category', "Small/Risky Firms")
    target_fred_key = "N/A"
    int_exp = inp.get('int_exp', 0.0)
    ebit = inp.get('ebit', 0.0)
    is_financial_firm = (_fin_subtype in ("bank", "insurance"))

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
        
        is_financial_firm = (_fin_subtype in ("bank", "insurance"))
        
        if is_financial_firm:
            # Financial Firms: Use observed (raw) beta directly
            # Hamada unlevering/relevering is not appropriate for financials
            # because their debt is operational (deposits), not financial leverage
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
            
            target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
            # No unlevered/relevered columns for financial firms
            calc_df["Unlevered Beta"] = calc_df["Adj Beta"]  # Display: same as observed
            calc_df["Re-levered Beta"] = calc_df["Adj Beta"]  # Display: same as observed
            sel_unlev = target_relevered_beta  # For display consistency
        else:
            # Non-Financial: Standard Hamada unlevering → relevering
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
        
        if category == "Financial Firms" and _fin_subtype in ('bank', 'insurance'):
            # Bank/Insurance: ICR = (Numerator + IE) / IE
            # bank: Numerator = PPNR, insurance: Numerator = Adj.Pretax
            icr = (ebit + int_exp) / int_exp if int_exp > 0 else 100.0
        else:
            # Non-Financial & fin_nonbank: ICR = EBIT / Interest Expense
            icr = ebit / int_exp if int_exp > 0 else 100.0
        
        # 1. Get Table
        damodaran_dict = get_damodaran_spreads()
        rating_table, _ = damodaran_dict.get(category, (None, ""))
        
        implied_rating = "N/A"
        implied_spread_val = 2.00
        target_fred_key = "BB US High Yield"  # default fallback
        
        if rating_table is not None:
            for idx, row in rating_table.iterrows():
                try:
                    low_v = float(str(row.get('greater than','-')).replace('greater than','').replace('-','-99999').strip())
                    high_v = float(str(row.get('≤ to','-')).replace('-','99999').strip())
                    if low_v < icr <= high_v:
                        implied_rating = row['Rating']
                        spread_str = str(row['Spread']).replace('%','')
                        implied_spread_val = float(spread_str)
                        # Read FRED mapping directly from table
                        if 'FRED' in row and pd.notna(row['FRED']):
                            target_fred_key = row['FRED']
                        break
                except Exception as e:
                    logger.debug(f"Rating table parsing error: {str(e)}")
                    continue
        
        
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
            if is_financial_firm:
                st.info("🏦 **Financial Firms**: Adjusted Beta (Bloomberg) is used directly — Hamada unlevering/relevering is skipped because financial firms' debt is primarily operational (deposits), not financial leverage.")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**1. Raw Beta**")
                    st.latex(r"\beta_{raw} = \text{5Y Monthly Regression vs Market}")
                with mc2:
                    st.markdown("**2. Adjusted Beta (= used for Ke)**")
                    st.latex(r"\beta_{adj} = 0.67 \cdot \beta_{raw} + 0.33")
            else:
                mc1, mc2, mc3 = st.columns(3)
                with mc1: 
                    st.markdown("**1. Adjusted Beta**")
                    st.latex(r"\beta_{adj} = 0.67 \cdot \beta_{raw} + 0.33")
                with mc2: 
                    st.markdown("**2. Unlevered Beta**")
                    st.latex(r"\beta_U = \frac{\beta_{adj}}{1 + (1 - T_{peer}) \frac{D}{E}}")
                with mc3: 
                    st.markdown("**3. Re-levered Beta**")
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
        c4.metric("Adjusted Beta" if is_financial_firm else "Re-levered Beta", f"{target_relevered_beta:.2f}")
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
    k2.metric("Beta (Adjusted)" if category == "Financial Firms" else "Beta (Re-levered)", f"{target_relevered_beta:.2f}")
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
        if category == "Financial Firms" and _fin_subtype == 'bank':
            st.caption(
                f"Based on {category} Table. "
                f"ICR = (PPNR + Fin.Int) / Fin.Int = ({ebit:,.0f} + {int_exp:,.0f}) / {int_exp:,.0f}"
            )
        elif category == "Financial Firms" and _fin_subtype == 'insurance':
            st.caption(
                f"Based on {category} Table. "
                f"ICR = (Adj.Pretax + IE) / IE = ({ebit:,.0f} + {int_exp:,.0f}) / {int_exp:,.0f}"
            )
        else:
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
