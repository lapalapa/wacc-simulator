"""
US Listed Companies Peer Data Collector
Runs via GitHub Actions (daily) to collect ticker/industry/sector/marketCap/revenue
from NASDAQ screener (all US exchanges) + yfinance enrichment.

Output: sp1500_peers.csv (name kept for backward compatibility)
"""
import pandas as pd
import yfinance as yf
import requests
import time
import random

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_nasdaq_listed():
    """Fetch all US-listed companies from NASDAQ screener API (NYSE + NASDAQ + AMEX)"""
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {
        "tableonly": "true",
        "limit": 10000,
        "offset": 0,
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", {}).get("table", {}).get("rows", [])
        
        if not rows:
            print(f"[NASDAQ] No rows returned. Response keys: {data.keys()}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        print(f"[NASDAQ] Raw rows: {len(df)}, columns: {df.columns.tolist()}")
        
        # Standardize columns
        result = pd.DataFrame()
        result['ticker'] = df.get('symbol', df.get('Symbol', pd.Series(dtype=str))).str.strip()
        result['name'] = df.get('name', df.get('Name', ''))
        result['nasdaq_sector'] = df.get('sector', df.get('Sector', ''))
        result['nasdaq_industry'] = df.get('industry', df.get('Industry', ''))
        result['exchange'] = df.get('exchange', df.get('Exchange', ''))
        
        # Parse market cap string (e.g., "$1,234,567,890" or "1.23B")
        raw_cap = df.get('marketCap', df.get('Market Cap', pd.Series(dtype=str)))
        if raw_cap is not None:
            result['nasdaq_market_cap'] = raw_cap.apply(_parse_market_cap)
        else:
            result['nasdaq_market_cap'] = 0
        
        # Filter: only common stocks (exclude ETFs, warrants, etc.)
        result = result[result['ticker'].str.len() <= 5]
        result = result[~result['ticker'].str.contains(r'[^A-Za-z]', na=False)]
        result = result[result['ticker'].str.len() > 0]
        
        result = result.drop_duplicates(subset='ticker', keep='first')
        print(f"[NASDAQ] After filter: {len(result)} tickers")
        return result
        
    except Exception as e:
        print(f"[NASDAQ] Failed: {e}")
        return pd.DataFrame()


def _parse_market_cap(val):
    """Parse NASDAQ market cap string to float"""
    if pd.isna(val) or val == '' or val is None:
        return 0
    s = str(val).replace('$', '').replace(',', '').strip()
    if not s or s == 'N/A':
        return 0
    try:
        return float(s)
    except ValueError:
        return 0


def enrich_with_yfinance(df):
    """
    Enrich with Yahoo Finance: industry, sector, marketCap, totalRevenue.
    Only enrich top ~3000 by market cap to save API calls.
    Smaller companies keep NASDAQ-provided sector/industry.
    """
    df = df.sort_values('nasdaq_market_cap', ascending=False).reset_index(drop=True)
    
    ENRICH_LIMIT = 3000
    enrich_mask = df.index < ENRICH_LIMIT
    
    print(f"\nEnriching top {min(ENRICH_LIMIT, len(df))} tickers with yfinance...")
    
    yahoo_industries = [''] * len(df)
    yahoo_sectors = [''] * len(df)
    market_caps = [0] * len(df)
    revenues = [0] * len(df)
    
    batch_count = 0
    for i in df[enrich_mask].index:
        ticker = df.at[i, 'ticker']
        batch_count += 1
        
        if batch_count % 100 == 0:
            print(f"  Progress: {batch_count}/{ENRICH_LIMIT} ({batch_count/ENRICH_LIMIT*100:.0f}%)")
        
        try:
            time.sleep(random.uniform(0.2, 0.5))
            t = yf.Ticker(ticker)
            info = t.info or {}
            yahoo_industries[i] = info.get('industry', '')
            yahoo_sectors[i] = info.get('sector', '')
            market_caps[i] = info.get('marketCap', 0) or 0
            revenues[i] = info.get('totalRevenue', 0) or 0
        except Exception as e:
            if batch_count % 200 == 0:
                print(f"  ⚠️ {ticker}: {e}")
    
    df['yahoo_industry'] = yahoo_industries
    df['yahoo_sector'] = yahoo_sectors
    df['market_cap'] = market_caps
    df['revenue'] = revenues
    
    # For non-enriched tickers, use NASDAQ data as fallback
    mask_no_ind = df['yahoo_industry'] == ''
    df.loc[mask_no_ind, 'yahoo_industry'] = df.loc[mask_no_ind, 'nasdaq_industry']
    mask_no_sec = df['yahoo_sector'] == ''
    df.loc[mask_no_sec, 'yahoo_sector'] = df.loc[mask_no_sec, 'nasdaq_sector']
    mask_no_cap = df['market_cap'] == 0
    df.loc[mask_no_cap, 'market_cap'] = df.loc[mask_no_cap, 'nasdaq_market_cap']
    
    return df


def main():
    print("=" * 60)
    print("US Listed Companies — Peer Data Collection")
    print("=" * 60)
    
    # Step 1: Get all US-listed tickers from NASDAQ screener
    print("\n[1/2] Fetching US-listed companies from NASDAQ...")
    df = get_nasdaq_listed()
    
    if df.empty:
        print("ERROR: No tickers fetched. Exiting.")
        return
    
    print(f"  → {len(df)} tickers from NASDAQ/NYSE/AMEX")
    
    # Step 2: Enrich with yfinance (Yahoo industry/sector + market cap + revenue)
    print("\n[2/2] Enriching with Yahoo Finance data...")
    df = enrich_with_yfinance(df)
    
    # Sort by market cap descending
    df = df.sort_values('market_cap', ascending=False).reset_index(drop=True)
    
    # Select final columns
    final = df[['ticker', 'name', 'yahoo_sector', 'yahoo_industry',
                 'market_cap', 'revenue', 'exchange']].copy()
    
    # Save
    final.to_csv('sp1500_peers.csv', index=False)
    
    # Stats
    enriched = (final['yahoo_industry'] != '').sum()
    with_revenue = (final['revenue'] > 0).sum()
    
    print(f"\n{'=' * 60}")
    print(f"DONE: sp1500_peers.csv")
    print(f"  Total tickers: {len(final)}")
    print(f"  With Yahoo industry: {enriched}")
    print(f"  With revenue data: {with_revenue}")
    print(f"  Unique industries: {final['yahoo_industry'].nunique()}")
    print(f"\nTop 10 by market cap:")
    print(final[['ticker', 'name', 'yahoo_industry', 'market_cap', 'revenue']].head(10).to_string())
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
