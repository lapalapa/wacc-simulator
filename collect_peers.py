"""
S&P 1500 Peer Data Collector
Runs via GitHub Actions (daily) to collect ticker/industry/sector/marketCap
from Wikipedia S&P 500/400/600 lists + yfinance enrichment.

Output: sp1500_peers.csv
"""
import pandas as pd
import yfinance as yf
import time
import random
import sys

def get_sp500():
    """S&P 500 from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0][['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
    df.columns = ['ticker', 'name', 'gics_sector', 'gics_sub_industry']
    df['index'] = 'SP500'
    df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)  # BRK.B → BRK-B
    return df

def get_sp400():
    """S&P 400 MidCap from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        # Column names vary — find the right ones
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if 'symbol' in cl or 'ticker' in cl:
                col_map['ticker'] = c
            elif 'company' in cl or 'security' in cl or 'name' in cl:
                col_map['name'] = c
            elif 'sector' in cl and 'sub' not in cl:
                col_map['gics_sector'] = c
            elif 'sub' in cl and 'industr' in cl:
                col_map['gics_sub_industry'] = c
        
        if 'ticker' not in col_map:
            print("[SP400] Could not find ticker column, skipping")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['ticker'] = df[col_map['ticker']].str.replace('.', '-', regex=False)
        result['name'] = df.get(col_map.get('name', ''), '')
        result['gics_sector'] = df.get(col_map.get('gics_sector', ''), '')
        result['gics_sub_industry'] = df.get(col_map.get('gics_sub_industry', ''), '')
        result['index'] = 'SP400'
        return result
    except Exception as e:
        print(f"[SP400] Failed: {e}")
        return pd.DataFrame()

def get_sp600():
    """S&P 600 SmallCap from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if 'symbol' in cl or 'ticker' in cl:
                col_map['ticker'] = c
            elif 'company' in cl or 'security' in cl or 'name' in cl:
                col_map['name'] = c
            elif 'sector' in cl and 'sub' not in cl:
                col_map['gics_sector'] = c
            elif 'sub' in cl and 'industr' in cl:
                col_map['gics_sub_industry'] = c
        
        if 'ticker' not in col_map:
            print("[SP600] Could not find ticker column, skipping")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['ticker'] = df[col_map['ticker']].str.replace('.', '-', regex=False)
        result['name'] = df.get(col_map.get('name', ''), '')
        result['gics_sector'] = df.get(col_map.get('gics_sector', ''), '')
        result['gics_sub_industry'] = df.get(col_map.get('gics_sub_industry', ''), '')
        result['index'] = 'SP600'
        return result
    except Exception as e:
        print(f"[SP600] Failed: {e}")
        return pd.DataFrame()

def enrich_with_yfinance(df, batch_size=50):
    """
    Enrich with Yahoo Finance industry (Yahoo's own classification).
    Wikipedia uses GICS, Yahoo uses its own system.
    We need Yahoo industry for matching with wacc-simulator's classification.
    """
    print(f"\nEnriching {len(df)} tickers with yfinance data...")
    
    yahoo_industries = []
    yahoo_sectors = []
    market_caps = []
    
    for i, ticker in enumerate(df['ticker']):
        if i > 0 and i % batch_size == 0:
            print(f"  Progress: {i}/{len(df)} ({i/len(df)*100:.0f}%)")
        
        try:
            time.sleep(random.uniform(0.3, 0.8))  # Rate limit protection
            t = yf.Ticker(ticker)
            info = t.info or {}
            yahoo_industries.append(info.get('industry', ''))
            yahoo_sectors.append(info.get('sector', ''))
            market_caps.append(info.get('marketCap', 0) or 0)
        except Exception as e:
            yahoo_industries.append('')
            yahoo_sectors.append('')
            market_caps.append(0)
            if i % 100 == 0:
                print(f"  ⚠️ {ticker}: {e}")
    
    df['yahoo_industry'] = yahoo_industries
    df['yahoo_sector'] = yahoo_sectors
    df['market_cap'] = market_caps
    
    return df

def main():
    print("=" * 60)
    print("S&P 1500 Peer Data Collection")
    print("=" * 60)
    
    # Step 1: Get S&P 500/400/600 from Wikipedia
    print("\n[1/4] Fetching S&P 500...")
    sp500 = get_sp500()
    print(f"  → {len(sp500)} tickers")
    
    print("[2/4] Fetching S&P 400 MidCap...")
    sp400 = get_sp400()
    print(f"  → {len(sp400)} tickers")
    
    print("[3/4] Fetching S&P 600 SmallCap...")
    sp600 = get_sp600()
    print(f"  → {len(sp600)} tickers")
    
    # Combine
    combined = pd.concat([sp500, sp400, sp600], ignore_index=True)
    combined = combined.drop_duplicates(subset='ticker', keep='first')
    print(f"\nTotal unique tickers: {len(combined)}")
    
    # Step 2: Enrich with yfinance (Yahoo industry/sector + market cap)
    print("\n[4/4] Enriching with Yahoo Finance data...")
    combined = enrich_with_yfinance(combined)
    
    # Fill missing yahoo_industry with GICS sub-industry
    mask = combined['yahoo_industry'] == ''
    combined.loc[mask, 'yahoo_industry'] = combined.loc[mask, 'gics_sub_industry']
    combined.loc[combined['yahoo_sector'] == '', 'yahoo_sector'] = combined.loc[combined['yahoo_sector'] == '', 'gics_sector']
    
    # Sort by market cap descending
    combined = combined.sort_values('market_cap', ascending=False)
    
    # Save
    combined.to_csv('sp1500_peers.csv', index=False)
    
    print(f"\n{'=' * 60}")
    print(f"DONE: sp1500_peers.csv ({len(combined)} tickers)")
    print(f"Columns: {combined.columns.tolist()}")
    print(f"Top 10 by market cap:")
    print(combined[['ticker', 'name', 'yahoo_industry', 'market_cap']].head(10).to_string())
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
