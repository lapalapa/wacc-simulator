"""
WACC Simulator — Market Data Collector (FRED, KPMG, NYU, SP500)
Runs via GitHub Actions (monthly) — separate from peer data collection.

Outputs:
  - fred_data.csv          FRED time series (GDP, Rf, 7 OAS spreads)
  - sp500_monthly.csv      S&P 500 monthly prices (for beta calc)
  - kpmg_tax_rates.csv     Country corporate tax rates
  - nyu_sp_earnings.csv    S&P 500 buyback/dividend/earnings (Damodaran)
"""
import pandas as pd
import requests
import time
import random
import io
import sys

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ==============================================================================
# [1] FRED Data (GDP, Rf, OAS Spreads)
# ==============================================================================
def collect_fred_data():
    targets = [
        ("GDP", "A191RP1A027NBEA"),
        ("RF", "DGS10"),
        ("AAA", "BAMLC0A1CAAA"),
        ("AA", "BAMLC0A2CAA"),
        ("A", "BAMLC0A3CA"),
        ("BBB", "BAMLC0A4CBBB"),
        ("BB", "BAMLH0A1HYBB"),
        ("B", "BAMLH0A2HYB"),
        ("CCC", "BAMLH0A3HYC"),
    ]
    
    all_rows = []
    for key, series_id in targets:
        try:
            time.sleep(random.uniform(0.5, 1.5))
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
            df = df.dropna()
            df["series"] = key
            df["series_id"] = series_id
            all_rows.append(df)
            print(f"  ✓ {key} ({series_id}): {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {key} ({series_id}): {e}")
    
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv("fred_data.csv", index=False)
        print(f"  → fred_data.csv: {len(combined)} total rows, {len(all_rows)}/{len(targets)} series")
        return True
    else:
        print("  → FAILED: No FRED data collected")
        return False


# ==============================================================================
# [2] S&P 500 Monthly Prices (for Beta calculation)
# ==============================================================================
def collect_sp500_monthly():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["DATE", "VALUE"]
        df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors='coerce')
        df = df.dropna().sort_values("DATE")
        
        df = df.set_index("DATE")
        monthly = df["VALUE"].resample("ME").last().dropna()
        
        cutoff = monthly.index.max() - pd.DateOffset(years=6)
        monthly = monthly[monthly.index >= cutoff]
        
        result = monthly.reset_index()
        result.columns = ["DATE", "SP500"]
        result.to_csv("sp500_monthly.csv", index=False)
        print(f"  → sp500_monthly.csv: {len(result)} months")
        return True
    except Exception as e:
        print(f"  → FAILED: {e}")
        return False


# ==============================================================================
# [3] KPMG Tax Rates
# ==============================================================================
def collect_kpmg_tax_rates():
    url = "https://kpmg.com/dk/en/services/tax/corporate-tax/corporate-tax-rates-table.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        r.raise_for_status()
        dfs = pd.read_html(io.StringIO(r.text))
        df = dfs[0]
        df.rename(columns={df.columns[0]: "Country"}, inplace=True)
        col_name = df.columns[-1]
        result = df[["Country", col_name]].copy()
        result.columns = ["Country", "Rate"]
        result["Rate"] = pd.to_numeric(result["Rate"], errors='coerce')
        result = result.dropna(subset=["Rate"])
        
        overrides = pd.DataFrame([
            {"Country": "United States", "Rate": 25.57},
            {"Country": "Korea", "Rate": 26.40},
        ])
        result = pd.concat([result, overrides], ignore_index=True)
        result = result.drop_duplicates(subset="Country", keep="last")
        
        result.to_csv("kpmg_tax_rates.csv", index=False)
        print(f"  → kpmg_tax_rates.csv: {len(result)} countries")
        return True
    except Exception as e:
        print(f"  → FAILED: {e}")
        return False


# ==============================================================================
# [4] NYU Stern (Damodaran) S&P Earnings/Buyback Data
# ==============================================================================
def collect_nyu_sp_earnings():
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        r.raise_for_status()
        dfs = pd.read_html(io.StringIO(r.text), header=0)
        df = dfs[0].dropna(subset=[dfs[0].columns[0]])
        df.to_csv("nyu_sp_earnings.csv", index=False)
        print(f"  → nyu_sp_earnings.csv: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"  → FAILED: {e}")
        return False


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("WACC Simulator — Market Data Collection")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/4] FRED Data (GDP, Rf, 7 OAS Spreads)...")
    results['fred'] = collect_fred_data()
    
    print("\n[2/4] S&P 500 Monthly Prices (Beta)...")
    results['sp500'] = collect_sp500_monthly()
    
    print("\n[3/4] KPMG Corporate Tax Rates...")
    results['kpmg'] = collect_kpmg_tax_rates()
    
    print("\n[4/4] NYU Stern S&P Earnings/Buyback...")
    results['nyu'] = collect_nyu_sp_earnings()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("COLLECTION SUMMARY:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} — {name}")
    
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n⚠️ {len(failed)} collection(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✓ All collections successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
