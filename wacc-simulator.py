import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
import time
import random

# 페이지 설정
st.set_page_config(page_title="Strategic WACC Simulator", layout="wide")

# ==============================================================================
# [MODULE] Helper: Safe Fetcher with Retry
# ==============================================================================
def safe_yf_info(ticker_obj, max_retries=3):
    """
    yfinance info fetching with retry logic to handle Rate Limiting (429).
    """
    for i in range(max_retries):
        try:
            return ticker_obj.info
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = (2 ** (i + 1)) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue
            else:
                if i == max_retries - 1: return {}
                time.sleep(1)
    return {}

# ==============================================================================
# [MODULE] Data Fetcher 1: NYU Stern (Buyback & Dividend)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    default_bb_yield = 2.0 
    default_div_yield = 1.5
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        dfs = pd.read_html(io.StringIO(response.text), header=0)
        df = None
        for d in dfs:
            cols_str = [str(c).lower() for c in d.columns]
            if "year" in cols_str and "s&p 500" in cols_str:
                df = d; break
        if df is None: return default_bb_yield, default_div_yield, None, ["Error"]

        cols_map = {}
        for c in df.columns:
            c_lower = str(c).lower().strip()
            if "year" in c_lower: cols_map["Period"] = c
            elif "s&p 500" in c_lower and "yield" not in c_lower: cols_map["S&P 500"] = c
            elif "dividends" in c_lower and "+" not in c_lower and "yield" not in c_lower: cols_map["Dividends"] = c 
            elif "dividends + buybacks" in c_lower or ("buybacks" in c_lower and "+" in c_lower): cols_map["TotalCash"] = c 

        clean_df = pd.DataFrame()
        clean_df["Year"] = df[cols_map["Period"]]
        clean_df["S&P 500"] = df[cols_map["S&P 500"]]
        clean_df["Dividends"] = df[cols_map["Dividends"]]
        clean_df["TotalCash"] = df[cols_map["TotalCash"]]
        clean_df["Year"] = pd.to_numeric(clean_df["Year"], errors='coerce')
        clean_df = clean_df.dropna(subset=["Year"]).sort_values(by="Year", ascending=False)
        for c in ["S&P 500", "Dividends", "TotalCash"]:
            clean_df[c] = pd.to_numeric(clean_df[c], errors='coerce')

        clean_df["Buybacks"] = clean_df["TotalCash"] - clean_df["Dividends"]
        clean_df["Buyback Yield"] = clean_df["Buybacks"] / clean_df["S&P 500"]
        clean_df["Dividend Yield"] = clean_df["Dividends"] / clean_df["S&P 500"]
        clean_df["Buyback Yield %"] = clean_df["Buyback Yield"] * 100
        clean_df["Dividend Yield %"] = clean_df["Dividend Yield"] * 100
        clean_df["Total Yield %"] = (clean_df["Buyback Yield"] + clean_df["Dividend Yield"]) * 100

        valid_rows = clean_df[clean_df["Buyback Yield"] > 0].head(5)
        avg_bb_yield = valid_rows["Buyback Yield %"].mean()
        avg_div_yield = valid_rows["Dividend Yield %"].mean()
        return avg_bb_yield, avg_div_yield, clean_df, []
    except: return default_bb_yield, default_div_yield, None, ["Error"]

# ==============================================================================
# [MODULE] Data Fetcher 2 & 3: FRED Data (GDP & RF) + [NEW] OAS Spreads
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_fred_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r_gdp = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=A191RP1A027NBEA", headers=headers)
        df_gdp = pd.read_csv(io.StringIO(r_gdp.text))
        df_gdp.columns = ["Date", "GDP Growth %"]
        df_gdp["Date"] = pd.to_datetime(df_gdp["Date"])
        df_gdp = df_gdp.sort_values(by="Date", ascending=False)
        latest_gdp = df_gdp["GDP Growth %"].iloc[0]
        df_gdp_disp = df_gdp.head(10).copy()
    except: latest_gdp = 2.0; df_gdp_disp = None

    try:
        r_rf = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", headers=headers)
        df_rf = pd.read_csv(io.StringIO(r_rf.text))
        df_rf.columns = ["Date", "Rate"]
        df_rf["Date"] = pd.to_datetime(df_rf["Date"])
        df_rf["Rate"] = pd.to_numeric(df_rf["Rate"], errors='coerce')
        df_rf = df_rf.dropna().sort_values(by="Date", ascending=False)
        latest_rf = df_rf["Rate"].iloc[0]
        cutoff = df_rf["Date"].max() - timedelta(days=365*5)
        df_rf_trend = df_rf[df_rf["Date"] >= cutoff].copy()
    except: latest_rf = 4.0; df_rf_trend = None

    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend

@st.cache_data(ttl=3600*24)
def get_fred_oas_data():
    """
    Fetches US Corporate Option-Adjusted Spread (OAS) for various ratings.
    """
    series_map = {
        "AAA US Corporate": "BAMLC0A1CAAA",
        "AA US Corporate": "BAMLC0A2CAA",
        "Single-A US Corporate": "BAMLC0A3CA",
        "BBB US Corporate": "BAMLC0A4CBBB",
        "BB US High Yield": "BAMLH0A1HYBB",
        "Single-B US High Yield": "BAMLH0A2HYB",
        "CCC & Lower US High Yield": "BAMLH0A3HYC"
    }
    
    headers = {"User-Agent": "Mozilla/5.0"}
    data_list = []
    
    for name, series_id in series_map.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=5)
            df = pd.read_csv(io.StringIO(r.text))
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df.sort_values(by="DATE", ascending=False)
            
            # Get latest valid value
            latest_val = df.iloc[0, 1]
            latest_date = df.iloc[0, 0].strftime('%Y-%m-%d')
            
            data_list.append({
                "OAS Name": name,
                "Latest Spread (%)": float(latest_val),
                "Date": latest_date,
                "Series ID": series_id
            })
        except:
            data_list.append({
                "OAS Name": name,
                "Latest Spread (%)": None,
                "Date": "N/A",
                "Series ID": series_id
            })
            
    return pd.DataFrame(data_list)

# ==============================================================================
# [MODULE] Data Fetcher 4: KPMG Tax Rates (Live Scraping)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_kpmg_tax_rates():
    url = "https://kpmg.com/dk/en/services/tax/corporate-tax/corporate-tax-rates-table.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        dfs = pd.read_html(io.StringIO(r.text))
        target_df = None
        for df in dfs:
            cols_str = [str(c) for c in df.columns]
            if any("2024" in c or "2025" in c for c in cols_str):
                target_df = df; break
        if target_df is None: return None, {}, 2025

        target_df.rename(columns={target_df.columns[0]: "Country"}, inplace=True)
        available_years = [int(c) for c in target_df.columns if str(c).isdigit()]
        if not available_years: return None, {}, 2025
        
        latest_year = max(available_years)
        col_name = str(latest_year)
        
        result_df = target_df[["Country", col_name]].copy()
        result_df.columns = ["Country", f"Rate"]
        result_df["Rate"] = pd.to_numeric(result_df["Rate"], errors='coerce')
        result_df = result_df.dropna().sort_values(by="Rate", ascending=False).reset_index(drop=True)
        
        # Mapping Dictionary
        tax_dict = dict(zip(result_df["Country"].str.upper().str.strip(), result_df["Rate"]))
        tax_dict["UNITED STATES"] = tax_dict.get("UNITED STATES OF AMERICA", 25.57)
        tax_dict["KOREA"] = tax_dict.get("KOREA (SOUTH)", 26.40)
        
        return result_df, tax_dict, latest_year
    except: return None, {}, 2025

# ==============================================================================
# [MODULE] Peer Recommender
# ==============================================================================
class PeerRecommender:
    def get_revenue(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            rev = info.get('totalRevenue')
            if not rev:
                fin = t.financials
                if not fin.empty and 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].iloc[0]
            return rev if rev else 0
        except: return 0

    def recommend(self, target_ticker, progress_bar=None):
        try:
            t = yf.Ticker(target_ticker)
            info = safe_yf_info(t)
            ind_key = info.get('industryKey')
            sec_key = info.get('sectorKey')
            group_name = "Unknown"
            top_df = None
            
            if ind_key: industry = yf.Industry(ind_key); top_df = industry.top_companies; group_name = f"Industry: {ind_key}"
            elif sec_key: sector = yf.Sector(sec_key); top_df = sector.top_companies; group_name = f"Sector: {sec_key}"
            
            if top_df is None or top_df.empty: return None, group_name, ["No peers found"]

            raw_list = top_df['symbol'].tolist() if 'symbol' in top_df.columns else top_df.index.tolist()
            candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5]
            
            if progress_bar: progress_bar.progress(0.2, text="Scanning...")
            revenue_map = []
            for idx, ticker in enumerate(candidates):
                time.sleep(random.uniform(1.0, 2.0))
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                if progress_bar: progress_bar.progress(0.2 + (0.8 * (idx/len(candidates))), text=f"Analyzing {ticker}...")
            
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_5 = [item[0] for item in revenue_map][:5]
            return ", ".join(top_5), group_name, []
        except Exception as e: return None, "Error", [str(e)]

# ==============================================================================
# [LOGIC] WACC Engine
# ==============================================================================
class DetailWACCModel:
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
        currency = currency.upper()
        if currency == 'USD': return 1.0, "USD"
        if currency in self.fx_cache: return self.fx_cache[currency], currency
        try:
            for i in range(2):
                try:
                    t = yf.Ticker(f"{currency}USD=X")
                    hist = t.history(period="1d")
                    if not hist.empty: 
                        rate = hist['Close'].iloc[-1]
                        self.fx_cache[currency] = rate
                        return rate, currency
                    else:
                        t = yf.Ticker(f"USD{currency}=X")
                        hist = t.history(period="1d")
                        if not hist.empty: 
                            rate = 1 / hist['Close'].iloc[-1]
                            self.fx_cache[currency] = rate
                            return rate, currency
                except:
                    time.sleep(1)
            return 1.0, currency
        except: return 1.0, currency

    def get_financials_latest(self, ticker):
        # [STRICT MODE] Fail if critical data is missing
        try:
            t = yf.Ticker(ticker)
            info = safe_yf_info(t)
            
            if not info:
                return None, f"⚠️ {ticker}: No data found in Yahoo Finance."

            curr = info.get('currency', 'USD')
            country = info.get('country', 'Unknown')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            mkt_cap_raw = info.get('marketCap', 0)
            
            # Debt Fallback Logic
            debt_raw = info.get('totalDebt', 0)
            if debt_raw == 0:
                try:
                    bs = t.balance_sheet
                    if not bs.empty:
                        for item in ['Total Debt', 'Long Term Debt', 'Total Liab']:
                            if item in bs.index: debt_raw = bs.loc[item].iloc[0]; break
                except: pass
            
            # Revenue Fallback Logic
            rev_raw = info.get('totalRevenue', 0)
            ebitda_raw = info.get('ebitda', 0)
            ebit_raw = 0
            
            try:
                fin = t.financials
                if not fin.empty:
                    if 'EBIT' in fin.index: ebit_raw = fin.loc['EBIT'].iloc[0]
                    elif 'Operating Income' in fin.index: ebit_raw = fin.loc['Operating Income'].iloc[0]
                    if rev_raw == 0 and 'Total Revenue' in fin.index: rev_raw = fin.loc['Total Revenue'].iloc[0]
            except: pass
            
            # [STRICT VALIDATION]
            if mkt_cap_raw == 0: 
                try:
                    mkt_cap_raw = t.fast_info['market_cap']
                except:
                    return None, f"⚠️ {ticker}: Excluded (Missing Market Cap/Price Data)."
            
            if rev_raw == 0:
                 return None, f"⚠️ {ticker}: Excluded (Missing Revenue Data)."

            # Tax Rate Lookup (KPMG)
            tax_rate = self.kpmg_map.get(country.upper(), 25.0) 

            data = {
                "name": info.get('longName', ticker),
                "country": country,
                "currency": curr_code,
                "fx_rate": fx,
                "tax_rate": tax_rate,
                "vals": {
                    "Revenue": rev_raw * fx,
                    "EBIT": ebit_raw * fx,
                    "EBITDA": ebitda_raw * fx,
                    "Total Debt": debt_raw * fx,
                    "Market Cap": mkt_cap_raw * fx
                }
            }
            return data, None
        except Exception as e: 
            return None, f"⚠️ {ticker}: Excluded (API Error: {str(e)})"

    def get_5y_monthly_beta_analysis(self):
        try:
            tickers = self.peers + [self.market_index]
            tickers = list(set([t.strip().upper() for t in tickers if t.strip()]))
            data = yf.download(tickers, period="5y", interval="1mo", progress=False)
            
            prices = data['Adj Close'] if 'Adj Close' in data else data['Close'] if 'Close' in data else None
            if prices is None: return None, None, None, ["Price Error: Failed to download price data."]
            if isinstance(prices, pd.Series): prices = prices.to_frame()

            returns = prices.pct_change()
            if self.market_index not in returns.columns: return None, None, None, ["Market Index Error: ^GSPC not found."]
            
            beta_list = []
            
            for t in self.peers:
                t_up = t.strip().upper()
                if t_up in returns.columns:
                    pair = returns[[t_up, self.market_index]].dropna()
                    if len(pair) < 12:
                        beta_list.append({"Ticker": t, "Raw Beta": np.nan, "Adj Beta": np.nan})
                        continue
                    cov = pair[t_up].cov(pair[self.market_index])
                    var = pair[self.market_index].var()
                    raw = cov / var
                    adj = (0.67 * raw) + (0.33 * 1.0)
                    beta_list.append({"Ticker": t, "Raw Beta": raw, "Adj Beta": adj})
                else:
                    beta_list.append({"Ticker": t, "Raw Beta": np.nan, "Adj Beta": np.nan})
            
            prices_disp = prices.copy(); prices_disp.index = prices_disp.index.strftime('%Y-%m-%d')
            return pd.DataFrame(beta_list), prices_disp, None, []
        except Exception as e: return None, None, None, [str(e)]

    def run(self):
        # 1. Get Beta Data
        beta_df, prices, _, beta_err = self.get_5y_monthly_beta_analysis()
        error_logs = beta_err if beta_err else []
        
        # 2. Get Financials with Throttling & Strict Filtering
        peer_data = []
        progress_text = st.empty()
        
        for idx, p in enumerate(self.peers):
            progress_text.text(f"⏳ Analyzing {p} ({idx+1}/{len(self.peers)})...")
            time.sleep(random.uniform(1.5, 3.0)) 
            
            fin, err = self.get_financials_latest(p)
            if err:
                error_logs.append(err)
                continue # Skip bad peer
            
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
                    "Company": fin['name'], # Alias for easier access
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
                    "Debt/TIC Ratio": dtic_ratio
                })
        
        progress_text.empty()
        
        df_peers = pd.DataFrame(peer_data)
        
        # 3. Merge
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
            "prices": prices,
            "market_params": {"Rm": rm, "MRP": mrp},
            "rf_trend": self.rf_trend_df,
            "gdp_df": self.gdp_df,
            "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
with st.sidebar:
    st.header("Target & Peers")
    target_ticker = st.text_input("Target Ticker", "WOLF")
    
    col1, col2 = st.columns([1,1])
    if col1.button("🤖 Auto-Recommend Peers (Top 5)", type="secondary", use_container_width=True):
        with st.spinner("Finding..."):
            rec = PeerRecommender()
            res_peers, group, logs = rec.recommend(target_ticker)
            if res_peers: st.session_state['peers'] = res_peers
            else: st.warning("추천 실패")
            
    peers_input = st.text_area("Peer Tickers", value=st.session_state.get('peers', "ON, STM, IFX.DE"), height=100)
    st.caption("※ Top 5 revenue companies in the industry\n(Source: Yahoo Finance Industry/Sector Data)")
    
    st.divider()
    st.header("Assumptions")
    with st.expander("Cost of Equity / Debt", expanded=True):
        latest_gdp, df_gdp_disp, latest_rf, rf_trend_df = get_fred_data()
        rf_in = st.number_input(f"Risk Free Rate (Latest: {latest_rf:.2f}%)", value=latest_rf, step=0.01)
        crp_in = st.number_input("Country Risk Premium (%)", value=0.0, step=0.1)
        size_in = st.number_input("Size Premium (%)", value=0.0, step=0.1)
    
    with st.expander("Implied Return", expanded=True):
        avg_bb, avg_div, _, _ = get_sp_buyback_data()
        bb_in = st.number_input(f"Buyback Yield (5Y Avg: {avg_bb:.2f}%)", value=avg_bb, step=0.1)
        div_in = st.number_input(f"Dividend Yield (5Y Avg: {avg_div:.2f}%)", value=avg_div, step=0.1)
        g_in = st.number_input(f"Growth Rate (Latest GDP: {latest_gdp:.2f}%)", value=latest_gdp, step=0.1)
        
    with st.expander("Target Assumptions", expanded=True):
        tax_in = st.slider("Tax Rate (%)", 0.0, 40.0, 25.0, 1.0)

    st.divider()
    if st.button("Calculate WACC", type="primary", use_container_width=True):
        model = DetailWACCModel(
            target_ticker, peers_input, rf_in, crp_in, size_in, 
            bb_in, div_in, g_in, tax_in, rf_trend_df, df_gdp_disp
        )
        with st.spinner("Calculating..."):
            st.session_state['result'] = model.run()
            st.session_state['inputs'] = {
                'rf': rf_in, 'crp': crp_in, 'sp': size_in, 'tax': tax_in,
                'bb': bb_in, 'div': div_in, 'g': g_in
            }

if 'result' in st.session_state:
    res = st.session_state['result']
    inp = st.session_state['inputs']
    df_init = res['full_df']
    m = res['market_params']
    
    # 1. Top Container for Results
    results_container = st.container()
    
    # 2. Beta Analysis Section
    st.subheader("Beta Analysis")
    sens_method = st.radio("Sensitivity Selection (Aggregation Method)", 
                           ["Average", "Median", "Maximum", "Minimum"], horizontal=True, index=1)

    target_relevered_beta=0; ke=0; kd=0; wacc=0; wd=0; we=0; target_de=0; sel_dtic=0

    # Display Excluded Peers (Transparency)
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
        spread = 0.02
        kd = ((inp['rf']/100) + spread) * (1 - inp['tax']/100)
        wd = sel_dtic
        we = 1 - sel_dtic
        wacc = (we * ke) + (wd * kd)

        with st.expander("5-Year Monthly Beta Analysis Table", expanded=True):
            cols_show = ["Ticker", "Company Name", "Country", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", "Tax Rate", "Raw Beta", "Adj Beta", "Unlevered Beta", "Re-levered Beta"]
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
                st.info(f"({inp['rf']:.2f}% + 2.00%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**")
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
    st.latex(r"K_d = (R_f + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
    spread = 2.0
    st.info(f"**Calculation:** ({inp['rf']:.2f}% + {spread:.2f}%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    d2.metric("Credit Spread", f"{spread:.2f}%")
    d3.metric("Pre-tax Cost of Debt", f"{(inp['rf'] + spread):.2f}%")
    d4.metric("Tax Rate", f"{inp['tax']:.1f}%")
    d5.metric("After-tax Cost of Debt", f"{kd:.2%}")

    st.markdown("---")
    st.subheader("Peer Group Analysis (Financials)")
    if not df_init.empty:
        fin_cols = ["Ticker", "Company Name", "Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio"]
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
    t1, t2, t3, t4, t5 = st.tabs(["📉 Risk Free Rate", "📈 US GDP Growth", "📊 S&P 500 Yields", "🏛️ KPMG Corp Tax", "📉 US Corp Spreads"])
    with t1:
        st.caption("Source: FRED (St. Louis Fed) - Series DGS10")
        if res.get('rf_trend') is not None: st.line_chart(res['rf_trend'].set_index("Date")["Rate"], color="#FF4B4B")
    with t2:
        st.caption("Source: FRED (St. Louis Fed) - Series A191RP1A027NBEA")
        if res.get('gdp_df') is not None:
            st.dataframe(res['gdp_df'], use_container_width=True, hide_index=True,
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
        oas_df = get_fred_oas_data()
        if not oas_df.empty:
            st.dataframe(oas_df, use_container_width=True, hide_index=True, 
                         column_config={
                             "Latest Spread (%)": st.column_config.NumberColumn(format="%.2f%%"),
                             "Series ID": st.column_config.LinkColumn(display_text="View on FRED")
                         })
