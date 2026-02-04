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
# [MODULE] Data Fetcher 1: NYU Stern (Buyback & Dividend)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_sp_buyback_data():
    """
    NYU Stern (Aswath Damodaran) S&P 500 Earnings & Dividends HTML 데이터 크롤링
    """
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    default_bb_yield = 2.0 
    default_div_yield = 1.5
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # HTML 표 읽기
        dfs = pd.read_html(io.StringIO(response.text), header=0)
        
        df = None
        for d in dfs:
            cols_str = [str(c).lower() for c in d.columns]
            if "year" in cols_str and "s&p 500" in cols_str:
                df = d
                break
        
        if df is None:
            return default_bb_yield, default_div_yield, None, ["⚠️ NYU Stern: HTML Table Structure Changed"]

        # 컬럼 매핑
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

        display_df = clean_df[["Year", "S&P 500", "Dividends", "Buybacks", "Dividend Yield %", "Buyback Yield %", "Total Yield %"]].copy()
        
        return avg_bb_yield, avg_div_yield, display_df, []

    except Exception as e:
        return default_bb_yield, default_div_yield, None, [f"⚠️ NYU Stern Error: {str(e)}"]

# ==============================================================================
# [MODULE] Data Fetcher 2 & 3: FRED Data (GDP & RF)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_fred_data():
    """
    Fetches GDP Growth and Risk Free Rate (DGS10) from FRED.
    Returns: latest_gdp, gdp_df, latest_rf, rf_trend_df
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    # 1. GDP
    try:
        url_gdp = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A191RP1A027NBEA"
        r_gdp = requests.get(url_gdp, headers=headers, timeout=10)
        df_gdp = pd.read_csv(io.StringIO(r_gdp.text))
        df_gdp.columns = ["Date", "GDP Growth %"]
        df_gdp["Date"] = pd.to_datetime(df_gdp["Date"])
        df_gdp["Year"] = df_gdp["Date"].dt.year
        df_gdp = df_gdp.sort_values(by="Date", ascending=False)
        latest_gdp = df_gdp["GDP Growth %"].iloc[0]
        df_gdp_disp = df_gdp.head(10).copy()
    except: 
        latest_gdp = 2.0
        df_gdp_disp = None

    # 2. Risk Free Rate (DGS10)
    try:
        url_rf = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        r_rf = requests.get(url_rf, headers=headers, timeout=10)
        df_rf = pd.read_csv(io.StringIO(r_rf.text))
        df_rf.columns = ["Date", "Rate"]
        df_rf["Date"] = pd.to_datetime(df_rf["Date"])
        df_rf["Rate"] = pd.to_numeric(df_rf["Rate"], errors='coerce')
        df_rf = df_rf.dropna().sort_values(by="Date", ascending=False)
        
        latest_rf = df_rf["Rate"].iloc[0]
        
        # 5-Year Trend for Graph
        cutoff = df_rf["Date"].max() - timedelta(days=365*5)
        df_rf_trend = df_rf[df_rf["Date"] >= cutoff].copy()
    except: 
        latest_rf = 4.0
        df_rf_trend = None

    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend

# ==============================================================================
# [MODULE] Data Fetcher 4: OECD Tax Rates (Manual/Static Fallback)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_oecd_tax_rates():
    """
    Returns OECD Combined Corporate Income Tax Rates (2023/2024 estimates).
    Scraping the OECD Data Explorer directly is unstable due to JS rendering.
    """
    # Data Source: OECD Tax Database (Combined Corporate Income Tax Rates)
    data = {
        "Country": [
            "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
            "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania", "Luxembourg",
            "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal", "Slovak Republic",
            "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States"
        ],
        "Combined Tax Rate (%)": [
            30.0, 23.0, 25.0, 26.2, 27.0, 35.0, 30.0, 21.0,
            22.0, 20.0, 20.0, 25.8, 29.9, 22.0, 9.0, 20.0,
            12.5, 23.0, 27.8, 29.7, 23.2, 20.0, 15.0, 24.9,
            30.0, 25.8, 28.0, 22.0, 19.0, 21.0, 21.0,
            19.0, 25.0, 20.6, 19.7, 25.0, 25.0, 25.8
        ]
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="Combined Tax Rate (%)", ascending=False).reset_index(drop=True)
    return df

# ==============================================================================
# [MODULE] Peer Recommender (Anti-Ban)
# ==============================================================================
class PeerRecommender:
    def get_revenue(self, ticker):
        try:
            t = yf.Ticker(ticker)
            rev = t.info.get('totalRevenue')
            if not rev:
                fin = t.financials
                if not fin.empty and 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].iloc[0]
            return rev if rev else 0
        except: return 0

    def recommend(self, target_ticker, progress_bar=None):
        logs = []
        try:
            t = yf.Ticker(target_ticker)
            info = t.info
            ind_key = info.get('industryKey')
            sec_key = info.get('sectorKey')
            
            group_name = "Unknown"
            top_df = None
            
            if ind_key:
                industry = yf.Industry(ind_key)
                top_df = industry.top_companies
                group_name = f"Industry: {ind_key}"
            elif sec_key:
                sector = yf.Sector(sec_key)
                top_df = sector.top_companies
                group_name = f"Sector: {sec_key}"
            
            if top_df is None or top_df.empty:
                return None, group_name, ["No peer data found in Yahoo Finance."]

            if 'symbol' in top_df.columns: raw_list = top_df['symbol'].tolist()
            elif 'Symbol' in top_df.columns: raw_list = top_df['Symbol'].tolist()
            else: raw_list = top_df.index.tolist()
            
            # Target 제외, 최대 5개
            candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5]
            
            if progress_bar: progress_bar.progress(0.2, text="Collecting Revenue Data...")
            
            revenue_map = []
            for idx, ticker in enumerate(candidates):
                # Anti-ban Delay
                time.sleep(random.uniform(1.5, 3.0))
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                
                if progress_bar:
                    progress_bar.progress(0.2 + (0.8 * (idx / len(candidates))), text=f"Analyzing {ticker}...")
            
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_5 = [item[0] for item in revenue_map][:5]
            
            return ", ".join(top_5), group_name, logs

        except Exception as e:
            return None, "Error", [f"Recommendation Error: {str(e)}"]

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
        self.gdp_df = gdp_df # For passing to result
        self.market_index = "^GSPC"
        self.fx_cache = {}

    def get_exchange_rate_to_usd(self, currency):
        currency = currency.upper()
        if currency == 'USD': return 1.0, "USD"
        if currency in self.fx_cache: return self.fx_cache[currency], currency
        try:
            # Try Direct Quote
            t = yf.Ticker(f"{currency}USD=X")
            hist = t.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
            else:
                # Try Indirect Quote
                t = yf.Ticker(f"USD{currency}=X")
                hist = t.history(period="1d")
                if not hist.empty: rate = 1 / hist['Close'].iloc[-1]
                else: rate = 1.0
            
            self.fx_cache[currency] = rate
            return rate, currency
        except: return 1.0, currency

    def get_financials_latest(self, ticker):
        """
        Fetches latest financial data and converts to USD.
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info
            curr = info.get('currency', 'USD')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            mkt_cap_raw = info.get('marketCap', 0)
            mkt_cap_usd = mkt_cap_raw * fx
            
            debt_raw = info.get('totalDebt', 0)
            if debt_raw == 0:
                bs = t.balance_sheet
                if not bs.empty:
                    for item in ['Total Debt', 'Long Term Debt', 'Total Liab']:
                        if item in bs.index:
                            debt_raw = bs.loc[item].iloc[0]
                            break
            debt_usd = debt_raw * fx
            
            rev_raw = info.get('totalRevenue', 0)
            ebitda_raw = info.get('ebitda', 0)
            ebit_raw = 0
            
            # Fallback for I/S items
            fin = t.financials
            if not fin.empty:
                if 'EBIT' in fin.index: ebit_raw = fin.loc['EBIT'].iloc[0]
                elif 'Operating Income' in fin.index: ebit_raw = fin.loc['Operating Income'].iloc[0]
                
                if rev_raw == 0 and 'Total Revenue' in fin.index:
                    rev_raw = fin.loc['Total Revenue'].iloc[0]
            
            vals_usd = {
                "Revenue": rev_raw * fx,
                "EBIT": ebit_raw * fx,
                "EBITDA": ebitda_raw * fx,
                "Total Debt": debt_usd,
                "Market Cap": mkt_cap_usd
            }
            
            return {
                "name": info.get('longName', ticker),
                "currency": curr_code,
                "fx_rate": fx,
                "vals": vals_usd
            }
        except:
            return None

    def get_5y_monthly_beta_analysis(self):
        """
        Calculates 5-Year Monthly Beta with robust fetching.
        """
        try:
            tickers = self.peers + [self.market_index]
            tickers = list(set([t.strip().upper() for t in tickers if t.strip()]))
            
            data = yf.download(tickers, period="5y", interval="1mo", progress=False)
            
            prices = None
            if 'Adj Close' in data:
                prices = data['Adj Close']
            elif 'Close' in data:
                prices = data['Close']
            else:
                if isinstance(data, pd.DataFrame): prices = data # Fallback
            
            if prices is None: return None, None, None, ["Failed to download price data."]

            returns = prices.pct_change()
            
            if self.market_index not in returns.columns:
                return None, None, None, ["Market Index (^GSPC) data missing."]
            
            beta_list = []
            
            # Original list order
            check_list = [p.strip().upper() for p in self.peers]
            
            for t in check_list:
                if t in returns.columns:
                    # Drop NA only for the pair
                    pair_data = returns[[t, self.market_index]].dropna()
                    
                    if len(pair_data) < 12: # Less than 1 year data
                        beta_list.append({
                            "Ticker": t, "Raw Beta": np.nan, "Adj Beta": np.nan, 
                            "Correlation": np.nan, "Note": "Insufficient Data"
                        })
                        continue

                    stock_ret = pair_data[t]
                    mkt_ret = pair_data[self.market_index]
                    
                    cov = stock_ret.cov(mkt_ret)
                    var = mkt_ret.var()
                    
                    raw_beta = cov / var
                    adj_beta = (0.67 * raw_beta) + (0.33 * 1.0)
                    
                    beta_list.append({
                        "Ticker": t, 
                        "Raw Beta": raw_beta,
                        "Adj Beta": adj_beta,
                        "Correlation": stock_ret.corr(mkt_ret),
                        "Note": f"{len(pair_data)} mo"
                    })
            
            # Format prices/returns for display
            prices_disp = prices.copy()
            prices_disp.index = prices_disp.index.strftime('%Y-%m-%d')
            returns_disp = returns.copy()
            returns_disp.index = returns_disp.index.strftime('%Y-%m-%d')
            
            # Move Market Index to first column
            cols = [self.market_index] + [c for c in returns_disp.columns if c != self.market_index]
            returns_disp = returns_disp[cols]
            
            return pd.DataFrame(beta_list), prices_disp, returns_disp, []
            
        except Exception as e:
            return None, None, None, [f"Beta Analysis Error: {str(e)}"]

    def run(self):
        # 1. Get Beta
        beta_df, prices, rets, beta_logs = self.get_5y_monthly_beta_analysis()
        error_logs = beta_logs if beta_logs else []
        
        # 2. Get Financials & Calculate Ratios
        peer_data = []
        for p in self.peers:
            fin = self.get_financials_latest(p)
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
        
        df_peers = pd.DataFrame(peer_data)
        
        # 3. Merge & Unlever Beta
        if beta_df is not None and not beta_df.empty and not df_peers.empty:
            full_df = pd.merge(df_peers, beta_df, on="Ticker", how="left")
            # Unlevered Beta = Levered Beta / (1 + (1 - T) * D/E)
            # Using Global Tax Rate Assumption
            full_df["Unlevered Beta"] = full_df["Adj Beta"] / (1 + (1 - self.tax) * full_df["D/E Ratio"])
        else:
            full_df = pd.DataFrame()

        # 4. Market Return Parameters
        rm = self.div_yield + self.buyback_yield + self.growth_rate
        mrp = rm - self.rf

        return {
            "full_df": full_df,
            "prices": prices,
            "returns": rets,
            "market_params": {"Rm": rm, "MRP": mrp},
            "rf_trend": self.rf_trend_df, # Pass for graphing
            "gdp_df": self.gdp_df, # Pass for Table
            "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard Layout
# ==============================================================================
# Sidebar
with st.sidebar:
    st.header("Target & Peers")
    target_ticker = st.text_input("Target Ticker", "WOLF")
    
    col1, col2 = st.columns([1,1])
    if col1.button("🤖 경쟁사 자동 추천 (Top 5)", type="secondary"):
        with st.spinner("Finding Peers..."):
            rec = PeerRecommender()
            res_peers, group, logs = rec.recommend(target_ticker)
            if res_peers: st.session_state['peers'] = res_peers
            else: st.warning("No peers found.")
            
    peers_input = st.text_area("Peer Tickers", value=st.session_state.get('peers', "ON, STM, IFX.DE"), height=100)
    st.caption("※ 산업 내 매출액(Revenue) 상위 5개 기업")
    
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

# Main Content
if 'result' in st.session_state:
    res = st.session_state['result']
    inp = st.session_state['inputs']
    df = res['full_df']
    m = res['market_params']
    
    # -------------------------------------------------------------------------
    # [LOGIC] Dynamic Sensitivity (Calculated BEFORE Displaying WACC)
    # -------------------------------------------------------------------------
    st.subheader("Beta Analysis")
    
    sens_method = st.radio("Sensitivity Selection (Aggregation Method)", 
                           ["Average", "Median", "Maximum", "Minimum"], horizontal=True)
    
    # Initialize variables to avoid NameError if df is empty
    target_relevered_beta = 0.0
    ke = 0.0
    kd = 0.0
    wacc = 0.0
    wd = 0.0
    we = 0.0
    target_de = 0.0
    sel_dtic = 0.0
    sel_unlev_beta = 0.0

    if not df.empty:
        # 1. Aggregate
        if sens_method == "Average":
            sel_unlev_beta = df["Unlevered Beta"].mean()
            sel_dtic = df["Debt/TIC Ratio"].mean()
        elif sens_method == "Median":
            sel_unlev_beta = df["Unlevered Beta"].median()
            sel_dtic = df["Debt/TIC Ratio"].median()
        elif sens_method == "Maximum":
            sel_unlev_beta = df["Unlevered Beta"].max()
            sel_dtic = df["Debt/TIC Ratio"].max()
        else:
            sel_unlev_beta = df["Unlevered Beta"].min()
            sel_dtic = df["Debt/TIC Ratio"].min()
            
        # 2. Derive Target D/E from Target Debt/TIC
        # D/E = (D/TIC) / (1 - D/TIC)
        target_de = sel_dtic / (1 - sel_dtic) if (1 - sel_dtic) != 0 else 0
        
        # 3. Re-lever Beta for Target (Hamada)
        target_relevered_beta = sel_unlev_beta * (1 + (1 - inp['tax']/100) * target_de)
        
        # 4. Update Table Column "Re-levered Beta" (What if Peers had Target Structure?)
        df["Re-levered Beta"] = df["Unlevered Beta"] * (1 + (1 - inp['tax']/100) * target_de)
        
        # 5. Cost of Equity
        ke = (inp['rf']/100) + (target_relevered_beta * m['MRP']) + (inp['crp']/100) + (inp['sp']/100)
        
        # 6. Cost of Debt
        spread = 0.02 # Assumption
        kd = ((inp['rf']/100) + spread) * (1 - inp['tax']/100)
        
        # 7. WACC
        wd = sel_dtic
        we = 1 - sel_dtic
        wacc = (we * ke) + (wd * kd)
    
    # -------------------------------------------------------------------------
    # [SECTION 1] WACC Results (Dynamic)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("WACC Calculation & Results")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final WACC", f"{wacc:.2%}")
    c2.metric("Cost of Equity", f"{ke:.2%}")
    c3.metric("Cost of Debt (A-T)", f"{kd:.2%}")
    c4.metric("Re-levered Beta", f"{target_relevered_beta:.2f}")
    
    st.caption(f"**Target Structure ({sens_method}):** Debt {wd:.1%} | Equity {we:.1%} (Implied D/E: {target_de:.2%})")

    # -------------------------------------------------------------------------
    # [SECTION 2] Beta Analysis Details
    # -------------------------------------------------------------------------
    st.markdown("---")
    # Subheader 2 already displayed above
    
    with st.expander("Target Capital Structure & Beta Summary", expanded=True):
        st.info(f"**Method:** {sens_method} of Peers. **Target D/E:** {target_de:.2%}. **Unlevered Beta:** {sel_unlev_beta:.2f}")
        st.write("The 'Re-levered Beta' column below applies the Target's Capital Structure to each peer's Unlevered Beta.")

    with st.expander("5-Year Monthly Beta Analysis Table", expanded=True):
        if not df.empty:
            disp_df = df.copy()
            # Requested Column Order
            cols_show = ["Ticker", "Company Name", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", 
                         "Raw Beta", "Adj Beta", "Unlevered Beta", "Re-levered Beta"]
            
            # Format Large Numbers
            disp_df["Total Debt"] = disp_df.apply(lambda x: f"{x['Currency']} {x['Total Debt']/1e9:,.2f}B", axis=1)
            disp_df["Market Cap"] = disp_df.apply(lambda x: f"{x['Currency']} {x['Market Cap']/1e9:,.2f}B", axis=1)
            
            st.dataframe(
                disp_df[cols_show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "D/E Ratio": st.column_config.NumberColumn(format="%.2f"),
                    "Debt/TIC Ratio": st.column_config.NumberColumn(format="%.2f"),
                    "Raw Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Adj Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Unlevered Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Re-levered Beta": st.column_config.NumberColumn(format="%.2f"),
                }
            )
            st.caption(f"* Tax Rate used: {inp['tax']}%")
        else:
            st.error("No Beta Data Available. Check Tickers.")

    # -------------------------------------------------------------------------
    # [SECTION 3] Cost of Equity
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Cost of Equity")
    st.latex(r"K_e = R_f + \beta_{L} \times (R_m - R_f) + CRP + SP")
    
    # Show Numerical Breakdown
    k_calc_info = f"**Calculation:** {inp['rf']:.2f}% + ({target_relevered_beta:.2f} × {(m['MRP']*100):.2f}%) + {inp['crp']:.2f}% + {inp['sp']:.2f}% = **{ke*100:.2f}%**"
    st.info(k_calc_info)

    k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
    k_col1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    k_col2.metric("Beta (Re-levered)", f"{target_relevered_beta:.2f}")
    k_col3.metric("Market Risk Prem", f"{m['MRP']*100:.2f}%")
    k_col4.metric("Country Risk Prem", f"{inp['crp']:.2f}%")
    k_col5.metric("Size Premium", f"{inp['sp']:.2f}%")
    
    with st.expander("Implied Market Return Details"):
        st.write(f"**Implied Market Return ($R_m$): {m['Rm']:.2%}**")
        st.write(f"= Buyback Yield ({inp['bb']:.2f}%) + Dividend Yield ({inp['div']:.2f}%) + Growth Rate ({inp['g']:.2f}%)")

    # -------------------------------------------------------------------------
    # [SECTION 4] Cost of Debt
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Cost of Debt")
    st.latex(r"K_d = (R_f + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
    
    # Show Numerical Breakdown
    d_spread = 2.0
    d_calc_info = f"**Calculation:** ({inp['rf']:.2f}% + {d_spread:.2f}%) × (1 - {inp['tax']:.2f}%) = **{kd*100:.2f}%**"
    st.info(d_calc_info)

    d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
    d_col1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    d_col2.metric("Credit Spread", f"{d_spread:.2f}%")
    d_col3.metric("Pre-tax Cost of Debt", f"{(inp['rf'] + d_spread):.2f}%")
    d_col4.metric("Tax Rate", f"{inp['tax']:.1f}%")
    d_col5.metric("After-tax Cost of Debt", f"{kd:.2%}")

    # -------------------------------------------------------------------------
    # [SECTION 5] Peer Group Analysis (Financials)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Peer Group Analysis (Financials)")
    if not df.empty:
        fin_cols = ["Ticker", "Company Name", "Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio"]
        fin_df = df.copy()
        
        # Convert to Billions
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
                "D/E Ratio": st.column_config.NumberColumn(format="%.2f"),
                "Debt/TIC Ratio": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        st.caption("Note: Financial figures converted to USD Billions using latest FX rates.")
        st.markdown("**Applied FX Rates (to USD):**")
        st.dataframe(df[["Ticker", "Currency", "FX Rate"]].T, use_container_width=True)

    # -------------------------------------------------------------------------
    # [SECTION 6] Market Data Reference
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Market Data Reference")
    t1, t2, t3, t4 = st.tabs(["📉 Risk Free Rate", "📈 US GDP Growth", "📊 S&P 500 Yields", "🏛️ OECD Corp Tax"])
    
    with t1:
        st.caption("Source: FRED (St. Louis Fed) - Series DGS10")
        if res.get('rf_trend') is not None: 
            st.line_chart(res['rf_trend'].set_index("Date")["Rate"], color="#FF4B4B")
        else: st.warning("Trend Chart data unavailable")
        
    with t2:
        st.caption("Source: FRED (St. Louis Fed) - Series A191RP1A027NBEA")
        if res.get('gdp_df') is not None:
            st.dataframe(
                res['gdp_df'],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "GDP Growth %": st.column_config.NumberColumn("GDP Growth (%)", format="%.2f%%")
                }
            )
        else: st.warning("GDP Data unavailable")

    with t3:
        st.caption("Source: Aswath Damodaran (NYU Stern)")
        _, _, sp_table, _ = get_sp_buyback_data()
        if sp_table is not None: st.dataframe(sp_table, use_container_width=True)

    with t4:
        st.caption("Source: OECD Data Explorer (Combined Corporate Income Tax Rates, 2023-2024 Estimates)")
        oecd_df = get_oecd_tax_rates()
        st.dataframe(
            oecd_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Combined Tax Rate (%)": st.column_config.NumberColumn(format="%.1f%%")
            }
        )
