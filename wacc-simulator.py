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
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    default_bb_yield = 2.0 
    default_div_yield = 1.5
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        dfs = pd.read_html(io.StringIO(response.text), header=0)
        df = None
        for d in dfs:
            cols_str = [str(c).lower() for c in d.columns]
            if "year" in cols_str and "s&p 500" in cols_str:
                df = d
                break
        
        if df is None: return default_bb_yield, default_div_yield, None, ["⚠️ NYU Stern Data Fetch Error"]

        # Column Mapping
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
    except Exception as e:
        return default_bb_yield, default_div_yield, None, [str(e)]

# ==============================================================================
# [MODULE] Data Fetcher 2 & 3: FRED Data
# ==============================================================================
@st.cache_data(ttl=3600*24)
def get_fred_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    # GDP
    try:
        r_gdp = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=A191RP1A027NBEA", headers=headers)
        df_gdp = pd.read_csv(io.StringIO(r_gdp.text))
        df_gdp["Date"] = pd.to_datetime(df_gdp["Date"])
        df_gdp.columns = ["Date", "Value"]
        latest_gdp = df_gdp.sort_values(by="Date", ascending=False)["Value"].iloc[0]
        df_gdp_disp = df_gdp.sort_values(by="Date", ascending=False).head(10)
    except: latest_gdp = 2.0; df_gdp_disp = None

    # Risk Free Rate
    try:
        r_rf = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", headers=headers)
        df_rf = pd.read_csv(io.StringIO(r_rf.text))
        df_rf["Date"] = pd.to_datetime(df_rf["Date"])
        df_rf.columns = ["Date", "Rate"]
        df_rf["Rate"] = pd.to_numeric(df_rf["Rate"], errors='coerce')
        df_rf = df_rf.dropna().sort_values(by="Date", ascending=False)
        latest_rf = df_rf["Rate"].iloc[0]
        cutoff = df_rf["Date"].max() - timedelta(days=365*5)
        df_rf_trend = df_rf[df_rf["Date"] >= cutoff]
    except: latest_rf = 4.0; df_rf_trend = None

    return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend

# ==============================================================================
# [MODULE] Peer Recommender
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
            
            group_name = f"Industry: {ind_key}" if ind_key else f"Sector: {sec_key}"
            
            if ind_key: industry = yf.Industry(ind_key); top_df = industry.top_companies
            elif sec_key: sector = yf.Sector(sec_key); top_df = sector.top_companies
            else: return None, "Unknown", ["Industry info not found"]

            if top_df is not None and not top_df.empty:
                raw_list = top_df['symbol'].tolist() if 'symbol' in top_df.columns else top_df.index.tolist()
                candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:5] # Top 5 limit
            else: return None, group_name, ["No peers found"]

            revenue_map = []
            if progress_bar: progress_bar.progress(0.1, text="Scanning Peers...")
            
            for idx, ticker in enumerate(candidates):
                time.sleep(random.uniform(1.0, 2.0)) # Anti-ban delay
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                if progress_bar: progress_bar.progress(0.1 + (0.8 * (idx/len(candidates))), text=f"Scanning {ticker}...")
            
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_5 = [item[0] for item in revenue_map][:5]
            
            return ", ".join(top_5), group_name, logs
        except Exception as e:
            return None, "Error", [str(e)]

# ==============================================================================
# [LOGIC] WACC Model
# ==============================================================================
class DetailWACCModel:
    def __init__(self, target, peers, rf_rate, crp, size_prem, buyback, div_yield, growth, tax):
        self.target = target
        self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.rf = rf_rate / 100
        self.crp = crp / 100
        self.size_prem = size_prem / 100
        self.buyback_yield = buyback / 100
        self.div_yield = div_yield / 100
        self.growth_rate = growth / 100
        self.tax = tax / 100
        self.market_index = "^GSPC"
        self.fx_cache = {}

    def get_exchange_rate_to_usd(self, currency):
        currency = currency.upper()
        if currency == 'USD': return 1.0, "USD"
        if currency in self.fx_cache: return self.fx_cache[currency], currency
        try:
            pair = f"{currency}USD=X"
            if currency in ['EUR', 'GBP', 'AUD']: pair = f"{currency}USD=X" # Direct
            else: pair = f"{currency}=X" # Indirect? No, standard is pair=X. Let's try direct fetch.
            
            # Trying standard pairs
            t = yf.Ticker(f"{currency}USD=X")
            hist = t.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
            else:
                # Try Inverse
                t = yf.Ticker(f"USD{currency}=X")
                hist = t.history(period="1d")
                if not hist.empty: rate = 1 / hist['Close'].iloc[-1]
                else: rate = 1.0
            
            self.fx_cache[currency] = rate
            return rate, currency
        except: return 1.0, currency

    def get_financials_latest(self, ticker):
        # Fetch LATEST data for Peer Group Analysis
        try:
            t = yf.Ticker(ticker)
            info = t.info
            curr = info.get('currency', 'USD')
            fx, curr_code = self.get_exchange_rate_to_usd(curr)
            
            # Market Cap
            mkt_cap_raw = info.get('marketCap', 0)
            mkt_cap_usd = mkt_cap_raw * fx
            
            # Debt (Total Debt)
            debt_raw = info.get('totalDebt', 0)
            if debt_raw == 0:
                # Fallback to balance sheet
                bs = t.balance_sheet
                if not bs.empty:
                    if 'Total Debt' in bs.index: debt_raw = bs.loc['Total Debt'].iloc[0]
                    elif 'Long Term Debt' in bs.index: debt_raw = bs.loc['Long Term Debt'].iloc[0]
            debt_usd = debt_raw * fx
            
            # Revenue, EBIT, EBITDA (TTM or Latest FY) - Instruction says "Latest data"
            rev_raw = info.get('totalRevenue', 0)
            ebitda_raw = info.get('ebitda', 0)
            # EBIT approximation
            ebit_raw = 0
            fin = t.financials
            if not fin.empty:
                if 'EBIT' in fin.index: ebit_raw = fin.loc['EBIT'].iloc[0]
                elif 'Operating Income' in fin.index: ebit_raw = fin.loc['Operating Income'].iloc[0]
            
            # TTM prefered for I/S items in valuation, but if 0, take latest FY
            if rev_raw == 0 and not fin.empty and 'Total Revenue' in fin.index: rev_raw = fin.loc['Total Revenue'].iloc[0]
            
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

    def get_beta_data_5y(self):
        # 5Y Monthly Adjusted Beta
        try:
            tickers = self.peers + [self.market_index]
            tickers = list(set([x.strip().upper() for x in tickers if x.strip()]))
            data = yf.download(tickers, period="5y", interval="1mo", progress=False)['Adj Close']
            
            returns = data.pct_change()
            if self.market_index not in returns.columns: return None
            
            mkt_ret = returns[self.market_index]
            mkt_var = mkt_ret.var()
            
            beta_results = []
            
            for p in self.peers:
                p = p.strip().upper()
                if p in returns.columns:
                    # Pairwise dropna to maximize data usage
                    valid = returns[[p, self.market_index]].dropna()
                    if len(valid) < 12:
                        beta_results.append({"Ticker": p, "Raw Beta": np.nan, "Adj Beta": np.nan})
                        continue
                    
                    cov = valid[p].cov(valid[self.market_index])
                    raw_beta = cov / mkt_var
                    adj_beta = (0.67 * raw_beta) + (0.33 * 1.0)
                    beta_results.append({"Ticker": p, "Raw Beta": raw_beta, "Adj Beta": adj_beta})
            
            return pd.DataFrame(beta_results), data, returns
        except: return None, None, None

    def run(self):
        # 1. Beta Analysis
        beta_df, prices, rets = self.get_beta_data_5y()
        
        # 2. Financials for Weighting
        peer_data = []
        for p in self.peers:
            fin = self.get_financials_latest(p)
            if fin:
                d = fin['vals']
                equity = d['Market Cap']
                debt = d['Total Debt']
                tic = equity + debt
                de_ratio = debt / equity if equity > 0 else 0
                dtic_ratio = debt / tic if tic > 0 else 0
                
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
        
        # Merge Beta with Financials
        if beta_df is not None and not df_peers.empty:
            full_df = pd.merge(df_peers, beta_df, on="Ticker", how="left")
            
            # Unlevered Beta Calculation (Hamada)
            # Unlevered Beta = Levered Beta / (1 + (1 - Tax) * D/E)
            # Using Global Tax Rate Assumption
            full_df["Unlevered Beta"] = full_df["Adj Beta"] / (1 + (1 - self.tax) * full_df["D/E Ratio"])
        else:
            full_df = pd.DataFrame()

        # 3. Market Return
        rm = self.div_yield + self.buyback_yield + self.growth_rate
        mrp = rm - self.rf

        return {
            "full_df": full_df,
            "prices": prices,
            "returns": rets,
            "market_params": {"Rm": rm, "MRP": mrp}
        }

# ==============================================================================
# [UI] Dashboard Layout
# ==============================================================================
# Sidebar Inputs
with st.sidebar:
    st.header("1. Target & Peers")
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
    st.header("2. Assumptions")
    
    with st.expander("2-1) Cost of Equity / Debt", expanded=True):
        latest_gdp, _, latest_rf, _ = get_fred_data()
        rf_in = st.number_input(f"Risk Free Rate (Latest: {latest_rf:.2f}%)", value=latest_rf, step=0.01)
        crp_in = st.number_input("Country Risk Premium (%)", value=0.0, step=0.1)
        size_in = st.number_input("Size Premium (%)", value=0.0, step=0.1)
    
    with st.expander("2-2) Implied Return", expanded=True):
        avg_bb, avg_div, _, _ = get_sp_buyback_data()
        bb_in = st.number_input(f"Buyback Yield (5Y Avg: {avg_bb:.2f}%)", value=avg_bb, step=0.1)
        div_in = st.number_input(f"Dividend Yield (5Y Avg: {avg_div:.2f}%)", value=avg_div, step=0.1)
        g_in = st.number_input(f"Growth Rate (Latest GDP: {latest_gdp:.2f}%)", value=latest_gdp, step=0.1)
        
    with st.expander("2-3) Target Assumptions", expanded=True):
        tax_in = st.slider("Tax Rate (%)", 0.0, 40.0, 25.0, 1.0)

    st.divider()
    if st.button("Calculate WACC", type="primary", use_container_width=True):
        model = DetailWACCModel(target_ticker, peers_input, rf_in, crp_in, size_in, bb_in, div_in, g_in, tax_in)
        with st.spinner("Calculating..."):
            st.session_state['result'] = model.run()
            st.session_state['inputs'] = {
                'rf': rf_in, 'crp': crp_in, 'sp': size_in, 'tax': tax_in,
                'bb': bb_in, 'div': div_in, 'g': g_in
            }

# Main Dashboard
if 'result' in st.session_state:
    res = st.session_state['result']
    inp = st.session_state['inputs']
    df = res['full_df']
    m = res['market_params']
    
    # Sensitivity Analysis Button (Top of Beta Section, but affects global WACC)
    # To implement the layout requested: 1. WACC, 2. Beta, 3. Ke, 4. Kd...
    # But Beta Sensitivity controls WACC. So we calculate dynamic values first.
    
    # -------------------------------------------------------------------------
    # [LOGIC] Dynamic Re-calculation based on Sensitivity
    # -------------------------------------------------------------------------
    st.subheader("2. Beta Analysis") # Displaying Header here but placing control first as requested
    
    sens_method = st.radio("Sensitivity Selection (Aggregation Method)", 
                           ["Average", "Median", "Maximum", "Minimum"], horizontal=True)
    
    if not df.empty:
        # 1. Aggregate Unlevered Beta & Capital Structure (Debt/TIC)
        if sens_method == "Average":
            sel_unlev_beta = df["Unlevered Beta"].mean()
            sel_dtic = df["Debt/TIC Ratio"].mean()
        elif sens_method == "Median":
            sel_unlev_beta = df["Unlevered Beta"].median()
            sel_dtic = df["Debt/TIC Ratio"].median()
        elif sens_method == "Maximum":
            sel_unlev_beta = df["Unlevered Beta"].max()
            sel_dtic = df["Debt/TIC Ratio"].max()
        else: # Minimum
            sel_unlev_beta = df["Unlevered Beta"].min()
            sel_dtic = df["Debt/TIC Ratio"].min()
            
        # 2. Derive Target D/E from Target Debt/TIC
        # D/E = (D/TIC) / (1 - D/TIC)
        target_de = sel_dtic / (1 - sel_dtic) if (1-sel_dtic) != 0 else 0
        
        # 3. Re-lever Beta for Target
        # Levered Beta = Unlevered * (1 + (1-t)*D/E)
        target_relevered_beta = sel_unlev_beta * (1 + (1 - inp['tax']/100) * target_de)
        
        # 4. Calculate Column "Re-levered Beta" for TABLE (per peer, assuming Target Structure)
        # Instruction: "Re-levered Beta는 Target Capital Structure에 따라 계산할 것"
        df["Re-levered Beta"] = df["Unlevered Beta"] * (1 + (1 - inp['tax']/100) * target_de)
        
        # 5. Cost of Equity
        # Ke = Rf + Beta * MRP + CRP + SP
        ke = (inp['rf']/100) + (target_relevered_beta * m['MRP']) + (inp['crp']/100) + (inp['sp']/100)
        
        # 6. Cost of Debt (Simplified)
        spread = 0.02 # Assumed 2% spread for BBB
        kd = ((inp['rf']/100) + spread) * (1 - inp['tax']/100)
        
        # 7. WACC
        wd = sel_dtic
        we = 1 - sel_dtic
        wacc = (we * ke) + (wd * kd)
    else:
        target_relevered_beta = 0; ke=0; kd=0; wacc=0; wd=0; we=0; target_de=0
    
    # -------------------------------------------------------------------------
    # [SECTION 1] WACC Calculation & Results
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("1. WACC Calculation & Results")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final WACC", f"{wacc:.2%}")
    c2.metric("Cost of Equity", f"{ke:.2%}")
    c3.metric("Cost of Debt (A-T)", f"{kd:.2%}")
    c4.metric("Re-levered Beta", f"{target_relevered_beta:.2f}")
    
    st.caption(f"**Target Structure (from {sens_method}):** Debt {wd:.1%} | Equity {we:.1%} (D/E: {target_de:.2%})")

    # -------------------------------------------------------------------------
    # [SECTION 2] Beta Analysis (Details)
    # -------------------------------------------------------------------------
    st.markdown("---")
    # Subheader already displayed above for Layout logic
    
    with st.expander("2-1) Target Capital Structure & Beta Summary", expanded=True):
        st.info(f"**Method:** {sens_method} of Peers. **Target D/E:** {target_de:.2%} (derived from Debt/TIC {sel_dtic:.2%}).")
        st.write("The Re-levered Beta column below shows what each peer's beta would be if they had the *Target's* Capital Structure.")

    with st.expander("2-2) 5-Year Monthly Beta Analysis Table", expanded=True):
        if not df.empty:
            # Column Formatting
            disp_df = df.copy()
            
            # Reorder columns as requested
            # Ticker, Name, Total Debt, Market Cap, D/E, Debt/TIC, Raw Beta, Adj Beta, Unlev Beta, Re-lev Beta
            cols_show = ["Ticker", "Company Name", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio", 
                         "Raw Beta", "Adj Beta", "Unlevered Beta", "Re-levered Beta"]
            
            # Format Numbers
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
                    "Re-levered Beta": st.column_config.NumberColumn(format="%.2f", help="Unlevered Beta * (1 + (1-t)*Target D/E)"),
                }
            )
            st.caption(f"* Tax Rate used for Unlevering/Re-levering: {inp['tax']}% (Global Assumption)")
        else:
            st.error("No Data Available")

    # -------------------------------------------------------------------------
    # [SECTION 3] Cost of Equity
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Cost of Equity")
    st.latex(r"K_e = R_f + \beta_{L} \times (R_m - R_f) + CRP + SP")
    
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    k_col1.metric("Risk Free Rate", f"{inp['rf']:.2f}%")
    k_col2.metric("Market Risk Prem", f"{m['MRP']*100:.2f}%")
    k_col3.metric("Country Risk Prem", f"{inp['crp']:.2f}%")
    k_col4.metric("Size Premium", f"{inp['sp']:.2f}%")
    
    with st.expander("3-1) Implied Market Return Details"):
        st.write(f"**Implied Market Return ($R_m$): {m['Rm']:.2%}**")
        st.write(f"= Buyback Yield ({inp['bb']:.2f}%) + Dividend Yield ({inp['div']:.2f}%) + Growth Rate ({inp['g']:.2f}%)")

    # -------------------------------------------------------------------------
    # [SECTION 4] Cost of Debt
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Cost of Debt")
    st.latex(r"K_d = (R_f + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
    
    d_col1, d_col2 = st.columns(2)
    d_col1.metric("Pre-tax Cost of Debt", f"{(inp['rf'] + 2.0):.2f}%", help="Rf + 2.0% Spread assumption")
    d_col2.metric("After-tax Cost of Debt", f"{kd:.2%}")

    # -------------------------------------------------------------------------
    # [SECTION 5] Peer Group Analysis
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("5. Peer Group Analysis (Financials)")
    if not df.empty:
        # Columns: Ticker, Name, Revenue, EBIT, EBITDA, Total Debt, Market Cap, D/E, Debt/TIC
        fin_cols = ["Ticker", "Company Name", "Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap", "D/E Ratio", "Debt/TIC Ratio"]
        fin_df = df.copy()
        
        # Display as USD for comparison consistency, but note original currency in name or tooltip
        # Instruction said: "Display in USD, show FX Rate"
        # I calculated everything in USD in the model. Let's format it.
        
        for c in ["Revenue", "EBIT", "EBITDA", "Total Debt", "Market Cap"]:
            fin_df[c] = fin_df[c] / 1e9 # Billions
            
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
        st.caption("Note: All financial figures are converted to USD (Billions) using the latest FX rate.")
        # Show FX table
        st.markdown("**Applied FX Rates (to USD):**")
        st.dataframe(df[["Ticker", "Currency", "FX Rate"]].T, use_container_width=True)

    # -------------------------------------------------------------------------
    # [SECTION 6] Market Data Reference
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("6. Market Data Reference")
    t1, t2 = st.tabs(["Fred (Risk Free / GDP)", "NYU Stern (Yields)"])
    with t1:
        st.write("Recent Risk Free Rate Trend")
        if res['prices'] is not None: st.line_chart(res['prices']) # Placeholder for RF trend if available
        else: st.write("Chart data unavailable")
    with t2:
        st.write("S&P 500 Buyback & Dividend Yields (Damodaran)")
        # Fetching fresh for display table (cached)
        _, _, sp_table, _ = get_sp_buyback_data()
        if sp_table is not None: st.dataframe(sp_table, use_container_width=True)
