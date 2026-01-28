import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Strategic WACC Simulator", layout="wide")

# ==============================================================================
# [MODULE] Data Fetcher (NYU Stern HTML Integration)
# ==============================================================================
@st.cache_data(ttl=3600*24) # 24시간 캐싱
def get_sp_buyback_data():
    """
    NYU Stern (Aswath Damodaran) S&P 500 Earnings & Dividends HTML 데이터 크롤링
    Target URL: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html
    """
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html"
    default_bb_yield = 2.0 
    default_div_yield = 1.5
    
    try:
        # 1. HTML 표 읽기
        dfs = pd.read_html(url, header=0)
        
        # 2. 올바른 테이블 찾기
        df = None
        for d in dfs:
            cols_str = [str(c).lower() for c in d.columns]
            if "year" in cols_str and "s&p 500" in cols_str:
                df = d
                break
        
        if df is None:
            return default_bb_yield, default_div_yield, None, ["⚠️ HTML 테이블을 찾을 수 없습니다."]

        # 3. 컬럼 매핑
        cols_map = {}
        for c in df.columns:
            c_lower = str(c).lower().strip()
            if "year" in c_lower: cols_map["Period"] = c
            elif "s&p 500" in c_lower and "yield" not in c_lower: cols_map["S&P 500"] = c
            elif "dividends" in c_lower and "+" not in c_lower and "yield" not in c_lower: cols_map["Dividends"] = c 
            elif "dividends + buybacks" in c_lower or ("buybacks" in c_lower and "+" in c_lower): cols_map["TotalCash"] = c 

        if not all(k in cols_map for k in ["Period", "S&P 500", "Dividends", "TotalCash"]):
             return default_bb_yield, default_div_yield, None, ["⚠️ 필요한 컬럼을 식별하지 못했습니다."]

        # 데이터 추출
        clean_df = pd.DataFrame()
        clean_df["Year"] = df[cols_map["Period"]]
        clean_df["S&P 500"] = df[cols_map["S&P 500"]]
        clean_df["Dividends"] = df[cols_map["Dividends"]]
        clean_df["TotalCash"] = df[cols_map["TotalCash"]]

        # 4. 전처리
        clean_df["Year"] = pd.to_numeric(clean_df["Year"], errors='coerce')
        clean_df = clean_df.dropna(subset=["Year"])
        clean_df = clean_df.sort_values(by="Year", ascending=False)

        for c in ["S&P 500", "Dividends", "TotalCash"]:
            clean_df[c] = pd.to_numeric(clean_df[c], errors='coerce')

        # 5. Yield 계산
        clean_df["Buybacks"] = clean_df["TotalCash"] - clean_df["Dividends"]
        clean_df["Buyback Yield"] = clean_df["Buybacks"] / clean_df["S&P 500"]
        clean_df["Dividend Yield"] = clean_df["Dividends"] / clean_df["S&P 500"]
        clean_df["Total Yield"] = clean_df["Buyback Yield"] + clean_df["Dividend Yield"]

        # % 단위 변환
        clean_df["Buyback Yield %"] = clean_df["Buyback Yield"] * 100
        clean_df["Dividend Yield %"] = clean_df["Dividend Yield"] * 100
        clean_df["Total Yield %"] = clean_df["Total Yield"] * 100

        # 6. 최근 5개년 평균 계산
        valid_rows = clean_df[clean_df["Buyback Yield"] > 0].head(5)
        
        avg_bb_yield = valid_rows["Buyback Yield %"].mean()
        avg_div_yield = valid_rows["Dividend Yield %"].mean()

        # UI 표시용 DF
        display_df = clean_df[["Year", "S&P 500", "Dividends", "Buybacks", "Dividend Yield %", "Buyback Yield %", "Total Yield %"]].copy()
        
        return avg_bb_yield, avg_div_yield, display_df, []

    except Exception as e:
        return default_bb_yield, default_div_yield, None, [f"⚠️ HTML 파싱 실패: {str(e)}"]

# ==============================================================================
# [MODULE] Peer Recommender
# ==============================================================================
class PeerRecommender:
    def get_revenue(self, ticker):
        try:
            t = yf.Ticker(ticker)
            rev = t.info.get('totalRevenue')
            if rev is None:
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
            
            if not ind_key:
                logs.append(f"Industry Key 없음. Sector({sec_key})로 대체.")
                if not sec_key: return None, "Unknown", logs
                sector = yf.Sector(sec_key)
                top_df = sector.top_companies
                group_name = f"Sector: {sec_key}"
            else:
                industry = yf.Industry(ind_key)
                top_df = industry.top_companies
                group_name = f"Industry: {ind_key}"
            
            logs.append(f"✅ {group_name} 리스트 확보")
            
            if top_df is not None and not top_df.empty:
                if 'symbol' in top_df.columns: raw_list = top_df['symbol'].tolist()
                elif 'Symbol' in top_df.columns: raw_list = top_df['Symbol'].tolist()
                else: raw_list = top_df.index.tolist()
                candidates = [c for c in raw_list if c.upper() != target_ticker.upper()][:20]
            else:
                return None, group_name, logs

            if progress_bar: progress_bar.progress(0.3, text="매출 데이터 정렬 중...")
            
            revenue_map = []
            total_cand = len(candidates)
            for idx, ticker in enumerate(candidates):
                rev = self.get_revenue(ticker)
                revenue_map.append((ticker, rev))
                if progress_bar: 
                    progress_bar.progress(0.3 + (0.4 * (idx / total_cand)), text=f"Scanning: {ticker}")
            
            revenue_map.sort(key=lambda x: x[1], reverse=True)
            top_10 = [item[0] for item in revenue_map][:10]
            
            return ", ".join(top_10), group_name, logs

        except Exception as e:
            logs.append(f"❌ Error: {str(e)}")
            return None, "Error", logs

# ==============================================================================
# [LOGIC] WACC Engine
# ==============================================================================
class DetailWACCModel:
    def __init__(self, target, peers, buyback, div_yield, growth, tax):
        self.target = target
        self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.buyback_yield = buyback / 100
        self.div_yield = div_yield / 100
        self.growth_rate = growth / 100
        self.tax = tax / 100
        self.rf_ticker = "^TNX"
        self.market_index = "^GSPC"
        self.fx_cache = {}

    def get_exchange_rate_to_usd(self, currency):
        currency = currency.upper()
        if currency == 'USD': return 1.0
        if currency in self.fx_cache: return self.fx_cache[currency]
        try:
            if currency == 'KRW': ticker = "KRW=X"
            elif currency == 'EUR': ticker = "EURUSD=X"
            elif currency == 'CNY': ticker = "CNY=X"
            elif currency == 'JPY': ticker = "JPY=X"
            else: ticker = f"{currency}USD=X"
            hist = yf.Ticker(ticker).history(period="1d")
            if hist.empty: return 1.0
            rate = hist['Close'].iloc[-1]
            final_rate = rate if currency in ['EUR', 'GBP', 'AUD'] else 1/rate
            self.fx_cache[currency] = final_rate
            return final_rate
        except: return 1.0

    def get_financials_detailed(self, ticker):
        log = []
        try:
            t = yf.Ticker(ticker)
            try: 
                info = t.info
                full_name = info.get('longName', ticker)
                currency = info.get('currency', 'USD')
            except: 
                full_name = ticker
                currency = 'USD' 
                info = {}

            fx_rate = self.get_exchange_rate_to_usd(currency)
            
            rev_ttm = info.get('totalRevenue')
            ebitda_ttm = info.get('ebitda')
            ebit_ttm = None
            if rev_ttm and info.get('operatingMargins'):
                ebit_ttm = rev_ttm * info.get('operatingMargins')
            
            rev_fy = 0; ebit_fy = 0; ebitda_fy = 0
            fy_date = "-"
            try:
                fin = t.financials
                if not fin.empty:
                    fy_col = fin.columns[0]
                    fy_date = fy_col.strftime('%Y-%m')
                    if 'Total Revenue' in fin.index: rev_fy = fin.loc['Total Revenue'].iloc[0]
                    if 'Operating Income' in fin.index: ebit_fy = fin.loc['Operating Income'].iloc[0]
                    elif 'EBIT' in fin.index: ebit_fy = fin.loc['EBIT'].iloc[0]
                    if 'EBITDA' in fin.index: ebitda_fy = fin.loc['EBITDA'].iloc[0]
                    elif 'Normalized EBITDA' in fin.index: ebitda_fy = fin.loc['Normalized EBITDA'].iloc[0]
            except: pass
            
            if not rev_ttm: rev_ttm = rev_fy
            if not ebit_ttm: ebit_ttm = ebit_fy
            if not ebitda_ttm: ebitda_ttm = ebitda_fy

            rev_ttm = rev_ttm if rev_ttm else 0
            ebit_ttm = ebit_ttm if ebit_ttm else 0
            ebitda_ttm = ebitda_ttm if ebitda_ttm else 0
            
            try: mkt_cap_local = t.fast_info['market_cap']
            except: mkt_cap_local = 0
            if mkt_cap_local is None: mkt_cap_local = 0

            debt_local = 0
            try:
                bs = t.balance_sheet
                if not bs.empty:
                    for item in ['Total Debt', 'Long Term Debt', 'Total Liab']:
                        if item in bs.index:
                            debt_local = bs.loc[item].iloc[0]
                            break
            except: pass
            
            data_usd = {
                "mkt_cap": mkt_cap_local * fx_rate,
                "debt": debt_local * fx_rate,
                "rev_ttm": rev_ttm * fx_rate,
                "ebit_ttm": ebit_ttm * fx_rate,
                "ebitda_ttm": ebitda_ttm * fx_rate,
                "rev_fy": rev_fy * fx_rate,
                "ebit_fy": ebit_fy * fx_rate,
                "ebitda_fy": ebitda_fy * fx_rate
            }
            if data_usd['mkt_cap'] == 0: log.append("시총 데이터 부재")
            return {"name": full_name, "currency": currency, "fy_date": fy_date, "usd": data_usd, "logs": log}
        except Exception as e:
            return {"name": ticker, "logs": [str(e)]}

    def get_risk_free_rate(self):
        try: return yf.Ticker(self.rf_ticker).history(period="5d")['Close'].iloc[-1]/100
        except: return 0.040

    def get_implied_market_return(self):
        # [UPDATED] Use user input (5Y Avg from Damodaran) instead of SPY fetch
        return self.div_yield + self.buyback_yield + self.growth_rate, self.div_yield

    def calculate_beta(self, ticker):
        try:
            s_df = yf.download(ticker, period="2y", interval="1wk", progress=False)
            b_df = yf.download(self.market_index, period="2y", interval="1wk", progress=False)
            s = s_df['Adj Close'] if 'Adj Close' in s_df.columns else s_df['Close']
            b = b_df['Adj Close'] if 'Adj Close' in b_df.columns else b_df['Close']
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            if isinstance(b, pd.DataFrame): b = b.iloc[:, 0]
            s.index = s.index.tz_localize(None)
            b.index = b.index.tz_localize(None)
            df = pd.DataFrame({'s': s, 'b': b}).dropna()
            if len(df) < 10: return np.nan
            ret = df.pct_change().dropna()
            return np.cov(ret['s'], ret['b'])[0, 1] / np.cov(ret['s'], ret['b'])[1, 1]
        except: return np.nan

    def run(self):
        rf = self.get_risk_free_rate()
        rm, div_yield = self.get_implied_market_return()
        mrp = rm - rf
        peer_results = []
        betas = []
        des = []
        error_logs = []
        
        my_bar = st.progress(0, text="Analyzing Market & Peers...")
        
        for idx, p in enumerate(self.peers):
            my_bar.progress((idx + 1) / len(self.peers), text=f"Analyzing {p}...")
            beta = self.calculate_beta(p)
            fin = self.get_financials_detailed(p)
            if 'logs' in fin and fin['logs']: error_logs.append(f"**{p}**: {', '.join(fin['logs'])}")
            d = fin.get('usd', {})
            equity = d.get('mkt_cap', 0)
            debt = d.get('debt', 0)
            if np.isnan(beta) or equity == 0: continue
            de = debt / equity
            unlev_beta = beta / (1 + (1 - self.tax) * de)
            rev_ttm = d.get('rev_ttm', 0)
            rev_fy = d.get('rev_fy', 0)
            peer_results.append({
                "Ticker": p, "Company Name": fin.get('name', p), "Currency": fin.get('currency', 'USD'), "FY Date": fin.get('fy_date', '-'),
                "Revenue (TTM)": rev_ttm, "EBIT (TTM)": d.get('ebit_ttm', 0), "EBIT % (TTM)": d.get('ebit_ttm', 0)/rev_ttm if rev_ttm > 0 else 0,
                "EBITDA (TTM)": d.get('ebitda_ttm', 0), "EBITDA % (TTM)": d.get('ebitda_ttm', 0)/rev_ttm if rev_ttm > 0 else 0,
                "Revenue (FY)": rev_fy, "EBIT (FY)": d.get('ebit_fy', 0), "EBIT % (FY)": d.get('ebit_fy', 0)/rev_fy if rev_fy > 0 else 0,
                "EBITDA (FY)": d.get('ebitda_fy', 0), "EBITDA % (FY)": d.get('ebitda_fy', 0)/rev_fy if rev_fy > 0 else 0,
                "Levered Beta": beta, "Unlevered Beta": unlev_beta, "D/E Ratio": de, "Market Cap": equity, "Total Debt": debt
            })
            betas.append(unlev_beta); des.append(de)
        my_bar.empty()
        if not peer_results: st.error("No valid data."); return None
        median_unlev_beta = np.median(betas)
        median_de = np.median(des)
        target_we = 1 / (1 + median_de)
        target_wd = median_de / (1 + median_de)
        relevered_beta = median_unlev_beta * (1 + (1 - self.tax) * median_de)
        ke = rf + relevered_beta * mrp
        credit_spread = 0.02
        kd = (rf + credit_spread) * (1 - self.tax)
        wacc = (target_we * ke) + (target_wd * kd)
        return {
            "market": {"Rf": rf, "Rm": rm, "MRP": mrp, "Div": div_yield},
            "peer_df": pd.DataFrame(peer_results),
            "target": {"WACC": wacc, "Ke": ke, "Kd": kd, "Spread": credit_spread, 
                       "Beta": relevered_beta, "DE": median_de, "We": target_we, "Wd": target_wd},
            "errors": error_logs
        }

# ==============================================================================
# [UI] Dashboard
# ==============================================================================
st.title("📊 Strategic WACC Simulator")
st.markdown("##### :chart_with_upwards_trend: 기업가치평가를 위한 최적의 할인율 산출 대시보드")
st.markdown("---")

if 'peers_input_val' not in st.session_state: st.session_state['peers_input_val'] = "ON, STM, IFX.DE"
if 'rec_logs' not in st.session_state: st.session_state['rec_logs'] = []
if 'rec_success_msg' not in st.session_state: st.session_state['rec_success_msg'] = ""

# 앱 시작 시 NYU Stern Buyback & Dividend 데이터 로드
sp_avg_bb_yield, sp_avg_div_yield, sp_df, sp_logs = get_sp_buyback_data()
if sp_logs:
    st.toast(sp_logs[0], icon="⚠️")

with st.sidebar:
    st.header("1. Target & Peers")
    target = st.text_input("Target Ticker", "WOLF")
    
    col1, col2 = st.columns([1,1])
    if col1.button("🤖 경쟁사 자동 추천 (Top 10)", type="secondary"):
        prog_bar = st.progress(0, text="산업 데이터 수집 중...")
        rec_engine = PeerRecommender()
        rec_peers, industry_name, logs = rec_engine.recommend(target, prog_bar)
        prog_bar.empty()
        
        if rec_peers:
            st.session_state['peers_input_val'] = rec_peers
            st.session_state['rec_success_msg'] = f"Found: {industry_name}"
            st.session_state['rec_logs'] = logs
        else:
            st.session_state['rec_success_msg'] = "추천 실패"
            st.session_state['rec_logs'] = logs
    
    if st.session_state['rec_success_msg']:
        if "Found" in st.session_state['rec_success_msg']: st.success(st.session_state['rec_success_msg'])
        else: st.warning(st.session_state['rec_success_msg'])
    if st.session_state['rec_logs']:
        with st.expander("Logs"):
            for l in st.session_state['rec_logs']: st.write(l)
    
    peers = st.text_area("Peer Tickers", key='peers_input_val', height=100)
    st.caption("※ 산업 내 매출액(Revenue) 상위 10개 기업")
    
    st.divider()
    st.header("2. Assumptions")
    tax = st.slider("Tax Rate (%)", 0.0, 40.0, 25.0, 1.0)
    
    # [LIVE DATA] S&P 500 5년 평균 Buyback & Dividend Yield 자동 적용 (Source: NYU Stern)
    buyback = st.number_input(f"Buyback Yield (5Y Avg: {sp_avg_bb_yield:.2f}%)", value=sp_avg_bb_yield, step=0.1)
    div_yield_in = st.number_input(f"Dividend Yield (5Y Avg: {sp_avg_div_yield:.2f}%)", value=sp_avg_div_yield, step=0.1)
    growth = st.number_input("Growth Rate (%)", value=5.5, step=0.1)
    
    st.divider()
    btn = st.button("Calculate WACC", type="primary", use_container_width=True)

if btn:
    model = DetailWACCModel(target, peers, buyback, div_yield_in, growth, tax)
    with st.spinner("Calculating..."):
        res = model.run()
        
    if res:
        with st.expander("📘 WACC 가이드 (처음 사용자용)", expanded=False):
            st.markdown(f"$$ WACC = (\\text{{Equity Ratio}} \\times \\text{{Cost of Equity}}) + (\\text{{Debt Ratio}} \\times \\text{{Cost of Debt}} \\times (1 - \\text{{Tax}})) $$")

        t = res['target']
        m = res['market']
        st.success(f"### 🎯 Final WACC: {t['WACC']:.2%}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cost of Equity", f"{t['Ke']:.2%}")
        c2.metric("Cost of Debt (After-Tax)", f"{t['Kd']:.2%}")
        c3.metric("Target D/E Ratio", f"{t['DE']:.2%}")
        c4.metric("Capital Structure", f"Eq {t['We']:.0%} : Dt {t['Wd']:.0%}")
        
        st.markdown("---")
        st.subheader("🧮 Detailed Calculation Process (Breakdown)")
        col_ke, col_kd = st.columns(2)
        
        with col_ke:
            st.markdown("#### 1. Cost of Equity (자기자본비용)")
            st.latex(r"\text{Cost of Equity} = \text{Risk Free Rate} + (\beta \times \text{Market Risk Premium})")
            ke_data = {
                "Item": ["Risk-Free Rate", "Re-levered Beta", "Market Risk Premium"],
                "Value": [f"{m['Rf']:.2%}", f"{t['Beta']:.2f}", f"{m['MRP']:.2%}"],
                "Source URL": ["https://finance.yahoo.com/quote/%5ETNX", None, None],
                "Source": ["🔗 Yahoo Finance (^TNX)", "Peer Group Median", "Implied Return - Rf"],
                "Logic": ["Risk-free asset yield (5-day avg)", "Unlevered Median adjusted for Target D/E", "Expected Equity Return excess over Rf"]
            }
            st.dataframe(pd.DataFrame(ke_data), hide_index=True, use_container_width=True, 
                         column_config={"Source URL": st.column_config.LinkColumn("Reference", display_text="Source")})
            st.info(f"💡 **Calculation:** {m['Rf']:.2%} + ({t['Beta']:.2f} × {m['MRP']:.2%}) = **{t['Ke']:.2%}**")

        with col_kd:
            st.markdown("#### 2. Cost of Debt (타인자본비용)")
            st.latex(r"\text{Cost of Debt} = (\text{Risk Free Rate} + \text{Credit Spread}) \times (1 - \text{Tax Rate})")
            kd_data = {
                "Item": ["Risk-Free Rate", "Credit Spread", "Tax Rate"],
                "Value": [f"{m['Rf']:.2%}", f"{t['Spread']:.2%}", f"{tax:.1f}%"],
                "Source URL": ["https://finance.yahoo.com/quote/%5ETNX", "https://fred.stlouisfed.org/series/BAMLC0A0CM", None],
                "Source": ["🔗 Yahoo Finance (^TNX)", "🔗 FRED (ICE BofA BBB)", "User Input"],
                "Logic": ["Base rate for debt pricing", "Proxy: US Corp BBB Option-Adjusted Spread", "Corporate Tax Shield"]
            }
            st.dataframe(pd.DataFrame(kd_data), hide_index=True, use_container_width=True, 
                         column_config={"Source URL": st.column_config.LinkColumn("Reference", display_text="Source")})
            st.info(f"💡 **Calculation:** ({m['Rf']:.2%} + {t['Spread']:.2%}) × (1 - {tax/100:.2f}) = **{t['Kd']:.2%}**")

        st.markdown("#### 3. Target Capital Structure Logic (Capital Weights)")
        col_str1, col_str2 = st.columns(2)
        with col_str1:
            st.markdown("**A. Target D/E Ratio**")
            st.info(f"Logic: **Peer Group Median** D/E Ratio.\n\nMedian Value: **{t['DE']:.2%}**")
        with col_str2:
            st.markdown("**B. Weights Conversion**")
            st.latex(r"W_{Equity} = \frac{1}{1 + D/E}, \quad W_{Debt} = \frac{D/E}{1 + D/E}")
            st.write(f"- **Equity:** **{t['We']:.1%}** | **Debt:** **{t['Wd']:.1%}**")

        st.markdown("#### 4. Implied Market Return Basis")
        st.caption("Market Risk Premium 산출을 위한 시장 기대수익률($R_m$) 구성 요소")
        rm_data = {
            "Item": ["Dividend Yield", "Buyback Yield", "Growth Rate"],
            "Value": [f"{div_yield_in:.2f}%", f"{buyback:.2f}%", f"{growth:.2f}%"],
            "Source URL": ["https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html", "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html", "https://insight.factset.com/"],
            "Source": ["🔗 NYU Stern (Damodaran)", "🔗 NYU Stern (Damodaran)", "🔗 FactSet Insight"],
            "Logic": [f"5Y Avg (Live Data: {sp_avg_div_yield:.2f}%)", f"5Y Avg (Live Data: {sp_avg_bb_yield:.2f}%)", "S&P 500 Long-term EPS Growth Consensus"]
        }
        st.dataframe(pd.DataFrame(rm_data), hide_index=True, use_container_width=True, 
                     column_config={"Source URL": st.column_config.LinkColumn("Reference", display_text="Source")})
        st.info(f"💡 **Calculation:** {div_yield_in:.2f}% + {buyback:.2f}% + {growth:.2f}% = **{m['Rm']:.2%}**")
        
        st.divider()
        st.subheader("🏢 Peer Group Analysis (TTM vs FY)")
        df = res['peer_df'].copy()
        cols_order = ["Ticker", "Company Name", "Currency", "FY Date", "Revenue (TTM)", "Revenue (FY)", "EBIT (TTM)", "EBIT % (TTM)", "EBIT (FY)", "EBIT % (FY)", "EBITDA (TTM)", "EBITDA % (TTM)", "EBITDA (FY)", "EBITDA % (FY)", "Levered Beta", "Unlevered Beta", "D/E Ratio", "Market Cap", "Total Debt"]
        cols_order = [c for c in cols_order if c in df.columns]
        df = df[cols_order]
        def fmt_usd(x): return f"${x/1e9:.2f}B" if x != 0 else "-"
        def fmt_pct(x): return f"{x:.1%}" if x != 0 else "-"
        df_disp = df.copy()
        for c in ["Revenue (TTM)", "Revenue (FY)", "EBIT (TTM)", "EBIT (FY)", "EBITDA (TTM)", "EBITDA (FY)", "Market Cap", "Total Debt"]: df_disp[c] = df_disp[c].apply(fmt_usd)
        for c in ["EBIT % (TTM)", "EBIT % (FY)", "EBITDA % (TTM)", "EBITDA % (FY)", "D/E Ratio"]: df_disp[c] = df_disp[c].apply(fmt_pct)
        for c in ["Levered Beta", "Unlevered Beta"]: df_disp[c] = df_disp[c].apply(lambda x: f"{x:.2f}")
        st.dataframe(df_disp, use_container_width=True, column_config={"Company Name": st.column_config.TextColumn(width="medium")})
        
        if res['errors']:
            with st.expander("⚠️ 데이터 경고"):
                for e in res['errors']: st.write(e)

        # [MARKET DATA] NYU Stern Buyback Data Reference
        st.divider()
        st.subheader("📉 Market Data Reference")
        with st.expander("📊 S&P 500 Buyback & Dividend Historical Data (Source: NYU Stern / A. Damodaran)", expanded=False):
            if sp_df is not None:
                disp_sp = sp_df.copy()
                st.dataframe(
                    disp_sp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Year": st.column_config.NumberColumn("Year", format="%d"),
                        "S&P 500": st.column_config.NumberColumn("S&P 500", format="%d"),
                        "Dividends": st.column_config.NumberColumn("Dividends", format="%.2f"),
                        "Buybacks": st.column_config.NumberColumn("Buybacks", format="%.2f"),
                        "Dividend Yield %": st.column_config.NumberColumn("Div Yield", format="%.2f%%"),
                        "Buyback Yield %": st.column_config.NumberColumn("Buyback Yield", format="%.2f%%"),
                        "Total Yield %": st.column_config.NumberColumn("Total Yield", format="%.2f%%"),
                    }
                )
                st.caption(f"Source: Aswath Damodaran (NYU Stern) | Fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            else:
                st.error("데이터를 불러오지 못했습니다.")
