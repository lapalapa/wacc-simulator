import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Strategic WACC Simulator", layout="wide")

# ==============================================================================
# [MODULE] Peer Recommender (Revenue Based)
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
    def __init__(self, target, peers, buyback, growth, tax):
        self.target = target
        self.peers = [p.strip() for p in peers.split(',') if p.strip()]
        self.buyback_yield = buyback / 100
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
                    if 'EBITDA' in fin.index: ebitda_fy = fin.
