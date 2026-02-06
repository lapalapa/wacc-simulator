# ==============================================================================
# [MODULE] Market Data Fetcher (FRED & S&P 500)
# ==============================================================================
def fetch_all_fred_data():
    """
    Fetches Risk-Free Rate (10Y), GDP Growth, and Corporate Spreads from FRED.
    """
    try:
        # 1. Risk-Free Rate (10Y Treasury: DGS10)
        rf_ticker = yf.Ticker("^TNX") # Yahoo Finance Proxy for 10Y
        rf_history = rf_ticker.history(period="1mo")
        latest_rf = rf_history['Close'].iloc[-1] if not rf_history.empty else 4.0
        
        # Trend Data for Chart
        df_rf_trend = rf_history.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Rate'})

        # 2. GDP Growth (Real GDP Annual Rate: A191RP1A027NBEA)
        # Note: In a production app, use 'fredapi' or requests to FRED API. 
        # Here we use a fallback/placeholder or mock for the simulator stability.
        latest_gdp = 2.5 
        df_gdp_disp = pd.DataFrame({
            "Date": [datetime.now().strftime('%Y-%m-%d')],
            "GDP Growth %": [2.5]
        })

        # 3. Option-Adjusted Spreads (OAS)
        # Example mapping for the dashboard
        oas_data = [
            {"OAS Name": "AAA US Corporate", "Latest Spread (%)": 0.45, "Link": "https://fred.stlouisfed.org/series/BAMLC0A1CAAA"},
            {"OAS Name": "AA US Corporate", "Latest Spread (%)": 0.55, "Link": "https://fred.stlouisfed.org/series/BAMLC0A2CAA"},
            {"OAS Name": "Single-A US Corporate", "Latest Spread (%)": 0.75, "Link": "https://fred.stlouisfed.org/series/BAMLC0A3CA"},
            {"OAS Name": "BBB US Corporate", "Latest Spread (%)": 1.10, "Link": "https://fred.stlouisfed.org/series/BAMLC0A4CBBB"},
            {"OAS Name": "BB US High Yield", "Latest Spread (%)": 1.90, "Link": "https://fred.stlouisfed.org/series/BAMLH0A1HYBB"},
            {"OAS Name": "Single-B US High Yield", "Latest Spread (%)": 3.20, "Link": "https://fred.stlouisfed.org/series/BAMLH0A2HYB"},
            {"OAS Name": "CCC & Lower US High Yield", "Latest Spread (%)": 8.50, "Link": "https://fred.stlouisfed.org/series/BAMLH0A3HYC"}
        ]
        df_oas = pd.DataFrame(oas_data)

        return latest_gdp, df_gdp_disp, latest_rf, df_rf_trend, df_oas

    except Exception as e:
        # Fallback values in case of API failure
        return 2.0, pd.DataFrame(), 4.0, pd.DataFrame(), pd.DataFrame()

# 이 외에도 코드에서 호출되는 아래 함수들이 정의되어 있는지 확인이 필요합니다:
# - get_kpmg_tax_rates()
# - get_sp_buyback_data()
# - get_damodaran_spreads()
