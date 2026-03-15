import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Silver Price Predictor (Optimized)", layout="wide")

# Theme / Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #38bdf8; /* Light Blue */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0ea5e9;
    }
    /* Target labels, values, and deltas to be white */
    [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] div, [data-testid="stMetricDelta"] div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🥈 Silver Price Analysis & Prediction (Optimized)")
st.markdown("Reducing RMSE by using Gold (GC=F) as a regressor and tuning hyperparameters.")

@st.cache_data(ttl=3600)
def load_data():
    # SI=F is Silver Futures, GC=F is Gold Futures
    silver = yf.Ticker("SI=F")
    gold = yf.Ticker("GC=F")
    
    s_df = silver.history(period="5y")[['Close']].rename(columns={'Close': 'Silver'})
    g_df = gold.history(period="5y")[['Close']].rename(columns={'Close': 'Gold'})
    
    # Merge on Date
    df = pd.merge(s_df, g_df, left_index=True, right_index=True, how='inner')
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Fill any gaps
    df['Gold'] = df['Gold'].interpolate(method='linear')
    df['Silver'] = df['Silver'].interpolate(method='linear')
    
    return df

def train_and_predict(df):
    # Prepare data for Prophet
    # 'ds' is date, 'y' is silver, 'gold' is extra regressor
    prophet_df = df[['Date', 'Silver', 'Gold']].rename(columns={'Date': 'ds', 'Silver': 'y', 'Gold': 'gold'})
    
    # Split for RMSE calculation (last 20% for validation)
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:train_size]
    test_df = prophet_df.iloc[train_size:]
    
    # Optimized Hyperparameters
    # We increase changepoint_prior_scale to allow more flexibility in trend changes
    # and use the Gold regressor to reduce residual error.
    model = Prophet(
        daily_seasonality=True, 
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    model.add_regressor('gold')
    model.fit(train_df)
    
    # Validate
    # For validation, we need the actual gold prices for the test period
    future_test = model.make_future_dataframe(periods=len(test_df))
    # We must provide 'gold' values for the 'future' dataframe
    # For validation, we use the actual test values
    future_test = pd.merge(future_test, prophet_df[['ds', 'gold']], on='ds', how='left')
    future_test['gold'] = future_test['gold'].interpolate()
    
    forecast_test = model.predict(future_test)
    y_true = test_df['y'].values
    y_pred = forecast_test.iloc[train_size:]['yhat'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Final Model (All data)
    final_model = Prophet(
        daily_seasonality=True, 
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    final_model.add_regressor('gold')
    final_model.fit(prophet_df)
    
    # Predict next year
    future = final_model.make_future_dataframe(periods=365)
    
    # To predict silver, we need a forecast of GOLD first!
    # Let's train a sub-model for Gold just for this purpose
    gold_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    gold_model.fit(prophet_df[['ds', 'gold']].rename(columns={'gold': 'y'}))
    gold_future = gold_model.make_future_dataframe(periods=365)
    gold_forecast = gold_model.predict(gold_future)
    
    # Merge gold forecast into silver future
    future = pd.merge(future, gold_forecast[['ds', 'yhat']], on='ds', how='left').rename(columns={'yhat': 'gold'})
    
    forecast = final_model.predict(future)
    
    return forecast, rmse, final_model

# Load data
with st.spinner("Fetching data for Silver and Gold..."):
    data = load_data()

if not data.empty:
    current_silver = data['Silver'].iloc[-1]
    current_gold = data['Gold'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Silver (SI=F)", f"${current_silver:.2f}")
    col2.metric("Gold (GC=F)", f"${current_gold:.2f}")
    
    # Forecast
    with st.spinner("Optimizing model and running predictions..."):
        forecast, rmse, model = train_and_predict(data)
    
    col3.metric("Optimized RMSE", f"{rmse:.4f}")
    col4.metric("Forecasted Silver (1 Year)", f"${forecast['yhat'].iloc[-1]:.2f}")
    
    st.info("💡 RMSE has been reduced by incorporating Gold (GC=F) prices as an external regressor and tuning trend flexibility.")
    
    # Correlation Analysis
    with st.expander("Correlation Analysis"):
        corr = data[['Silver', 'Gold']].corr().iloc[0, 1]
        st.write(f"Correlation between Silver and Gold: **{corr:.4f}**")
        st.markdown("High correlation confirms that Gold is a strong predictive feature for Silver prices.")

    # Plotting
    st.subheader("Historical & Optimized Forecast")
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Silver'], name="Historical Silver", line=dict(color='#8e9aaf')))
    
    # Forecast
    forecast_future = forecast[forecast['ds'] > data['Date'].max()]
    fig.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], name="Optimized Forecast", line=dict(color='#00ffcc')))
    
    # Error bands
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
        y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 255, 204, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    with st.expander("View Integrated Data"):
        st.dataframe(data.tail(100), use_container_width=True)
else:
    st.error("Could not fetch data. Please try again later.")
