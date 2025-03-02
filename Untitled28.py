#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Load Data Function
def load_data(uploaded_file):
    """Load and preprocess uploaded CSV or Excel file."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Ensure 'Date' column exists and is parsed correctly
        if "Date" not in df.columns:
            st.error("Missing 'Date' column. Please check your data file.")
            return None
        
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])  # Remove rows with invalid dates
        df.set_index("Date", inplace=True)
        
        # Ensure 'Flights' column exists and is numeric
        if "Flights" not in df.columns:
            st.error("Missing 'Flights' column. Please check your data file.")
            return None
        
        df["Flights"] = pd.to_numeric(df["Flights"], errors='coerce')
        df = df.dropna(subset=["Flights"])  # Remove rows with invalid 'Flights' data
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Forecast Function
def arima_forecast(df, steps):
    """Perform ARIMA forecasting on the 'Flights' column."""
    try:
        if len(df) < 10:  # Ensure there's enough data for ARIMA
            st.error("Not enough data points for ARIMA modeling. Please provide more data.")
            return None
        
        model = ARIMA(df["Flights"], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.error(f"Error during ARIMA forecasting: {e}")
        return None

# Streamlit UI
st.title("Aviation Traffic Prediction")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Raw Data Preview")
        st.write(df.head())

        st.subheader("Traffic Trend Over Time")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(data=df, x=df.index, y="Flights", ax=ax, marker="o", color="blue")
        ax.set_title("Air Traffic Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Flights")
        ax.grid(True)
        st.pyplot(fig)

        # Forecasting
        steps = st.slider("Predict Days Ahead:", 1, 365, 7)  # Increased max to 365 days
        if st.button("Predict Traffic"):
            forecast = arima_forecast(df, steps)
            
            if forecast is not None:
                st.subheader("Predicted Traffic")
                forecast_dates = pd.date_range(start=df.index[-1], periods=steps+1, freq="D")[1:]
                forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Flights": forecast})
                
                fig, ax = plt.subplots(figsize=(10,5))
                sns.lineplot(data=df, x=df.index, y="Flights", ax=ax, label="Actual", color="blue")
                sns.lineplot(data=forecast_df, x="Date", y="Predicted Flights", ax=ax, label="Forecast", color="red", linestyle="dashed")
                ax.set_title("Traffic Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Number of Flights")
                ax.legend()
                st.pyplot(fig)

                # Display forecasted values
                st.write("Forecasted Values:")
                st.write(forecast_df)

                # Download Prediction
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions", data=csv, file_name="traffic_forecast.csv", mime="text/csv")


# In[2]:


pip install -r requirements.txt


# In[ ]:




