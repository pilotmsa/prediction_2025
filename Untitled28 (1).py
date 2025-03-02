#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Example model
import pickle

# Load your pre-trained model (replace with your actual model)
# Example: model = pickle.load(open('aviation_traffic_model.pkl', 'rb'))

# Placeholder function for prediction (replace with your actual model prediction)
def predict_traffic(year, day, hour):
    # Example: Replace this with your model's prediction logic
    # model.predict([[year, day, hour]])
    return 1000 + (year - 2020) * 50 + day * 10 + hour * 5  # Dummy prediction

# Streamlit app
st.title("Aviation Traffic Prediction")
st.write("This app predicts total aviation traffic based on year, day, and hour.")

# Input fields
year = st.number_input("Enter the year", min_value=2020, max_value=2030, value=2023)
day = st.number_input("Enter the day of the year (1-365)", min_value=1, max_value=365, value=1)
hour = st.number_input("Enter the hour of the day (0-23)", min_value=0, max_value=23, value=12)

# Predict button
if st.button("Predict Traffic"):
    prediction = predict_traffic(year, day, hour)
    st.success(f"Predicted Total Traffic: {prediction:.2f} passengers")

# Optional: Add a description or visualization
st.write("### How it works:")
st.write("The prediction is based on a machine learning model trained on historical aviation traffic data. Input the year, day, and hour to get the predicted traffic.")

# Optional: Add a sample dataset or visualization
sample_data = pd.DataFrame({
    'Year': [2020, 2021, 2022],
    'Day': [1, 100, 200],
    'Hour': [12, 15, 18],
    'Traffic': [1200, 1500, 1800]  # Example traffic data
})
st.write("### Sample Data:")
st.write(sample_data)

# Optional: Add a plot
st.line_chart(sample_data.set_index('Year')['Traffic'])


# In[ ]:




