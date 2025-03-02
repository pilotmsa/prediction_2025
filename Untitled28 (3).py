#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import numpy as np
import joblib  # For loading pre-trained models (if applicable)
from sklearn.linear_model import LinearRegression

# Sample model (replace with actual trained model if available)
def load_model():
    model = LinearRegression()
    model.coef_ = np.array([0.8])  # Example coefficient
    model.intercept_ = 10  # Example intercept
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Aviation Traffic Prediction")

st.write("Enter the time period to predict air traffic volume:")

# User input fields
period = st.selectbox("Select Time Period", ["Year", "Day", "Hour"])
value = st.number_input(f"Enter Number of {period}s", min_value=1, step=1)

# Prediction
if st.button("Predict Traffic"):
    features = np.array([[value]])
    prediction = model.predict(features)
    st.success(f"Predicted Air Traffic Volume for {value} {period}(s): {prediction[0]:.2f}")


# In[ ]:





# In[ ]:




