#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install matplotlib


# In[2]:


get_ipython().system('pip install matplotlib')


# In[3]:


import matplotlib.pyplot as plt
print("Matplotlib is installed and working!")


# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Title
st.title("Aviation Traffic Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Determine file type and read data accordingly
    if uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Ensure 'Date' column is properly parsed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])  # Remove rows with invalid dates
        df = df.sort_values(by="Date")  # Ensure chronological order
    else:
        st.error("The uploaded file must contain a 'Date' column.")
        st.stop()
    
    # Ensure 'Flights' column exists
    if "Flights" not in df.columns:
        st.error("The uploaded file must contain a 'Flights' column.")
        st.stop()
    
    # Data preview
    st.write("### Data Preview")
    st.write(df.head())
    
    # Plot
    st.write("### Traffic Trend Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Flights"], marker='o', linestyle='-')
    plt.title("Air Traffic Trend")
    plt.xlabel("Date")
    plt.ylabel("Number of Flights")
    plt.xticks(rotation=45)
    st.pyplot(plt)
else:
    st.write("Please upload a CSV or Excel file to get started.")


# In[ ]:




