import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import requests

# URL for Solana daily data from Binance on CryptoDataDownload
DATA_URL = "https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_d.csv"

@st.cache
def load_data():
    # Download the CSV file from CryptoDataDownload
    response = requests.get(DATA_URL)
    response.raise_for_status()  # Ensure the request was successful

    # Read data, skip extra rows, and display column names to check
    data = pd.read_csv(StringIO(response.text), skiprows=1)  # Skip initial row with text header

    # Check column names
    st.write("Columns in the downloaded data:", data.columns.tolist())
    
    # Ensure the expected columns exist
    if 'date' in data.columns.str.lower():
        data.columns = map(str.lower, data.columns)  # Convert columns to lowercase for consistency
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    else:
        st.error("The expected 'date' column was not found in the data.")
        return pd.DataFrame()

    # Return only the 'close' price column if it exists
    if 'close' in data.columns:
        return data[['close']].sort_index()  # Sort by date
    else:
        st.error("The 'close' price column was not found in the data.")
        return pd.DataFrame()

# Load data
st.title("Historical Daily Price Data for Solana (SOL)")
st.write("Data source: [CryptoDataDownload](https://www.cryptodatadownload.com)")

data = load_data()

if data.empty:
    st.write("No data available.")
else:
    # Display data and plot
    st.write("Showing daily closing prices for Solana since its inception:")
    st.line_chart(data['close'])

    # Matplotlib plot for additional customization (optional)
    st.subheader("Daily Closing Price of Solana (SOL) Over Time")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['close'], label="SOL-USD")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price in USD")
    ax.set_title("Historical Daily Closing Price of Solana (SOL)")
    st.pyplot(fig)