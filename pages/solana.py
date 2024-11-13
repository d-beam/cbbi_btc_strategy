import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO

# App title
st.title('Solana Daily Historical Price Data')

# URL to download Solana historical data from Binance (via CryptoDataDownload)
data_url = 'https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_d.csv'

# Function to download the data
def download_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading data: {e}")
        return None

# Load data into a DataFrame
data_text = download_data(data_url)

if data_text:
    try:
        # Convert downloaded text to DataFrame
        data = pd.read_csv(StringIO(data_text), skiprows=1)  # Skip the header row

        # Check for necessary columns
        required_columns = ['date', 'close']
        if all(col in data.columns for col in required_columns):
            # Convert date column to datetime format
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
            
            # Sort data by date
            data = data.sort_values(by='date')

            # Set date as index for plotting
            data.set_index('date', inplace=True)

            # Plotting the closing prices
            st.subheader('Daily Closing Prices for Solana')
            fig, ax = plt.subplots()
            ax.plot(data.index, data['close'], label='SOL Closing Price', color='blue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USDT)')
            ax.set_title('Solana Daily Closing Prices')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Missing required columns in the data. Please ensure 'date' and 'close' columns are available.")
    except pd.errors.ParserError:
        st.error("Error parsing the data. Please check the CSV formatting.")
    except ValueError as ve:
        st.error(f"Value error: {ve}")
else:
    st.error("Unable to load data. Please try again later.")

# UI: Help section for user experience
st.sidebar.title("Help & Information")
st.sidebar.write("This app visualizes the historical daily closing prices of Solana (SOL) using data from CryptoDataDownload.")
st.sidebar.write("If data fails to load, please ensure you have an active internet connection and that the data source is available.")
