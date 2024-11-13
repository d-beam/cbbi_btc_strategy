import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO

# App title
st.title('Solana and Bitcoin Daily Historical Price Data')

# URLs to download Solana and Bitcoin historical data from Binance (via CryptoDataDownload)
sol_data_url = 'https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_d.csv'
btc_data_url = 'https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv'

# Function to download the data
def download_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading data: {e}")
        return None

# Load Solana data into a DataFrame
sol_data_text = download_data(sol_data_url)
# Load Bitcoin data into a DataFrame
btc_data_text = download_data(btc_data_url)

if sol_data_text and btc_data_text:
    try:
        # Convert downloaded text to DataFrames
        sol_data = pd.read_csv(StringIO(sol_data_text), skiprows=1)  # Skip the header row
        btc_data = pd.read_csv(StringIO(btc_data_text), skiprows=1)  # Skip the header row

        # Rename columns to lowercase for consistency
        sol_data.columns = [col.strip().lower() for col in sol_data.columns]
        btc_data.columns = [col.strip().lower() for col in btc_data.columns]

        # Check for necessary columns
        required_columns = ['date', 'close']
        if all(col in sol_data.columns for col in required_columns) and all(col in btc_data.columns for col in required_columns):
            # Convert date columns to datetime format
            sol_data['date'] = pd.to_datetime(sol_data['date'], format='%Y-%m-%d')
            btc_data['date'] = pd.to_datetime(btc_data['date'], format='%Y-%m-%d')
            
            # Sort data by date
            sol_data = sol_data.sort_values(by='date')
            btc_data = btc_data.sort_values(by='date')

            # Set date as index for plotting
            sol_data.set_index('date', inplace=True)
            btc_data.set_index('date', inplace=True)

            # Plotting the closing prices for Solana and Bitcoin
            st.subheader('Daily Closing Prices for Solana and Bitcoin')
            fig, ax = plt.subplots()
            ax.plot(sol_data.index, sol_data['close'], label='SOL Closing Price', color='blue')
            ax.plot(btc_data.index, btc_data['close'], label='BTC Closing Price', color='orange')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USDT)')
            ax.set_title('Solana and Bitcoin Daily Closing Prices')
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
st.sidebar.write("This app visualizes the historical daily closing prices of Solana (SOL) and Bitcoin (BTC) using data from CryptoDataDownload.")
st.sidebar.write("If data fails to load, please ensure you have an active internet connection and that the data source is available.")
