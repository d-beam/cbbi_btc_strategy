import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

            # Merge the two dataframes on the date column
            merged_data = pd.merge(sol_data[['date', 'close']], btc_data[['date', 'close']], on='date', suffixes=('_sol', '_btc'))

            # Plotting the closing prices for Solana and Bitcoin using Plotly
            st.subheader('Daily Closing Prices for Solana and Bitcoin')
            fig = go.Figure()
            
            # Add Solana trace
            fig.add_trace(go.Scatter(x=merged_data['date'], y=merged_data['close_sol'],
                                     mode='lines', name='SOL Closing Price', line=dict(color='blue')))
            
            # Add Bitcoin trace with a secondary y-axis
            fig.add_trace(go.Scatter(x=merged_data['date'], y=merged_data['close_btc'],
                                     mode='lines', name='BTC Closing Price', line=dict(color='orange'), yaxis='y2'))

            # Update layout for dual y-axis
            fig.update_layout(
                title='Solana and Bitcoin Daily Closing Prices',
                xaxis_title='Date',
                yaxis_title='Solana Price (USDT)',
                yaxis2=dict(title='Bitcoin Price (USDT)', overlaying='y', side='right'),
                legend_title='Cryptocurrency',
            )

            st.plotly_chart(fig)

            # Find the date of the tops in the 2021 bull cycle
            sol_top = sol_data[(sol_data['date'] >= '2021-01-01') & (sol_data['date'] <= '2021-12-31')].sort_values(by='close', ascending=False).iloc[0]
            btc_top = btc_data[(btc_data['date'] >= '2021-01-01') & (btc_data['date'] <= '2021-12-31')].sort_values(by='close', ascending=False).iloc[0]

            # Correct the comparison logic to determine which asset topped first
            if sol_top['date'] > btc_top['date']:
                topping_order = f"Bitcoin topped first on {btc_top['date'].date()}, followed by Solana on {sol_top['date'].date()} ({(sol_top['date'] - btc_top['date']).days} days later)."
            else:
                topping_order = f"Solana topped first on {sol_top['date'].date()}, followed by Bitcoin on {btc_top['date'].date()} ({(btc_top['date'] - sol_top['date']).days} days later)."

            # Display the results
            st.subheader('2021 Bull Cycle Top Analysis')
            st.write(f"Bitcoin reached its peak price on {btc_top['date'].date()} with a closing price of {btc_top['close']} USDT.")
            st.write(f"Solana reached its peak price on {sol_top['date'].date()} with a closing price of {sol_top['close']} USDT.")
            st.write(topping_order)
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
