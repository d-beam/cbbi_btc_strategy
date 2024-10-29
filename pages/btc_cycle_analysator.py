import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import linregress

st.set_page_config(page_title="CBBI BTC App", page_icon="🚀", layout="wide")
DATA_FILE = 'bitcoin_halving_data.csv'
bitcoin_halving_data = pd.read_csv(DATA_FILE)

# Define the check_persistence function
def check_persistence(series, threshold, above=True, min_days=5):
    count = 0
    dates = series.index.tolist()

    for i, (date, value) in enumerate(series.items()):
        valid = value >= threshold if above else value <= threshold
        if valid:
            count += 1
            if count == min_days:
                return dates[i - min_days + 1]
        else:
            count = 0
    return None

# Define fetch_and_process_data function to get CBBI data
def fetch_and_process_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        processed_data = [{
            "Date": pd.to_datetime(int(timestamp), unit='s'),
            "Price": float(data["Price"][timestamp]),
            "CBBI": float(data.get("Confidence", {}).get(timestamp, "N/A")) * 100
        } for timestamp in data.get("Price", {})]
        df = pd.DataFrame(processed_data)
        df['CBBI'] = pd.to_numeric(df['CBBI'], errors='coerce')
        return df
    except requests.HTTPError as http_err:
        st.error(f'HTTP error occurred: {http_err}')
    except Exception as err:
        st.error(f'Other error occurred: {err}')
    return pd.DataFrame()

# Main data processing function
def process_data():
    url = "https://colintalkscrypto.com/cbbi/data/latest.json"
    df = fetch_and_process_data(url)
    halving_dates = pd.to_datetime(bitcoin_halving_data['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Organize data by chunks and calculate cycle statistics
    chunks, cycle_stats_list = [], []
    start_index, cycle_number, is_first_chunk = 0, 0, True

    for halving_date in halving_dates:
        end_index = df[df['Date'] >= halving_date].index.min()
        is_complete = not is_first_chunk and halving_date in df.loc[start_index:end_index]['Date'].values
        chunk = df.loc[start_index:end_index] if is_complete or is_first_chunk else df.loc[start_index:end_index-1]
        chunk['Cycle Number'] = cycle_number
        chunks.append((chunk, is_complete))
        
        # Calculate stats for each cycle
        cycle_stats = {
            "Cycle Number": chunk['Cycle Number'].iloc[0],
            "Cycle Start Date": chunk.iloc[0]['Date'].date(),
            "Cycle Length": (chunk['Date'].iloc[-1] - chunk['Date'].iloc[0]).days,
            "First CBBI >= 85 Date": None,
            "Days CBBI >= 85": 0,
            "Relative First CBBI >= 85 Position": None,
        }
        
        first_above_85_index = check_persistence(chunk['CBBI'], 85, above=True)
        if first_above_85_index is not None:
            first_above_85_date = chunk.loc[first_above_85_index, 'Date']
            cycle_stats["First CBBI >= 85 Date"] = first_above_85_date.date()
            cycle_stats["Days CBBI >= 85"] = chunk[chunk['Date'] >= first_above_85_date]['CBBI'].ge(85).sum()
            cycle_length = cycle_stats["Cycle Length"]
            relative_position = (first_above_85_date - chunk.iloc[0]['Date']).days / cycle_length
            cycle_stats["Relative First CBBI >= 85 Position"] = relative_position
        
        cycle_stats_list.append(cycle_stats)
        start_index, cycle_number = end_index, cycle_number + 1
        is_first_chunk = False

    cycles_stats_df = pd.DataFrame(cycle_stats_list)
    last_chunk = df.loc[start_index:]

    # Additional calculations for extrapolated values
    extrapolated_values = {
        "top_date": None,  # Placeholder values; add logic here as needed
        "top_value": None,
        "bottom_date": None,
        "bottom_value": None,
    }

    return df, cycles_stats_df, chunks, extrapolated_values

# Display functions
def display_btc_price_cbbi(df, extrapolated_values):
    st.header("BTC Price & CBBI")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="BTC Price", mode='lines', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['CBBI'], name="CBBI", mode='lines', line=dict(color='blue'), yaxis='y2'))
    for index, row in bitcoin_halving_data.iterrows():
        halving_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
        fig.add_shape(type="line", x0=halving_date, y0=0, x1=halving_date, y1=1, xref="x", yref="paper", line=dict(color="gray", dash="dash"))

    fig.add_shape(type="rect", xref="paper", yref="y2", x0=0, y0=85, x1=1, y1=100, fillcolor="red", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y2", x0=0, y0=0, x1=1, y1=15, fillcolor="green", opacity=0.2, layer="below", line_width=0)

    fig.update_layout(title="", xaxis=dict(title='Date'), yaxis=dict(title='BTC Price', type='log'), yaxis2=dict(title='CBBI', overlaying='y', side='right', range=[0, 100]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_cycle_analysis(cycles_stats_df):
    st.header("Cycle Analysis")
    st.write(cycles_stats_df)

def display_extended_analysis(cycles_stats_df):
    with st.expander("Extended Analysis (Click to view)"):
        pass  # Add your extended analysis plots here as before

# Main function to run the app
def main():
    df, cycles_stats_df, chunks, extrapolated_values = process_data()
    display_btc_price_cbbi(df, extrapolated_values)
    display_cycle_analysis(cycles_stats_df)
    display_extended_analysis(cycles_stats_df)

if __name__ == "__main__":
    main()
