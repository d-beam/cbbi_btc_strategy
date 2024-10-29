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

# Utility Functions
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

def process_data():
    url = "https://colintalkscrypto.com/cbbi/data/latest.json"
    df = fetch_and_process_data(url)
    halving_dates = pd.to_datetime(bitcoin_halving_data['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    chunks, normalized_chunks, cycle_stats_list = [], [], []
    start_index, cycle_number, is_first_chunk = 0, 0, True

    for halving_date in halving_dates:
        end_index = df[df['Date'] >= halving_date].index.min()
        is_complete = not is_first_chunk and halving_date in df.loc[start_index:end_index]['Date'].values
        chunk = df.loc[start_index:end_index] if is_complete or is_first_chunk else df.loc[start_index:end_index-1]
        chunk['Cycle Number'] = cycle_number
        chunks.append((chunk, is_complete))
        
        start_index, cycle_number = end_index, cycle_number + 1
        is_first_chunk = False

    last_chunk = df.loc[start_index:]
    if not last_chunk.empty:
        last_chunk['Cycle Number'] = cycle_number
        chunks.append((last_chunk, False))

    for chunk, is_complete in chunks:
        if chunk.empty:
            continue
        cycle_stats = {
            "Cycle Number": chunk['Cycle Number'].iloc[0],
            "Cycle Start Date": chunk.iloc[0]['Date'].date(),
            "Cycle Length": (chunk['Date'].iloc[-1] - chunk['Date'].iloc[0]).days,
            "First CBBI >= 85 Date": None,
            "Days CBBI >= 85": 0,
            "Relative First CBBI >= 85 Position": None,
            "Cycle Top": None,
            "Date of Cycle Top": None,
            "Days to Cycle Top": None,
            "Relative Top Position": None,
            "First CBBI <= 15 Date": None,
            "Days CBBI <= 15": 0,
            "Cycle Bottom": None,
            "Date of Cycle Bottom": None,
            "Days to Cycle Bottom": None,
            "Relative Bottom Position": None,
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
    
    cycles_stats_df = pd.DataFrame(cycle_stats_list)

    # Additional calculations for extrapolated dates
    # Calculate extrapolated values here (top date, bottom date, etc.)

    return df, cycles_stats_df, chunks, normalized_chunks, extrapolated_values

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

    # Extrapolated top and bottom points
    fig.add_trace(go.Scatter(x=[extrapolated_values['top_date']], y=[extrapolated_values['top_value']], mode='markers', name='Extrapolated Cycle Top', marker=dict(color='lightgreen', size=10, symbol='cross')))
    fig.add_trace(go.Scatter(x=[extrapolated_values['bottom_date']], y=[extrapolated_values['bottom_value']], mode='markers', name='Extrapolated Cycle Bottom', marker=dict(color='blue', size=10, symbol='cross')))

    fig.update_layout(title="", xaxis=dict(title='Date'), yaxis=dict(title='BTC Price', type='log'), yaxis2=dict(title='CBBI', overlaying='y', side='right', range=[0, 100]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_cycle_analysis(cycles_stats_df):
    st.header("Cycle Analysis")
    st.write(cycles_stats_df)

def display_extended_analysis(cycles_stats_df):
    with st.expander("Extended Analysis (Click to view)"):
        # Code for plotting extended analysis (subplots, extrapolations, etc.)
        pass

# Main function to run the app
def main():
    df, cycles_stats_df, chunks, normalized_chunks, extrapolated_values = process_data()
    display_btc_price_cbbi(df, extrapolated_values)
    display_cycle_analysis(cycles_stats_df)
    display_extended_analysis(cycles_stats_df)

if __name__ == "__main__":
    main()
