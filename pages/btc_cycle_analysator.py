import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

# Set the page to wide mode
st.set_page_config(layout="wide")

# Load the Bitcoin halving data from a CSV file one level above the working directory
DATA_FILE = 'bitcoin_halving_data.csv'
bitcoin_halving_data = pd.read_csv(DATA_FILE)

def fetch_and_process_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError if the status is 4XX or 5XX
        data = response.json()

        processed_data = [{
            "Date": pd.to_datetime(int(timestamp), unit='s'),
            "Price": float(data["Price"][timestamp]),
            "CBBI": float(data.get("Confidence", {}).get(timestamp, "N/A")) * 100  # Scale the CBBI values
        } for timestamp in data.get("Price", {})]

        df = pd.DataFrame(processed_data)
        df['CBBI'] = pd.to_numeric(df['CBBI'], errors='coerce')  # Convert 'CBBI' to numeric, set 'N/A' to NaN
        return df
    except requests.HTTPError as http_err:
        st.error(f'HTTP error occurred: {http_err}')
    except Exception as err:
        st.error(f'Other error occurred: {err}')
    return pd.DataFrame()

def check_persistence(series, threshold, above=True, min_days=5):
    count = 0
    dates = series.index.tolist()  # Ensure we have a list of dates/indexes to work with
    
    for i, (date, value) in enumerate(series.iteritems()):
        valid = value >= threshold if above else value <= threshold
        if valid:
            count += 1
            # When count reaches min_days, return the date at the start of this consecutive period
            if count == min_days:
                return dates[i - min_days + 1]  # Adjust to get the first date in the sequence
        else:
            count = 0  # Reset count if the sequence is broken

    return None

def interpolate_and_normalize(chunk, max_length):
    # Ensure chunk is sorted by Date
    chunk = chunk.sort_values('Date')

    # Original dates and values
    original_dates = np.linspace(0, 1, len(chunk))
    target_dates = np.linspace(0, 1, max_length)
    
    # Interpolate CBBI and Price
    price_interpolator = interp1d(original_dates, chunk['Price'], kind='linear')
    cbbi_interpolator = interp1d(original_dates, chunk['CBBI'], kind='linear')

    # Interpolated values
    interpolated_price = price_interpolator(target_dates)
    interpolated_cbbi = cbbi_interpolator(target_dates)

    # Normalization
    normalized_price = (interpolated_price - min(interpolated_price)) / (max(interpolated_price) - min(interpolated_price))
    normalized_cbbi = (interpolated_cbbi - min(interpolated_cbbi)) / (max(interpolated_cbbi) - min(interpolated_cbbi))

    # Create a new DataFrame for the interpolated and normalized data
    interpolated_normalized_chunk = pd.DataFrame({
        'Normalized Days': target_dates,
        'Normalized Price': normalized_price,
        'Normalized CBBI': normalized_cbbi,
    })

    return interpolated_normalized_chunk

def main():
    st.title("Interactive Plot of BTC Data")

    url = "https://colintalkscrypto.com/cbbi/data/latest.json"
    df = fetch_and_process_data(url)

    if not df.empty:
        # Create an empty figure
        fig = go.Figure()

        # Add BTC Price trace
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="BTC Price", mode='lines', line=dict(color='red')))

        # Add CBBI trace on the second y-axis
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CBBI'], name="CBBI", mode='lines', line=dict(color='blue'), yaxis='y2'))

        # Manually add vertical lines for Bitcoin halving events
        for index, row in bitcoin_halving_data.iterrows():
            halving_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')  # Ensure date format is YYYY-MM-DD
            fig.add_shape(type="line",
                          x0=halving_date, y0=0, x1=halving_date, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="gray", dash="dash"))
            fig.add_annotation(x=halving_date, y=0.10, xref="x", yref="paper",
                               text=row['Event'], showarrow=False, yanchor="top")

        # Add the green and red shading
        fig.add_shape(type="rect",  # Add a rectangular shape for CBBI > 85
              xref="paper",  # Reference the x-axis in paper terms (entire x-axis)
              yref="y2",  # Reference the right y-axis (where CBBI is plotted)
              x0=0,  # Start from the very left
              y0=85,  # Start from CBBI of 85
              x1=1,  # Extend to the very right
              y1=100,  # Extend to the maximum CBBI considered (100)
              fillcolor="red",  # Fill color
              opacity=0.2,  # Make the fill transparent
              layer="below",  # Ensure the shading is below data points
              line_width=0,  # No border line
        )

        fig.add_shape(type="rect",  # Add a rectangular shape for CBBI < 15
              xref="paper",  # Reference in paper terms (entire x-axis)
              yref="y2",  # Reference the right y-axis
              x0=0,  # Start from the very left
              y0=0,  # Start from the lowest CBBI considered (0)
              x1=1,  # Extend to the very right
              y1=15,  # Extend to a CBBI of 15
              fillcolor="green",  # Fill color
              opacity=0.2,  # Make the fill transparent
              layer="below",  # Ensure the shading is below data points
              line_width=0,  # No border line
        )

        # Update figure layout
        fig.update_layout(
            title="",
            xaxis=dict(title='Date'),
            yaxis=dict(title='BTC Price', type='log', showgrid=False),
            yaxis2=dict(title='CBBI', overlaying='y', side='right', dtick=5, showgrid=True, range=[0, 100],),
            legend=dict(x=0, y=1, traceorder='reversed', font_size=16, bgcolor='rgba(255,255,255,0.5)'),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Everyting from here on is about extracting cycle
        # Extracting halving dates
        halving_dates = pd.to_datetime(bitcoin_halving_data['Date'])

        # Ensure df['Date'] is in datetime format for comparison
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort just in case
        df.sort_values('Date', inplace=True)

        # Initialize variables for chunking
        chunks = []
        start_index = 0
        is_first_chunk = True

        # Iterate through halving dates to create chunks
        for halving_date in halving_dates:
            # Find the index of the first occurrence of the halving date in df
            end_index = df[df['Date'] >= halving_date].index.min()

            # Correctly include the halving date in the current chunk
            # Check if the chunk is complete (starts and ends with halving date)
            is_complete = not is_first_chunk and halving_date in df.loc[start_index:end_index]['Date'].values
            if is_complete or is_first_chunk:
                # Include the halving date in the chunk
                chunk = df.loc[start_index:end_index]
            else:
                # Exclude the first date of the next chunk (halving date) from the current chunk
                chunk = df.loc[start_index:end_index-1]

            chunks.append((chunk, is_complete))

            # Prepare for the next chunk by starting it from the current halving date
            start_index = end_index
            is_first_chunk = False

        # Add the remaining data as the last chunk, which will be incomplete
        last_chunk = df.loc[start_index:]
        chunks.append((last_chunk, False))

        # Identify the longest chunk for later processing
        max_length = max(len(chunk) for chunk, _ in chunks)

        cycles_stats = []

        for chunk, is_complete in chunks:
            if chunk.empty:
                continue  # Skip empty chunks

            chunk['Date'] = pd.to_datetime(chunk['Date'])  # Ensure Date is in datetime format
            cycle_stats = {
                "Cycle Start Date": chunk.iloc[0]['Date'].date(),
                "Cycle Length": (chunk['Date'].iloc[-1] - chunk['Date'].iloc[0]).days,  # Cycle length in days
                "First CBBI >= 85 Date": None,
                "Days CBBI >= 85": 0,
                "Cycle Top": None,
                "Date of Cycle Top": None,
                "Days to Cycle Top": None,
                "Relative Top Position": None,  # Relative measure of when the top was reached
                "First CBBI <= 15 Date": None,
                "Days CBBI <= 15": 0,
                "Cycle Bottom": None,
                "Date of Cycle Bottom": None,
                "Days to Cycle Bottom": None,
                "Relative Bottom Position": None,  # Relative measure of when the bottom was reached
            }

            # Finding first valid CBBI >= 85 crossing
            first_above_85_index = check_persistence(chunk['CBBI'], 85, above=True)
            if first_above_85_index is not None:
                first_above_85_date = chunk.loc[first_above_85_index, 'Date']
                cycle_stats["First CBBI >= 85 Date"] = first_above_85_date.date()
                cycle_stats["Days CBBI >= 85"] = chunk[chunk['Date'] >= first_above_85_date]['CBBI'].ge(85).sum()

            # Finding first valid CBBI <= 15 crossing
            first_below_15_index = check_persistence(chunk['CBBI'], 15, above=False)
            if first_below_15_index is not None:
                first_below_15_date = chunk.loc[first_below_15_index, 'Date']
                cycle_stats["First CBBI <= 15 Date"] = first_below_15_date.date()
                cycle_stats["Days CBBI <= 15"] = chunk[chunk['Date'] <= first_below_15_date]['CBBI'].le(15).sum()

            # Directly find the Cycle Top considering all data up to the first CBBI <= 15 crossing
            if first_below_15_index:
                # Consider all data up to first_below_15_date for Cycle Top
                top_search_chunk = chunk[chunk['Date'] < first_below_15_date]
                if not top_search_chunk.empty:
                    top_price_row = top_search_chunk.loc[top_search_chunk['Price'].idxmax()]
                    cycle_stats["Cycle Top"] = top_price_row['Price']
                    cycle_stats["Date of Cycle Top"] = top_price_row['Date'].date()
                    cycle_stats["Days to Cycle Top"] = (top_price_row['Date'] - chunk.iloc[0]['Date']).days
            
            # Update for finding the first valid CBBI <= 15 crossing for Cycle Bottom determination
            if first_below_15_index:
                first_below_15_date = chunk.loc[first_below_15_index, 'Date']
                cycle_stats["First CBBI <= 15 Date"] = first_below_15_date.date()
                cycle_stats["Days CBBI <= 15"] = chunk[chunk['Date'] >= first_below_15_date]['CBBI'].le(15).sum()

                # Consider data after the first_below_15_date for Cycle Bottom
                bottom_search_chunk = chunk[chunk['Date'] > first_below_15_date]
                if not bottom_search_chunk.empty:
                    bottom_price_row = bottom_search_chunk.loc[bottom_search_chunk['Price'].idxmin()]
                    cycle_stats["Cycle Bottom"] = bottom_price_row['Price']
                    cycle_stats["Date of Cycle Bottom"] = bottom_price_row['Date'].date()
                    cycle_stats["Days to Cycle Bottom"] = (bottom_price_row['Date'] - chunk.iloc[0]['Date']).days
            
            if cycle_stats["Days to Cycle Top"] is not None:
                cycle_stats["Relative Top Position"] = cycle_stats["Days to Cycle Top"] / cycle_stats["Cycle Length"]
            if cycle_stats["Days to Cycle Bottom"] is not None:
                cycle_stats["Relative Bottom Position"] = cycle_stats["Days to Cycle Bottom"] / cycle_stats["Cycle Length"]

            cycles_stats.append(cycle_stats)

        cycles_stats_df = pd.DataFrame(cycles_stats)
        
        # Display the cycle statistics table in Streamlit
        st.write("Cycle Statistics", cycles_stats_df)

        # Calculate interpolated and normalized chunks
        interpolated_normalized_chunks = [interpolate_and_normalize(chunk, max_length) for chunk, is_complete in chunks if not chunk.empty]

        # Plot interpolated and normalized Price
        fig = go.Figure()

        # Add a line trace for the normalized price of each chunk
        for i, chunk in enumerate(interpolated_normalized_chunks):
            fig.add_trace(go.Scatter(
                x=chunk['Normalized Days'],
                y=chunk['Normalized Price'],
                mode='lines',
                name=f'Cycle {i}'
            ))

        # Update the layout to suit your preferences
        fig.update_layout(
            title='Normalized Price of Each Cycle',
            xaxis_title='Normalized Days',
            yaxis_title='Normalized Price',
            legend_title='Cycles'
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Create a new Plotly figure for the CBBI
        fig_cbbi = go.Figure()

        # Add a line trace for the normalized CBBI of each chunk
        for i, chunk in enumerate(interpolated_normalized_chunks):
            fig_cbbi.add_trace(go.Scatter(
                x=chunk['Normalized Days'],
                y=chunk['Normalized CBBI'],
                mode='lines',
                name=f'Cycle {i}'  # Keeping the cycle count starting from 0
            ))

        # Update the layout for the CBBI plot
        fig_cbbi.update_layout(
            title='Normalized CBBI of Each Cycle',
            xaxis_title='Normalized Days',
            yaxis_title='Normalized CBBI',
            legend_title='Cycles'
        )

        # Display the CBBI figure in Streamlit
        st.plotly_chart(fig_cbbi, use_container_width=True)

if __name__ == "__main__":
    main()
