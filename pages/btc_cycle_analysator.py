import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import linregress

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

# Checks whether a cbbi threshold crossing persists
def check_persistence(series, threshold, above=True, min_days=5):
    count = 0
    dates = series.index.tolist()  # Ensure we have a list of dates/indexes to work with

    for i, (date, value) in enumerate(series.items()):
        valid = value >= threshold if above else value <= threshold
        if valid:
            count += 1
            # When count reaches min_days, return the date at the start of this consecutive period
            if count == min_days:
                return dates[i - min_days + 1]  # Adjust to get the first date in the sequence
        else:
            count = 0  # Reset count if the sequence is broken

    return None

# Normalizes cycle chunks data for comparison
def normalize_cycle_data(chunk):
    # Normalize time (x-axis) from 0 to 1
    chunk['Normalized Time'] = (chunk['Date'] - chunk['Date'].min()) / (chunk['Date'].max() - chunk['Date'].min())

    # Normalize price (y-axis) from 0 to 1
    chunk['Normalized Price'] = (chunk['Price'] - chunk['Price'].min()) / (chunk['Price'].max() - chunk['Price'].min())

    # Normalize CBBI (y-axis) from 0 to 1
    chunk['Normalized CBBI'] = (chunk['CBBI'] - chunk['CBBI'].min()) / (chunk['CBBI'].max() - chunk['CBBI'].min())

    return chunk

# Interpolate data to a common time scale
def interpolate_data(filtered_chunks):
    # Determine the shortest time series length
    min_length = min(len(chunk) for chunk in filtered_chunks)

    # Common time points for interpolation (from 0 to 1 with min_length points)
    common_time = np.linspace(0, 1, min_length)

    # Interpolate all chunks to the common time scale
    interpolated_chunks = []
    for chunk in filtered_chunks:
        f_price = interp1d(chunk['Normalized Time'], chunk['Normalized Price'], kind='linear')
        f_cbbi = interp1d(chunk['Normalized Time'], chunk['Normalized CBBI'], kind='linear')

        interpolated_chunk = pd.DataFrame({
            'Normalized Time': common_time,
            'Normalized Price': f_price(common_time),
            'Normalized CBBI': f_cbbi(common_time),
            'Cycle Number': chunk['Cycle Number'].iloc[0]
        })
        interpolated_chunks.append(interpolated_chunk)

    return interpolated_chunks, common_time

# Calculate the mean curve for normalized price and CBBI
def calculate_mean_curve(interpolated_chunks):
    mean_price = np.mean([chunk['Normalized Price'] for chunk in interpolated_chunks], axis=0)
    mean_cbbi = np.mean([chunk['Normalized CBBI'] for chunk in interpolated_chunks], axis=0)

    return mean_price, mean_cbbi


def main():
    st.title("BTC Cycle & CBBI Analysis")

    url = "https://colintalkscrypto.com/cbbi/data/latest.json"
    df = fetch_and_process_data(url)

    if not df.empty:

# Chunking
        # Extracting halving dates
        halving_dates = pd.to_datetime(bitcoin_halving_data['Date'])

        # Ensure df['Date'] is in datetime format for comparison
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort just in case
        df.sort_values('Date', inplace=True)

        # Initialize variables for chunking
        chunks = []
        normalized_chunks = []
        start_index = 0
        is_first_chunk = True
        # Initialize cycle number
        cycle_number = 0

        # Iterate through halving dates to create chunks (Chunk processing)
        for halving_date in halving_dates:
            # Find the index of the first occurrence of the halving date in df
            end_index = df[df['Date'] >= halving_date].index.min()

            # Check if the chunk is complete (starts and ends with halving date)
            is_complete = not is_first_chunk and halving_date in df.loc[start_index:end_index]['Date'].values
            # Correctly include the halving date in the current chunk
            if is_complete or is_first_chunk:
                # Include the halving date in the chunk
                chunk = df.loc[start_index:end_index]
            else:
                # Exclude the first date of the next chunk (halving date) from the current chunk
                chunk = df.loc[start_index:end_index-1]

            # Add cycle number & Store the original chunk
            chunk['Cycle Number'] = cycle_number
            chunks.append((chunk, is_complete))

            # Normalize & Store the normalized chunk
            normalized_chunk = normalize_cycle_data(chunk)
            normalized_chunk['Cycle Number'] = cycle_number  # Add cycle number to the normalized chunk
            normalized_chunks.append(normalized_chunk)

            # Prepare for the next chunk by starting it from the current halving date
            start_index = end_index
            cycle_number += 1
            is_first_chunk = False

        # Add the remaining data as the last chunk, which will be incomplete
        last_chunk = df.loc[start_index:]
        if not last_chunk.empty:
            last_chunk['Cycle Number'] = cycle_number
            chunks.append((last_chunk, False))
            normalized_last_chunk = normalize_cycle_data(last_chunk)
            normalized_last_chunk['Cycle Number'] = cycle_number
            normalized_chunks.append(normalized_last_chunk)

# Cycle Stats
        cycles_stats = []

        for chunk, is_complete in chunks:
            if chunk.empty:
                continue  # Skip empty chunks

            chunk['Date'] = pd.to_datetime(chunk['Date'])  # Ensure Date is in datetime format
            cycle_stats = {
                "Cycle Number": chunk['Cycle Number'].iloc[0],
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

            # Calculate the relative Cycle Top/Bottom Position
            if cycle_stats["Days to Cycle Top"] is not None:
                cycle_stats["Relative Top Position"] = cycle_stats["Days to Cycle Top"] / cycle_stats["Cycle Length"]
            if cycle_stats["Days to Cycle Bottom"] is not None:
                cycle_stats["Relative Bottom Position"] = cycle_stats["Days to Cycle Bottom"] / cycle_stats["Cycle Length"]

            cycles_stats.append(cycle_stats)

        cycles_stats_df = pd.DataFrame(cycles_stats)

        # Display the cycle Analysis table in Streamlit
        st.subheader("Cycle Analysis")
        st.write(cycles_stats_df)


# Extended Analysis
        with st.expander("Extended Analysis (Click to view)"):

            # Start creating the layout
            col1, col2 = st.columns(2)  # First row with two columns

            with col1:
                # First diagram: Days CBBI >= 85
                st.subheader("Days CBBI >= 85")

                # Place checkboxes below the title
                neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_1", value=True)
                neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_1", value=True)


                # Filter data based on checkbox state
                start_idx = 1 if neglect_first else 0
                end_idx = -1 if neglect_last else None
                col1_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]

                # First diagram: Days CBBI >= 85
                fig1 = go.Figure(data=go.Scatter(
                    x=cycles_stats_df.index,
                    y=cycles_stats_df['Days CBBI >= 85'],
                    mode='markers+lines',
                    name='Days CBBI >= 85'
                ))
                # Filtered plot overlayed in red
                fig1.add_trace(go.Scatter(
                    x=col1_filtered_cycles_stats_df.index,
                    y=col1_filtered_cycles_stats_df['Days CBBI >= 85'],
                    mode='markers+lines',
                    name='Filtered Days CBBI >= 85',
                    line=dict(color='red')
                ))

                # Perform linear regression on filtered data
                slope, intercept, r_value, p_value, std_err = linregress(col1_filtered_cycles_stats_df.index, col1_filtered_cycles_stats_df['Days CBBI >= 85'])


                # Determine the x-range for the linear fit (from min to extrapolated point)
                x_min = col1_filtered_cycles_stats_df.index.min()
                x_max = max(col1_filtered_cycles_stats_df.index) + 1
                extended_x = np.linspace(x_min, x_max, num=100)  # Create 100 points between min and extrapolated point

                # Compute the linear fit line values
                extended_y = intercept + slope * extended_x

                # Plot the linear fit extending to the extrapolated point
                fig1.add_trace(go.Scatter(
                    x=extended_x,
                    y=extended_y,
                    mode='lines',
                    name='Extended Linear Fit',
                    line=dict(color='red', dash='dash')
                ))

                # Extrapolate for the next cycle
                col1_next_cycle = max(col1_filtered_cycles_stats_df.index) + 1
                col1_extrapolated_value = intercept + slope * col1_next_cycle
                fig1.add_trace(go.Scatter(
                    x=[col1_next_cycle],
                    y=[col1_extrapolated_value],
                    mode='markers',
                    name='Extrapolated Point',
                    marker=dict(color='red', size=10, symbol='cross')
                ))


                fig1.update_layout(
                    xaxis_title="BTC Cycle",
                    yaxis_title="Days CBBI >= 85",
                    xaxis=dict(type='linear', dtick=1, tick0=0, tickformat=".0f")  # Format for integer ticks
                )
                st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    # Second diagram: Relative Cycle Top
                    st.subheader("Relative Cycle Top")

                    # Place checkboxes below the title
                    neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_2", value=True)
                    neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_2", value=True)

                    # Filter data based on checkbox state
                    start_idx = 1 if neglect_first else 0
                    end_idx = -1 if neglect_last else None
                    col2_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]

                    fig2 = go.Figure(data=go.Scatter(
                        x=cycles_stats_df.index,
                        y=cycles_stats_df['Relative Top Position'],
                        mode='markers+lines',
                        name='Relative Cycle Top'
                    ))

                    # Filtered plot overlayed in red
                    fig2.add_trace(go.Scatter(
                        x=col2_filtered_cycles_stats_df.index,
                        y=col2_filtered_cycles_stats_df['Relative Top Position'],
                        mode='markers+lines',
                        name='Filtered Relative Cycle Top',
                        line=dict(color='red')
                    ))

                    # Perform linear regression on filtered data
                    slope, intercept, r_value, p_value, std_err = linregress(col2_filtered_cycles_stats_df.index, col2_filtered_cycles_stats_df['Relative Top Position'])


                    # Determine the x-range for the linear fit (from min to extrapolated point)
                    x_min = col2_filtered_cycles_stats_df.index.min()
                    x_max = max(col2_filtered_cycles_stats_df.index) + 1
                    extended_x = np.linspace(x_min, x_max, num=100)

                    # Compute the linear fit line values
                    extended_y = intercept + slope * extended_x

                    # Plot the linear fit extending to the extrapolated point
                    fig2.add_trace(go.Scatter(
                        x=extended_x,
                        y=extended_y,
                        mode='lines',
                        name='Extended Linear Fit',
                        line=dict(color='red', dash='dash')
                    ))

                    # Extrapolate for the next cycle
                    col2_next_cycle = max(col2_filtered_cycles_stats_df.index) + 1
                    col2_extrapolated_value = intercept + slope * col2_next_cycle
                    fig2.add_trace(go.Scatter(
                        x=[col2_next_cycle],
                        y=[col2_extrapolated_value],
                        mode='markers',
                        name='Extrapolated Point',
                        marker=dict(color='red', size=10, symbol='cross')
                    ))

                    fig2.update_layout(
                        xaxis_title="BTC Cycle",
                        yaxis_title="Relative Cycle Top",
                        xaxis=dict(type='linear', dtick=1, tick0=0, tickformat=".0f")  # Format for integer ticks
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)  # Second row with two columns
            with col3:
                # Third diagram: Days CBBI <= 15
                st.subheader("Days CBBI <= 15")

                # Place checkboxes below the title
                neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_3", value=False)
                neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_3", value=True)

                # Filter data based on checkbox state
                start_idx = 1 if neglect_first else 0
                end_idx = -1 if neglect_last else None
                col3_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]


                fig3 = go.Figure(data=go.Scatter(
                    x=cycles_stats_df.index,
                    y=cycles_stats_df['Days CBBI <= 15'],
                    mode='markers+lines',
                    name='Days CBBI <= 15'
                ))

                # Filtered plot overlayed in red
                fig3.add_trace(go.Scatter(
                    x=col3_filtered_cycles_stats_df.index,
                    y=col3_filtered_cycles_stats_df['Days CBBI <= 15'],
                    mode='markers+lines',
                    name='Filtered Days CBBI <= 15',
                    line=dict(color='red')
                ))

                # Perform linear regression on filtered data
                slope, intercept, r_value, p_value, std_err = linregress(col3_filtered_cycles_stats_df.index, col3_filtered_cycles_stats_df['Days CBBI <= 15'])

                # Determine the x-range for the linear fit (from min to extrapolated point)
                x_min = col3_filtered_cycles_stats_df.index.min()
                x_max = max(col3_filtered_cycles_stats_df.index) + 1
                extended_x = np.linspace(x_min, x_max, num=100)

                # Compute the linear fit line values
                extended_y = intercept + slope * extended_x

                # Plot the linear fit extending to the extrapolated point
                fig3.add_trace(go.Scatter(
                    x=extended_x,
                    y=extended_y,
                    mode='lines',
                    name='Extended Linear Fit',
                    line=dict(color='red', dash='dash')
                ))

                # Extrapolate for the next cycle
                col3_next_cycle = max(col3_filtered_cycles_stats_df.index) + 1
                col3_extrapolated_value = intercept + slope * col3_next_cycle
                fig3.add_trace(go.Scatter(
                    x=[col3_next_cycle],
                    y=[col3_extrapolated_value],
                    mode='markers',
                    name='Extrapolated Point',
                    marker=dict(color='red', size=10, symbol='cross')
                ))

                fig3.update_layout(
                    xaxis_title="BTC Cycle",
                    yaxis_title="Days CBBI <= 15",
                    xaxis=dict(type='linear', dtick=1, tick0=0, tickformat=".0f")  # Format for integer ticks
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                # Fourth diagram: Relative Cycle Bottom
                st.subheader("Relative Cycle Bottom")

                # Place checkboxes below the title
                neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_4", value=True)
                neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_4", value=True)

                # Filter data based on checkbox state
                start_idx = 1 if neglect_first else 0
                end_idx = -1 if neglect_last else None
                col4_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]


                fig4 = go.Figure(data=go.Scatter(
                    x=cycles_stats_df.index,
                    y=cycles_stats_df['Relative Bottom Position'],
                    mode='markers+lines',
                    name='Relative Cycle Bottom'
                ))

                # Filtered plot overlayed in red
                fig4.add_trace(go.Scatter(
                    x=col4_filtered_cycles_stats_df.index,
                    y=col4_filtered_cycles_stats_df['Relative Bottom Position'],
                    mode='markers+lines',
                    name='Filtered Relative Cycle Bottom',
                    line=dict(color='red')
                ))

                # Perform linear regression on filtered data
                slope, intercept, r_value, p_value, std_err = linregress(col4_filtered_cycles_stats_df.index, col4_filtered_cycles_stats_df['Relative Bottom Position'])

                # Determine the x-range for the linear fit (from min to extrapolated point)
                x_min = col4_filtered_cycles_stats_df.index.min()
                x_max = max(col4_filtered_cycles_stats_df.index) + 1
                extended_x = np.linspace(x_min, x_max, num=100)

                # Compute the linear fit line values
                extended_y = intercept + slope * extended_x

                # Plot the linear fit extending to the extrapolated point
                fig4.add_trace(go.Scatter(
                    x=extended_x,
                    y=extended_y,
                    mode='lines',
                    name='Extended Linear Fit',
                    line=dict(color='red', dash='dash')
                ))

                # Extrapolate for the next cycle
                col4_next_cycle = max(col4_filtered_cycles_stats_df.index) + 1
                col4_extrapolated_value = intercept + slope * col4_next_cycle
                fig4.add_trace(go.Scatter(
                    x=[col4_next_cycle],
                    y=[col4_extrapolated_value],
                    mode='markers',
                    name='Extrapolated Point',
                    marker=dict(color='red', size=10, symbol='cross')
                ))

                fig4.update_layout(
                    xaxis_title="BTC Cycle",
                    yaxis_title="Relative Cycle Bottom",
                    xaxis=dict(type='linear', dtick=1, tick0=0, tickformat=".0f")  # Format for integer ticks
                )
                st.plotly_chart(fig4, use_container_width=True)

            # Adding a new row for Cycle Length and Cycle Top plots
            col5, col6 = st.columns(2)  # Third row with two columns

            with col5:
                # Fifth diagram: Cycle Length
                st.subheader("Cycle Length")

                # Place checkboxes below the title
                neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_5", value=True)
                neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_5", value=True)

                # Filter data based on checkbox state
                start_idx = 1 if neglect_first else 0
                end_idx = -1 if neglect_last else None
                col5_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]

                fig5 = go.Figure(data=go.Scatter(
                    x=cycles_stats_df.index,
                    y=cycles_stats_df['Cycle Length'],
                    mode='markers+lines',
                    name='Cycle Length',
                    marker=dict(color='blue')
                ))

                # Filtered plot overlayed in red
                fig5.add_trace(go.Scatter(
                    x=col5_filtered_cycles_stats_df.index,
                    y=col5_filtered_cycles_stats_df['Cycle Length'],
                    mode='markers+lines',
                    name='Filtered Cycle Length',
                    marker=dict(color='red')
                ))

                # Perform linear regression on filtered data
                slope, intercept, r_value, p_value, std_err = linregress(col5_filtered_cycles_stats_df.index, col5_filtered_cycles_stats_df['Cycle Length'])

                # Determine the x-range for the linear fit (from min to extrapolated point)
                x_min = col5_filtered_cycles_stats_df.index.min()
                x_max = max(col5_filtered_cycles_stats_df.index) + 1
                extended_x = np.linspace(x_min, x_max, num=100)  # Create 100 points between min and extrapolated point

                # Compute the linear fit line values
                extended_y = intercept + slope * extended_x

                # Plot the linear fit extending to the extrapolated point
                fig5.add_trace(go.Scatter(
                    x=extended_x,
                    y=extended_y,
                    mode='lines',
                    name='Extended Linear Fit',
                    line=dict(color='red', dash='dash')
                ))

                # Extrapolate for the next cycle
                col5_next_cycle = max(col5_filtered_cycles_stats_df.index) + 1
                col5_extrapolated_value = intercept + slope * col5_next_cycle
                fig5.add_trace(go.Scatter(
                    x=[col5_next_cycle],
                    y=[col5_extrapolated_value],
                    mode='markers',
                    name='Extrapolated Point',
                    marker=dict(color='red', size=10, symbol='cross')
                ))

                fig5.update_layout(
                    xaxis_title="BTC Cycle",
                    yaxis_title="Cycle Length (Days)",
                    xaxis=dict(type='linear', dtick=1, tickformat=".0f")
                )
                st.plotly_chart(fig5, use_container_width=True)

            with col6:
                # Sixth diagram: Cycle Top
                st.subheader("Cycle Top")

                # Place checkboxes below the title
                neglect_first = st.checkbox('Neglect First Cycle', key="neglect_first_6", value=True)
                neglect_last = st.checkbox('Neglect Last Cycle', key="neglect_last_6", value=True)

                # Filter data based on checkbox state
                start_idx = 1 if neglect_first else 0
                end_idx = -1 if neglect_last else None
                col6_filtered_cycles_stats_df = cycles_stats_df.iloc[start_idx:end_idx]

                fig6 = go.Figure(data=go.Scatter(
                    x=cycles_stats_df.index,
                    y=cycles_stats_df['Cycle Top'],
                    mode='markers+lines',
                    name='Cycle Top'
                ))

                # Filtered plot overlayed in red
                fig6.add_trace(go.Scatter(
                    x=col6_filtered_cycles_stats_df.index,
                    y=col6_filtered_cycles_stats_df['Cycle Top'],
                    mode='markers+lines',
                    name='Filtered Cycle Top',
                    line=dict(color='red')
                ))

                # Perform linear regression on filtered data
                slope, intercept, r_value, p_value, std_err = linregress(col6_filtered_cycles_stats_df.index, col6_filtered_cycles_stats_df['Cycle Top'])

                # Determine the x-range for the linear fit (from min to extrapolated point)
                x_min = col6_filtered_cycles_stats_df.index.min()
                x_max = max(col6_filtered_cycles_stats_df.index) + 1
                extended_x = np.linspace(x_min, x_max, num=100)

                # Compute the linear fit line values
                extended_y = intercept + slope * extended_x

                # Plot the linear fit extending to the extrapolated point
                fig6.add_trace(go.Scatter(
                    x=extended_x,
                    y=extended_y,
                    mode='lines',
                    name='Extended Linear Fit',
                    line=dict(color='red', dash='dash')
                ))

                # Extrapolate for the next cycle
                col6_next_cycle = max(col6_filtered_cycles_stats_df.index) + 1
                col6_extrapolated_value = intercept + slope * col6_next_cycle
                fig6.add_trace(go.Scatter(
                    x=[col6_next_cycle],
                    y=[col6_extrapolated_value],
                    mode='markers',
                    name='Extrapolated Point',
                    marker=dict(color='red', size=10, symbol='cross')
                ))

                fig6.update_layout(
                    xaxis_title="BTC Cycle",
                    yaxis_title="Cycle Top Value",
                    xaxis=dict(type='linear', dtick=1, tickformat=".0f")
                )
                st.plotly_chart(fig6, use_container_width=True)

# Current Cycle (Extrapolated Data)
        # Convert the 'Date' column to datetime
        bitcoin_halving_data['Date'] = pd.to_datetime(bitcoin_halving_data['Date'], format='%B %d, %Y')

        # Get the most recent halving event date
        most_recent_halving = bitcoin_halving_data.sort_values(by='Date', ascending=False).iloc[0]
        current_cycle_start_date = most_recent_halving['Date']

        # Calculate the extrapolated cycle top date
        days_until_top = col5_extrapolated_value * col2_extrapolated_value
        extrapolated_top_date = current_cycle_start_date + timedelta(days=days_until_top)

        # Calculate the extrapolated cycle bottom date
        days_until_bottom = col5_extrapolated_value * col4_extrapolated_value
        extrapolated_bottom_date = current_cycle_start_date + timedelta(days=days_until_bottom)

        st.subheader("Current Cycle (Extrapolated Data)")

        # Create the Markdown table
        markdown_table = f"""
        Description             | Value                                                                                |
        |-----------------------|--------------------------------------------------------------------------------------|
        | Cycle Length          | {int(col5_extrapolated_value)} days                                                  |
        | Cycle Top             | {extrapolated_top_date.strftime('%d.%m.%Y')} with {int(col6_extrapolated_value)} USD |
        | Cycle Bottom Date     | {extrapolated_bottom_date.strftime('%d.%m.%Y')}                                      |
        | Days CBBI >= 85       | {int(col1_extrapolated_value)} days                                                  |
        | Days CBBI <= 15       | {int(col3_extrapolated_value)} days                                                  |
        """
        # Display the table in Streamlit
        st.markdown(markdown_table)


# Plot BTC Price & CBBI
        st.subheader("BTC Price & CBBI")
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
        # Add the new point to the existing plot
        fig.add_trace(go.Scatter(
            x=[extrapolated_top_date],  # Use a list to ensure compatibility with Plotly
            y=[col6_extrapolated_value],
            mode='markers+text',
            marker=dict(color='lightgreen', size=10, symbol='cross'),
            text=['Extrapolated Cycle Top'],
            textposition='top right',
            name='Analysis Results'
        ))
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
        # Display the updated figure
        st.plotly_chart(fig, use_container_width=True)

# Cycle Comparison
        st.subheader("Cycle Comparison")

        # Determine the cycle numbers to exclude (first and last)
        cycle_numbers = [chunk['Cycle Number'].iloc[0] for chunk in normalized_chunks]
        first_cycle = min(cycle_numbers)
        last_cycle = max(cycle_numbers)

        # Filter out the first and last cycles
        filtered_chunks = [chunk for chunk in normalized_chunks if chunk['Cycle Number'].iloc[0] != first_cycle and chunk['Cycle Number'].iloc[0] != last_cycle]

        # Interpolate data to a common time scale
        interpolated_chunks, common_time = interpolate_data(filtered_chunks)

        # Calculate the mean curve for the normalized price and CBBI
        mean_price, mean_cbbi = calculate_mean_curve(interpolated_chunks)

        # Extract the last chunk from the chunks list
        last_chunk, _ = chunks[-1]

        # Ensure last_chunk is a DataFrame
        if isinstance(last_chunk, pd.DataFrame):
            current_cycle_time = last_chunk['Date']
            current_cycle_price = last_chunk['Price']
            current_cycle_cbbi = last_chunk['CBBI']
        else:
            raise TypeError("The last chunk is not a DataFrame.")

        # Convert datetime to numeric if needed for plotting
        numeric_time = current_cycle_time.map(pd.Timestamp.toordinal)  # Convert to ordinal days
        time_min = numeric_time.min()

        # Apply rescaling
        # Normalize the time
        days_since_start = numeric_time - time_min  # Days since the last halving
        rescaled_time = days_since_start / col5_extrapolated_value  # Now x ranges from 0 to current progress in the cycle
        # Step 1: Calculate the minimum price of the current cycle
        min_current_price = current_cycle_price.min()
        # Step 2: Rescale the price data
        rescaled_price = (current_cycle_price - min_current_price) / (col6_extrapolated_value - min_current_price)
        rescaled_cbbi = current_cycle_cbbi / 100  # Normalize to the range [0, 1]

        # Initialize figures for normalized price and CBBI
        fig_price = go.Figure()
        fig_cbbi = go.Figure()

        # Plot each normalized cycle on the same graph
        for chunk in interpolated_chunks:
            cycle_number = chunk['Cycle Number'].iloc[0]  # Extract cycle number from the normalized chunk

            # Plot normalized price
            fig_price.add_trace(go.Scatter(
                x=chunk['Normalized Time'],
                y=chunk['Normalized Price'],
                mode='lines',
                name=f"Cycle {cycle_number}",
            ))

            # Plot normalized CBBI
            fig_cbbi.add_trace(go.Scatter(
                x=chunk['Normalized Time'],
                y=chunk['Normalized CBBI'],
                mode='lines',
                name=f"Cycle {cycle_number}",
            ))

        # Add mean curve for normalized price
        fig_price.add_trace(go.Scatter(
            x=common_time,
            y=mean_price,
            mode='lines',
            name="Mean Price",
            line=dict(color='black', width=4)  # Solid thick line for the mean
        ))

        # Add mean curve for normalized CBBI
        fig_cbbi.add_trace(go.Scatter(
            x=common_time,
            y=mean_cbbi,
            mode='lines',
            name="Mean CBBI",
            line=dict(color='black', width=4)  # Solid thick line for the mean
        ))
        
        # Plot the current cycle on the same graphs
        fig_price.add_trace(go.Scatter(
            x=rescaled_time,  # Use the rescaled time
            y=rescaled_price,
            mode='lines',
            name="Current Cycle",
            line=dict(color='lightgreen', width=4)
        ))

        # Plot the current cycle CBBI on the same graph as other cycles
        fig_cbbi.add_trace(go.Scatter(
            x=rescaled_time,  # Use the rescaled time (same as for price)
            y=rescaled_cbbi,
            mode='lines',
            name="Current Cycle",
            line=dict(color='lightgreen', width=4)
        ))

        # Add point plot for the extrapolated cycle top to price plot
        fig_price.add_trace(go.Scatter(
            x=[col2_extrapolated_value],  # x-value (normalized time) for the extrapolated cycle top
            y=[1],  # y-value is 1 since it's the cycle top in the normalized scale
            mode='markers',
            name='Extrapolated Cycle Top',
            marker=dict(symbol='cross', color='lightgreen', size=12),  # Light green cross
        ))

        # Add point plot for the extrapolated cycle top to cbbi plot
        fig_cbbi.add_trace(go.Scatter(
            x=[col2_extrapolated_value],  # x-value (normalized time) for the extrapolated cycle top
            y=[1],  # y-value is 1 since it's the cycle top in the normalized scale
            mode='markers',
            name='Extrapolated Cycle Top',
            marker=dict(symbol='cross', color='lightgreen', size=12),  # Light green cross
        ))

        # Update layout for normalized price plot
        fig_price.update_layout(
            title="Normalized BTC Price Across Cycles with Mean",
            xaxis_title="Normalized Cycle Time",
            yaxis_title="Normalized Price",
            xaxis=dict(type='linear'),
            yaxis=dict(type='linear'),
        )

        # Update layout for normalized CBBI plot
        fig_cbbi.update_layout(
            title="Normalized CBBI Across Cycles with Mean",
            xaxis_title="Normalized Cycle Time",
            yaxis_title="Normalized CBBI",
            xaxis=dict(type='linear'),
            yaxis=dict(type='linear'),
        )

        # Display the figures in Streamlit
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(fig_price, use_container_width=True)
        with col6:
            st.plotly_chart(fig_cbbi, use_container_width=True)

if __name__ == "__main__":
    main()
