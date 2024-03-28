import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

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

        # Prepare a list to hold the statistics for each cycle
        cycles_stats = []

        for chunk, is_complete in chunks:
            if chunk.empty:
                continue  # Skip empty chunks
    
            # Initialize the dictionary for this cycle's stats
            cycle_stats = {
                "Cycle Start Date": chunk['Date'].iloc[0],
                "First CBBI >= 85 Date": None,
                "Days CBBI >= 85": 0,
                "Max Price": None,
                "Date of Max Price": None,
                "Days to Max Price": None,
                "First CBBI <= 15 Date": None,
                "Days CBBI <= 15": 0,
                "Min Price": None,
                "Date of Min Price": None,
                "Days to Min Price": None,
            }
    
            # Check for the first date CBBI reached or rose above 85
            cbbi_85 = chunk[chunk['CBBI'] >= 85]
            if not cbbi_85.empty:
                cycle_stats["First CBBI >= 85 Date"] = cbbi_85['Date'].iloc[0]
                cycle_stats["Days CBBI >= 85"] = len(cbbi_85)
    
            # Maximum price and its date
            max_price_row = chunk.loc[chunk['Price'].idxmax()]
            cycle_stats["Max Price"] = max_price_row['Price']
            cycle_stats["Date of Max Price"] = max_price_row['Date']
            cycle_stats["Days to Max Price"] = (max_price_row['Date'] - chunk['Date'].iloc[0]).days
    
            # Check for the first date CBBI reached or fell below 15
            cbbi_15 = chunk[chunk['CBBI'] <= 15]
            if not cbbi_15.empty:
                cycle_stats["First CBBI <= 15 Date"] = cbbi_15['Date'].iloc[0]
                cycle_stats["Days CBBI <= 15"] = len(cbbi_15)
    
            # Minimum price and its date
            min_price_row = chunk.loc[chunk['Price'].idxmin()]
            cycle_stats["Min Price"] = min_price_row['Price']
            cycle_stats["Date of Min Price"] = min_price_row['Date']
            cycle_stats["Days to Min Price"] = (min_price_row['Date'] - chunk['Date'].iloc[0]).days
    
            # Add the stats for this cycle to the list
            cycles_stats.append(cycle_stats)

        # Convert the list of dictionaries to a DataFrame for easy display
        cycles_stats_df = pd.DataFrame(cycles_stats)

        # Assuming you're using Streamlit, display the DataFrame
        st.write("Cycle Statistics", cycles_stats_df)

if __name__ == "__main__":
    main()
