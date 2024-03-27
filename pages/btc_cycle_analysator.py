import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

# Load the Bitcoin halving data from a CSV file one level above the working directory
DATA_FILE = 'bitcoin_halving_data.csv'
bitcoin_halving_data = pd.read_csv(DATA_FILE)

def main():
    st.title("Interactive Plot of BTC Data")

    # Assume fetching and processing external data works as before,
    # resulting in a DataFrame df with 'Date', 'Price', and 'CBBI' columns
    # Placeholder for fetching and processing external data
    # For demonstration purposes, let's create an empty DataFrame
    df = pd.DataFrame()

    # Create an empty figure
    fig = go.Figure()

    # Add traces for BTC Price and CBBI if df is not empty
    # Placeholder code for adding traces
    # For demonstration, these traces will not be added

    # Manually add vertical lines for Bitcoin halving events
    for index, row in bitcoin_halving_data.iterrows():
        halving_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')  # Ensure date format is YYYY-MM-DD
        fig.add_shape(type="line",
                      x0=halving_date, y0=0, x1=halving_date, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="gray", dash="dash"))
        fig.add_annotation(x=halving_date, y=0.95, xref="x", yref="paper",
                           text=row['Event'], showarrow=False, yanchor="top")

    # Update figure layout
    fig.update_layout(
        title="BTC Price and CBBI Over Time",
        xaxis=dict(title='Date'),
        yaxis=dict(title='BTC Price', type='log', showgrid=False),
        yaxis2=dict(title='CBBI', overlaying='y', side='right', dtick=5, showgrid=True),
        legend=dict(x=0, y=1, traceorder='reversed', font_size=16, bgcolor='rgba(255,255,255,0.5)'),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # Display the Bitcoin halving data as a table below the chart
    st.write("Bitcoin Halving Events Table:")
    st.dataframe(bitcoin_halving_data)

if __name__ == "__main__":
    main()
