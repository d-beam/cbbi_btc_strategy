import streamlit as st
import pandas as pd

st.title('CBBI BTC Strategy Dashboard')

st.markdown("""
Welcome to the CBBI BTC Strategy Dashboard! Navigate through the tabs to access different tools.
Außerdem habe ich das noch hinzugefügt.
""")

# Path to your CSV file
DATA_FILE = 'bitcoin_halving_data.csv'
data = pd.read_csv(DATA_FILE)

# Display the current dataset in a table
st.write('Bitcoin Halving Data', data)
