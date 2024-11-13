import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Funktion zum Abrufen historischer Preisdaten von CoinGecko
def get_solana_price_data():
    url = 'https://api.coingecko.com/api/v3/coins/solana/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': 'max',  # Abruf aller verfügbaren Daten
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Umwandeln der Preisdaten in ein DataFrame
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.date  # Konvertieren der Zeitstempel in Datum
    df.set_index('Date', inplace=True)
    df.drop(columns='Timestamp', inplace=True)
    return df

# Laden der Daten
df = get_solana_price_data()

# Titel der App
st.title('Historische Solana Preisdaten')

# Anzeigen der Daten
st.write("Hier sind die historischen Solana-Preisdaten:")
st.line_chart(df['Price'])

# Plot der Preisdaten
st.subheader("Preisverlauf von Solana über die Zeit")
fig, ax = plt.subplots()
ax.plot(df.index, df['Price'])
ax.set_xlabel("Datum")
ax.set_ylabel("Preis in USD")
ax.set_title("Historischer Solana Preisverlauf")
st.pyplot(fig)

