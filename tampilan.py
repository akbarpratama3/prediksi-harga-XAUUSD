import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import seaborn as sns

# Load model and scaler
model = load_model('model_lstm.h5')

with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

df = pd.read_pickle('dataframe.pkl')

# Function to predict and plot
def predict_range_and_plot(start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if end_date <= start_date:
        st.warning("Tanggal akhir harus lebih besar dari tanggal awal.")
        return None

    total_days = (end_date - start_date).days + 1
    recent_data = df['Price'][-60:].values.reshape(-1, 1)
    recent_scaled = sc.transform(recent_data)
    x_input = np.reshape(recent_scaled, (1, recent_scaled.shape[0], 1))

    predictions = []
    for _ in range(total_days):
        next_prediction = model.predict(x_input)
        next_prediction_scaled = sc.inverse_transform(next_prediction)
        predictions.append(next_prediction_scaled[0][0])
        x_input = np.append(x_input[:, 1:, :], [[next_prediction[0]]], axis=1)

    prediction_dates = pd.date_range(start=start_date, end=end_date)
    predicted_df = pd.DataFrame({
        'Tanggal': prediction_dates,
        'Prediksi Harga Penutupan (USD)': predictions
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-100:], df['Price'][-100:], label='Real Price', color='#252F75')
    ax.plot(predicted_df['Tanggal'], predicted_df['Prediksi Harga Penutupan (USD)'], label='Predictions', color='#EA2641', linestyle='--')
    ax.set_title('Prediksi Harga XAU/USD', fontsize=16)
    ax.set_xlabel('Tanggal', fontsize=12)
    ax.set_ylabel('Harga Penutupan (USD)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.write(predicted_df)

# Streamlit UI
st.title('Prediksi Harga XAU/USD')

start_date = st.date_input("Masukkan tanggal awal", datetime.today())
end_date = st.date_input("Masukkan tanggal akhir", datetime.today())

if st.button('Prediksi Harga'):
    predict_range_and_plot(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
