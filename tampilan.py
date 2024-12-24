import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta
import streamlit as st

# Muat kembali DataFrame
df = pd.read_pickle('/content/dataframe.pkl')

# Fungsi untuk melakukan prediksi dan menampilkan hasil
def predict_range_and_plot(start_date, end_date):
    # Pastikan start_date dan end_date dalam format datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Validasi input tanggal
    if end_date <= start_date:
        print("Tanggal akhir harus lebih besar dari tanggal awal.")
        return None

    # Hitung jumlah hari yang akan diprediksi
    total_days = (end_date - start_date).days + 1

    # Ambil data terbaru untuk memulai prediksi (60 hari terakhir)
    recent_data = df['Price'][-60:].values.reshape(-1, 1)
    recent_scaled = sc.transform(recent_data)

    # Reshape untuk input ke model LSTM
    x_input = np.reshape(recent_scaled, (1, recent_scaled.shape[0], 1))

    # Inisialisasi array untuk menyimpan prediksi
    predictions = []

    # Prediksi untuk setiap hari dalam rentang tanggal
    for _ in range(total_days):
        # Prediksi hari berikutnya
        next_prediction = model.predict(x_input)

        # Inverse transform untuk mendapatkan nilai asli
        next_prediction_scaled = sc.inverse_transform(next_prediction)

        # Tambahkan prediksi ke dalam list
        predictions.append(next_prediction_scaled[0][0])

        # Update x_input dengan prediksi terbaru untuk prediksi berikutnya
        x_input = np.append(x_input[:, 1:, :], [[next_prediction[0]]], axis=1)

    # Buat DataFrame untuk prediksi dengan tanggal
    prediction_dates = pd.date_range(start=start_date, end=end_date)
    predicted_df = pd.DataFrame({
        'Tanggal': prediction_dates,
        'Prediksi Harga Penutupan (USD)': predictions
    })

    # Plot grafik
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], df['Price'][-100:], label='Harga Historis', color='#227E19')  # Menampilkan data historis 100 hari terakhir
    plt.plot(predicted_df['Tanggal'], predicted_df['Prediksi Harga Penutupan (USD)'], label='Prediksi Harga', color='#EA2641', linestyle='--')
    plt.title('Prediksi Harga XAU/USD', fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Harga Penutupan (USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Tampilkan tabel prediksi
    print(predicted_df)

# Input tanggal awal dan akhir
start_date = input("Masukkan tanggal awal (format: YYYY-MM-DD): ")
end_date = input("Masukkan tanggal akhir (format: YYYY-MM-DD): ")

# Melakukan prediksi dan menampilkan hasil
predict_range_and_plot(start_date, end_date)
