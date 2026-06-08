import io
import base64
import pandas as pd
import numpy as np
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as pltt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

app.secret_key = "supersecretkey"  # ganti sesuai kebutuhan

# Helper: plot ke base64
def plot_series_to_base64(dates, values, title="Total Pendapatan Harian"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Total Pendapatan")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Helper: buat sequence LSTM
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Pipeline utama: baca, bersihkan, latih, prediksi
def train_and_predict_next_month(df, window_size=30, epochs=50):
    # Validasi kolom
    if 'Tanggal' not in df.columns or 'total_pendapatan' not in df.columns:
        raise ValueError("File harus punya kolom 'Tanggal' dan 'total_pendapatan'.")

    # Bersihkan data
    df = df[['Tanggal', 'total_pendapatan']].copy()
    # Konversi tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce', dayfirst=True)
    # Buang baris invalid
    df = df.dropna(subset=['Tanggal'])
    # Koersi numerik total_pendapatan dan isi kosong dengan 0
    df['total_pendapatan'] = pd.to_numeric(df['total_pendapatan'], errors='coerce').fillna(0)

    # Sort by tanggal
    df = df.sort_values('Tanggal').reset_index(drop=True)

    # Ambil seri nilai
    values = df['total_pendapatan'].values.reshape(-1, 1)

    # Skala
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(values)

    # Minimal length check
    if len(data_scaled) <= window_size + 1:
        raise ValueError(f"Data terlalu sedikit untuk window_size={window_size}. Butuh minimal {window_size + 2} baris.")

    # Sequence
    X, y = create_sequences(data_scaled, window_size)

    # Split train/val (80/20)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Model
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(window_size, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_val, y_val), callbacks=[es], verbose=0)

    # Prediksi 30 hari ke depan secara autoregressive
    last_window = data_scaled[-window_size:].reshape(1, window_size, 1)
    preds_scaled = []
    for _ in range(30):
        next_scaled = model.predict(last_window, verbose=0)
        preds_scaled.append(next_scaled[0, 0])
        # Update window: geser dan tambah prediksi
        new_window = np.append(last_window.flatten()[1:], next_scaled[0, 0]).reshape(1, window_size, 1)
        last_window = new_window

    # Kembalikan ke skala rupiah
    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds_rupiah = scaler.inverse_transform(preds_scaled_arr).flatten()

    # Tanggal prediksi: mulai dari tanggal terakhir + 1 hari
    start_date = df['Tanggal'].iloc[-1] + timedelta(days=1)
    pred_dates = [start_date + timedelta(days=i) for i in range(30)]

    # Total bulan depan (30 hari) sebagai agregat
    total_next_month = np.sum(preds_rupiah)

    return {
        "df_clean": df,
        "daily_plot_base64": plot_series_to_base64(df['Tanggal'], df['total_pendapatan'], "Total Pendapatan Harian (Aktual)"),
        "pred_dates": pred_dates,
        "pred_values": preds_rupiah,
        "pred_plot_base64": plot_series_to_base64(pred_dates, preds_rupiah, "Prediksi Harian 30 Hari Ke Depan"),
        "total_next_month": int(total_next_month)
    }

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("Tidak ada file yang diupload.")
            return redirect(url_for('index'))
        f = request.files['file']
        if f.filename == "":
            flash("Pilih file Excel terlebih dahulu.")
            return redirect(url_for('index'))

        try:
            # Baca Excel dari file upload (in-memory)
            df = pd.read_excel(f)
            # Buat grafik harian awal untuk ditampilkan sebelum training
            df_display = df.copy()
            if 'Tanggal' in df_display.columns and 'total_pendapatan' in df_display.columns:
                df_display['Tanggal'] = pd.to_datetime(df_display['Tanggal'], errors='coerce', dayfirst=True)
                df_display = df_display.dropna(subset=['Tanggal'])
                df_display['total_pendapatan'] = pd.to_numeric(df_display['total_pendapatan'], errors='coerce').fillna(0)
                daily_plot_base64 = plot_series_to_base64(df_display['Tanggal'], df_display['total_pendapatan'])
            else:
                daily_plot_base64 = None

            # Simpan sementara ke session-like memory? Di sini kita kirim DataFrame lewat konteks global sederhana
            # Agar sederhana, kita kirim df sebagai HTML preview tanpa sesi.
            return render_template("index.html", daily_plot_base64=daily_plot_base64, has_data=True, preview_rows=df.head(20).to_dict(orient='records'))
        except Exception as e:
            flash(f"Error membaca file: {e}")
            return redirect(url_for('index'))

    return render_template("index.html", daily_plot_base64=None, has_data=False)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        flash("Tidak ada file untuk diproses. Upload terlebih dahulu.")
        return redirect(url_for('index'))

    f = request.files['file']
    if f.filename == "":
        flash("Pilih file Excel terlebih dahulu.")
        return redirect(url_for('index'))

    try:
        df = pd.read_excel(f)
        result = train_and_predict_next_month(df, window_size=30, epochs=50)

        return render_template(
            "result.html",
            daily_plot_base64=result["daily_plot_base64"],
            pred_plot_base64=result["pred_plot_base64"],
            total_next_month=result["total_next_month"],
        )
    except Exception as e:
        flash(f"Gagal melakukan prediksi: {e}")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
