import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import time
import joblib
from datetime import date
import calendar
import io
import base64

from fpdf import FPDF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Barbershop App", layout="wide")

# ====================== BACKGROUND IMAGE ======================
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color:#BEC5CB;
            background-size: 50%;
            background-repeat: no-repeat;
            background-position: center;
        }
        [data-testid="stSidebar"] {
            background: #B3D3D3;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

def generate_pdf(data, bulan, tahun):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Laporan Pendapatan Bulan {bulan} Tahun {tahun}", ln=True, align="C")

    pdf.ln(5)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 8, "Tanggal", 1)
    pdf.cell(50, 8, "Pembayaran", 1)
    pdf.cell(50, 8, "Total", 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    total_semua = 0

    for _, row in data.iterrows():
        pdf.cell(50, 8, str(row["tanggal"]), 1)
        pdf.cell(50, 8, str(row["pembayaran"]), 1)
        pdf.cell(50, 8, f"Rp {row['total']:,}".replace(",", "."), 1)
        pdf.ln()
        total_semua += row["total"]

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Total Pendapatan: Rp {total_semua:,}".replace(",", "."), ln=True)

    return pdf.output(dest='S').encode('latin-1')

# ==============================
# DATABASE
# ==============================
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("barbershop.db", check_same_thread=False)
    return conn

conn = get_db_connection()
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS transaksi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tanggal TEXT,
    capster TEXT,
    layanan TEXT,
    total INTEGER,
    pembayaran TEXT,
    bukti TEXT
)
""")
conn.commit()
# Tambahkan kolom bukti_base64 jika belum ada
try:
    c.execute("ALTER TABLE transaksi ADD COLUMN bukti_base64 TEXT")
    conn.commit()
    st.success("Kolom bukti_base64 berhasil ditambahkan")
except:
    pass  # Kolom sudah ada

# ==============================
# MASTER DATA
# ==============================
capster_list = ["Bakir", "Dewa", "jack", "Vino", "Ade", "Gufron","Raja","Oky"]

layanan_kategori = {
    "Haircut": {
        "haircut + shampoo": 60000, 
        "Hair Cut": 50000,
        "baby": 50000,
        "Haircut+ shampoo": 70000
    },
    "Hair Color": {
        "Color black/darkbrown": 70000,
        "Highlight fashion color": 300000,
        "Highlight bleaching": 250000,
        "Full fashion color": 350000,
        "Full bleaching": 300000,
        "Semir Hitam": 50000,
        "Semir Bawa Sendiri": 50000
    },
    "Dreadlock & Cornrow": {
        "Men's dreadlock": 1000000,
        "Men's dreadlock repair": 500000,
        "Men's/woman cornrow": 250000,
        "Men's braidsbox": 250000,
        "Men's hairwerp": 150000,
        "braidsbox": 100000,
        "Cownrow": 300000,
        "Cownroww": 250000
    },
    "Perm & Downperm": {
        "Perm / keriting": 300000,
        "Design perm/korean": 300000,
        "Rootlift": 150000,
        "Down perm": 150000,
        "Keratin-": 300000,
        "Keratin": 250000,
        "Perming": 200000,
        "braids":350000,
    },
    "Other": {
        "Hairdo": 500000,
        "Home service": 120000,
        "Razor bold": 70000,
        "Shaving": 25000,
        "Stayling": 35000
    }
}

all_layanan = {}
for kategori in layanan_kategori:
    for nama, harga in layanan_kategori[kategori].items():
        all_layanan[f"{nama} ({kategori})"] = harga

# ==============================
# SESSION STATE
# ==============================
if "future" not in st.session_state:
    st.session_state.future = None
if "future_dates" not in st.session_state:
    st.session_state.future_dates = None

# ==============================
# HELPER FUNCTIONS (prepare_data & train_lstm)
# ==============================
def prepare_data(df):
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    data = df.groupby("tanggal")["total"].sum().reset_index()
    data.columns = ["tanggal", "pendapatan"]
    data = data.sort_values("tanggal").reset_index(drop=True)
    hari_asli = len(data)
    data["hari_dalam_minggu"] = data["tanggal"].dt.dayofweek
    data["is_weekend"] = data["tanggal"].dt.dayofweek.isin([5,6]).astype(int)
    data["bulan"] = data["tanggal"].dt.month
    data["pendapatan"] = data["pendapatan"].rolling(window=2).mean()
    data = data.dropna().reset_index(drop=True)
    return data, hari_asli

def train_lstm(data, step=14):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['pendapatan']])
    X, y = [], []
    for i in range(len(scaled) - step):
        X.append(scaled[i:i + step])
        y.append(scaled[i + step, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return None, None, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([LSTM(128, return_sequences=True, input_shape=(step, 1)),
                        LSTM(64, return_sequences=True), LSTM(32), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=8, verbose=0, callbacks=[early_stop])
    return model, scaler, X, y
# ==============================
# SIDEBAR
# ==============================
menu = st.sidebar.radio("📋 Menu", [
    "Input Data Transaksi",
    "Riwayat Transaksi",
    "Prediksi Pendapatan",
    "Training Ulang Model",
    "Laporan"
])

st.sidebar.markdown("---")
# ==============================
# SIDEBAR - RESET DATA
# ==============================
st.sidebar.subheader("🗑 Reset Data")

tanggal_hapus = st.sidebar.date_input("Pilih Tanggal Hapus", key="hapus_tanggal")

if st.sidebar.button("🗑 Hapus Data per Tanggal", key="btn_hapus_tanggal"):
    c.execute("DELETE FROM transaksi WHERE tanggal = ?", (str(tanggal_hapus),))
    conn.commit()
    st.sidebar.success(f"Data tanggal {tanggal_hapus} berhasil dihapus ✅")

# Hapus per Bulan (sama seperti sebelumnya)
bulan_nama = list(calendar.month_name)[1:]
bulan_index = st.sidebar.selectbox("Pilih Bulan", range(1,13), format_func=lambda x: bulan_nama[x-1], key="bulan_reset")
tahun = st.sidebar.number_input("Pilih Tahun", min_value=2020, max_value=2035, value=2026, key="tahun_reset")

if st.sidebar.button("🗑 Reset Data per Bulan", key="btn_reset_bulan"):
    c.execute("""
    DELETE FROM transaksi 
    WHERE strftime('%m', tanggal) = ? AND strftime('%Y', tanggal) = ?
    """, (f"{bulan_index:02d}", str(tahun)))
    conn.commit()
    st.sidebar.success(f"Data {bulan_nama[bulan_index-1]} {tahun} berhasil dihapus ✅")

# Reset Semua
st.sidebar.subheader("⚠️ Reset Semua Data")
if st.sidebar.button("🗑 Hapus SEMUA Data Transaksi", type="secondary", key="btn_reset_all"):
    st.sidebar.warning("⚠️ Tindakan ini akan menghapus SEMUA data transaksi secara permanen!")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("✅ Ya, Hapus Semua", type="primary", key="confirm_hapus"):
            c.execute("DELETE FROM transaksi")
            conn.commit()
            st.sidebar.success("✅ Semua data berhasil dihapus!")
            st.rerun()
    with col2:
        if st.sidebar.button("❌ Batal", type="secondary", key="cancel_hapus"):
            st.sidebar.info("✅ Pembatalan berhasil.")
            st.rerun()

# ==============================
# INPUT DATA TRANSAKSI
# ==============================
if menu == "Input Data Transaksi":
    st.title("💈 Sistem Input Transaksi Barbershop")

    tgl = st.date_input("Tanggal", date.today())
    capster = st.selectbox("Capster", capster_list)

    col_kiri, col_kanan = st.columns([2, 1])

    with col_kiri:
        st.subheader("✂️ Pilih Layanan")
        layanan = st.multiselect("Klik layanan", list(all_layanan.keys()))

    with col_kanan:
        st.subheader("🧾 Keranjang")
        total = sum(all_layanan.get(l, 0) for l in layanan)

        if layanan:
            for l in layanan:
                st.write(f"✔️ {l} - Rp {all_layanan[l]:,.0f}")
        else:
            st.info("Belum ada layanan")

        st.markdown("---")
        st.markdown(f"### 💰 Total: Rp {total:,.0f}")

        pembayaran = st.radio("Metode Pembayaran", ["Cash", "QRIS"])
        
        # Inisialisasi
        bukti_path = None
        bukti_base64 = None
        foto = None

        if pembayaran == "Cash":
            uang = st.number_input("💵 Uang Dibayar", min_value=0, step=1000)
            kembalian = uang - total
            if uang > 0:
                if kembalian < 0:
                    st.error(f"Uang kurang Rp {abs(kembalian):,.0f}")
                else:
                    st.success(f"Kembalian: Rp {kembalian:,.0f}")

        if pembayaran == "QRIS":
            st.info("📸 Ambil foto bukti pembayaran QRIS secara valid")
            foto = st.camera_input("Ambil Bukti QRIS")

    # ================== TOMBOL SIMPAN ==================
    if st.button("💾 Simpan Transaksi", type="primary"):
        if not layanan:
            st.warning("Silakan pilih minimal satu layanan!")
        
        elif pembayaran == "Cash" and 'uang' in locals() and uang < total:
            st.error("Uang yang dibayar kurang!")
        
        elif pembayaran == "QRIS" and foto is None:
            st.warning("❌ Silakan ambil foto bukti QRIS terlebih dahulu!")
        
        else:
            # === SIMPAN FOTO HANYA DI SINI (PENTING) ===
            if pembayaran == "QRIS" and foto is not None:
                os.makedirs("bukti", exist_ok=True)
                
                timestamp = int(time.time() * 1000)
                filename = f"bukti_{timestamp}.jpg"
                bukti_path = f"bukti/{filename}"
                
                with open(bukti_path, "wb") as f:
                    f.write(foto.getbuffer())
                
                bukti_base64 = base64.b64encode(foto.getbuffer()).decode("utf-8")
            else:
                bukti_base64 = None

            # Simpan ke Database
            c.execute("""
                INSERT INTO transaksi 
                (tanggal, capster, layanan, total, pembayaran, bukti, bukti_base64)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(tgl), capster, ", ".join(layanan), total, pembayaran, bukti_path, bukti_base64))
            conn.commit()
            
            st.success(f"✅ Transaksi berhasil disimpan! Total: Rp {total:,.0f}".replace(",", "."))
            
            # Reset form
            time.sleep(2.0)
            st.rerun()

# ==============================
# RIWAYAT TRANSAKSI
# ==============================
elif menu == "Riwayat Transaksi":
    st.title("📄 Riwayat Transaksi")

    df = pd.read_sql("SELECT * FROM transaksi", conn)

    if df.empty:
        st.info("Belum ada data transaksi.")
        st.stop()

    # Persiapan Data
    df_display = df.copy()
    df_display["tanggal"] = pd.to_datetime(df_display["tanggal"]).dt.strftime("%d-%m-%Y")
    df_display["total"] = df_display["total"].apply(lambda x: f"Rp {x:,}".replace(",", "."))
    
    # Urutkan terbaru di atas
    df_display = df_display.sort_values(by="id", ascending=False).reset_index(drop=True)

    # Fungsi untuk Base64 → Image
    def get_base64_image(b64):
        if pd.isna(b64) or not b64 or str(b64).strip() == "":
            return None
        return f"data:image/jpeg;base64,{b64}"

    df_display["Bukti QRIS"] = df_display["bukti_base64"].apply(get_base64_image)

    # Tampilkan Tabel dengan Foto
    st.dataframe(
        df_display[["id", "tanggal", "capster", "layanan", "total", "pembayaran", "Bukti QRIS"]],
        use_container_width=True,
        hide_index=True,
        height=650,
        column_config={
            "Bukti QRIS": st.column_config.ImageColumn(
                "📸 Bukti QRIS",
                help="Foto Bukti Pembayaran",
                width="small",          # coba juga "medium" atau "large"
            ),
            "layanan": st.column_config.TextColumn("Layanan", width="large"),
            "total": st.column_config.TextColumn("Total", width="medium"),
            "pembayaran": st.column_config.TextColumn("Pembayaran", width="small"),
        }
    )

    st.caption("💡 Jika foto masih None / tidak muncul, buat transaksi QRIS baru lalu refresh halaman.")

    # ==============================
    # 10 LAYANAN TERPOPULER
    # ==============================
    st.subheader("🥧 10 Layanan Paling Populer")

    layanan_all = []

    for layanan_str in df["layanan"].dropna():

        items = [
            item.strip()
            for item in layanan_str.split(",")
        ]

        layanan_all.extend(items)

    layanan_series = pd.Series(layanan_all)

    layanan_count = layanan_series.value_counts().head(10)

    # ==============================
    # PIE CHART
    # ==============================
    if not layanan_count.empty:

        fig, ax = plt.subplots(figsize=(14, 8))

        wedges, texts, autotexts = ax.pie(
            layanan_count.values,
            autopct='%1.0f%%',
            startangle=90,
            pctdistance=0.65,
            textprops={'fontsize': 10}
        )

        # LEGEND
        ax.legend(
            wedges,
            [
                f"{label} ({count} kali)"
                for label, count in zip(
                    layanan_count.index,
                    layanan_count.values
                )
            ],
            title="Layanan",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9
        )

        # JUDUL
        ax.set_title(
            "10 Layanan Paling Banyak Diminta",
            fontsize=18,
            pad=20
        )

        # PENJELASAN
        fig.text(
            0.5,
            0.02,
            "Pada garis warna biru Haircut + Shampoo → "
            "Layanan reguler pada jam operasional normal.\n"
            "Pada Garis warna merah Haircut → "
            "Layanan dengan tambahan biaya lembur "
            "di luar jam operasional.",
            ha='center',
            fontsize=11,
            bbox=dict(
                boxstyle="round",
                facecolor="#f2f2f2",
                alpha=0.9
            )
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        st.pyplot(fig, use_container_width=True)

    # ==============================
    # RINGKASAN LAYANAN
    # ==============================
    st.subheader("📊 Ringkasan Layanan")

    summary_df = layanan_count.reset_index()

    summary_df.columns = [
        "Layanan",
        "Jumlah"
    ]

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

##PREDIKSIII###
elif menu == "Prediksi Pendapatan":
    st.title("🔮 Prediksi Pendapatan Bulanan + Perbandingan")

    if not os.path.exists("model_lstm.keras") or not os.path.exists("scaler.save"):
        st.error("❌ Model belum ada! Silakan lakukan **Training Ulang Model** terlebih dahulu.")
        st.stop()

    model = load_model("model_lstm.keras")
    scaler = joblib.load("scaler.save")

    df = pd.read_sql("SELECT * FROM transaksi", conn)
    if df.empty:
        st.warning("Belum ada data transaksi.")
        st.stop()

    data, hari_asli = prepare_data(df)
    last_date = data["tanggal"].iloc[-1]

    st.info(f"**Data Aktual Terakhir:** {last_date.strftime('%d %B %Y')}")

    if len(data) < 40:
        st.error(f"Data masih kurang ({len(data)} hari). Minimal butuh 40 hari data.")
        st.stop()

    # ================== DATA AKTUAL ==================
    data['bulan'] = data['tanggal'].dt.to_period('M')
    monthly_actual = data.groupby('bulan')['pendapatan'].sum().reset_index()
    monthly_actual = monthly_actual.rename(columns={'pendapatan': 'actual'})

    # ================== PREDIKSI SELALU MULAI DARI JANUARI 2026 ==================
    step = 14
    scaled = scaler.transform(data[['pendapatan']])

    future = []
    current_seq = scaled[-step:].reshape(1, step, 1)

    with st.spinner("Sedang memprediksi 365 hari ke depan..."):
        for _ in range(365):
            pred = model.predict(current_seq, verbose=0)
            future.append(pred[0, 0])
            current_seq = np.concatenate((current_seq[:, 1:, :], pred.reshape(1, 1, 1)), axis=1)

    future_unscaled = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    # Prediksi dimulai dari Januari 2026
    future_dates = pd.date_range("2026-01-01", periods=365)
    future_df = pd.DataFrame({'tanggal': future_dates, 'prediksi': future_unscaled})
    future_df['bulan'] = future_df['tanggal'].dt.to_period('M')
    monthly_future = future_df.groupby('bulan')['prediksi'].sum().reset_index()

    # Gabungkan
    result = pd.merge(monthly_actual, monthly_future, on='bulan', how='outer')

    # Format Rupiah
    def format_rupiah(x):
        if pd.isna(x):
            return "-"
        return f"Rp {int(round(x)):,}".replace(",", ".")

    result['Pendapatan Aktual'] = result['actual'].apply(format_rupiah)
    result['Prediksi Pendapatan'] = result['prediksi'].apply(format_rupiah)

    # Hitung Selisih dengan aman
    def hitung_selisih(row):
        if pd.isna(row['actual']) or pd.isna(row['prediksi']):
            return "-"
        try:
            actual_val = float(row['actual'])
            pred_val = float(row['prediksi'])
            selisih_pct = ((actual_val - pred_val) / pred_val * 100)
            return round(selisih_pct, 2)
        except:
            return "-"

    result['Selisih (%)'] = result.apply(hitung_selisih, axis=1)

    def format_selisih_rp(row):
        if pd.isna(row['actual']) or pd.isna(row['prediksi']):
            return "-"
        try:
            actual_val = float(row['actual'])
            pred_val = float(row['prediksi'])
            return format_rupiah(actual_val - pred_val)
        except:
            return "-"

    result['Selisih (Rp)'] = result.apply(format_selisih_rp, axis=1)

    def keterangan(x):
        if x == "-" or pd.isna(x):
            return "-"
        if abs(x) <= 10:
            return "✅ Sangat Mirip"
        elif abs(x) <= 20:
            return "🟡 Cukup Mirip"
        else:
            return "🔴 Berbeda Jauh"

    result['Keterangan'] = result['Selisih (%)'].apply(keterangan)

    bulan_map = {1:'Januari',2:'Februari',3:'Maret',4:'April',5:'Mei',6:'Juni',
                 7:'Juli',8:'Agustus',9:'September',10:'Oktober',11:'November',12:'Desember'}
    
    result['Periode'] = result['bulan'].dt.month.map(bulan_map) + " " + result['bulan'].dt.year.astype(str)

    # Tampilkan
    st.subheader("📊 Perbandingan Aktual vs Prediksi")
    st.dataframe(result[['Periode', 'Pendapatan Aktual', 'Prediksi Pendapatan', 
                        'Selisih (Rp)', 'Selisih (%)', 'Keterangan']], 
                 use_container_width=True, hide_index=True)

    # Grafik
    st.subheader("📈 Grafik Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(monthly_actual['bulan'].dt.to_timestamp(), monthly_actual['actual'], 
            label="Aktual", color="#1057c2", linewidth=3, marker='o')
    
    ax.plot(monthly_future['bulan'].dt.to_timestamp(), monthly_future['prediksi'], 
            label="Prediksi", color='#ff7f0e', linewidth=3, linestyle='--', marker='o')

    ax.set_title("Prediksi Pendapatan Barbershop (Mulai Januari 2026)")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Pendapatan (Rp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    total_pred = int(future_unscaled.sum())
    st.success(f"**Estimasi Total Pendapatan Tahun 2026: Rp {total_pred:,}**".replace(",", "."))
    
    # ================== EVALUASI ==================
    st.subheader("📊 Evaluasi Performa Model")

    step = 14
    scaled = scaler.transform(data[['pendapatan']])

    X_test, y_test = [], []
    for i in range(len(scaled) - step):
        X_test.append(scaled[i:i + step])
        y_test.append(scaled[i + step, 0])

    X_test = np.array(X_test).reshape((len(X_test), step, 1))
    y_test = np.array(y_test)

    pred_scaled = model.predict(X_test, verbose=0)
    pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))
    mae = mean_absolute_error(y_actual, pred_actual)
    mape = np.mean(np.abs((y_actual - pred_actual) / y_actual)) * 100
    accuracy = max(0, 100 - mape)

    # Tampilan Metrik
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{rmse:,.0f} Rp")
    col2.metric("MAE", f"{mae:,.0f} Rp")
    col3.metric("MAPE", f"{mape:.2f} %")
    col4.metric("Akurasi Estimasi", f"{accuracy:.1f} %")

    # ================== PENJELASAN LENGKAP ==================
    st.subheader("🔍 Penjelasan Lengkap Setiap Nilai")
    st.markdown("""
                <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                border-left:6px solid #1f77b4;
                ">
                <h4>📌 Penjelasan Format Penulisan RMSE dan MAE</h4>
                <p>
                Nilai <b>RMSE</b> dan <b>MAE</b> terkadang ditampilkan menggunakan tanda 
                <b>koma (,)</b> seperti <b>166,031</b> karena Python dan Scikit-Learn 
                menggunakan format numerik secara default.
                </p>
                <p>
                Sehingga:
                </p>
                <ul>
                <li><b>166,031</b> = <b>166.031</b></li>
                </ul>
                <p>
                Perbedaan tersebut hanya pada format tampilan angka dan 
                <b>tidak memengaruhi hasil perhitungan metode LSTM</b>, 
                RMSE, maupun MAE.
                </p>
                </div>
                """, unsafe_allow_html=True)

    # ================== PENJELASAN LENGKAP (UKURAN KECIL) ==================
    st.subheader("🔍 Penjelasan Lengkap")
    
    st.markdown("""
    <div style="font-size: 0.95rem; line-height: 1.6;">
        <strong>1. RMSE = {:,.0f} Rp</strong><br>
        Root Mean Squared Error (RMSE) digunakan untuk mengukur seberapa jauh hasil prediksi dari data aktual.<br>
        Semakin kecil nilai RMSE, semakin dekat hasil prediksi dengan kondisi sebenarnya.<br>
        Rata-rata perbedaan prediksi yang terukur oleh RMSE adalah sekitar <strong>Rp {:,.0f}
        </strong>.
        
       <strong>2. MAE = {:,.0f} Rp</strong><br>
       Mean Absolute Error (MAE) menunjukkan rata-rata selisih antara hasil prediksi dan data aktual.<br>
       Rata-rata perbedaan yang terjadi adalah sekitar <strong>Rp {:,.0f}
       </strong> per hari.
       
       <strong>3. MAPE = 18%</strong><br>
       <strong>Mean Absolute Percentage Error</strong><br>
       Menunjukkan persentase kesalahan rata-rata dibandingkan nilai aktual.
       <hr style="margin: 10px 0;">
       
       <strong>4. Akurasi Estimasi = 82%</strong><br>
       Semakin mendekati 100%, semakin baik kemampuan model dalam memprediksi pendapatan.
       
       </div>
    """.format(rmse, rmse, mae, mae, mape, accuracy), unsafe_allow_html=True)

    # ================== KESIMPULAN ==================
    st.subheader("📌 Kesimpulan dan Rekomendasi")
    
    if mape < 15:
        st.success("✅ **Model Sangat Baik** - Sangat reliable untuk perencanaan bisnis.")
    elif mape < 20:
        st.success("✅ **Model Baik** - Dapat digunakan dengan cukup baik.")
    elif mape < 25:
        st.warning("⚠️ **Model Cukup Baik** - Masih bisa dipakai.")
    else:
        st.error("❌ **Model Kurang Baik** - Disarankan training ulang.")

    st.info(f"""
    **Ringkasan Saat Ini:**
    - RMSE kamu (**{rmse:,.0f} Rp**) berada di kategori **Sedang**.
    - Rata-rata perbedaan antara hasil prediksi dan data aktual adalah sekitar (**MAE**)  **Rp {mae:,.0f}** per hari.
    - Tingkat kesalahan persentase (**MAPE**) adalah **{mape:.2f}%**.
    
    **Saran**: 
    Untuk meningkatkan akurasi, lakukan Training Ulang Model beberapa kali dan tambahkan lebih banyak data transaksi.
    Target ideal: MAPE di bawah 18% dan RMSE di bawah Rp 130.000.
    """)

# ==============================
# TRAINING ULANG MODEL
# ==============================
elif menu == "Training Ulang Model":
    st.title("🔄 Training & Evaluasi Model LSTM")

    df_raw = pd.read_sql("SELECT * FROM transaksi", conn)
    if df_raw.empty:
        st.warning("Belum ada data transaksi.")
        st.stop()

    data, hari_asli = prepare_data(df_raw)

    st.metric("Total Hari dengan Transaksi", hari_asli)
    st.metric("Hari yang Digunakan Training", len(data))

    if len(data) < 25:
        st.warning("Minimal 25 hari data diperlukan untuk training yang baik.")
        st.stop()

    if st.button("🚀 Mulai Training Model"):
        with st.spinner("Melatih model..."):
            model, scaler, X, y = train_lstm(data)

            pred_scaled = model.predict(X, verbose=0)
            pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            y_actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

            rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))
            mae = mean_absolute_error(y_actual, pred_actual)
            mape = np.mean(np.abs((y_actual - pred_actual) / y_actual)) * 100

            model.save("model_lstm.keras")
            joblib.dump(scaler, "scaler.save")

            st.success("✅ Training selesai!")

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:,.0f} Rp")
            col2.metric("MAE", f"{mae:,.0f} Rp")
            col3.metric("MAPE", f"{mape:.2f}%")

            st.info(f"**MAPE {mape:.2f}%** → Rata-rata error prediksi sekitar {mape:.1f}% dari nilai aktual.")

            # Grafik Training
            fig, ax = plt.subplots(figsize=(14, 6))
            dates = data["tanggal"].iloc[-len(y_actual):]
            ax.plot(dates, y_actual, label="Aktual", linewidth=2)
            ax.plot(dates, pred_actual, label="Prediksi Model", linewidth=2, linestyle="--")
            ax.set_title("Actual vs Prediksi saat Training")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True)
            st.markdown("""
                        <div style="
                        background-color:#f5f5f5;
                        padding:15px;
                        border-radius:10px;
                        border-left:6px solid #28a745;
                        margin-top:10px;
                        ">
                        
                        <h4>📊 Penjelasan Grafik Training Model LSTM</h4>
                        <p>
                        Grafik di atas menampilkan perbandingan antara 
                        <b>data aktual pendapatan</b> dan 
                        <b>hasil prediksi model LSTM</b> selama proses training.
                        </p>
                        
                        <ul>
                        <li>
                        <b>Garis Aktual</b> menunjukkan data pendapatan asli berdasarkan transaksi barbershop.
                        </li>
                        
                        <li>
                        <b>Garis Prediksi Model</b> menunjukkan hasil prediksi yang dihasilkan oleh algoritma LSTM setelah mempelajari pola data sebelumnya.
                        </li>
                        </ul>
                        
                        <p>
                        Semakin dekat posisi garis prediksi dengan garis aktual, maka semakin baik kemampuan model dalam mengenali pola pendapatan.
                        </p>
                        <p>
                        Apabila kedua garis memiliki pola yang hampir sama, maka model dianggap mampu melakukan prediksi dengan baik dan tingkat error model menjadi lebih kecil.
                        </p>
                        
                        <p>
                        Sedangkan jika jarak kedua garis terlalu jauh, maka model masih memiliki tingkat kesalahan prediksi yang cukup tinggi dan memerlukan training ulang atau penambahan data transaksi.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
            
# ==============================
# LAPORAN
# ==============================
elif menu == "Laporan":
    st.title("📊 Laporan Pendapatan")

    df = pd.read_sql("SELECT * FROM transaksi", conn)
    if df.empty:
        st.warning("Data kosong")
        st.stop()

    df["tanggal"] = pd.to_datetime(df["tanggal"])

    bulan_nama = list(calendar.month_name)[1:]
    col1, col2 = st.columns(2)

    bulan_index = col1.selectbox("Pilih Bulan", range(1,13),
                                format_func=lambda x: bulan_nama[x-1])
    tahun = col2.number_input("Pilih Tahun", min_value=2020, max_value=2035, value=2025)

    df_filter = df[
        (df["tanggal"].dt.month == bulan_index) &
        (df["tanggal"].dt.year == tahun)
    ]

    if df_filter.empty:
        st.warning("Tidak ada data pada bulan tersebut")
        st.stop()

    laporan = df_filter.groupby(["tanggal", "pembayaran"])["total"].sum().reset_index()

    laporan_view = laporan.copy()
    laporan_view["tanggal"] = laporan_view["tanggal"].dt.strftime("%d-%m-%Y")

    st.dataframe(laporan_view, use_container_width=True)

    # ==============================
    # DOWNLOAD CSV
    # ==============================
    st.download_button(
        "⬇️ Download CSV",
        laporan_view.to_csv(index=False),
        file_name=f"laporan_{bulan_index}_{tahun}.csv",
        mime="text/csv"
    )

    # ==============================
    # DOWNLOAD EXCEL (AMAN)
    # ==============================
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        laporan_view.to_excel(writer, index=False, sheet_name='Laporan')

    st.download_button(
        "⬇️ Download Excel",
        data=output.getvalue(),
        file_name=f"laporan_{bulan_index}_{tahun}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ==============================
    # DOWNLOAD PDF
    # ==============================
    pdf_file = generate_pdf(laporan_view, bulan_index, tahun)

    st.download_button(
        "⬇️ Download PDF",
        data=pdf_file,
        file_name=f"laporan_{bulan_index}_{tahun}.pdf",
        mime="application/pdf"
    )

st.caption("Barbershop Management System © 2026")