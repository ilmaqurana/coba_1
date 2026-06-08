import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def predict_lstm(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['pendapatan']])

    def create_dataset(ds, step=7):
        X, y = [], []
        for i in range(len(ds)-step):
            X.append(ds[i:i+step, 0])
            y.append(ds[i+step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled)

    # fallback jika data kecil
    if len(X) == 0:
        avg = data["pendapatan"].mean()
        future = np.array([avg]*30).reshape(-1,1)
        future_dates = pd.date_range(
            data["tanggal"].iloc[-1] + pd.Timedelta(days=1),
            periods=30
        )
        return future, future_dates, 0, 0

    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(7,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0
    )

    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred)) if len(y_test) > 0 else 0
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100 if len(y_test) > 0 else 0

    last = scaled[-7:].reshape(1,7,1)
    future = []

    for _ in range(30):
        p = model.predict(last, verbose=0)
        future.append(p[0,0])
        last = np.concatenate((last[:,1:,:], p.reshape(1,1,1)), axis=1)

    future = scaler.inverse_transform(np.array(future).reshape(-1,1))

    future_dates = pd.date_range(
        data["tanggal"].iloc[-1] + pd.Timedelta(days=1),
        periods=30
    )

    return future, future_dates, rmse, mape