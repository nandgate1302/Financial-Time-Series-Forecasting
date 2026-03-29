import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import stft
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# Data Collection and Preprocessing

sensex = yf.download("^BSESN", start='2020-01-01', end='2024-01-01')['Close']
usd_inr = yf.download("INR=X", start='2020-01-01', end='2024-01-01')['Close']

# Multivariate Signal Representation

def create_signal(stock_name):
    # Download all data inside function (important fix)
    stock_df = yf.download(stock_name, start='2020-01-01', end='2024-01-01', auto_adjust=True)
    sensex_df = yf.download("^BSESN", start='2020-01-01', end='2024-01-01', auto_adjust=True)
    usd_df = yf.download("INR=X", start='2020-01-01', end='2024-01-01', auto_adjust=True)

    # Extract Close
    stock = stock_df['Close']
    sensex = sensex_df['Close']
    usd_inr = usd_df['Close']

    # Combine safely using concat (BEST METHOD)
    df = pd.concat([stock, sensex, usd_inr], axis=1)
    df.columns = ['price', 'sensex', 'usd_inr']

    # Drop missing
    df.dropna(inplace=True)

    # Feature engineering
    df['moving_avg'] = df['price'].rolling(10).mean()
    df['volatility'] = df['price'].rolling(10).std()

    df.dropna(inplace=True)

    # Normalize
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    return df, df_scaled

# Time, Frequency, and Time–Frequency Visualization

def visualize_signal(df_scaled, title):
    signal = df_scaled[:, 0]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    # --- Time domain ---
    axes[0].plot(signal)
    axes[0].set_title(f"{title} - Time Domain")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")

    # --- FFT ---
    fft_vals = np.abs(fft(signal))
    freq = np.fft.fftfreq(len(signal))

    axes[1].plot(freq, fft_vals)
    axes[1].set_title("Frequency Spectrum")
    axes[1].set_xlabel("Frequency")

    # --- Spectrogram ---
    f, t, Zxx = stft(signal, nperseg=16)

    im = axes[2].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    axes[2].set_title("Spectrogram")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Frequency")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

# Dataset Preparation using Sliding Window

def create_dataset(data):
    window_size = 64
    X, y = [], []

    for i in range(len(data) - window_size):
        segment = data[i:i+window_size]

        channel_spectrograms = []

        for ch in range(segment.shape[1]):
            f, t, Zxx = stft(segment[:, ch], nperseg=16)
            spec = np.abs(Zxx)**2
            spec = spec / (np.max(spec) + 1e-8)
            channel_spectrograms.append(spec)

        X.append(np.stack(channel_spectrograms, axis=-1))
        y.append(data[i+window_size, 0])

    return np.array(X), np.array(y)

# CNN Model Architecture

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3,1), padding='same', activation='relu'),
        layers.MaxPooling2D((2,1)),

        layers.Conv2D(64, (3,1), padding='same', activation='relu'),
        layers.MaxPooling2D((2,1)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

df_temp, scaled_temp = create_signal('RELIANCE.NS')
X_temp, y_temp = create_dataset(scaled_temp)

input_shape = X_temp.shape[1:]

# Build model
model = build_model(input_shape)

print("Model architecture saved as cnn_arch_diag.png")

plot_model(
    model,
    to_file='cnn_arch_diag.png',
    show_shapes=True,
    show_layer_names=True,
    dpi=100
)

# Model Training

stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
results = {}

for stock in stocks:
    print(f"\nProcessing {stock}...\n")

    df, scaled = create_signal(stock)

    visualize_signal(scaled, stock)

    X, y = create_dataset(scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(X.shape[1:])

    model.fit(X_train, y_train, epochs=10, verbose=0)

    preds = model.predict(X_test)

    mse = np.mean((preds.flatten() - y_test)**2)

    results[stock] = mse

    print(f"{stock} MSE: {mse}")

# Results and Evaluation

print("\nFinal Comparison:")
for stock, mse in results.items():
    print(f"{stock}: MSE = {mse}")