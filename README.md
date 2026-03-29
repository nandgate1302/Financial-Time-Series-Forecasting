# **Multivariate Financial Pattern Recognition using CNN & STFT**

**Name :** Nandana Sasikumar

**University Registration Number :** TCR24CS050

**Course :** Pattern Recognition

**Institution :** Government Engineering College, Thrissur

---
## Project Overview

This project focuses on forecasting stock prices using a hybrid approach that combines **signal processing** and **deep learning** techniques.

Financial time series data is inherently **non-stationary**, meaning its statistical properties change over time. To capture these complex patterns, the raw time-series data is transformed into a **time–frequency representation** using the Short-Time Fourier Transform (STFT).

The resulting **spectrograms** provide a two-dimensional view of how frequency components evolve over time. These spectrograms are then used as inputs to a **Convolutional Neural Network (CNN)**, which learns spatial patterns to predict future stock prices.

The pipeline is applied independently to multiple companies (RELIANCE, TCS, INFY), and performance is evaluated using Mean Squared Error (MSE).

---

## Objective
To combine time-frequency signal processing and deep learning techniques to predict stock prices.

---

## Approach

Financial data is modeled as a multivariate signal:

X(t) = [p(t), r(t), g(t), s(t), d(t)]

Where:
- p(t): Stock price
- r(t): Moving average (proxy for revenue)
- g(t): Volatility (proxy for profit)
- s(t): Sensex
- d(t): USD-INR exchange rate

---

## Methodology

1. Data Collection using Yahoo Finance  
2. Signal Processing:
   - Fourier Transform (FFT)
   - Short-Time Fourier Transform (STFT)
3. Spectrogram Generation  
4. CNN-based Regression Model  

---

## CNN Architecture

Input → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout → Output

---

## Results

| Stock | MSE |
|------|------|
| RELIANCE |  0.01053 |
| TCS | 0.00747 |
| INFY | 0.00774 |

---

## Conclusion

The project demonstrates that financial time series can be effectively analyzed using signal processing techniques and CNN models. Spectrogram-based representations help capture hidden patterns and improve prediction performance.

---

## Repository Structure

```
notebooks/    → Jupyter notebook version (recommended)
src/          → Python script version
plots/        → Generated visualizations and CNN diagram
README.md     → Project documentation
requirements.txt → Dependencies
```

## How to Run

1. Clone the Repository

```bash
git clone https://github.com/nandgate1302/Financial-Time-Series-Forecasting.git
cd Financial-Time-Series-Forecasting
```
2. Install Dependencies

```bash
pip install -r requirements.txt
```
### Option 1: Run using Jupyter Notebook (Recommended)

```bash
jupyter notebook
```
Then open: 
```bash
notebooks/financial_time_series_forecasting.ipynb
```
This will:

- Execute all steps interactively
- Show plots (time domain, FFT, spectrograms)
- Display results clearly

### Option 2: Run using Python Script

```bash
python src/financial_time_series_forecasting.py
```

This will:

- Run the full pipeline
- Train the CNN model
- Output MSE values in terminal
- Save CNN architecture diagram (cnn_arch_diag.png)

Note:

- Plots will open in separate windows
- Ensure required libraries are installed

## Output

- Spectrogram and analysis plots are stored in the ```plots/``` directory
- CNN architecture diagram is saved as: ```plots/cnn_arch_diag.png``` 
