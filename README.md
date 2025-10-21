## Part A — Supervised Learning

**File:** `explore.ipynb`  
**Contents:** Data collection, technical feature engineering, label generation, and multi-class classification of S&P 500 stock price movements using LSTM, LightGBM, and Transformer models.

---

### Dataset Features

The dataset is generated dynamically from online sources and includes the following technical and fundamental features:

| Feature Category | Features | Description |
|-----------------|----------|-------------|
| **Trend Indicators** | `SMA_5`, `SMA_20`, `SMA_ratio`, `EMA_12`, `EMA_26` | Moving averages and exponential moving averages for trend detection |
| **Momentum Indicators** | `RSI_14`, `MACD_line`, `MACD_signal`, `MACD_hist`, `Streak_length` | Relative Strength Index and MACD components for momentum analysis |
| **Volatility Indicators** | `Realized_vol_20d`, `Realized_vol_60d`, `Bollinger_bandwidth`, `Rolling_High_Low_20d` | Rolling volatility measures and price range indicators |
| **Volume Indicators** | `OBV`, `Volume_zscore_20d` | On-Balance Volume and volume anomaly detection |
| **Returns & Lags** | `Return_1d` to `Return_5d`, `Momentum_20d`, `Momentum_60d`, `Rolling_mean_return_20d`, `Rolling_var_return_20d` | Historical returns over various time horizons |
| **Price Data** | `Open`, `High`, `Low`, `Close`, `Volume` | Raw OHLCV data |
| **Fundamentals** | `Market_Cap` | Market capitalization derived from share price and shares outstanding |

**Total**: 32 features per time step for sequence models

---

### Target Variable (Labels)

A **3-class classification** target is generated for each stock-date observation:

| Label | Condition | Description |
|-------|-----------|-------------|
| `up` | `ForwardReturn ≥ RollingVol` | Price increases by at least one standard deviation over next 5 days |
| `down` | `ForwardReturn ≤ -RollingVol` | Price decreases by at least one standard deviation over next 5 days |
| `flat` | Otherwise | Price movement within ±1 standard deviation (low volatility period) |

**Labeling Parameters:**
- `horizon = 5`: Labels are based on 5-day forward returns
- `vol_window = 20`: Rolling volatility computed over 20 trading days
- `min_vol_threshold = 0.005`: Minimum volatility threshold to avoid division by zero

**Label Distribution** (Train Set):
- Up: ~35.5%
- Flat: ~37.4%
- Down: ~27.1%

---

### Data Sources

This project does **not rely on any pre-saved CSV files** (except for intermediate caching during execution).  
All data is fetched directly from publicly available online sources.

| Source | Description | Link |
|--------|-------------|------|
| **DataHub – S&P 500 Companies** | List of S&P 500 company ticker symbols | [https://datahub.io/core/s-and-p-500-companies/r/constituents.csv](https://datahub.io/core/s-and-p-500-companies/r/constituents.csv) |
| **Yahoo Finance API** (via [`yfinance`](https://pypi.org/project/yfinance/)) | Historical OHLCV data, market cap, and fundamental ratios | [https://finance.yahoo.com](https://finance.yahoo.com) |

Both sources are open and free to access for educational purposes.

---

### Data Workflow

1. **Data Collection**: Download S&P 500 ticker list from DataHub and retrieve 10 years of historical price data (2015-2024) from Yahoo Finance for all 503 stocks.

2. **Technical Feature Engineering**: Compute 32 technical indicators including trend (SMA, EMA), momentum (RSI, MACD), volatility (Bollinger Bands, realized volatility), volume (OBV), and return-based features.

3. **Fundamental Data Integration**: Merge market capitalization data from Yahoo Finance fundamental API (EPS, PE Ratio, and Dividend Yield were initially extracted but later excluded to avoid sparsity).

4. **Label Generation**: Create 3-class labels (`up`, `down`, `flat`) based on forward returns normalized by rolling volatility, with a 5-day prediction horizon.

5. **Dataset Construction**:
   - **GBDT Dataset**: Flatten features to create tabular data; split by date (before 2023-01-01 = train, after = test)
   - **LSTM/Transformer Dataset**: Create 60-day sliding windows with per-stock standardization; split by date before windowing to prevent leakage

6. **Model Training**: Train three supervised learning models:
   - **LSTM**: 2-layer LSTM with 128 hidden units, dropout regularization, early stopping (final val acc: ~60.4%)
   - **LightGBM**: Gradient boosted decision trees with 64 leaves, early stopping (final val acc: ~47%)
   - **Transformer**: Multi-head attention with 4 heads, 2 layers, positional encoding (final val acc: ~61%)

---

### Model Architectures

| Model | Architecture | Input Shape | Output | Performance (Val Acc) |
|-------|-------------|-------------|--------|---------------------|
| **LSTM** | 2-layer LSTM (128 hidden) + Dropout + FC layers | (batch, 60, 32) | 3 classes | ~60.4% |
| **LightGBM** | GBDT with 64 leaves, min_child_samples=5 | (batch, 1920) | 3 classes | ~47% |
| **Transformer** | 4-head attention, 2 encoder layers, model_dim=64 | (batch, 60, 32) | 3 classes | ~61% |

**Train/Test Split**: Temporal split at 2023-01-01 (946,121 train samples, 248,309 test samples for GBDT)

**Saved Models**:
- `lstm_model_long.pth`
- `lgbm_model.txt`
- `transformer_model_long.pth`

---

This supervised learning pipeline demonstrates end-to-end stock price movement prediction using technical analysis features and modern deep learning approaches.


## Part B — Unsupervised Learning

**File:** `PartB_Unsupervised_PRICE.ipynb`  
**Contents:** Data manipulation, feature engineering, clustering (K-Means and Agglomerative), PCA, and UMAP visualization of S&P 500 stocks.

---

### Dataset Features

The dataset used in this notebook is generated dynamically from online sources and includes the following financial and statistical features:

| Feature | Description |
|----------|-------------|
| `mean_return` | Average daily return of each stock over the selected period |
| `volatility` | Standard deviation of daily returns (measure of price risk) |
| `corr_market` | Correlation of each stock’s daily returns with the market average (S&P 500 mean return) |
| `pe` | Price-to-Earnings ratio (valuation metric) |
| `eps` | Earnings per share |
| `market_cap` | Market capitalization (total value of outstanding shares) |
| `dividend_yield` | Annual dividend as a percentage of stock price |

These features are calculated automatically within the notebook after downloading market and fundamental data.

---

### Data Sources

This project does **not rely on any pre-saved CSV files**.  
All data is fetched directly from publicly available online sources each time the notebook is executed.

| Source | Description | Link |
|---------|-------------|------|
| **Wikipedia – S&P 500 Companies** | Official list of companies and GICS Sector classifications | [https://en.wikipedia.org/wiki/List_of_S%26P_500_companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) |
| **Yahoo Finance API** (via [`yfinance`](https://pypi.org/project/yfinance/)) | Historical stock prices, financial ratios, and market data | [https://finance.yahoo.com](https://finance.yahoo.com) |

Both sources are open and free to access for educational purposes.

---

### Data Workflow

1. The notebook retrieves the official S&P 500 company list and sector classifications from **Wikipedia**.  
2. Historical price and financial data for each ticker are downloaded using the **Yahoo Finance API** (`yfinance` Python package).  
3. Financial features such as mean return, volatility, and correlation with the market are computed in-memory.  
4. The resulting dataset is standardized and used for **unsupervised learning** (K-Means and Agglomerative Clustering).  
5. Dimensionality reduction is applied using **PCA** and **UMAP** for visualization.

No local data storage or pre-existing CSV files are required.

---
