---Part A Supervised Learning----

1. explore.ipynb: Download data, preprocess data, training models, finetune. All data can be accessed, downloaded, and preprocessed by the method in this notebook. The api access is free.
2. model_test.ipynb: Build evaluation data, evaluation pipeline, investment simulation, result analysis, graph generations.


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
