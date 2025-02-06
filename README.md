
# 📈 S&P 500 Trading Strategy with Support Vector Classifier (SVC)

![Cumulative Returns Plot](https://via.placeholder.com/800x400.png?text=Cumulative+Returns+Plot)  
*Example strategy performance on test data (16% returns).*

A machine learning-driven trading strategy using **Support Vector Classifier (SVC)** to predict S&P 500 price movements. This project demonstrates end-to-end workflow from data preprocessing to strategy backtesting.

---

## 🚀 Key Features
- **Predictive Modeling**: Uses SVM to classify buy/sell signals based on OHLC data.
- **Feature Engineering**: `Open-Close` and `High-Low` price-derived features.
- **Backtesting**: Evaluates strategy performance with cumulative returns visualization.
- **Accuracy**: Achieves **54.32% test accuracy** in predicting market trends.

---

## 📋 Project Structure
```bash
├── data/                   # Contains SPY.csv (S&P 500 historical data)
├── S&P SVM.ipynb           # Main Jupyter notebook with code
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/svm-sp500-trading.git
   cd svm-sp500-trading
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Libraries used*: `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---

## 🧠 Workflow Overview

### 1. Data Preparation
- Load S&P 500 OHLC data from `SPY.csv`.
- Calculate features:  
  ```python
  df['Open-Close'] = df.Open - df.Close
  df['High-Low'] = df.High - df.Low
  ```

### 2. Target Variable
- Binary classification (`1` for price increase, `0` otherwise):  
  ```python
  y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
  ```

### 3. Model Training
- Split data (80% train, 20% test).
- Train SVC model:  
  ```python
  from sklearn.svm import SVC
  cls = SVC().fit(X_train, y_train)
  ```

### 4. Strategy Implementation
- Generate signals and calculate returns:  
  ```python
  df['Strategy_Returns'] = df.Returns * df.Predicted_Signal.shift(1)
  ```
- Visualize performance:  
  ![Matplotlib Plot](https://matplotlib.org/stable/_static/logo2_compressed.svg)

---

## 📊 Results
| Metric          | Value       |
|-----------------|-------------|
| Test Accuracy   | 54.32%      |
| Strategy Returns| ~16%        |

---

## 💡 Tweak the Code
1. **Experiment with features**:
   - Add technical indicators (RSI, MACD).
   ```python
   df['SMA_20'] = df['Close'].rolling(20).mean()
   ```

2. **Try different models**:
   - Random Forest, LSTM, or Gradient Boosting.

3. **Optimize parameters**:
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
   grid = GridSearchCV(SVC(), param_grid)
   ```

---

## 🙏 Acknowledgments
- Dataset: S&P 500 historical data (SPY)
- Inspired by QuantInsti's algorithmic trading course
``` 
🚀
