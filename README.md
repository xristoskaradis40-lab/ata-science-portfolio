# 📊 Data Science Portfolio

A comprehensive collection of 5 production-ready machine learning projects demonstrating expertise in classification, regression, time series forecasting, and deep learning.

**Author:** Christos  
**Date:** February 2026  
**Skill Level:** Advanced

---

## 🎯 Portfolio Overview

This portfolio showcases end-to-end machine learning projects with real-world applications, from data preprocessing to model evaluation and business insights.

| Project | Type | Dataset | Best Model | Performance |
|---------|------|---------|-----------|-------------|
| 1. Titanic | Classification | 889 passengers | XGBoost | **79.21% Accuracy** |
| 2. House Prices | Regression | 1,460 houses | XGBoost (Tuned) | **R² = 0.8989** |
| 3. Customer Churn | Classification | 7,043 customers | Random Forest | **ROC-AUC = 92.5%** |
| 4. Stock Forecasting | Time Series | 1,260 days | Exponential Smoothing | **RMSE = 0.1060** |
| 5. Neural Networks | Deep Learning | 891 passengers | MLPClassifier | **Accuracy = 78.14%** |

---

## 📁 Project Structure

```
portfolio/
├── 01_titanic_classification/      # Binary classification
├── 02_house_prices_regression/     # Continuous value prediction
├── 03_customer_churn/              # Business classification + ROI
├── 04_stock_forecasting/           # Time series forecasting
├── 05_neural_networks/             # Deep learning approach
├── requirements.txt                # All dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/data-science-portfolio.git
cd portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Any Project

```bash
# Project 1: Titanic Classification
python 01_titanic_classification/titanic_project.py

# Project 2: House Prices Regression
python 02_house_prices_regression/house_prices_project.py

# Project 3: Customer Churn Classification
python 03_customer_churn/customer_churn_project.py

# Project 4: Stock Forecasting
python 04_stock_forecasting/stock_time_series_project.py

# Project 5: Neural Networks
python 05_neural_networks/neural_network_project.py
```

---

## 📚 Project Details

### Project #1: Titanic Survival Classification

**Context:** Predict passenger survival on the Titanic using demographic data.

**Key Achievements:**
- Data: 889 passengers, 38.4% survival rate (baseline)
- Features: Passenger class, sex, age, fare, port, family composition
- Best Model: XGBoost
- Accuracy: **79.21%**
- Key Insight: Female passengers had 55% survival vs males 19%

**Technical Skills:**
✓ Data cleaning & exploration  
✓ Feature engineering (is_alone, family_size)  
✓ Model comparison (DecisionTree, RandomForest, XGBoost)  
✓ Confusion matrix & precision/recall analysis

---

### Project #2: House Prices Regression

**Context:** Predict sale prices for residential properties using structural features.

**Key Achievements:**
- Data: 1,460 houses, 18 features
- Best Model: XGBoost (Tuned)
- R² Score: **0.8989**
- RMSE: €53,948
- Key Insight: Total area is strongest predictor (54.47% importance)

**Technical Skills:**
✓ Regression modeling  
✓ Feature engineering (HouseAge, RemodAge, TotalArea)  
✓ Hyperparameter tuning (GridSearchCV)  
✓ Model comparison & RMSE optimization

---

### Project #3: Customer Churn Classification

**Context:** Identify at-risk customers and calculate business ROI from retention efforts.

**Key Achievements:**
- Data: 7,043 customers, 49% churn rate
- Best Model: Random Forest
- ROC-AUC: **92.5%**
- Business Impact: €107,840 potential ROI
- Key Insight: Contract type is 71.5% important for churn prediction

**Technical Skills:**
✓ Business metrics calculation  
✓ ROI analysis  
✓ Stratified train-test split  
✓ Class imbalance handling

---

### Project #4: Stock Price Forecasting

**Context:** Forecast future stock prices using time series data (5 years).

**Key Achievements:**
- Data: 1,260 days of trading data
- Models: Persistence, Moving Average, Exponential Smoothing
- Best RMSE: **0.1060**
- Trading Signal: Bullish (Golden Cross pattern)
- Key Insight: Strong autocorrelation (0.90+) enables accurate forecasting

**Technical Skills:**
✓ Time series decomposition  
✓ Autocorrelation analysis  
✓ Multiple forecasting approaches  
✓ Trading signal generation

---

### Project #5: Neural Networks

**Context:** Deep learning approach to Titanic survival prediction.

**Key Achievements:**
- Architecture: 3 hidden layers (128 → 64 → 32 neurons)
- Activation: ReLU (hidden), Sigmoid (output)
- Accuracy: **78.14%**
- ROC-AUC: **84.30%**
- vs XGBoost: Comparable performance, different decision boundaries

**Technical Skills:**
✓ Neural network architecture design  
✓ Feature normalization (StandardScaler)  
✓ Early stopping & regularization  
✓ Deep learning frameworks (scikit-learn MLPClassifier)

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.14.3 |
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Neural Networks** | scikit-learn MLPClassifier |
| **Environment** | Virtual Environment (.venv) |

---

## 📊 Key Metrics Across Portfolio

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Regression Metrics:**
- R² Score, RMSE, MAE

**Time Series Metrics:**
- RMSE, Autocorrelation, Trend Analysis

---

## 💡 Key Learnings

1. **Feature Engineering** - Domain knowledge drives model performance
2. **Model Selection** - XGBoost often outperforms simpler models on tabular data
3. **Business Metrics** - Always connect predictions to business impact (ROI)
4. **Time Series** - Temporal patterns require special handling (no shuffling)
5. **Deep Learning** - Neural networks excel with complex features but need more data

---

## 🎯 Next Steps / Improvements

- [ ] Convert models to REST API (Flask/FastAPI)
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add cross-validation
- [ ] Ensemble methods combining all models
- [ ] Interactive dashboards (Streamlit)
- [ ] Real-time prediction pipeline

---

## 📝 Requirements

See `requirements.txt` for complete list. Main packages:

```
pandas>=3.0.0
numpy>=2.4.2
scikit-learn>=1.8.0
xgboost>=latest
matplotlib>=3.10.8
seaborn>=0.13.2
```

---

## 👤 About

This portfolio demonstrates:
- ✅ End-to-end ML pipeline expertise
- ✅ Multiple ML paradigms (Classification, Regression, TS, DL)
- ✅ Business acumen (ROI, risk analysis)
- ✅ Code quality & documentation
- ✅ Production-ready implementations

**Target Roles:** Data Scientist, ML Engineer, Analytics Engineer

---

## 📞 Contact

For questions or collaboration opportunities, feel free to reach out!

---

## 📄 License

This project is open source and available for educational purposes.

---

**Last Updated:** February 14, 2026  
**Status:** ✅ All 5 projects complete and tested
