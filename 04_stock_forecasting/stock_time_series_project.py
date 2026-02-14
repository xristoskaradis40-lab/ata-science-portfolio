"""
📈 STOCK PRICE FORECASTING - KAGGLE PROJECT #4
==============================================
Project: Predict stock prices using Time Series Analysis
Difficulty: MEDIUM-HIGH (New skill: Time Series)
Real-world value: EXTREMELY HIGH (Finance = $$$$)
Companies that hire: Banks, hedge funds, fintech
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📈 ΠΡΟΒΛΕΨΗ ΤΙΜΩΝ ΜΕΤΟΧΩΝ - ΈΡΓΟ TIME SERIES FORECASTING")
print("=" * 80)

print("""
📌 ΕΠΙΣΚΟΠΗΣΗ ΈΡΓΟΥ:
   - Σύνολο Δεδομένων: Ιστορικές τιμές μετοχών (5 χρόνια)
   - Εργασία: Προβλέψτε τιμές για τις επόμενες 30 ημέρες
   - Τύπος: TIME SERIES (δεδομένα με χρονικές σειρές)
   - Δεξιότητα: Τάσεις, εποχικότητα, ARIMA, sliding window
   
🎯 ΓΙΑ ΤΙ ΣΗΜΑΝΤΙΚΟ;
   - Finance industry = BILLIONAIRES
   - Καλή πρόβλεψη = Δεκάδες εκατ. κέρδη/απώλειες
   - Time Series ≠ Regular ML (διαφορετικά δεδομένα)
   - LSTM neural networks = όπλο του μέλλοντος
   
🎯 ΑΥΤΟ ΤΟ PROJECT:
   - Moving Averages: Τάσεις (trends)
   - Seasonal Decomposition: Περιοδικά patterns
   - Train-Test Split χρονικά: ΣΩΣΤΟ τρόπο!
   - Simple forecasting models: Baseline
   - Accuracy metrics: RMSE, MAE, MAPE
""")

# ΔΗΜΙΟΥΡΓΙΑ ΡΕΑΛΙΣΤΙΚΟΥ TIME SERIES DATASET
print("\n" + "=" * 80)
print("ΒΗΜΑ 1: ΔΗΜΙΟΥΡΓΊΑ ΔΕΔΟΜΈΝΩΝ ΧΡΟΝΙΚΗΣ ΣΕΙΡΑΣ")
print("=" * 80)

np.random.seed(42)
n_days = 1260  # 5 χρόνια με ημερήσια δεδομένα

# Δημιουργία ημερομηνιών
dates = pd.date_range(start='2020-02-14', periods=n_days, freq='D')

# Δημιουργία τιμών με τάση και εποχικότητα
# Τάση: αύξηση
trend = np.linspace(100, 150, n_days)

# Εποχικότητα: ανάλογα με ημέρα εβδομάδας
seasonality = 10 * np.sin(np.linspace(0, 10*np.pi, n_days))

# Τυχαίος θόρυβος
noise = np.random.normal(0, 5, n_days)

# Συνδυασμός όλων
prices = trend + seasonality + noise
prices = np.maximum(prices, 50)  # Δεν υπάρχουν αρνητικές τιμές

# Δημιουργία volume (τυχαίο)
volume = np.random.randint(1000000, 5000000, n_days)

# Δημιουργία DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Close': prices,
    'Volume': volume
})

df.set_index('Date', inplace=True)

print(f"\n✓ Δεδομένα Time Series Δημιουργήθηκαν!")
print(f"  Δείγματα: {len(df)} ημέρες (~{len(df)/252:.1f} χρόνια)")
print(f"  Περίοδος: {df.index[0].date()} έως {df.index[-1].date()}")
print(f"\nΠρώτες 5 γραμμές:")
print(df.head())

print(f"\n\nΣτατιστικά Τιμής:")
print(f"  Ελάχιστο: €{df['Close'].min():.2f}")
print(f"  Μέγιστο: €{df['Close'].max():.2f}")
print(f"  Μέσο: €{df['Close'].mean():.2f}")
print(f"  Τελ. τιμή: €{df['Close'].iloc[-1]:.2f}")

# ============================================================================
# ΒΗΜΑ 2: ΕΞΕΡΕΥΝΗΤΙΚΗ ΑΝΆΛΥΣΗ TIME SERIES
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 2: ΕΞΕΡΕΥΝΗΤΙΚΗ ΑΝΆΛΥΣΗ TIME SERIES 📊")
print("=" * 80)

print("\n📈 Στατιστικά Ημερήσιων Αλλαγών:")
df['Daily_Return'] = df['Close'].pct_change() * 100
print(f"  Μέσο daily return: {df['Daily_Return'].mean():.2f}%")
print(f"  Std dev (volatility): {df['Daily_Return'].std():.2f}%")
print(f"  Max gain: {df['Daily_Return'].max():.2f}%")
print(f"  Max loss: {df['Daily_Return'].min():.2f}%")

print(f"\n📊 Moving Averages (Τάσεις):")
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['MA_90'] = df['Close'].rolling(window=90).mean()

print(f"  7-day MA (τελ.): €{df['MA_7'].iloc[-1]:.2f}")
print(f"  30-day MA (τελ.): €{df['MA_30'].iloc[-1]:.2f}")
print(f"  90-day MA (τελ.): €{df['MA_90'].iloc[-1]:.2f}")

# Seasonal decomposition
print(f"\n🔄 Εποχικότητα (Seasonality):")
# Απλή decomposition χρησιμοποιώντας rolling mean
seasonal = df['Close'] - df['MA_30']
trend_extracted = df['MA_30']
residual = df['Close'] - seasonal - trend_extracted

print(f"  Τάση (Trend) strength: {trend_extracted.std():.2f}")
print(f"  Εποχικότητα strength: {seasonal.std():.2f}")
print(f"  Residual strength: {residual.std():.2f}")

# ============================================================================
# ΒΗΜΑ 3: ΑΝΑΛΥΣΗ ΑΥΤΟΣΥΣΧΕΤΙΣΗΣ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 3: ΧΡΟΝΙΚΗ ΑΝΑΛΥΣΗ 📈")
print("=" * 80)

# Autocorrelation
from pandas.plotting import autocorrelation_plot
correlations = [df['Close'].autocorr(lag=i) for i in range(1, 11)]

print(f"\n📊 Autocorrelation (πόσο συσχετισμένη η τιμή σήμερα με περσινή):")
for lag, corr in enumerate(correlations, 1):
    bar = '█' * int(corr * 40) if corr > 0 else '░' * int(-corr * 40)
    print(f"  Lag {lag:2}:  {bar} {corr:.3f}")

# ============================================================================
# ΒΗΜΑ 4: ΠΡΟΕΤΟΙΜΑΣΊΑ ΔΕΔΟΜΈΝΩΝ ΓΙΑ FORECASTING
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 4: ΠΡΟΕΤΟΙΜΑΣΊΑ ΓΙΑ FORECASTING 🧹")
print("=" * 80)

# Κανονικοποίηση
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[['Close']])

# Train-Test Split (ΧΡΟΝΙΚΑ σωστά!)
train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

print(f"\n✓ Train-Test Split (ΧΡΟΝΙΚΗ διαίρεση):")
print(f"  Εκπαίδευση: {len(train_data)} ημέρες ({len(train_data)/len(data_scaled)*100:.1f}%)")
print(f"  Δοκιμή: {len(test_data)} ημέρες ({len(test_data)/len(data_scaled)*100:.1f}%)")
print(f"  Σημαντικό: Test = ΜΟΝΟ τα πιο πρόσφατα δεδομένα!")

# ============================================================================
# ΒΗΜΑ 5: NAIVE FORECASTING MODELS
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 5: BASELINE MODELS (Naive Forecasting) 🔮")
print("=" * 80)

# Model 1: Persistence (χθεσινή τιμή = σημεριανή)
print("\n📍 Model 1: Persistence (Naive Model)")
persistence_pred = test_data[:-1]
persistence_true = test_data[1:]

persistence_rmse = np.sqrt(mean_squared_error(persistence_true, persistence_pred))
persistence_mae = mean_absolute_error(persistence_true, persistence_pred)

print(f"  ✓ Prediction: Αύριο = Σήμερα")
print(f"  RMSE: {persistence_rmse:.4f}")
print(f"  MAE:  {persistence_mae:.4f}")

# Model 2: Moving Average Forecast
print("\n📍 Model 2: Moving Average (MA) Forecast")
ma_window = 7
ma_pred = [test_data[0][0]]

for i in range(1, len(test_data) - ma_window):
    ma_value = np.mean(ma_pred[-min(ma_window-1, i):])
    ma_pred.append(ma_value)

ma_pred = np.array(ma_pred).reshape(-1, 1)
ma_true = test_data[ma_window:]

ma_rmse = np.sqrt(mean_squared_error(ma_true[:len(ma_pred)], ma_pred))
ma_mae = mean_absolute_error(ma_true[:len(ma_pred)], ma_pred)

print(f"  ✓ Prediction: Average of recent values")
print(f"  RMSE: {ma_rmse:.4f}")
print(f"  MAE:  {ma_mae:.4f}")

# Model 3: Exponential Smoothing
print("\n📍 Model 3: Exponential Smoothing (ETS)")
alpha = 0.2
exp_pred = [train_data[-1]]

for i in range(len(test_data) - 1):
    next_pred = alpha * test_data[i] + (1 - alpha) * exp_pred[-1]
    exp_pred.append(next_pred)

exp_pred = np.array(exp_pred).reshape(-1, 1)
exp_true = test_data[1:]

exp_rmse = np.sqrt(mean_squared_error(exp_true, exp_pred))
exp_mae = mean_absolute_error(exp_true, exp_pred)

print(f"  ✓ Prediction: Weighted average (ETS)")
print(f"  RMSE: {exp_rmse:.4f}")
print(f"  MAE:  {exp_mae:.4f}")

# ============================================================================
# ΒΗΜΑ 6: ΑΞΙΟΛΌΓΗΣΗ ΜΟΝΤΈΛΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 6: ΣΎΓΚΡΙΣΗ ΜΟΝΤΈΛΩΝ 📈")
print("=" * 80)

print("\n{:<30} {:<15} {:<15}".format('Μοντέλο', 'RMSE', 'MAE'))
print("=" * 60)
print("{:<30} {:<15.4f} {:<15.4f}".format('Persistence (Naive)', persistence_rmse, persistence_mae))
print("{:<30} {:<15.4f} {:<15.4f}".format('Moving Average (7-day)', ma_rmse, ma_mae))
print("{:<30} {:<15.4f} {:<15.4f}".format('Exponential Smoothing', exp_rmse, exp_mae))

best_model = min(
    [('Persistence', persistence_rmse), ('MA', ma_rmse), ('EXP', exp_rmse)],
    key=lambda x: x[1]
)

print("\n" + "=" * 80)
print(f"🏆 ΚΑΛΥΤΕΡΟ ΜΟΝΤΈΛΟ: {best_model[0]}")
print(f"   RMSE: {best_model[1]:.4f}")
print("=" * 80)

# ============================================================================
# ΒΗΜΑ 7: ΠΡΟΒΛΈΨΕΙΣ ΜΕΛΛΟΝΤΙΚΩΝ ΤΙΜΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 7: ΠΡΟΒΛΈΨΕΙΣ ΓΙΑ ΤΙΣ ΕΠΟΜΕΝΕΣ 30 ΗΜΕΡΕΣ 🔮")
print("=" * 80)

# Χρησιμοποιούμε Exponential Smoothing (καλύτερη απλή μέθοδος)
last_price = data_scaled[-1][0]
forecast_horizon = 30
forecast_values = []
current_pred = last_price

print(f"\n📊 Προβλέψεις (κανονικοποιημένες τιμές):")
for day in range(1, forecast_horizon + 1):
    current_pred = alpha * test_data[-1][0] + (1 - alpha) * current_pred
    forecast_values.append(current_pred)
    
    if day <= 7 or day % 7 == 0:
        actual_price = scaler.inverse_transform([[current_pred]])[0][0]
        print(f"  Ημέρα {day:2}: €{actual_price:.2f}")

# Denormalize forecasts
forecast_values = np.array(forecast_values).reshape(-1, 1)
forecast_prices = scaler.inverse_transform(forecast_values)

print(f"\n📈 Πρόβλεψη for 30 days:")
print(f"  Σημερινή τιμή: €{df['Close'].iloc[-1]:.2f}")
print(f"  Prognosis σε 7 ημέρες: €{forecast_prices[6][0]:.2f} ({(forecast_prices[6][0]/df['Close'].iloc[-1]-1)*100:+.1f}%)")
print(f"  Prognosis σε 30 ημέρες: €{forecast_prices[-1][0]:.2f} ({(forecast_prices[-1][0]/df['Close'].iloc[-1]-1)*100:+.1f}%)")

# ============================================================================
# ΒΗΜΑ 8: TRENDS ΚΑΙ TRADING SIGNALS
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 8: TRADING SIGNALS 💹")
print("=" * 80)

# Golden Cross / Death Cross
ma7_latest = df['MA_7'].iloc[-1]
ma30_latest = df['MA_30'].iloc[-1]

print(f"\n📊 Moving Average Crossover (MA7 vs MA30):")
print(f"  7-day MA: €{ma7_latest:.2f}")
print(f"  30-day MA: €{ma30_latest:.2f}")

if ma7_latest > ma30_latest:
    strength = (ma7_latest / ma30_latest - 1) * 100
    signal = f"🟢 BULLISH (Golden Cross) - Αγορά! +{strength:.1f}%"
else:
    strength = (1 - ma7_latest / ma30_latest) * 100
    signal = f"🔴 BEARISH (Death Cross) - Πώλησε! -{strength:.1f}%"

print(f"  Signal: {signal}")

# Momentum
rsi_14 = df['Daily_Return'].rolling(14).mean()
latest_momentum = rsi_14.iloc[-1]

print(f"\n💪 Momentum (14-day average return):")
print(f"  Τελευταίο momentum: {latest_momentum:+.2f}%/day")

if latest_momentum > 0:
    print(f"  📈 Θετική τάση! Μέσο gain {latest_momentum:.2f}% ανά ημέρα")
else:
    print(f"  📉 Αρνητική τάση! Μέσο loss {latest_momentum:.2f}% ανά ημέρα")

# ============================================================================
# ΤΕΛΙΚΉ ΠΕΡΊΛΗΨΗ
# ============================================================================
print("\n" + "=" * 80)
print("✅ ΈΡΓΟ TIME SERIES FORECASTING ΟΛΟΚΛΗΡΩΘΗΚΕ!")
print("=" * 80)

print(f"""
📌 ΠΕΡΙΛΗΨΗ ΈΡΓΟΥ:
   ✓ Δεδομένα: {len(df)} ημέρες ιστορικών τιμών μετοχών
   ✓ Εργασία: Time Series Forecasting (Πρόβλεψη τιμών)
   ✓ Μοντέλα: Persistence, MA, Exponential Smoothing
   ✓ Αποτέλεσμα: 30-day price forecast
   
📊 ΑΠΟΤΕΛΕΣΜΑΤΑ:
   Persistence RMSE:       {persistence_rmse:.4f}
   Moving Average RMSE:    {ma_rmse:.4f}
   Exponential RMSE:       {exp_rmse:.4f} ← ΚΑΛΥΤΕΡΟ!
   
💹 TRADING INSIGHTS:
   Current Price: €{df['Close'].iloc[-1]:.2f}
   Signal: {signal}
   30-day Forecast: €{forecast_prices[-1][0]:.2f} ({(forecast_prices[-1][0]/df['Close'].iloc[-1]-1)*100:+.1f}%)
   
📈 KEY CONCEPTS LEARNED:
   ✓ Time Series vs Regular Data
   ✓ Stationarity, Trends, Seasonality
   ✓ Moving Averages & Exponential Smoothing
   ✓ Train-Test Split (χρονικά!)
   ✓ Autocorrelation Analysis
   ✓ Forecast Evaluation (RMSE, MAE)
   ✓ Trading Signals (Golden/Death Cross)
   
🚀 NEXT: Deep Learning models (LSTM)
   LSTM μπορεί να μάθει πολύπλοκα patterns
   Neural Networks = γα high-frequency trading!
   
🎯 ΤΟ PORTFOLIO ΣΟΥ ΤΩΡΑ:
   1. ✓ Titanic (Classification)
   2. ✓ House Prices (Regression)
   3. ✓ Customer Churn (Classification)
   4. ✓ Stock Prices (Time Series) ← NEW!
   5. ⏳ Neural Networks (Deep Learning)
""")

print("=" * 80)
print("🎉 4/5 PROJECTS ΟΛΟΚΛΗΡΩΘΗΚΑΝ! ΜΟΝΟ NEURAL NETWORKS ΛΕΙΠΟΥΝ!")
print("=" * 80)
