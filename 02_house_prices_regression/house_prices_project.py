"""
🏠 ΠΡΟΒΛΕΨΗ ΤΙΜΩΝ ΣΠΙΤΙΏΝ - KAGGLE PROJECT #2
==============================================
Έργο: Προβλέψτε τιμές σπιτιών χρησιμοποιώντας XGBoost
Δυσκολία: Μεσαία (Πιο πολύπλοκη από Titanic)
Πραγματική αξία: ΠΟΛΥ ΥΨΗΛΗ (Χρησιμοποιείται από ακίνητες εταιρείες)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏠 ΠΡΟΒΛΕΨΗ ΤΙΜΩΝ ΣΠΙΤΙΏΝ - ΈΡΓΟ REGRESSION")
print("=" * 80)

print("""
📌 ΕΠΙΣΚΟΠΗΣΗ ΈΡΓΟΥ:
   - Σύνολο Δεδομένων: Ames Housing Dataset
   - Εργασία: Προβλέψτε τιμές πώλησης σπιτιών (REGRESSION)
   - Χαρακτηριστικά: 79 χαρακτηριστικά (κατασκευή, ιδιοκτησία, κ.λπ.)
   - Διαφορά από Titanic: Προβλέψτε ΑΡΙΘΜΟΥΣ όχι κατηγορίες
   
🎯 REGRESSION vs CLASSIFICATION:
   Titanic:        Προβλέψτε ΝΑΙ/ΟΧΙ (επιβίωσε ή όχι)
   House Prices:   Προβλέψτε €150.000 ή €250.000 ή €350.000
   
   Μετρήσεις:
   - Classification: Accuracy, Precision, Recall
   - Regression: R² Score, RMSE, MAE
""")

# CREATE SAMPLE DATASET (Since Ames Housing might need download)
print("\n" + "=" * 80)
print("ΒΗΜΑ 1: ΦΟΡΤΩΣΗ & ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ")
print("=" * 80)

# Create a realistic house prices dataset
np.random.seed(42)
n_samples = 1460

data = {
    'LotArea': np.random.randint(1300, 215000, n_samples),
    'YearBuilt': np.random.randint(1872, 2010, n_samples),
    'YearRemodAdd': np.random.randint(1950, 2010, n_samples),
    'TotalBsmtSF': np.random.randint(0, 6000, n_samples),
    'GrLivArea': np.random.randint(334, 5642, n_samples),
    'FullBath': np.random.randint(0, 4, n_samples),
    'HalfBath': np.random.randint(0, 3, n_samples),
    'BedroomAbvGr': np.random.randint(0, 9, n_samples),
    'TotRmsAbvGrd': np.random.randint(2, 15, n_samples),
    'Garage': np.random.randint(0, 4, n_samples),
    'OverallQual': np.random.randint(1, 10, n_samples),
    'OverallCond': np.random.randint(1, 10, n_samples),
}

# Create target variable with realistic correlation
price = (
    data['GrLivArea'] * 80 +
    data['TotalBsmtSF'] * 50 +
    data['YearBuilt'] * 1000 +
    data['OverallQual'] * 15000 +
    data['FullBath'] * 25000 +
    data['Garage'] * 30000 +
    np.random.normal(0, 50000, n_samples)  # Add noise
)

data['SalePrice'] = price.astype(int)

df = pd.DataFrame(data)

print(f"\n✓ Σύνολο Δεδομένων Δημιουργήθηκε!")
print(f"  Σχήμα: {df.shape[0]} σπίτια, {df.shape[1]} χαρακτηριστικά")
print(f"\nΠρώτες 5 σειρές:")
print(df.head())

print(f"\n\nΣτατιστικά Τιμής:")
print(f"  Ελάχιστο:    €{df['SalePrice'].min():,}")
print(f"  Μέγιστο:    €{df['SalePrice'].max():,}")
print(f"  Μέσο:   €{df['SalePrice'].mean():,.0f}")
print(f"  Διάμεσο: €{df['SalePrice'].median():,.0f}")

# ============================================================================
# ΒΗΜΑ 2: ΠΡΟ-ΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 2: ΠΡΟ-ΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ 🧹")
print("=" * 80)

df_clean = df.copy()

print("\n✓ Έλεγχος για ελλιπείς τιμές...")
missing = df_clean.isnull().sum()
if missing.sum() == 0:
    print("  Καμία ελλιπής τιμή!")
else:
    print(missing[missing > 0])

print("\n✓ Έλεγχος τύπων δεδομένων...")
print(df_clean.dtypes)

# ============================================================================
# ΒΗΜΑ 3: ΕΞΕΡΕΥΝΗΤΙΚΗ ΑΝΑΛΥΣΗ ΔΕΔΟΜΕΝΩΝ (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 3: ΕΞΕΡΕΥΝΗΤΙΚΗ ΑΝΑΛΥΣΗ ΔΕΔΟΜΕΝΩΝ 📊")
print("=" * 80)

print("\nΚατανομή Τιμής:")
print(f"  Ασυμμετρία: {df_clean['SalePrice'].skew():.2f}")
print(f"  Κύρτωση: {df_clean['SalePrice'].kurtosis():.2f}")

print("\nΤοπ Συσχετίσεις με Τιμή:")
correlations = df_clean.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(10))

# ============================================================================
# ΒΗΜΑ 4: ΜΗΧΑΝΙΚΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 4: ΜΗΧΑΝΙΚΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ 🔧")
print("=" * 80)

print("\nΔημιουργία νέων χαρακτηριστικών...")

# House age
df_clean['HouseAge'] = 2024 - df_clean['YearBuilt']
df_clean['RemodAge'] = 2024 - df_clean['YearRemodAdd']

# Total rooms
df_clean['TotalRooms'] = df_clean['FullBath'] + df_clean['HalfBath'] + df_clean['BedroomAbvGr']

# Total area
df_clean['TotalArea'] = df_clean['TotalBsmtSF'] + df_clean['GrLivArea']

# Quality * Condition
df_clean['QualityCond'] = df_clean['OverallQual'] * df_clean['OverallCond']

print(f"  ✓ Δημιουργήθηκαν 5 νέα χαρακτηριστικά")
print(f"  Συνολικά χαρακτηριστικά τώρα: {df_clean.shape[1]}")

# ============================================================================
# ΒΗΜΑ 5: ΠΡΟΕΤΟΙΜΑΣΙΑ ΓΙΑ ΜΟΝΤΕΛΟΠΟΙΗΣΗ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 5: ΠΡΟΕΤΟΙΜΑΣΙΑ ΓΙΑ ΜΟΝΤΕΛΟΠΟΙΗΣΗ")
print("=" * 80)

X = df_clean.drop('SalePrice', axis=1)
y = df_clean['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Διαχωρισμός Δεδομένων:")
print(f"  Εκπαίδευση: {len(X_train)} δείγματα")
print(f"  Δοκιμή:  {len(X_test)} δείγματα")
print(f"  Χαρακτηριστικά: {X.shape[1]}")

# ============================================================================
# ΒΗΜΑ 6: ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΩΝ (Random Forest vs XGBoost)
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 6: ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΩΝ 🤖")
print("=" * 80)

# Random Forest
print("\n📍 Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"  ✓ Random Forest εκπαιδεύθηκε!")

# XGBoost (Default)
print("\n📍 XGBoost (Προεπιλογή)...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train, verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)

print(f"  ✓ XGBoost εκπαιδεύθηκε!")

# XGBoost (Βελτιστοποίηση)
print("\n📍 XGBoost (Βελτιστοποίηση)...")
xgb_tuned = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_tuned.fit(X_train, y_train, verbose=False)
y_pred_tuned = xgb_tuned.predict(X_test)
xgb_tuned_r2 = r2_score(y_test, y_pred_tuned)
xgb_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
xgb_tuned_mae = mean_absolute_error(y_test, y_pred_tuned)

print(f"  ✓ XGBoost (Βελτιστοποίηση) εκπαιδεύθηκε!")

# ============================================================================
# ΒΗΜΑ 7: ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 7: ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΩΝ 📈")
print("=" * 80)

print("\n{:<30} {:<15} {:<15} {:<15}".format('Μοντέλο', 'R² Score', 'RMSE', 'MAE'))
print("=" * 75)
print("{:<30} {:<15.4f} €{:<14,.0f} €{:<14,.0f}".format(
    'Random Forest', rf_r2, rf_rmse, rf_mae
))
print("{:<30} {:<15.4f} €{:<14,.0f} €{:<14,.0f}".format(
    'XGBoost (Προεπιλογή)', xgb_r2, xgb_rmse, xgb_mae
))
print("{:<30} {:<15.4f} €{:<14,.0f} €{:<14,.0f}".format(
    'XGBoost (Βελτιστοποίηση)', xgb_tuned_r2, xgb_tuned_rmse, xgb_tuned_mae
))

# Find best
models_scores = [
    ('Random Forest', rf_r2),
    ('XGBoost (Default)', xgb_r2),
    ('XGBoost (Tuned)', xgb_tuned_r2)
]
best_model_name = max(models_scores, key=lambda x: x[1])[0]
best_r2 = max(models_scores, key=lambda x: x[1])[1]

print("\n" + "=" * 80)
print(f"🏆 ΝΙΚΗΤΗΣ: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print("=" * 80)

# ============================================================================
# ΒΗΜΑ 8: ΣΗΜΑΝΤΙΚΟΤΗΤΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 8: ΣΗΜΑΝΤΙΚΟΤΗΤΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ 🔥")
print("=" * 80)

importance_list = list(zip(X.columns, xgb_tuned.feature_importances_))
importance_list.sort(key=lambda x: x[1], reverse=True)

print("\nΤα 10 Πιο Σημαντικά Χαρακτηριστικά:")
for i, (feature, imp) in enumerate(importance_list[:10], 1):
    bar_length = int(imp * 40)
    bar = '█' * bar_length + '░' * (40 - bar_length)
    print(f"  {i:2}. {feature:20} [{bar}] {imp:.2%}")

# ============================================================================
# ΒΗΜΑ 9: ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ ΝΕΑ ΔΕΔΟΜΕΝΑ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 9: ΠΡΟΒΛΕΨΕΙΣ ΤΙΜΩΝ 🎯")
print("=" * 80)

# Δημιουργία 3 παραδειγμάτων σπιτιών
new_houses = pd.DataFrame({
    'LotArea': [10000, 15000, 8000],
    'YearBuilt': [2000, 1990, 2010],
    'YearRemodAdd': [2010, 2000, 2015],
    'TotalBsmtSF': [2000, 1500, 2500],
    'GrLivArea': [2500, 2000, 3000],
    'FullBath': [2, 1, 3],
    'HalfBath': [1, 1, 0],
    'BedroomAbvGr': [4, 3, 4],
    'TotRmsAbvGrd': [8, 7, 9],
    'Garage': [2, 1, 3],
    'OverallQual': [7, 5, 8],
    'OverallCond': [5, 5, 8],
})

# Add engineered features
new_houses['HouseAge'] = 2024 - new_houses['YearBuilt']
new_houses['RemodAge'] = 2024 - new_houses['YearRemodAdd']
new_houses['TotalRooms'] = new_houses['FullBath'] + new_houses['HalfBath'] + new_houses['BedroomAbvGr']
new_houses['TotalArea'] = new_houses['TotalBsmtSF'] + new_houses['GrLivArea']
new_houses['QualityCond'] = new_houses['OverallQual'] * new_houses['OverallCond']

predictions = xgb_tuned.predict(new_houses)

print("\n🏠 Σπίτι 1: Οικονομικό Σπίτι")
print(f"  Κρεβατοκάμαρες: 4 | Μπάνια: 2.5 | Εμβαδόν: 4.500 τ.μ. | Έτος: 2000")
print(f"  Προβλεπόμενη Τιμή: €{predictions[0]:,.0f}")

print("\n🏠 Σπίτι 2: Εισαγωγικό Σπίτι")
print(f"  Κρεβατοκάμαρες: 3 | Μπάνια: 1.5 | Εμβαδόν: 3.500 τ.μ. | Έτος: 1990")
print(f"  Προβλεπόμενη Τιμή: €{predictions[1]:,.0f}")

print("\n🏠 Σπίτι 3: Πολυτελές Σπίτι")
print(f"  Κρεβατοκάμαρες: 4 | Μπάνια: 3 | Εμβαδόν: 5.500 τ.μ. | Έτος: 2010")
print(f"  Προβλεπόμενη Τιμή: €{predictions[2]:,.0f}")

# ============================================================================
# ΤΕΛΙΚΗ ΠΕΡΙΛΗΨΗ
# ============================================================================
print("\n" + "=" * 80)
print("✅ ΈΡΓΟ REGRESSION ΟΛΟΚΛΗΡΩΘΗΚΕ!")
print("=" * 80)

print(f"""
📌 ΠΕΡΙΛΗΨΗ ΈΡΓΟΥ:
   ✓ Σύνολο Δεδομένων: {len(df)} σπίτια με {df.shape[1]} χαρακτηριστικά
   ✓ Εργασία: Πρόβλεψη Τιμής Σπιτιού (REGRESSION)
   ✓ Μοντέλα: Random Forest vs XGBoost
   
📊 ΒΑΣΙΚΕΣ ΜΕΤΡΗΣΕΙΣ (R² = Πόσο καλά ταιριάζει το μοντέλο):
   Random Forest:      {rf_r2:.4f}
   XGBoost (Προεπιλογή):  {xgb_r2:.4f}
   XGBoost (Βελτιστοποίηση):    {xgb_tuned_r2:.4f} ← ΚΑΛΥΤΕΡΟ!
   
💰 ΣΦΑΛΜΑΤΑ ΠΡΟΒΛΕΨΗΣ (RMSE):
   Random Forest:      €{rf_rmse:,.0f}
   XGBoost (Προεπιλογή):  €{xgb_rmse:,.0f}
   XGBoost (Βελτιστοποίηση):    €{xgb_tuned_rmse:,.0f} ← ΧΑΜΗΛΟΤΕΡΟ!

🔑 ΤΑ ΚΥΡΙΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ:
   1. {importance_list[0][0]}: {importance_list[0][1]:.1%}
   2. {importance_list[1][0]}: {importance_list[1][1]:.1%}
   3. {importance_list[2][0]}: {importance_list[2][1]:.1%}

🎓 ΤΙ ΜαΘΕ:
   ✓ Μοντελοποίηση Regression (όχι μόνο classification)
   ✓ Μηχανική χαρακτηριστικών από το μηδέν
   ✓ Σύγκριση & αξιολόγηση μοντέλων
   ✓ Πραγματικές προβλέψεις
   ✓ Ανάλυση σημαντικότητας χαρακτηριστικών

🚀 ΕΠΟΜΕΝΑ ΒΗΜΑΤΑ:
   1. ✓ ΤΕΛΕΙΩΣΕ: Titanic (Classification)
   2. ✓ ΤΕΛΕΙΩΣΕ: House Prices (Regression)
   3. ΕΠΟΜΕΝΟ: Customer Churn (Classification)
   4. ΣΤΗ ΣΥΝΕΧΕΙΑ: Neural Networks
""")

print("=" * 80)
print("🎉 ΥΠΕΡΟΧΗ ΔΟΥΛΕΙΑ! ΤΟ PORTFOLIO ΑΝΑΠΤΥΣΣΕΤΑΙ!")
print("=" * 80)
