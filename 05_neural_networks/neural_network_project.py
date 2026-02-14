# ============================================================================
# PROJECT #5: NEURAL NETWORKS FOR TITANIC SURVIVAL PREDICTION
# ============================================================================
#
# Objective: Build a neural network using scikit-learn MLPClassifier
#            to predict Titanic passenger survival

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PROJECT #5: NEURAL NETWORKS - TITANIC SURVIVAL PREDICTION")
print("=" * 80)

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

print("\n✓ Data Loaded")
print(f"  Shape: {df.shape}")
print(f"  Survival rate: {df['survived'].mean()*100:.1f}%")

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

print("✓ Missing values handled")

# Feature engineering
df['is_alone'] = (df['sibsp'] + df['parch'] == 0).astype(int)
df['family_size'] = df['sibsp'] + df['parch'] + 1

print("✓ Features engineered: is_alone, family_size")

# Select and encode features
features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'is_alone', 'family_size']
X = df[features].copy()

# Encode categorical variables and handle NaN
X['sex'] = (X['sex'] == 'male').astype(int)
port_map = {'S': 0, 'C': 1, 'Q': 2}
# Map embarked and fill any NaN before converting to int
X['embarked'] = X['embarked'].map(port_map).fillna(0)

# Drop any rows with NaN values in X
X = X.dropna()

y = df.loc[X.index, 'survived'].values

print(f"✓ Features selected: {features}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Data split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")

# ============================================================================
# NORMALIZE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE NORMALIZATION (Critical for Neural Networks)")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features normalized")

# ============================================================================
# BUILD NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("BUILD NEURAL NETWORK ARCHITECTURE")
print("=" * 80)

print("""
Architecture:
  Input (7) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
""")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    verbose=0
)

print("✓ Model created")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING NEURAL NETWORK")
print("=" * 80)

print("Training...", end=" ", flush=True)
model.fit(X_train_scaled, y_train)
print("✓ Complete!")
print(f"  Iterations: {model.n_iter_}")
print(f"  Training loss: {model.loss_:.6f}")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

# Neural Network predictions
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

nn_accuracy = accuracy_score(y_test, y_pred)
nn_precision = precision_score(y_test, y_pred)
nn_recall = recall_score(y_test, y_pred)
nn_f1 = f1_score(y_test, y_pred)
nn_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Neural Network Results:")
print(f"  Accuracy:  {nn_accuracy:.4f}")
print(f"  Precision: {nn_precision:.4f}")
print(f"  Recall:    {nn_recall:.4f}")
print(f"  F1-Score:  {nn_f1:.4f}")
print(f"  ROC-AUC:   {nn_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# ============================================================================
# COMPARE WITH XGBOOST
# ============================================================================
print("\n" + "=" * 80)
print("NEURAL NETWORK vs XGBOOST")
print("=" * 80)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_pred_prob_xgb)

print("\n📊 MODEL COMPARISON:")
print("\n{:<20} {:<12} {:<12} {:<12} {:<12} {:<10}".format(
    'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'
))
print("-" * 78)
print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<10.4f}".format(
    'Neural Network', nn_accuracy, nn_precision, nn_recall, nn_f1, nn_auc
))
print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<10.4f}".format(
    'XGBoost', xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc
))
print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<10.4f}".format(
    'Difference (NN-XGB)', 
    nn_accuracy-xgb_accuracy, 
    nn_precision-xgb_precision, 
    nn_recall-xgb_recall, 
    nn_f1-xgb_f1, 
    nn_auc-xgb_auc
))

winner = "🏆 Neural Network" if nn_accuracy > xgb_accuracy else "🏆 XGBoost"
print(f"\n{winner}")

# ============================================================================
# PREDICTIONS ON NEW PASSENGERS
# ============================================================================
print("\n" + "=" * 80)
print("PREDICTIONS ON NEW PASSENGERS")
print("=" * 80)

example_passengers = pd.DataFrame({
    'pclass': [1, 2, 3],
    'sex': ['female', 'male', 'female'],
    'age': [25, 35, 20],
    'fare': [512.33, 26.0, 7.65],
    'embarked': ['S', 'S', 'S'],
    'is_alone': [1, 0, 1],
    'family_size': [1, 3, 1]
})

example_encoded = example_passengers.copy()
example_encoded['sex'] = (example_encoded['sex'] == 'male').astype(int)
example_encoded['embarked'] = example_encoded['embarked'].map(port_map)
example_scaled = scaler.transform(example_encoded)

nn_pred_proba = model.predict_proba(example_scaled)[:, 1]
xgb_pred_proba = xgb_model.predict_proba(example_scaled)[:, 1]

print("\n{:<12} {:<8} {:<8} {:<6} {:<17} {:<17}".format(
    'Passenger', 'Class', 'Sex', 'Age', 'NN Surv %', 'XGB Surv %'
))
print("-" * 70)

for i in range(len(example_passengers)):
    passenger = f'Passenger {i+1}'
    pclass = example_passengers.iloc[i]['pclass']
    sex = example_passengers.iloc[i]['sex']
    age = example_passengers.iloc[i]['age']
    nn_surv = nn_pred_proba[i] * 100
    xgb_surv = xgb_pred_proba[i] * 100
    
    print("{:<12} {:<8} {:<8} {:<6.0f} {:<17.2f} {:<17.2f}".format(
        passenger, pclass, sex, age, nn_surv, xgb_surv
    ))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ PROJECT #5 COMPLETE!")
print("=" * 80)

print(f"""
PORTFOLIO STATUS:
  ✓ Project #1: Titanic Classification (XGBoost) - 79.21%
  ✓ Project #2: House Prices Regression - R² = 0.8989
  ✓ Project #3: Customer Churn Classification - ROC-AUC = 92.5%
  ✓ Project #4: Stock Price Forecasting
  ✓ Project #5: Neural Networks - TODAY!

RESULTS:
  Neural Network Accuracy: {nn_accuracy:.4f}
  XGBoost Accuracy: {xgb_accuracy:.4f}
  Winner: {winner}
  
You now have production-ready data science skills! 🎉
""")
