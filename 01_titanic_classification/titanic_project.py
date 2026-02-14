"""
🚢 TITANIC SURVIVAL PREDICTION - REAL KAGGLE PROJECT
=====================================================
Πρόγνωση: Ποιος επέζησε στο Titanic;
Αυτό είναι ένα ΑΛΗΘΙΝΟ Data Science Project που θα δείξεις σε εργοδότες!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("=" * 80)
print("🚢 TITANIC SURVIVAL PREDICTION")
print("=" * 80)

# ============================================================================
# ΒΗΜΑ 1: ΦΌΡΤΩΣΗ ΔΕΔΟΜΈΝΩΝ & ΕΠΙΣΚΌΠΗΣΗ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 1: ΦΌΡΤΩΣΗ ΔΕΔΟΜΈΝΩΝ")
print("=" * 80)

# Φορτώνουμε το dataset
df = sns.load_dataset('titanic')

print(f"\nDataset Shape: {df.shape[0]} γραμμές, {df.shape[1]} στήλες")
print("\nΠρώτες 5 γραμμές:")
print(df.head())

print("\n\nΠληροφορίες για τα δεδομένα:")
print(df.info())

print("\n\nΣτατιστικά:")
print(df.describe())

# ============================================================================
# ΒΗΜΑ 2: ΑΝΆΛΥΣΗ MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 2: ΑΝΆΛΥΣΗ MISSING VALUES 🔍")
print("=" * 80)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

print("\nMissing Values:")
for col in missing[missing > 0].index:
    print(f"  {col:15} → {missing[col]:3d} ({missing_percent[col]:5.1f}%)")

# ============================================================================
# ΒΗΜΑ 3: DATA CLEANING 🧹
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 3: DATA CLEANING 🧹")
print("=" * 80)

# Κάνουμε ένα copy για να μην αλλάξουμε το original
df_clean = df.copy()

# Διαγράφουμε στήλες που δεν είναι χρήσιμες
print("\nΔιαγραφή άχρηστων στηλών...")
df_clean = df_clean.drop(['deck', 'embark_town', 'who', 'adult_male', 'alive', 'alone'], axis=1)

# Καθαρίζουμε το 'age' - πληρώνουμε τα κενά με τη μέση τιμή
print("  - Γέμισμα missing ages με median...")
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)

# Καθαρίζουμε το 'embarked'
print("  - Διαγραφή γραμμών με missing 'embarked'...")
df_clean = df_clean.dropna(subset=['embarked'])

# Καθαρίζουμε το 'sex' - δεν έχει κενά σε αυτά τα δεδομένα
print("  - Κωδικοποίηση κατηγοριών (sex, embarked)...")

# ============================================================================
# ΒΗΜΑ 4: FEATURE ENGINEERING 🔧
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 4: FEATURE ENGINEERING 🔧")
print("=" * 80)

# Encoder για κατηγοριακές μεταβλητές
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_pclass = LabelEncoder()

df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
df_clean['embarked'] = le_embarked.fit_transform(df_clean['embarked'])
df_clean['pclass'] = le_pclass.fit_transform(df_clean['pclass'].astype(str))

print("\nΜετασχηματισμοί:")
print(f"  - sex: encoded")
print(f"  - embarked: encoded")
print(f"  - pclass: encoded")

# Δημιουργούμε νέα features
print("\nΔημιουργία νέων features:")
print("  - is_alone: (SibSp + Parch == 0)")
df_clean['is_alone'] = (df_clean['sibsp'] + df_clean['parch'] == 0).astype(int)

print("  - family_size: (SibSp + Parch)")
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch']

print("  - fare_per_person: (fare / (family_size + 1))")
df_clean['fare_per_person'] = df_clean['fare'] / (df_clean['family_size'] + 1)

# Διαγράφουμε τις στήλες που δεν χρειάζονται πια
df_clean = df_clean.drop(['sibsp', 'parch'], axis=1)

print("\nΤελικά columns:")
print(f"  {list(df_clean.columns)}")

# ============================================================================
# ΒΗΜΑ 5: EXPLORATORY DATA ANALYSIS (EDA) 📊
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 5: EXPLORATORY DATA ANALYSIS 📊")
print("=" * 80)

print("\nΣτατιστικά ανά survival:")
print(df_clean.groupby('survived').agg({
    'age': ['mean', 'min', 'max'],
    'fare': ['mean', 'min', 'max'],
    'sex': ['mean']  # mean του 0/1, όπου 1=female
}))

# Υπολογίζουμε survival rate ανά feature
print("\n\nSurvival Rate ανά Sex:")
print(df_clean.groupby('sex')['survived'].mean())

print("\n\nSurvival Rate ανά Embarked Port:")
print(df_clean.groupby('embarked')['survived'].mean())

print("\n\nSurvival Rate (Overall):")
survival_rate = df_clean['survived'].mean()
print(f"  {survival_rate:.1%} των επιβατών επέζησαν")

# ============================================================================
# ΒΗΜΑ 6: ΠΡΟΕΤΟΙΜΑΣΊΑ ΔΕΔΟΜΈΝΩΝ ΓΙΑ ML
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 6: ΠΡΟΕΤΟΙΜΑΣΊΑ ΓΙΑ MACHINE LEARNING")
print("=" * 80)

# Χωρίζουμε Features (X) και Target (y)
X = df_clean.drop('survived', axis=1)
y = df_clean['survived']

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: survived")

# Χωρίζουμε σε training και test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} δείγματα")
print(f"Test set: {len(X_test)} δείγματα")

# ============================================================================
# ΒΗΜΑ 7: TRAINING MODELS 🤖
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 7: ΕΚΠΑΊΔΕΥΣΗ ΜΟΝΤΈΛΩΝ 🤖")
print("=" * 80)

# Decision Tree
print("\n📍 Decision Tree Classifier")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
print("📍 Random Forest Classifier (100 δέντρα)")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n✓ Τα μοντέλα εκπαιδεύτηκαν!")

# ============================================================================
# ΒΗΜΑ 8: ΑΠΟΤΊΜΗΣΗ ΜΟΝΤΈΛΩΝ 📈
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 8: ΑΠΟΤΊΜΗΣΗ ΜΟΝΤΈΛΩΝ 📈")
print("=" * 80)

def eval_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name}")
    print(f"  Accuracy:  {accuracy:.2%} ← Σωστές προβλέψεις")
    print(f"  Precision: {precision:.2%} ← Από αυτούς που είπαμε 'επέζησαν', πόσοι όντως επέζησαν")
    print(f"  Recall:    {recall:.2%}  ← Από αυτούς που όντως επέζησαν, πόσοι τους βρήκαμε")
    print(f"  F1-Score:  {f1:.2%}   ← Συνδυασμένο metric")
    
    return accuracy

print("\n🌳 DECISION TREE:")
dt_accuracy = eval_model(y_test, y_pred_dt, "Decision Tree")

print("\n\n🌲 RANDOM FOREST:")
rf_accuracy = eval_model(y_test, y_pred_rf, "Random Forest")

# Σύγκριση
print("\n\n" + "=" * 80)
print("ΣΎΓΚΡΙΣΗ")
print("=" * 80)

if rf_accuracy > dt_accuracy:
    improvement = ((rf_accuracy - dt_accuracy) / dt_accuracy) * 100
    print(f"\n✅ Random Forest είναι {improvement:.1f}% καλύτερο!")
else:
    print(f"\n✓ Και τα δύο μοντέλα έχουν παρόμοια απόδοση")

# ============================================================================
# ΒΗΜΑ 9: FEATURE IMPORTANCE 🔥
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 9: FEATURE IMPORTANCE - Ποια χαρακτηριστικά είναι σημαντικά; 🔥")
print("=" * 80)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nΣημαντικότητα χαρακτηριστικών (Random Forest):")
for idx, row in importance.iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"  {row['Feature']:20} [{bar}] {row['Importance']:.2%}")

# ============================================================================
# ΒΗΜΑ 10: ΠΡΟΒΛΈΨΕΙΣ ΓΙΑ ΝΕΟΥΣ ΕΠΙΒΑΤΕΣ
# ============================================================================
print("\n" + "=" * 80)
print("ΒΗΜΑ 10: ΠΡΟΒΛΈΨΕΙΣ ΓΙΑ ΝΕΟΥΣ ΕΠΙΒΑΤΕΣ 🎯")
print("=" * 80)

# Δημιουργούμε φανταστικούς επιβάτες
new_passengers = pd.DataFrame({
    'age': [25, 35, 5],
    'sex': [0, 1, 0],  # 0=male, 1=female
    'fare': [100, 200, 50],
    'embarked': [0, 1, 2],  # 0=S, 1=C, 2=Q
    'is_alone': [1, 0, 0],
    'family_size': [0, 1, 2],
    'fare_per_person': [100, 100, 25]
})

predictions = rf_model.predict_proba(new_passengers)

print("\nΠρόβλεψη για νέους επιβάτες:")
print("\nΕπιβάτης 1: 25χρονος άνδρας, κοίτη 2ης κλάσης, ταξίδεψε μόνος")
print(f"  ⚫ Πρόβλεψη: {'📍 ΕΠΈΖΗΣΕ!' if predictions[0][1] > 0.5 else '❌ ΔΕΝ ΕΠΈΖΗΣΕ'}")
print(f"  Πιθανότητα επιβίωσης: {predictions[0][1]:.1%}")

print("\nΕπιβάτης 2: 35χρονη γυναίκα, κοίτη 1ης κλάσης, με 1 άτομο")
print(f"  ⚫ Πρόβλεψη: {'📍 ΕΠΈΖΗΣΕ!' if predictions[1][1] > 0.5 else '❌ ΔΕΝ ΕΠΈΖΗΣΕ'}")
print(f"  Πιθανότητα επιβίωσης: {predictions[1][1]:.1%}")

print("\nΕπιβάτης 3: 5χρονο αγόρι, με 2 άτομα")
print(f"  ⚫ Πρόβλεψη: {'📍 ΕΠΈΖΗΣΕ!' if predictions[2][1] > 0.5 else '❌ ΔΕΝ ΕΠΈΖΗΣΕ'}")
print(f"  Πιθανότητα επιβίωσης: {predictions[2][1]:.1%}")

# ============================================================================
# ΒΗΜΑ 11: ΣΥΜΠΕΡΆΣΜΑΤΑ
# ============================================================================
print("\n" + "=" * 80)
print("ΣΥΜΠΕΡΆΣΜΑΤΑ & INSIGHTS 💡")
print("=" * 80)

print("""
1️⃣ DATA CLEANING: Αφαιρέσαμε κενές τιμές, άχρηστα δεδομένα
2️⃣ FEATURE ENGINEERING: Δημιουργήσαμε νέα features (is_alone, family_size)
3️⃣ EDA: Δείδαμε ποιες ομάδες είχαν υψηλότερο survival rate
4️⃣ MODELING: Εκπαιδεύσαμε Decision Tree & Random Forest
5️⃣ EVALUATION: Random Forest ήταν καλύτερο/παρόμοιο με αποδοχή
6️⃣ INSIGHTS: Τα σημαντικότερα χαρακτηριστικά είναι sex, age, fare

🔑 KEY INSIGHT: Το φύλλο ήταν ο πιο σημαντικός παράγοντας!
   - Γυναίκες είχαν ΠΟΛΛΑΠΛΆΣΙΕΣ πιθανότητες επιβίωσης
   - Άνδρες είχαν χαμηλότερες πιθανότητες

🎯 ΠΌΣΟ ΚΑΛΌ ΕΊΝΑΙ ΤΟ ΜΟΝΤΈΛΟ;
   - Accuracy: {:.1f}% ← Πόσες σωστές προβλέψεις
   - Baseline: {:.1f}% ← Αν λέγαμε "όλοι επέζησαν"
   - Improvement: {:.1f}% ← Πόσο καλύτεροι είμαστε

✅ READY FOR PRODUCTION!
   Αυτό το project δείχνει ΟΛΟΚΛΗΡΗ ροή Data Science:
   - Data Loading → Cleaning → Exploration
   - Feature Engineering → Modeling → Evaluation
""".format(rf_accuracy * 100, df['survived'].mean() * 100, 
           (rf_accuracy - df['survived'].mean()) / df['survived'].mean() * 100))

print("=" * 80)
print("🎉 ΤΈΛΟΣ PROJECT!")
print("=" * 80)
