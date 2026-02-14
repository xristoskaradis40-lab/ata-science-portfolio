# Project #5: Neural Networks (Deep Learning)

## 📌 Overview

Apply deep learning neural networks to Titanic survival prediction for comparison with traditional ML.

**Dataset:** Titanic (same as Project #1)  
**Samples:** 889 passengers  
**Target:** Survived (1) or Did not survive (0)  
**Architecture:** 3 hidden layers (128→64→32 neurons)

---

## 🎯 Problem Statement

While XGBoost achieved 79.21% accuracy (Project #1), neural networks offer:
- More complex feature interactions
- Better scalability to large datasets
- Foundation for advanced deep learning
- Different decision boundaries/patterns

**Business Question:** Can neural networks find patterns that tree-based models miss?

---

## 📊 Key Results

| Metric | Neural Network | XGBoost | Difference |
|--------|---|---|---|
| **Accuracy** | 78.14% | 80.93% | -2.79% |
| **Precision** | 73.81% | 78.75% | -4.94% |
| **Recall** | 71.26% | 72.41% | -1.15% |
| **F1-Score** | 72.51% | 75.45% | -2.93% |
| **ROC-AUC** | 84.30% | 84.83% | -0.53% |

### Winner: XGBoost (slight edge on this dataset)

```
Both models perform similarly on Titanic dataset.
XGBoost slightly better for tabular data with < 1M samples.
Neural Networks advantage emerges with larger datasets/complex patterns.
```

---

## 🧠 Neural Network Architecture

### Structure
```
Input Layer (7 features)
    ↓
Dense Layer 1: 128 neurons, ReLU activation
    ↓
Dense Layer 2: 64 neurons, ReLU activation
    ↓
Dense Layer 3: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron, Sigmoid activation
```

### Why This Design?

**Layer Progressions:**
- 7 → 128: Expand representation space
- 128 → 64: Compress, learn abstractions
- 64 → 32: Further compression
- 32 → 1: Final binary decision

**Activation Functions:**
- **ReLU (hidden):** f(x) = max(0, x)
  - Non-linear (learns complex patterns)
  - Efficient (fast computation)
  - Avoids vanishing gradient problem
  
- **Sigmoid (output):** f(x) = 1/(1+e^-x)
  - Outputs probability [0, 1]
  - Perfect for binary classification

**Regularization:**
- L2 regularization (alpha=0.0001)
- Early stopping (validation monitoring)
- Validation split (20% for validation)

---

## 📈 Training Details

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Optimizer | Adam | Adaptive learning rate |
| Solver | Adam | Gradient descent variant |
| Max Iterations | 1000 | Prevent endless training |
| Learning Rate Init | 0.001 | Initial step size |
| Learning Rate Adapt | Adaptive | Adjust during training |
| Early Stopping | True | Stop if validation↑ |
| Validation Split | 20% | Monitor overfitting |
| Alpha (L2) | 0.0001 | Regularization strength |

### Training Progress
```
Iterations to convergence: 33
Final training loss: 0.3887
Achieved convergence without overfitting ✓
```

---

## 📈 Key Insights

1. **Neural Networks vs XGBoost:**
   - XGBoost: 80.93% (better on small datasets)
   - NN: 78.14% (slightly worse, but comparable)
   - Difference: Minimal (-2.79%) on 889 samples

2. **Generalization:**
   - Both models generalize well (no gross overfitting)
   - Training/test discrepancy manageable
   - Early stopping prevented overfitting

3. **ROC-AUC:** 84.30% (excellent discrimination)
   - Model distinguishes survivors well
   - 84% chance NN ranks random survivor > non-survivor

4. **Feature Learning:**
   - Network learned meaningful representations
   - Hidden layers discovered feature interactions
   - Output layer made final binary decisions

5. **Computational Trade-off:**
   - Slower to train (but still <1 second)
   - Faster to predict (once trained)
   - Harder to interpret (black box)

---

## 🛠️ Technical Pipeline

### 1. Data Preparation (same as Project #1)
- Load 889 passenger records
- Handle missing values (age, fare, embarked)
- Create features (is_alone, family_size)

### 2. Feature Engineering
```python
Features used:
- Pclass (passenger class)
- Sex (male=1, female=0)
- Age (years)
- Fare (ticket price)
- Embarked (port: S/C/Q → 0/1/2)
- is_alone (traveled alone)
- family_size (total relatives)

Total: 7 input features
```

### 3. Feature Normalization (CRITICAL!)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Result: All features normalized to ~N(0,1)
Why: Gradient descent sensitive to feature scale
```

### 4. Train-Test Split
- Total: 714 non-null samples (after cleaning)
- Training: 499 passengers (70%)
- Test: 215 passengers (30%)
- Stratification: Maintain 38.4% survival rate in both

### 5. Model Training
```python
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2
)

model.fit(X_train_scaled, y_train)
```

### 6. Evaluation
- Accuracy: 78.14%
- Precision: 73.81%
- Recall: 71.26%
- F1-Score: 72.51%
- ROC-AUC: 84.30%

---

## 🎓 Skills Demonstrated

✅ Neural network architecture design  
✅ Activation functions (ReLU, Sigmoid)  
✅ Feature normalization (critical for NN!)  
✅ Regularization techniques  
✅ Early stopping (prevent overfitting)  
✅ Adam optimizer understanding  
✅ Binary classification with deep learning  
✅ Model comparison (NN vs XGBoost)  
✅ Deep learning fundamentals  

---

## 🚀 How to Run

```bash
python neural_network_project.py
```

**Output:**
- Network architecture summary
- Training progress
- Model evaluation metrics
- Confusion matrix
- NN vs XGBoost comparison
- Example predictions on 3 passengers

---

## 👥 Example Predictions

```
Passenger 1: 1st Class Female, Age 25, Paid £512 
  NN Prediction: 99.45% survival ✓ (Very confident)
  XGB Prediction: 90.84% survival
  → Both agree: LIKELY SURVIVED

Passenger 2: 2nd Class Male, Age 35, Paid £26
  NN Prediction: 18.40% survival ❌
  XGB Prediction: 7.27% survival ❌
  → Both agree: UNLIKELY TO SURVIVE

Passenger 3: 3rd Class Female, Age 20, Paid £7.65
  NN Prediction: 61.70% survival
  XGB Prediction: 84.48% survival
  → Models disagree! (Different decision boundaries)
```

---

## 📊 When to Use Neural Networks

### NN Advantages
✅ Large datasets (>100k samples)  
✅ Complex feature interactions  
✅ Unstructured data (images, text, audio)  
✅ Transfer learning (pre-trained models)  
✅ Deep patterns (multiple abstraction levels)  

### NN Disadvantages
❌ Overkill for small datasets  
❌ Black box (hard to interpret)  
❌ Slow to train (needs GPU for large data)  
❌ Hyperparameter tuning difficult  
❌ Requires more data for good performance  

### XGBoost Advantages
✅ Small-medium datasets (1k-1M samples)  
✅ Interpretable (feature importance)  
✅ Fast to train (CPU-friendly)  
✅ Minimal hyperparameter tuning  
✅ Tabular data specialist  

### XGBoost Disadvantages
❌ Not ideal for massive datasets  
❌ Struggles with images/sequences  
❌ May underfit on very large data  

---

## 📚 Related Concepts

- **Neurons:** Computational units applying activation
- **Layers:** Groups of neurons processing data
- **Weights:** Parameters learned during training
- **Activation:** Non-linear function (ReLU, Sigmoid)
- **Backpropagation:** Algorithm updating weights
- **Gradient Descent:** Optimization method
- **Overfitting:** Memorizing training data
- **Regularization:** Penalizing complexity

---

## 🔮 Possible Improvements

1. **More layers:** 4-5 hidden layers for deeper learning
2. **Dropout:** Randomly disable neurons (prevent overfitting)
3. **Batch normalization:** Normalize layer inputs
4. **Hyperparameter tuning:** Grid search optimal values
5. **Ensemble:** Combine NN with XGBoost
6. **Larger dataset:** Use more samples for NN advantage
7. **Transfer learning:** Pre-trained networks
8. **Attention mechanisms:** Focus on important features

---

## 💡 Key Takeaway

**Neural networks are powerful but not always better!**

On small tabular datasets like Titanic:
- XGBoost: 80.93% (simpler, faster, interpretable)
- NN: 78.14% (more complex, harder to interpret)

**Result:** XGBoost wins for this task.

**But:** With 100k+ samples and millions of features, neural networks shine!

---

## 🌟 Portfolio Progression

```
Project 1: Classical ML (XGBoost)
Project 2: Regression baseline
Project 3: Business metrics (ROI)
Project 4: Time series patterns
Project 5: Deep learning (NN) ← You are here

Complete spectrum of ML knowledge! 🎓
```

---

**Created:** February 2026  
**Status:** ✅ Complete & tested  
**Best Model:** XGBoost (79.21% > 78.14%)  
**Neural Network Accuracy:** 78.14%  
**ROC-AUC:** 84.30%  
**Conclusion:** Both models excellent; XGBoost slight edge on tabular data
