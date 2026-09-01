# Artificial Neural Network (ANN) Deep Learning Project Suite

Welcome to the Artificial Neural Network (ANN) repository. This project showcases end-to-end Deep Learning workflows for **Regression**, **Binary Classification**, and **Multiclass Classification** using TensorFlow/Keras.

---

## 📁 Repository Structure

```
d:\Projects\ANN\
│
├── ANN_Regression_Housing_Prices.ipynb            # Section 1: ANN Regression Model
├── ANN_Binary_Classification_IoT_Security.ipynb   # Section 2: ANN Binary Classification Model
├── ANN_Multiclass_Classification_IoT_Security.ipynb # Section 3: ANN Multiclass Classification Model
├── real_estate_dataset.csv                        # Housing Prices Dataset
├── train_test_network.csv                         # TON_IoT Network Security Dataset (211k+ rows)
└── README.md                                      # Project Documentation
```

---

# SECTION 1: ANN Regression – Real Estate Housing Prices Prediction

### Notebook: `ANN_Regression_Housing_Prices.ipynb`
* **Task Type**: Continuous Numerical Value Prediction (Regression)
* **Dataset**: `real_estate_dataset.csv` (1,000 samples, 6 input features)
* **Target Variable**: `Price` (Continuous USD Real Estate Price)

---

### 🧠 1.1 ANN Model Architecture

```python
def build_ann_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
```

#### **Architecture Breakdown & Layer Components**:
* **Layer Hierarchy (128 $\rightarrow$ 64 $\rightarrow$ 32 Neurons)**:
  - **Dense Layer 1 (128 neurons)**: Expands input features into a 128-dimensional latent space to capture complex non-linear feature interactions.
  - **Dense Layer 2 (64 neurons)** & **Dense Layer 3 (32 neurons)**: Progressive bottleneck layers that compress representations and distill key pricing signals.
  - **ReLU Activation ($\max(0, z)$)**: Used in all hidden layers to introduce non-linearity and prevent vanishing gradients.
* **Batch Normalization**:
  Applied after hidden Dense layers. It normalizes output activations across the mini-batch ($\mu \approx 0, \sigma \approx 1$), reducing internal covariate shift, stabilizing training, and allowing faster convergence.
* **Dropout Regularization (0.1)**:
  Randomly deactivates $10\%$ of neurons during each forward pass. This prevents neuron co-adaptation and overfitting on the training set.
* **Output Layer — `Dense(1, activation='linear')`**:
  Outputs a single continuous numerical price. Linear activation ($f(z) = z$) provides an unconstrained real-valued output suitable for regression.

---

### ⚙️ 1.2 Optimizer, Loss Function & Epochs

* **Adam Optimizer (`learning_rate=0.01`)**:
  Combines **Momentum** (exponentially decaying average of past gradients) and **RMSprop** (exponentially decaying average of past squared gradients). It adapts learning rates per parameter, providing faster convergence and robustness against noisy gradients compared to standard SGD or Adagrad.
* **Loss Function (`mse`)**: Mean Squared Error penalizes larger errors quadratically ($\frac{1}{N} \sum (y_i - \hat{y}_i)^2$).
* **Epochs & Training Strategy**: **200 Epochs** with batch size `16`, using `EarlyStopping` (patience 25) and `ReduceLROnPlateau` (factor 0.5, patience 10).

---

### 📊 1.3 Evaluation Metrics & $R^2$ Score Explanation

```
==================================================
  ANN Regression Performance Evaluation
==================================================
  R² Score (Coefficient of Determination) : 0.9421 (94.21%)
  1-MAPE Accuracy                        : 0.9385 (93.85%)
  Mean Absolute Error (MAE)              : $28,412.35
  Root Mean Squared Error (RMSE)         : $35,110.42
==================================================
[SUCCESS] Target Accuracy (>90% R² Score) achieved!
```

#### **What is the $R^2$ Score (Coefficient of Determination)?**
$R^2$ measures the proportion of total target variance explained by the model:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

* **Interpretation**: $R^2 = 1.0$ indicates perfect prediction; $R^2 = 0.0$ indicates baseline mean prediction.
* **Our Result**: **$0.9421$ (94.21%)**, exceeding the >90% target threshold.

---

### 📈 1.4 Visualizations

1. **Training & Validation Loss Plot**: Confirms steady loss reduction without overfitting.
2. **Actual vs. Predicted Scatter Plot**: Shows tight clustering of predictions along the ideal $y=x$ line.
3. **Residuals Distribution**: Displays a zero-centered Gaussian curve ($y_i - \hat{y}_i$), proving unbiased errors.

---

# SECTION 2: ANN Binary Classification – Network IoT Security Dataset

### Notebook: `ANN_Binary_Classification_IoT_Security.ipynb`
* **Task Type**: Binary Classification (Normal Traffic vs. Cyber Attack)
* **Dataset**: `train_test_network.csv` (211,043 rows, 42 features)
* **Target Variable**: `label` (0 = Normal, 1 = Attack)

---

### 🧠 2.1 ANN Model Architecture

```python
def build_ann_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

#### **Architecture Breakdown & Layer Components**:
* **Layer Hierarchy (128 $\rightarrow$ 64 $\rightarrow$ 32 Neurons)**:
  Processes 42 telemetry features through 3 ReLU-activated hidden layers to detect complex network intrusion patterns.
* **Batch Normalization & Dropout (0.2 / 0.1)**:
  Batch Normalization standardizes mini-batch activations across 211,043 samples. Dropout randomly deactivates $10\% - 20\%$ of neurons to prevent memorizing specific IP addresses or ports.
* **Output Layer — `Dense(1, activation='sigmoid')`**:
  Sigmoid activation ($\sigma(z) = \frac{1}{1 + e^{-z}}$) outputs a probability score $p \in [0, 1]$. Traffic is classified as Attack if $p \ge 0.5$, else Normal.

---

### ⚙️ 2.2 Optimizer, Loss Function & Epochs

* **Adam Optimizer (`learning_rate=0.001`)**: Automatically scales step sizes for sparse network features.
* **Loss Function (`binary_crossentropy`)**: Penalizes confident wrong binary predictions logarithmically ($-\frac{1}{N} \sum [y \log \hat{y} + (1-y) \log(1-\hat{y})]$).
* **Epochs & Batch Size**: **20 Epochs** with batch size `256`.

---

### 📊 2.3 Evaluation Metrics & Results

```
==================================================
  ANN Binary Classification Performance Evaluation
==================================================
  Accuracy Score                         : 0.9995 (99.95%)
  Precision                              : 0.9996
  Recall                                 : 0.9998
  F1 Score                               : 0.9997
  ROC-AUC Score                          : 1.0000
==================================================
[SUCCESS] Target Accuracy (>90%) achieved!
```

* **Metrics Summary**: Accuracy (**99.95%**), Precision (**0.9996**), Recall (**0.9998**), F1-Score (**0.9997**), and ROC-AUC (**1.0000**).

---

### 📈 2.4 Visualizations

1. **Learning Curves**: Shows rapid accuracy and loss convergence within 3 epochs.
2. **2x2 Confusion Matrix Heatmap**: Displays exact True Positive/Negative rates.
3. **ROC Curve**: Demonstrates near-perfect classification area ($\text{AUC} = 1.0000$).

---

# SECTION 3: ANN Multiclass Classification – Network IoT Security Dataset

### Notebook: `ANN_Multiclass_Classification_IoT_Security.ipynb`
* **Task Type**: Multiclass Classification (10 Network Traffic Categories)
* **Dataset**: `train_test_network.csv` (211,043 rows, 42 features)
* **Target Variable**: `type` (`normal`, `backdoor`, `ddos`, `dos`, `injection`, `password`, `scanning`, `ransomware`, `xss`, `mitm`)

---

### 🧠 3.1 ANN Model Architecture

```python
def build_ann_multiclass_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

#### **Architecture Breakdown & Layer Components**:
* **Layer Hierarchy (128 $\rightarrow$ 64 $\rightarrow$ 32 Neurons)**:
  Extracts distinct network signatures to separate 10 traffic types. Batch Normalization and Dropout ($0.2 / 0.1$) maintain gradient stability and prevent overfitting.
* **Output Layer — `Dense(10, activation='softmax')`**:
  Softmax activation ($\frac{e^{z_i}}{\sum e^{z_j}}$) converts raw logits into a 10-class probability distribution ($\sum p_i = 1.0$). The class with the highest probability is selected.

---

### ⚙️ 3.2 Optimizer, Loss Function & Epochs

* **Adam Optimizer (`learning_rate=0.001`)**: Efficient adaptive gradient updates across 10 output boundaries.
* **Loss Function (`sparse_categorical_crossentropy`)**: Computes categorical cross-entropy directly using integer target labels ($0..9$), avoiding memory-heavy One-Hot matrices.
* **Epochs & Batch Size**: **20 Epochs** with batch size `256`.

---

### 📊 3.3 Evaluation Metrics & Results

```
=======================================================
  ANN Multiclass Classification Performance Evaluation
=======================================================
  Overall Accuracy Score                  : 0.9716 (97.16%)
  Weighted Precision                      : 0.9722
  Weighted Recall                         : 0.9716
  Weighted F1 Score                       : 0.9715
  Macro F1 Score                          : 0.9425
=======================================================
[SUCCESS] Target Accuracy (>90%) achieved!
```

---

### 📈 3.4 Visualizations

1. **Learning Curves**: Tracks loss reduction and multi-class accuracy over 20 epochs.
2. **10x10 Confusion Matrix Heatmap**: Visualizes predictions across all 10 attack classes.
3. **Per-Class Metrics Bar Chart**: Compares Precision, Recall, and F1-Score for each individual traffic class.

---

# 🔄 Model Comparison & Summary

| Parameter / Metric | Section 1: Regression | Section 2: Binary Classification | Section 3: Multiclass Classification |
| :--- | :--- | :--- | :--- |
| **Notebook File** | `ANN_Regression_Housing_Prices.ipynb` | `ANN_Binary_Classification_IoT_Security.ipynb` | `ANN_Multiclass_Classification_IoT_Security.ipynb` |
| **Dataset** | `real_estate_dataset.csv` | `train_test_network.csv` | `train_test_network.csv` |
| **Target Variable** | `Price` (Continuous) | `label` (0 or 1) | `type` (10 classes) |
| **Hidden Layers** | 128 $\rightarrow$ 64 $\rightarrow$ 32 (ReLU) | 128 $\rightarrow$ 64 $\rightarrow$ 32 (ReLU) | 128 $\rightarrow$ 64 $\rightarrow$ 32 (ReLU) |
| **Output Activation** | `linear` | `sigmoid` | `softmax` |
| **Loss Function** | `mse` | `binary_crossentropy` | `sparse_categorical_crossentropy` |
| **Optimizer** | Adam ($\eta = 0.01$) | Adam ($\eta = 0.001$) | Adam ($\eta = 0.001$) |
| **Epochs** | 200 (EarlyStop patience=25) | 20 | 20 |
| **Primary Metric** | $R^2$ Score ($94.21\%$) | Binary Accuracy ($99.95\%$) | Multiclass Accuracy ($97.16\%$) |
| **Target Goal** | $>90.00\%$ ($R^2$) | $>90.00\%$ Accuracy | $>90.00\%$ Accuracy |
| **Goal Status** | **PASSED** | **PASSED** | **PASSED** |

---

## 🛠️ Requirements & Execution

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

Run notebooks in Jupyter:
```bash
jupyter notebook d:\Projects\ANN\ANN_Regression_Housing_Prices.ipynb
jupyter notebook d:\Projects\ANN\ANN_Binary_Classification_IoT_Security.ipynb
jupyter notebook d:\Projects\ANN\ANN_Multiclass_Classification_IoT_Security.ipynb
```
