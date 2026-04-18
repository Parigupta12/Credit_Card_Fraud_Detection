# 💳 AI Credit Card Fraud Detection System (Advanced ML + Streamlit)

## 📌 Project Overview

This project is an **Advanced Machine Learning-based Fraud Detection System** that detects fraudulent credit card transactions using:

* **XGBoost Classifier**
* **SMOTE (Imbalanced Data Handling)**
* **Feature Engineering**
* **Threshold Optimization**

The system is deployed using **Streamlit**, allowing users to:

* Upload CSV files
* Detect fraud transactions
* View probabilities
* Download results

---

## 🧠 Model Architecture (IMPORTANT)

### 🔍 Pipeline Used

1. **Data Cleaning**

   * Removed missing values
   * Ensured numeric data

2. **Feature Engineering**

   * Extracted `Hour` from `Time`

   ```python
   Hour = (Time // 3600) % 24
   ```

3. **Scaling**

   * StandardScaler applied on `Amount`

4. **Handling Imbalanced Data**

   * Used **SMOTE**
   * Balanced fraud vs normal samples

5. **Model**

   * **XGBoost Classifier**

   ```python
   n_estimators=700
   max_depth=10
   learning_rate=0.03
   scale_pos_weight=25
   ```

6. **Threshold Tuning**

   * Instead of default 0.5, used:

   ```python
   threshold = 0.01
   ```

   👉 This improves **fraud detection recall**

---

## 📊 Model Performance

### ⚠️ Important:

Dataset is highly imbalanced → Accuracy is misleading

### 📌 Metrics (Your Actual Model)

* **Fraud Detection Recall:** HIGH (optimized using low threshold)
* **Confusion Matrix used for evaluation**
* Model focuses on:

  * Detecting frauds (minimizing false negatives)

---

## 🎯 Key Insight

> The model prioritizes **detecting fraud transactions** even at the cost of slightly higher false positives.

This is **real-world banking approach**.

---

## 📂 Input Features

Model expects:

```id="f1"
V1, V2, ..., V28, Amount, Hour
```

### ⚠️ Notes:

* `Scaled_Amount` is internally handled via scaler
* `Time` is NOT required (converted to Hour)
* Column names must match

---

## 📁 Saved Artifacts

```id="f2"
fraud_model.pkl   → Trained XGBoost model
scaler.pkl        → StandardScaler for Amount
features.pkl      → List of feature names
```

---

## 🚀 Features of Web App

✔ Upload CSV for batch prediction
✔ Automatic preprocessing
✔ Fraud probability scoring
✔ Custom threshold-based prediction
✔ Download results
✔ Handles missing columns

---

## ⚙️ Installation

```bash id="f3"
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash id="f4"
streamlit run app.py
```

Open:

```id="f5"
http://localhost:8501
```

---

## 📊 Example Workflow

1. Upload dataset
2. App processes:

   * Scaling
   * Feature alignment
3. Model predicts:

   * Fraud probability
   * Fraud label
4. Results displayed
5. Download CSV

---

## ⚠️ Limitations

* Model is **dataset-specific (PCA-based features)**
* Cannot work on raw banking datasets
* Requires similar feature distribution

---

## 🔮 Future Improvements

* Real-time fraud detection system
* Deep Learning models (LSTM / Autoencoder)
* API integration with banking systems
* Advanced dashboard (analytics + visualization)

---

## 👩‍💻 Author

**Pari Gupta**
B.Tech CSE (AI/ML)

---

## ⭐ Support

If you found this project useful, consider ⭐ starring the repository!
