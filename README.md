# 🔍 Feature Selection Analysis App

This Streamlit application demonstrates three different **Feature Selection** methods using the **Bank Marketing** dataset. It allows users to interactively compare how different algorithms select the most important features.

## 📂 Dataset
* **File:** `bank-full.csv`
* **Goal:** Predict if a client will subscribe to a term deposit (target variable: `y`).
* **Preprocessing:** Categorical data is automatically converted to numbers using Label Encoding.

## ⚙️ Methods Implemented
1.  **Filter Method:** Chi-Square Test (Selects top 5 statistically significant features).
2.  **Wrapper Method:** Recursive Feature Elimination (RFE) with Logistic Regression.
3.  **Embedded Method:** Random Forest Feature Importance.

## 🚀 How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

3.  **View:** The app will open automatically in your browser (usually at `http://localhost:8501`).

---
