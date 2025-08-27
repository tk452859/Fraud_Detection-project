# Credit Card Fraud Detection using Machine Learning


A machine learning project focused on accurately identifying fraudulent credit card transactions from a highly imbalanced dataset. This project demonstrates a full data science workflow, from preprocessing and exploratory data analysis to model building, evaluation, and addressing class imbalance.

---

## 📖 Project Overview

Credit card fraud is a multi-billion dollar problem for the financial industry. This project tackles the challenge of building a predictive model that can **precisely identify fraudulent transactions** while minimizing false alarms. The core difficulty lies in the extreme **class imbalance**, where fraudulent transactions are a tiny fraction of all transactions.

**Main Objective:** To develop a robust machine learning model that achieves high **recall** (catching most fraud) without sacrificing too much **precision** (minimizing false positives).

## 📊 Dataset

**Source:** [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- The dataset contains transactions made by European cardholders in September 2013.
- It comprises **284,807 transactions** in total, with only **492 (0.172%) fraudulent** ones.
- All features (`V1` to `V28`) are numerical and are the result of a **PCA transformation** (Dimensionality Reduction) to protect confidentiality. The original features are not provided.
- Features `Time` and `Amount` are the only non-transformed features.
- **`Class`** is the target variable (1 = Fraud, 0 = Normal).

## 🛠️ Tech Stack & Libraries

- **Programming Language:** Python 3.8+
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Environment:** Jupyter Notebook / Google Colab

## 🔍 Methodology & Approach

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)
- **Handled Missing Values:** Checked for and confirmed no missing values.
- **Analyzed Class Imbalance:** Visualized the severe imbalance between fraudulent and non-fraudulent classes.
- **Scaled Features:** Standardized the `Time` and `Amount` features using `StandardScaler` to ensure all features were on a similar scale for the model.
- **Train-Test Split:** Performed a **stratified split** to maintain the same class distribution in both training and testing sets, ensuring a fair evaluation.

### 2. Dealing with Class Imbalance
To handle the extreme imbalance, two primary strategies were employed:
- **Algorithm Choice:** Used models like **Random Forest** that perform well on imbalanced data.
- **Resampling Techniques:** Experimented with **Random Under-Sampling** (reducing the majority class) to improve model performance on the minority class.

### 3. Model Building & Training
Trained and evaluated multiple classifiers to identify the best performer:
- **Logistic Regression** (Baseline)
- **Random Forest Classifier**
- **XGBoost Classifier**

The **Random Forest** model was selected for its strong performance and interpretability.

### 4. Model Evaluation
Models were evaluated based on metrics that are critical for imbalanced classification problems:
- **Confusion Matrix:** To visualize True/False Positives/Negatives.
- **Precision:** Ability to not label a legitimate transaction as fraudulent.
- **Recall (Sensitivity):** Ability to find all fraudulent transactions.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **Area Under the ROC Curve (AUC-ROC):** Overall measure of model performance across all classification thresholds.

## 📈 Results & Key Findings

The final Random Forest model, tuned with hyperparameters, achieved the following results on the **test set**:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.95% |
| **Precision** | 93.55% |
| **Recall** | 82.05% |
| **F1-Score** | 87.50% |
| **AUC-ROC Score** | 0.981 |

- **Key Insight:** The model successfully identifies **82% of all fraudulent transactions** while maintaining a high precision of **93.6%**, meaning that over 9 out of 10 transactions flagged as fraud are actually fraudulent. This balance is crucial for a practical fraud detection system.


