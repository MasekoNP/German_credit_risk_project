# German Credit Risk Classification & Predictive Pipeline

---

## Title
**End-to-End German Credit Risk Analysis and Machine Learning Pipeline**

---

## Overview
Credit scoring models are essential financial tools used by banking institutions to evaluate loan applicant risk. This project builds a machine learning classification pipeline using the German Credit dataset to predict whether a loan applicant presents a **Good** or **Bad** credit risk. By evaluating demographic, employment and financial features, the project balances default detection against false rejections to optimize credit decisioning.

---

## Objectives
* **Identify Key Risk Factors**: Perform exploratory data analysis (EDA) to determine financial and demographic indicators strongly associated with credit default.
* **Benchmark Machine Learning Algorithms**: Compare linear and ensemble classifiers (Logistic Regression, Decision Trees, Random Forest, Extra Trees, XGBoost) using robust evaluation metrics.
* **Mitigate Class Imbalance**: Optimize model evaluation for class imbalance (70:30 ratio) by prioritizing Macro F1-Score over raw accuracy.
* **Model Serialization**: Export the trained, top-performing model for seamless downstream deployment and inference.

---

## Dataset
* **Dataset Name**: German Credit Dataset
* **Records**: 1,000 applicant profile entries
* **Features**: 21 numerical and categorical attributes
* **Currency**: Euro (€)
* **Target Variable**: Credit Risk Classification (`good` vs. `bad`)
* **Class Distribution**: ~70% Good Risk / ~30% Bad Risk

---

## Tools
* **Programming Language**: Python 3.x
* **Data Processing & Analysis**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn`, `xgboost`
* **Model Persistence**: `pickle`, `joblib`
* **Environment**: Jupyter Notebook / Anaconda

---

## Workflow
1. **Data Ingestion & Cleaning**: Handle missing values, address identifier columns and format feature data types.
2. **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations and target variable relationships across key demographic and financial features.
3. **Data Preprocessing & Feature Engineering**: Encode categorical variables, apply numerical scaling and structure train-test splits with stratification.
4. **Model Training & Benchmarking**: Train multiple classifiers and evaluate baseline performance.
5. **Evaluation & Selection**: Measure model performance across Accuracy, Precision, Recall and Macro F1-Score.
6. **Serialization & Deployment Setup**: Export the top-performing model to a reusable file format (`.pkl` / `.joblib`).

---

## Data Cleaning
* **ID Removal**: Dropped non-predictive identifier columns (`id`) to prevent memorization artifacts.
* **Type Validation**: Categorized numerical features (e.g., credit amount, loan duration, age) and string/categorical features for separate pipeline handling.
* **Consistency Verification**: Verified categorical entry values to prevent data formatting errors during label transformation.

---

## EDA
* **Savings Account Status**: Applicants holding lower savings balances (`<€100`) accounted for the majority of default cases (~35.8%), whereas higher savings (`>=€1000`) demonstrated low default rates (~11.1%).
* **Present Employment Tenure**: Higher employment duration strongly correlates with creditworthiness. Applicants with `<1` year of present employment showed an elevated default rate (~40.7%), compared to those with `>=7` years (~25.2%).
* **Age Demographics**: Younger applicants (`18–25` years) exhibited higher default rates (~42%) relative to older borrowers (`50+` years), who presented higher financial stability.
* **Feature Correlation**: A strong positive correlation ($r = 0.62$) was identified between `Duration in months` and `Credit amount`, confirming that larger loan requests typically carry extended repayment periods.

---

## Feature Engineering
* **Categorical Encoding**: Transformed categorical features using label encoding for tree-based algorithm compatibility.
* **Feature Scaling**: Applied `StandardScaler` to numerical inputs to normalize variable ranges for distance- and gradient-based models like Logistic Regression.
* **Stratified Splitting**: Implemented an 80:20 train-test split using `stratify=y` to preserve target class proportions across training and testing sets.

---

## Modeling
The following classification models were implemented and evaluated under identical training conditions:
1. **Logistic Regression** (Linear baseline with `class_weight='balanced'`)
2. **Decision Tree Classifier** (Single tree baseline with `class_weight='balanced'`)
3. **Random Forest Classifier** (Ensemble bagging approach with 100 trees and `class_weight='balanced'`)
4. **Extra Trees Classifier** (Extremely randomized trees ensemble with `class_weight='balanced'`)
5. **XGBoost Classifier** (Gradient boosting algorithm)

---

## Evaluation
Because the dataset exhibits a 70:30 class imbalance, evaluation metrics beyond raw accuracy were prioritized:
* **Macro F1-Score**: Primary benchmark metric to weigh precision and recall equally across both classes.
* **Macro Precision & Recall**: Evaluated to track trade-offs between false positives and false negatives.
* **Accuracy**: Monitored as a secondary holistic measure.

---

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.7200** | **0.6667** | **0.6667** | **0.6667** | 🏆 **Top Performer** |
| **Extra Trees** | 0.7450 | 0.7235 | 0.6083 | 0.6123 | High Accuracy, Lower Recall |
| **XGBoost** | 0.7100 | 0.6453 | 0.6262 | 0.6322 | Competitive Baseline |
| **Decision Tree** | 0.6900 | 0.6243 | 0.6167 | 0.6197 | Moderate Variance |
| **Logistic Regression** | 0.6400 | 0.6212 | 0.6429 | 0.6160 | Underperformed Linear Boundaries |

* **Winner**: **Random Forest** achieved the highest **Macro F1-Score (0.6667)** with balanced Macro Precision (0.6667) and Macro Recall (0.6667).

---

## Business Insights
* **Liquidity Buffer**: Higher savings balances serve as a major safety buffer against loan default. Encouraging applicants to link active savings accounts can improve risk scoring accuracy.
* **Job Stability**: Employment duration acts as a primary proxy for income dependability. Short employment tenure (`<1` year) represents a key risk exposure zone.
* **Exposure Scaling**: Extended loan durations combined with high credit amounts carry compounding risk, requiring stricter qualification criteria for long-term borrowing requests.

---

## Recommendations
1. **Targeted Risk Policies**: Establish higher underwriting thresholds or require co-signers/collateral for higher-risk demographics (e.g., short job tenure or low liquidity balances).
2. **Savings Incentives**: Offer tiered interest rate discounts for borrowers maintaining high savings balances (`>=500` DM) to reward low-risk profiles.
3. **Refactor Pipeline Encoding**: Upgrade label encoding workflows to scikit-learn `Pipeline` and `ColumnTransformer` frameworks to prevent preprocessing leakage between training and testing splits.

---

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/german-credit-risk.git](https://github.com/your-username/german-credit-risk.git)
   cd german-credit-risk
