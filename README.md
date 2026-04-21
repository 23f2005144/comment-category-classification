# 🧠 Comment Category Prediction Challenge

### 📌 Multiclass Text Classification | End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-green)
![NLP](https://img.shields.io/badge/Task-NLP-purple)
![Text Classification](https://img.shields.io/badge/Problem-Multiclass--Classification-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.82-blue)

---

## 🚀 Project Summary

This project focuses on **classifying user-generated comments into multiple categories** using a structured machine learning pipeline.

The workflow covers:

* 🔍 Data understanding
* 🧹 Preprocessing
* 📊 Exploratory analysis
* 🧠 Feature engineering
* 🤖 Model training & evaluation

---

## 🎯 Objective

* Accurately classify comments into predefined categories
* Handle **imbalanced class distribution**
* Build a model that generalizes well

---

## 🧠 Approach

### 🔹 Data Understanding & Cleaning

* Removed irrelevant features
* Handled missing values
* Standardized dataset for modeling

---

### 🔹 Exploratory Data Analysis

* 📊 Feature distributions and outliers
* ⚖️ Class imbalance analysis
* 📝 Review of sample comments
* 🔗 Relationships between features and labels

---

### 📊 Visual: Class Distribution

*(Add your plot here)*

```id="viz1"
![Class Distribution](your-image-link-here)
```

---

### 🔹 Feature Engineering

* 📝 Text-based features (length, word count)
* ⏱️ Temporal feature extraction
* 📉 Log transformation
* ⚖️ Yeo-Johnson normalization
* 🔍 Mutual Information for feature selection

---

### 🔹 Text Representation

#### 📌 Word-Level TF-IDF

* Tuned n-grams and vocabulary

#### 🔬 Character-Level TF-IDF (`char_wb`)

* Captures subword patterns
* Handles noisy and inconsistent text

---

### 🔹 Modeling

Models trained under a consistent pipeline:

* Logistic Regression
* Linear SVM (LinearSVC)
* LightGBM Classifier

---

### 📊 Visual: Model Performance Comparison

*(Add your comparison chart here)*

```id="viz2"
![Model Comparison](your-image-link-here)
```

---

## 📊 Evaluation

### 🎯 Metric

**Macro F1 Score**

* Accounts for class imbalance
* Treats all classes equally

### 🏆 Final Score

**Macro F1: 0.82**

---

## ⚔️ Challenges & Observations

### ⚖️ Class Imbalance

* Some categories had significantly fewer samples
* Addressed by using **Macro F1-score** instead of accuracy

---

### 📝 Noisy Text Data

* Informal language, inconsistencies, and variations
* Improved robustness using **character-level TF-IDF**

---

### 📉 Skewed Feature Distributions

* Certain numerical features were highly skewed
* Applied **log and Yeo-Johnson transformations**

---

### 🔍 Feature Selection

* Not all engineered features contributed equally
* Used **Mutual Information** to identify useful signals

---

### ⚙️ Model Trade-offs

* Linear models performed strongly on sparse data
* Boosting methods required careful tuning to remain competitive

---

## 🧩 Key Insights

* Feature engineering significantly impacted performance
* Character-level TF-IDF improved generalization
* Linear models are highly effective for text classification
* Evaluation metric choice directly affects model selection

---

## 🛠️ Tech Stack  

- 🐍 **Core**: Python  
- 🤖 **Machine Learning**: Scikit-learn, LightGBM  
- 📊 **Data Handling**: Pandas, NumPy  
- 📉 **Visualization**: Matplotlib, Seaborn, WordCloud  
- 🔤 **Text Processing**: Regex (`re`), TF-IDF  
- 📐 **Statistical Transformations**: SciPy (`scipy.stats`)

---

## 📂 Repository Structure

```id="viz3"
.
├── notebook.ipynb        # Complete workflow
├── submission.csv        # Final predictions
└── README.md             # Documentation
```

---

## 🌟 Summary

A structured machine learning workflow for text classification, with focus on:

* Data understanding
* Feature engineering
* Consistent evaluation

---

## 💡 Notes for Reviewers

* The notebook contains full experimentation and intermediate results
* Visualizations and comparisons are included within the workflow
* Emphasis is placed on **decision-making and reasoning at each step**

---
* Review your **actual graphs and pick the best 2–3**
* Or turn this into a **GitHub Pages portfolio website layout** (that’s a big upgrade)
