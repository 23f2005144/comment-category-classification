# Multiclass Comment Classification Pipeline

High-dimensional NLP classification pipeline for noisy user-generated comments using TF-IDF representations, feature engineering, and class-imbalance-aware modeling.

![Project](https://img.shields.io/badge/Project-NLP%20Pipeline-blue)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-orange)
![Approach](https://img.shields.io/badge/Approach-TFIDF%20%2B%20Feature%20Engineering-purple)
![Model](https://img.shields.io/badge/Best%20Model-LightGBM-green)
![Metric](https://img.shields.io/badge/Macro%20F1-0.8344-teal)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Competition Context

This project was submitted to the **[Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)** on Kaggle.

The goal was to classify user-generated comments into multiple categories under real-world constraints — significant class imbalance, noisy informal text, and high-dimensional sparse feature spaces.

**Public Leaderboard Score: Macro F1 — 0.8344**

---

## Overview

This project explores multiclass classification of noisy user-generated comments under significant class imbalance.

The objective was not only to achieve strong predictive performance, but also to study how different text representations, feature transformations, and model families behave in sparse high-dimensional NLP settings.

The pipeline includes:

* Exploratory data analysis and linguistic pattern analysis
* Numerical and text-based feature engineering
* Sparse text representation using word-level and character-level TF-IDF
* Transformation and selection of skewed features
* Systematic comparison of linear and boosting-based models
* Evaluation focused on class imbalance using Macro F1 and Precision-Recall analysis

---

## Pipeline Overview

```text
Raw Comments + Metadata
           ↓
EDA & Text Analysis
           ↓
Feature Engineering
           ↓
TF-IDF + Engineered Features
           ↓
High-Dimensional Sparse Matrix
           ↓
Model Training & Tuning
           ↓
Evaluation & Interpretation
```

---

## Problem Context

The dataset presented several practical NLP challenges:

* Significant class imbalance across target categories
* Informal and noisy user-generated text
* High-dimensional sparse feature spaces
* Minority-class sensitivity under Macro F1 evaluation
* Overlapping linguistic patterns between semantically similar classes

These constraints made the project particularly useful for studying trade-offs between feature engineering, sparse text representations, and model behavior under imbalance.

---

## Objectives

The primary goals of the project were:

* Accurately classify comments into predefined categories
* Improve minority-class recall without severely degrading precision
* Build representations robust to noisy and inconsistent text
* Compare linear and boosting-based approaches on sparse NLP features
* Maintain strong generalization under leaderboard evaluation

---

## Exploratory Data Analysis

EDA was used not only for descriptive analysis, but to guide downstream feature engineering and modeling decisions.

### Key Observations

#### Class Imbalance

The dataset showed substantial imbalance across categories, motivating:

* Macro F1 as the primary evaluation metric
* Class-weight-aware optimization
* Precision-Recall analysis instead of ROC-based evaluation

### Linguistic Patterns by Class

Class-wise text analysis revealed that categories often exhibited distinct lexical and structural patterns.

Some categories contained:

* Repeated keywords and recurring semantic themes
* Short informal comments and noisy phrasing
* Distinct stylistic and token-frequency patterns
* Variations in comment length and vocabulary density

These observations motivated the use of:

* Word-level TF-IDF for semantic/contextual signals
* Character-level TF-IDF (`char_wb`) for robustness against misspellings and morphological variation
* Additional numerical text statistics such as word count and average word length

### Temporal Patterns

Temporal analysis suggested cyclic activity patterns across comments.

To preserve periodic relationships without introducing artificial ordinal assumptions, temporal features such as hour and month were encoded using sine-cosine transformations.

### Numerical Feature Distributions

Several engineered numerical features showed:

* Strong skewness
* Heavy zero inflation
* Long-tailed distributions

This motivated:

* Yeo-Johnson transformation for skew reduction
* Binary indicator conversion for zero-inflated variables
* Mutual-information-based comparison of raw vs transformed features

### Feature Importance Alignment with EDA

Mutual Information analysis later confirmed many patterns identified during EDA.

Features associated with:

* Comment length
* Vocabulary structure
* Temporal activity patterns
* Character-level token variation

showed meaningful predictive contribution across several categories.

This alignment between exploratory observations and downstream feature importance provided additional confidence that the engineered features were capturing genuine class-specific behavior rather than noise.

---

## Visualizations

### Class Distribution

![Class Distribution](https://raw.githubusercontent.com/23f2005144/comment-category-classification/main/Class%20Distribution.png)

### Model Performance Comparison

![Model Comparison](https://raw.githubusercontent.com/23f2005144/comment-category-classification/main/Model%20Performance%20Comparison.png)

---

## Feature Engineering

Feature engineering played a central role in improving model performance.

### Engineered Numerical Features

The following text-derived numerical features were created:

* Comment length
* Word count
* Average word length
* Temporal features extracted from timestamps

These features were intended to capture structural differences in how categories were expressed.

### Temporal Encoding

Hour and month features were cyclically encoded using sine-cosine transformations.

This preserved periodic relationships that would otherwise be distorted under standard ordinal encoding.

### Handling Zero-Inflated Features

Several engineered variables contained excessive zeros.

Instead of treating them purely as continuous features, binary indicators were introduced to explicitly model feature presence versus absence.

### Handling Skewed Features

Two transformation strategies were evaluated:

* Log transformation
* Yeo-Johnson transformation

Yeo-Johnson consistently reduced skewness more effectively while remaining stable for zero and negative values, and was therefore retained.

### Feature Selection and Validation

Mutual Information (MI) was used to evaluate feature relevance.

The project compared:

* Raw engineered features
* Transformed engineered features

The transformed feature set produced slightly stronger validation performance and was selected for the final pipeline.

---

## Text Representation

The final representation combined:

* Word-level TF-IDF
* Character-level TF-IDF (`char_wb`)
* Engineered numerical features

### Word-Level TF-IDF

Word-level vectorization was used to capture:

* Semantic patterns
* Contextual word relationships
* Class-specific vocabulary usage

Hyperparameters such as:

* `ngram_range`
* `min_df`
* `max_df`
* `sublinear_tf`

were systematically tuned.

### Character-Level TF-IDF (`char_wb`)

Character-level TF-IDF significantly improved robustness against:

* Informal language
* Misspellings
* Morphological variation
* Noisy user-generated text

This representation was particularly helpful for minority classes with inconsistent phrasing patterns.

### Final Sparse Representation

The final feature space contained approximately 125K sparse TF-IDF features.

This high-dimensional sparse representation favored models capable of efficiently handling sparse inputs, particularly linear methods and gradient boosting approaches.

---

## Modeling Strategy

Three model families were explored to compare how different approaches behave under sparse high-dimensional NLP representations.

### Logistic Regression

Used as a strong linear baseline with:

* Regularization tuning
* Tolerance optimization
* Class-weight balancing

### LinearSVC

Evaluated for its strong performance on sparse text representations.

Hyperparameters focused primarily on:

* Regularization strength
* Class-weight adjustments

### LightGBM

LightGBM was manually tuned across multiple parameter combinations to explore:

* Nonlinear interactions
* Sparse-feature handling
* Generalization behavior under imbalance

---

## Model Trade-Off Analysis

| Model               | Strengths                                                 | Limitations                          |
| ------------------- | --------------------------------------------------------- | ------------------------------------ |
| Logistic Regression | Stable baseline, interpretable behavior                   | Limited nonlinear learning           |
| LinearSVC           | Strong sparse-feature performance                         | Reduced probability interpretability |
| LightGBM            | Captured nonlinear interactions and feature relationships | Greater overfitting sensitivity      |

Linear models performed strongly on sparse TF-IDF representations, while LightGBM achieved the best overall Macro F1 by leveraging both engineered numerical features and nonlinear interactions.

Slight overfitting observed during training-validation comparison was considered acceptable because it improved minority-class recall while maintaining leaderboard generalization.

---

## Evaluation Strategy

### Primary Metric: Macro F1

Macro F1 was selected because:

* The dataset was highly imbalanced
* Equal importance was assigned to all classes
* Minority-class performance needed explicit weighting

### Evaluation Components

The models were evaluated using:

* Classification reports
* Confusion matrices
* Precision-Recall curves

### Why ROC Curves Were Avoided

ROC curves were intentionally not prioritized because the large number of true negatives in imbalanced settings can produce overly optimistic interpretations.

Precision-Recall analysis provided more informative insight into minority-class behavior and precision-recall trade-offs.

---

## Results

### Validation Performance

**Macro F1: 0.8350** (up from ~0.79 with a Logistic Regression baseline)

### Public Leaderboard Performance

**Macro F1: 0.8344**

The close alignment between validation and leaderboard scores suggested that the final pipeline generalized reliably without severe leaderboard overfitting.

---

## Failure Analysis and Observations

### Minority-Class Challenges

Minority classes were more sensitive to:

* Noisy phrasing
* Short comments
* Overlapping semantic patterns

This occasionally reduced recall under simpler linear baselines.

### Sparse NLP Trade-Offs

The project highlighted the balance between:

* Capturing minority-class patterns
* Avoiding excessive overfitting
* Preserving generalization under sparse high-dimensional features

### Feature Representation Insights

Character-level TF-IDF proved especially effective for handling:

* Informal language
* Misspellings
* Short noisy comments

while engineered numerical features provided additional structural signals complementary to text representations.

---

## Key Insights

* Feature engineering substantially improved overall model performance
* Character-level TF-IDF improved robustness against noisy user-generated text
* Feature importance analysis aligned closely with earlier EDA observations
* LightGBM benefited from combining sparse TF-IDF features with engineered numerical features
* Precision-Recall analysis was significantly more informative than ROC analysis under imbalance
* Careful trade-off balancing between minority recall and precision was critical for optimizing Macro F1

---

## Lessons Learned

- Character-level TF-IDF representations were especially effective for improving robustness on noisy minority-class samples
- Sparse linear models remained highly competitive despite the availability of more complex nonlinear approaches
- Evaluation metric selection significantly influenced optimization strategy under heavy class imbalance
- Incremental improvements in feature engineering produced larger gains than aggressively increasing model complexity
- Alignment between EDA observations and downstream feature importance analysis improved confidence in feature design decisions

---

## Repository Structure

```text
.
├── 23f2005144-comment-classification-notebook.ipynb
├── submission.csv
├── Class Distribution.png
├── Model Performance Comparison.png
└── README.md
```

---

## Reproducing the Project

The dataset is sourced from the Kaggle competition and cannot be redistributed here.

To run the project:

1. Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/comment-category-prediction-challenge)
2. Place them in the root directory alongside the notebook
3. Install dependencies and run:

```bash
pip install -r requirements.txt
jupyter notebook 23f2005144-comment-classification-notebook.ipynb
```

---

## Tech Stack

| Category                    | Tools                          |
| --------------------------- | ------------------------------ |
| Core Language               | Python                         |
| Machine Learning            | Scikit-learn, LightGBM         |
| Data Handling               | Pandas, NumPy                  |
| Visualization               | Matplotlib, Seaborn, WordCloud |
| NLP / Text Processing       | TF-IDF, Regex                  |
| Statistical Transformations | SciPy                          |

---

## Future Improvements

* Engineering additional engagement-based features (e.g. interaction signals, user activity patterns) to improve class separability
* Deeper error analysis on Class 3 specifically — understanding whether its poor performance stems from overlapping linguistic patterns, insufficient samples, or feature representation gaps
* More systematic hyperparameter search using automated tuning instead of manual combinations
* Ensemble combinations of linear and boosting models to balance sparse-feature strengths of LinearSVC with LightGBM's nonlinear capacity

---

## Notes for Reviewers

The notebook contains:

* Full exploratory analysis
* Feature engineering experiments
* Transformation comparisons
* Mutual Information feature analysis
* Hyperparameter tuning workflows
* Model comparison and evaluation
* Precision-Recall and confusion matrix analysis

The project emphasizes data-driven reasoning, trade-off analysis, and iterative experimentation throughout the pipeline.
