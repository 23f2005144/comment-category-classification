# Multiclass Comment Classification Pipeline

High-dimensional NLP classification pipeline for approximately 198K noisy user-generated comments using TF-IDF representations, feature engineering, and class-imbalance-aware modeling.

![Project](https://img.shields.io/badge/Project-NLP%20Pipeline-blue)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-orange)
![Approach](https://img.shields.io/badge/Approach-TFIDF%20%2B%20Feature%20Engineering-purple)
![Model](https://img.shields.io/badge/Best%20Model-LightGBM-green)
![Metric](https://img.shields.io/badge/Macro%20F1-0.8344-teal)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Competition Context

This project was developed as part of the **Machine Learning Practice (MLP) Project** course under the **BS in Data Science and Applications** programme at **IIT Madras**, independently executed using the **[Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)** on Kaggle as the problem statement.

The project was evaluated through a two-stage viva:

- **Level 1 Viva** — Focused on verifying end-to-end project implementation, including data loading, EDA, preprocessing, feature engineering, pipeline construction, hyperparameter tuning, model comparison, and the ability to clearly explain code structure and implementation decisions.
- **Level 2 Viva** — 1-hour technical viva with industry experts involving a detailed walkthrough of the complete pipeline. Evaluation focused on:
  - conceptual understanding
  - mathematical foundations
  - practical implementation
  - programming-level understanding

**Academic Score: 96/100 (S Grade)**  
**Public Leaderboard Score: Macro F1 — 0.8344** *(cutoff to qualify for viva: 0.80)*

Having completed foundational coursework in ML algorithms, this project was an opportunity to apply that theory end-to-end — building a full pipeline with Scikit-learn, LightGBM, Pandas, NumPy, Matplotlib, and Seaborn on a real problem with real constraints.

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

The dataset contained **198,000 comments** across 15 features, including engagement signals (`upvote`, `downvote`, emoticon counts), automated system indicators (`if_1`, `if_2`), demographic metadata (`race`, `religion`, `gender`), and the raw comment text.

Several practical NLP challenges were present:

* Significant class imbalance across target categories
* Informal and noisy user-generated text
* High-dimensional sparse feature spaces
* Minority-class sensitivity under Macro F1 evaluation
* Overlapping linguistic patterns between semantically similar classes

The target had four classes with no provided descriptions. Text analysis during EDA revealed their likely nature:

| Class | Observed Behaviour |
|---|---|
| 0 | Relatively neutral language and factual discussion |
| 1 | Hateful and discriminatory language; minority class with strong demographic signal |
| 2 | Ambiguous vocabulary with significant overlap across other classes; majority class |
| 3 | Violent and offensive expressions; smallest class with more distinct but rare patterns |

These characterisations were derived from sampling and linguistic pattern analysis during EDA, not provided as part of the competition. Class 2's broad vocabulary overlap was identified early as the likely primary driver of misclassification — a pattern confirmed across all models.

These constraints made the project particularly useful for studying trade-offs between feature engineering, sparse text representations, and model behaviour under imbalance.

---

## Objectives

The primary goals of the project were:

* Accurately classify comments into predefined categories
* Build representations robust to noisy and inconsistent text
* Maintain strong generalisation under leaderboard evaluation

---

## Exploratory Data Analysis

EDA was used not only for descriptive analysis, but to guide downstream feature engineering and modelling decisions.

### Data Preparation Decisions

**Missing Value Strategy**

The `race`, `religion`, and `gender` columns each contained approximately **73% missing values**. Three approaches were considered:

* Dropping rows — not viable at 73% missingness without severe data loss
* Imputing from existing categories — would introduce distributional bias since the true distribution of missing entries was unknown
* Creating a separate **"Missing"** category — selected, as it preserves the absence of demographic information as a signal in itself

The single missing value in the `comment` column was dropped safely.

The `post_id` column was investigated separately — comments within the same thread still had different labels, meaning thread identity carried no predictive signal and the column was dropped.

### Class Imbalance

The dataset exhibited approximately a **21:1 imbalance ratio** between the majority and minority classes, creating significant challenges for minority-class recall and stable Macro F1 optimisation.

This imbalance motivated:

* Macro F1 as the primary evaluation metric
* Class-weight-aware optimisation
* Precision-Recall analysis instead of ROC-based evaluation

Standard remedies such as oversampling and undersampling were considered but rejected: undersampling would cause significant information loss at 198K rows, while oversampling risked introducing duplicate samples and artificially inflating the training distribution. Class-weight penalisation was used instead.


### Linguistic Patterns by Class

Class-wise text analysis revealed distinct lexical and structural patterns across categories:

* **Class 0** reflected relatively neutral language with factual discussion patterns
* **Class 1** contained hateful and discriminatory language with strong demographic keyword associations
* **Class 2** showed significant vocabulary overlap with other classes, making it the hardest class to separate
* **Class 3** contained violent and offensive expressions with more distinct but rare vocabulary patterns

The overlap between Class 2 and other classes was consistently the most common source of misclassification across all models. These observations motivated the use of:

* Word-level TF-IDF for semantic and contextual signals
* Character-level TF-IDF (`char_wb`) for robustness against misspellings and morphological variation — particularly helpful for Class 3's sparse and inconsistent phrasing
* Additional numerical text statistics such as word count and average word length

### Temporal Patterns

Temporal analysis suggested cyclic activity patterns across comments. To preserve periodic relationships without introducing artificial ordinal assumptions, hour and month features were encoded using sine-cosine transformations.

### Numerical Feature Distributions

Several numerical features showed strong skewness, heavy zero inflation, and long-tailed distributions.

Among numerical features, `if_2` showed the most noticeable median differences across classes, suggesting stronger predictive signal compared to features like `upvote`, `downvote`, and emoticon counts, which showed heavy IQR overlap across all classes.

Categorical features (`race`, `religion`, `gender`) showed a disproportionately strong association with **Class 1** despite it being a minority class — an early signal that demographic context was a meaningful predictor for that specific category.

These observations motivated:

* Yeo-Johnson transformation for skew reduction
* Binary indicator conversion for zero-inflated variables
* Mutual-information-based comparison of raw vs transformed features

### Feature Importance Alignment with EDA

Mutual Information analysis later confirmed many patterns identified during EDA. Features associated with comment length, vocabulary structure, temporal activity patterns, and character-level token variation showed meaningful predictive contribution across several categories.

This alignment between exploratory observations and downstream feature importance provided additional confidence that the engineered features were capturing genuine class-specific behaviour rather than noise.

---

## Visualizations

### Class Distribution

![Class Distribution](https://raw.githubusercontent.com/23f2005144/comment-category-classification/main/Class%20Distribution.png)

### Class-Wise Word Clouds

![Class Wise Wordclouds](https://raw.githubusercontent.com/23f2005144/comment-category-classification/main/Class%20Wise%20Wordclouds.png)

### Model Performance Comparison

![Model Comparison](https://raw.githubusercontent.com/23f2005144/comment-category-classification/main/Model%20Performance%20Comparison.png)

---

## Feature Engineering

Feature engineering played a central role in improving model performance.

### Engineered Numerical Features

The following features were created from the raw data:

* **Comment length** — character count of the comment
* **Word count** — number of words in the comment
* **Average word length** — derived from comment length ÷ word count, capturing whether comments use simpler or more complex vocabulary
* **Hour, month, year** — extracted from the comment timestamp
* **`disability`** — boolean column converted to numeric binary (0/1) for pipeline compatibility

Comment length and word count showed strong positive correlation (both derived from the same source), which was noted but accepted as both carried complementary structural signals.

### Temporal Encoding

Hour and month features were cyclically encoded using sine-cosine transformations. This preserved periodic relationships that would otherwise be distorted under standard ordinal encoding.

### Handling Zero-Inflated Features

Features such as emoticon counts, `if_1`, and `downvote` were heavily zero-inflated. Instead of treating them purely as continuous features, binary indicators were introduced to explicitly model feature presence versus absence, reducing skewness and allowing models to learn meaningful patterns.

### Handling Skewed Features

Two transformation strategies were evaluated on features such as `upvote`, `if_2`, average word length, and word count:

* Log transformation
* Yeo-Johnson transformation

Yeo-Johnson consistently reduced skewness more effectively while remaining stable for zero and negative values, and was therefore retained.

### Feature Selection and Validation

Mutual Information (MI) was used to evaluate feature relevance across both raw and transformed feature sets.

* Raw features pipeline — cross-validated Macro F1: **0.6127**
* Transformed features pipeline — cross-validated Macro F1: **0.6182**

The transformed feature set marginally outperformed and was selected for the final pipeline. TF-IDF features were excluded from MI selection due to their high dimensionality and sparse nature, which can produce unstable importance estimates.

---

## Text Representation

The final representation combined word-level TF-IDF, character-level TF-IDF (`char_wb`), and engineered numerical features.

### Word-Level TF-IDF

Word-level vectorisation was used to capture semantic patterns, contextual word relationships, and class-specific vocabulary usage. Hyperparameters including `ngram_range`, `min_df`, `max_df`, and `sublinear_tf` were systematically tuned.

### Character-Level TF-IDF (`char_wb`)

Character-level TF-IDF improved robustness against informal language, misspellings, morphological variation, and noisy user-generated text. This representation was particularly helpful for minority classes with inconsistent phrasing patterns.

### Final Sparse Representation

The final tuned feature space contained approximately **75,000 sparse TF-IDF features**, combining word-level and character-level representations with tuned vocabulary limits.

### Why Dimensionality Reduction Was Not Applied

Truncated SVD was intentionally not applied. SVD captures global variance across the feature space, which risks diluting the rare but class-specific signals that minority classes (particularly Class 1 and Class 3) depend on. Preserving the full sparse representation was considered more important than reducing dimensionality.

---

## Modeling Strategy

Three model families were explored to compare behaviour under sparse high-dimensional NLP representations.

### Logistic Regression

Used as a strong linear baseline with regularisation tuning, tolerance optimisation, and class-weight balancing.

### LinearSVC

Evaluated for its strong performance on sparse text representations, with hyperparameter focus on regularisation strength and class-weight adjustments.

### LightGBM

Manually tuned across parameter combinations to explore nonlinear interactions, sparse-feature handling, and generalisation behaviour under imbalance.

---

## Model Trade-Off Analysis

| Model | Validation Macro F1 | Leaderboard Macro F1 | Strengths | Limitations |
|---|---|---|---|---|
| Logistic Regression | 0.8273 | 0.8243 | Stable baseline, interpretable | Limited nonlinear learning |
| LinearSVC | 0.8151 | 0.8181 | Strong sparse-feature performance | Reduced probability interpretability |
| LightGBM | **0.8353** | **0.8344** | Captured nonlinear interactions | Greater overfitting sensitivity |

LightGBM achieved the best overall Macro F1 (+0.008 over Logistic Regression on validation) by leveraging both engineered numerical features and nonlinear interactions. The manual tuning table across 5 parameter combinations is documented in the notebook.

Slight overfitting in the final LightGBM model (train F1: 0.9777 vs val F1: 0.8353) was considered acceptable — it improved minority-class recall while the leaderboard score confirmed generalisation was maintained.

---

## Evaluation Strategy

### Primary Metric: Macro F1

Macro F1 was selected because the dataset was highly imbalanced, equal importance was assigned to all classes, and minority-class performance needed explicit weighting.

### Evaluation Components

Models were evaluated using classification reports, confusion matrices, and Precision-Recall curves.

### Why ROC Curves Were Avoided

ROC curves were intentionally not prioritised because the large number of true negatives in imbalanced settings can produce overly optimistic interpretations. Precision-Recall analysis provided more informative insight into minority-class behaviour and precision-recall trade-offs.

---

## Results

| Stage | Macro F1 |
|---|---|
| Raw TF-IDF baseline (Logistic Regression) | 0.7968 |
| + Feature engineering | 0.8137 |
| + Hyperparameter tuning | 0.8273 |
| + `char_wb` TF-IDF + LightGBM (final) | **0.8353** |

### Public Leaderboard Performance

**Macro F1: 0.8344**

The close alignment between validation (0.8353) and leaderboard (0.8344) scores confirmed the pipeline generalised reliably. The progression table above shows that feature engineering contributed more to the final score than switching model families.

---

## Failure Analysis and Observations

### Class 3 — Consistently the Hardest to Classify

Class 3 (violent/offensive) showed the lowest Average Precision across all three models:

| Model | Class 1 AP | Class 3 AP |
|---|---|---|
| Logistic Regression | 0.84 | 0.71 |
| LinearSVC | 0.80 | 0.65 |
| LightGBM | 0.85 | 0.73 |

This gap likely stems from Class 3's combination of small sample size, vocabulary overlap with Class 1, and rare/inconsistent phrasing patterns. Class 2 was the most frequently misclassified overall due to vocabulary overlap across all categories.

### Sparse NLP Trade-Offs

The project highlighted the balance between capturing minority-class patterns, avoiding excessive overfitting, and preserving generalisation under sparse high-dimensional features.

### Feature Representation Insights

Character-level TF-IDF (`char_wb`) proved especially effective for handling informal language, misspellings, and short noisy comments, while engineered numerical features provided complementary structural signals.

---

## Key Insights

* Feature engineering substantially improved overall model performance
* Character-level TF-IDF (`char_wb`) improved robustness against noisy user-generated text
* Feature importance analysis aligned closely with earlier EDA observations
* LightGBM benefited from combining sparse TF-IDF features with engineered numerical features
* Precision-Recall analysis was significantly more informative than ROC analysis under imbalance
* Careful trade-off balancing between minority recall and precision was critical for optimising Macro F1

---

## Lessons Learned

* Character-level TF-IDF (`char_wb`) was especially effective for improving robustness on noisy minority-class samples
* Sparse linear models remained highly competitive despite the availability of more complex nonlinear approaches
* Evaluation metric selection significantly influenced optimisation strategy under heavy class imbalance
* Incremental improvements in feature engineering produced larger gains than aggressively increasing model complexity
* Alignment between EDA observations and downstream feature importance improved confidence in feature design decisions

---

## Repository Structure

```text
.
├── 23f2005144-comment-classification-notebook.ipynb
├── Class Distribution.png
├── Class Wise Wordclouds.png
├── Model Performance Comparison.png
├── README.md
├── requirements.txt
└── submission.csv
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

| Category | Tools |
|---|---|
| Core Language | Python |
| Machine Learning | Scikit-learn, LightGBM |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP / Text Processing | TF-IDF, Regex |
| Statistical Transformations | SciPy |

---

## Future Improvements

* Engineering additional engagement-based features (e.g. interaction signals, user activity patterns) to improve class separability
* Deeper error analysis on Class 3 specifically — understanding whether its poor performance stems from overlapping linguistic patterns, insufficient samples, or feature representation gaps
* Ensemble combinations of linear and boosting models to balance sparse-feature strengths of LinearSVC with LightGBM's nonlinear capacity

---

## Notes for Reviewers

The notebook is structured sequentially and mirrors this README:

| Section | Contents |
|---|---|
| EDA | Data overview, missing value handling, class distribution, text analysis, uni/bi/multivariate analysis |
| Feature Engineering | Binary indicators, sine-cosine encoding, skew transformations, MI comparison |
| Text Representation | TF-IDF word + `char_wb` tuning, feature space construction |
| Modeling | Baseline → tuned runs for all 3 models, with classification reports, confusion matrices, PR curves, and feature importance plots |
| Submission | Final model retrained on full data, all 3 submission scores documented |

The emphasis throughout is on data-driven decision-making — each engineering and modelling choice is motivated by a preceding observation.
