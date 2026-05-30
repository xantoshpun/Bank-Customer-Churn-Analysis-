# 📊 Bank Customer Churn Analysis

**Author:** Humendra Pun  
**Date:** August 2025  
**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn

---

## Overview

End-to-end Python analysis identifying the key drivers of customer churn for a bank with 10,000 customers across France, Germany, and Spain. The project covers the full data science workflow — raw data ingestion, cleaning, EDA, and actionable business recommendations.

**Dataset:** 10,002 rows × 14 columns → cleaned to **10,000 rows × 11 features**  
**Target variable:** `Exited` (1 = churned, 0 = retained)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Overall churn rate | **20.37%** (2,037 customers) |
| Highest churn by country | **Germany — 32.44%** |
| Highest churn by age group | **51–60 years — 56.21%** |
| Avg age of churned customers | **44.84** vs 37.41 (retained) |
| Dataset size after cleaning | **10,000 customers** |

---

## Project Structure

```
├── Bank Customer Churn Analysis.ipynb   # Full analysis notebook
├── Churn_Modelling.csv                  # Raw dataset
├── bank_churn_cleaned.csv               # Cleaned dataset (Phase 1 output)
└── README.md
```

---

## Phase 1: Data Understanding & Preparation 📁

### 🔍 1.1 Initial Data Exploration
- Load the dataset and check dimensions (rows × columns)
- Display first/last rows to understand structure
- Check data types of each column
- Understand what each column represents

<img width="1397" height="370" alt="Screenshot 2025-08-25 233900" src="https://github.com/user-attachments/assets/d0230849-de83-491a-86ba-9bee5dd4101d" />

<img width="1401" height="777" alt="Screenshot 2025-08-25 234011" src="https://github.com/user-attachments/assets/9c169ae6-f6c7-4492-8e75-3cbf73cb3e3a" />

<img width="1392" height="1441" alt="Screenshot 2025-08-25 234443" src="https://github.com/user-attachments/assets/d3c21de7-791b-4f60-9dfb-d8e3eac86ce0" />

---

### 📋 1.2 Data Quality Assessment
- Checked for missing values and their patterns
- Identified duplicates
- Validated data ranges (Age, CreditScore, Tenure, binary columns)

**Issues found:**
- 2 duplicate rows removed
- 4 columns had 1 missing value each → filled with mode/median
- Age had non-integer decimal values → corrected to integer
- CreditScore range valid: 350–850 ✅
- Tenure range valid: 0–10 years ✅

<img width="1384" height="1171" alt="Screenshot 2025-08-25 235039" src="https://github.com/user-attachments/assets/87f279e2-c605-4ba0-847f-0f5e396a8073" />

---

### 🧹 1.3 Data Cleaning

| Step | Action |
|------|--------|
| Duplicates | Removed 2 duplicate rows |
| Missing values | Filled Geography (mode: France), Age (median: 37), HasCrCard & IsActiveMember (mode: 1) |
| Column removal | Dropped RowNumber, CustomerId, Surname (3 irrelevant columns) |
| Data types | Converted HasCrCard, IsActiveMember, Age to integer |
| Outliers | Detected but retained — 15 in CreditScore (0.15%), 359 in Age (3.59%) |

**Final cleaned dataset: 10,000 rows × 11 columns — zero missing values**

![Phase 1 Summary Report](images/phase1/phase1_summary_report.png)

<img width="1388" height="1122" alt="Screenshot 2025-08-25 235349" src="https://github.com/user-attachments/assets/f67b882d-2767-4c17-95a2-68aeb19577af" />

<img width="1386" height="773" alt="Screenshot 2025-08-25 235420" src="https://github.com/user-attachments/assets/ce476c1f-b717-49b9-85e5-f3bae49509c7" />

<img width="1371" height="441" alt="Screenshot 2025-08-25 235759" src="https://github.com/user-attachments/assets/8c739fd7-ebf2-4353-90cf-1f60155dea49" />

---

## Phase 2: Exploratory Data Analysis (EDA) 📈

### 📊 2.1 Univariate Analysis

![Phase 2.1A — Univariate Distributions](images/phase2/univariate/phase2_univariate_a.png)

![Phase 2.1B — Distribution Insights](images/phase2/univariate/phase2_univariate_b.png)

**Customer geography split:**
- France: 5,014 (50.14%) · Germany: 2,509 (25.09%) · Spain: 2,477 (24.77%)

**Age distribution:**
- 18–30: 19.7% · 31–40: 44.5% · 41–50: 23.2% · 51–60: 8.0% · 60+: 4.6%

**Balance:** 36.16% of customers have zero balance

**Target variable:**
- Retained: 7,963 (79.63%)
- Churned: 2,037 (**20.37%**)

---

### 🔗 2.2 Bivariate Analysis

![Phase 2.2A — Churn by Category](images/phase2/bivariate/phase2_bivariate_a_churn_by_category.png)

#### Geography vs Churn

| Country | Customers | Churned | Churn Rate |
|---------|-----------|---------|------------|
| France | 5,014 | 810 | 16.15% |
| Germany | 2,509 | 814 | **32.44%** |
| Spain | 2,477 | 413 | 16.67% |

> Germany has double the churn rate of France and Spain despite having fewer customers.

---

#### Gender vs Churn
- Female customers churn at **25.1%** vs males at ~16%

---

#### Age Groups vs Churn

| Age Group | Customers | Churn Rate |
|-----------|-----------|------------|
| 18–30 | 1,967 | 7.52% |
| 31–40 | 4,452 | 12.08% |
| 41–50 | 2,320 | **33.97%** |
| 51–60 | 797 | **56.21%** |
| 60+ | 464 | 24.78% |

> Churn rate peaks in the 51–60 age group at 56.21%. Customers aged 41+ are the highest-risk segments.

---

![Phase 2.2B — Churn by Numeric Features](images/phase2/bivariate/phase2_bivariate_b_churn_by_numeric.png)

#### Churned vs Retained — Mean Comparison

| Feature | Churned | Retained | Difference |
|---------|---------|----------|------------|
| Age | 44.84 | 37.41 | +7.43 (+19.9%) |
| CreditScore | 645.35 | 651.85 | -6.50 (-1.0%) |

> Age is a far stronger churn predictor than CreditScore.

---

![Phase 2.2C — Age & Balance vs Churn](images/phase2/bivariate/phase2_bivariate_c_age_balance.png)

#### Number of Products vs Churn
- 1 product: 50.84% of customer base — significantly higher churn
- 2 products: 45.90% — lower churn, highest retention
- 3–4 products: small segment but disproportionately high churn rates

---

### 🎯 2.3 Correlation Analysis

![Phase 2.3 — Correlation Matrix](images/phase2/correlation/phase2_correlation_matrix.png)

- **Age** — strongest positive correlation with churn
- **IsActiveMember** — negative correlation; inactive members churn more
- **NumOfProducts** — non-linear relationship with churn
- **CreditScore & EstimatedSalary** — minimal correlation with churn

---

## Key Findings

1. **Germany is a critical risk market** — 32.44% churn vs ~16% in France and Spain
2. **Middle-to-older customers (41–60) are the highest risk** — 51–60 age band peaks at 56.21%
3. **Female customers churn at 25.1%** vs ~16% for males
4. **Inactive members churn at 26.9%** vs active members — engagement directly reduces attrition
5. **CreditScore is a weak predictor** — only 6.5 point difference between churned and retained
6. **Customers with 3–4 products have extremely high churn** — despite being a small segment (3.26%)
7. **Zero-balance customers churn less** — customers with money in the account have higher stickiness
8. **Males churn at 16.5%** vs females at 25.1% — 8.6 percentage point gap

---

## Business Recommendations

| Priority | Recommendation |
|----------|---------------|
| 🔴 High | Targeted retention campaigns in Germany with localised, personalised offers |
| 🔴 High | Age-segmented loyalty programmes for customers aged 41–60 |
| 🟡 Medium | Gender-specific engagement strategies for female customer segments |
| 🟡 Medium | Cross-sell products to single-product customers to increase stickiness |
| 🟡 Medium | Review product bundling strategy — 3–4 product customers show disproportionate churn |
| 🟢 Low | Re-engagement campaigns for inactive members (26.9% churn rate) |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations, outlier detection |
| `matplotlib` | Custom visualisations |
| `seaborn` | Statistical plots |
| `scikit-learn` | Pipeline ready for modelling phase |
