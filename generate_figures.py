"""
Runs the full Bank Customer Churn Analysis and saves figures to images/
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Output dirs ────────────────────────────────────────────
os.makedirs('images/phase1', exist_ok=True)
os.makedirs('images/phase2/univariate', exist_ok=True)
os.makedirs('images/phase2/bivariate', exist_ok=True)
os.makedirs('images/phase2/correlation', exist_ok=True)

# ── Load & clean ────────────────────────────────────────────
df = pd.read_csv('Churn_Modelling.csv')
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
for col in ['Geography', 'HasCrCard', 'IsActiveMember']:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['HasCrCard'] = df_clean['HasCrCard'].astype(float).round().astype(int)
df_clean['IsActiveMember'] = df_clean['IsActiveMember'].astype(float).round().astype(int)
df_clean['Age'] = df_clean['Age'].astype(float).round().astype(int)

print(f"Cleaned: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
print(f"Churn rate: {df_clean['Exited'].mean()*100:.2f}%")

# ── PHASE 1 SUMMARY REPORT ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHASE 1: INITIAL SUMMARY REPORT', fontsize=20, fontweight='bold')

# 1. Data completeness
ax = axes[0, 0]
completeness = (1 - df_clean.isnull().sum() / len(df_clean)) * 100
ax.barh(completeness.index, completeness.values, color='#19e6ce')
ax.set_xlim(0, 105)
ax.set_title('Data Completeness (%)')
ax.axvline(100, color='gray', linestyle='--', alpha=0.5)

# 2. Churn distribution
ax = axes[0, 1]
churn_counts = df_clean['Exited'].value_counts()
ax.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
       colors=['#19e6ce', '#7c3aed'], startangle=90)
ax.set_title('Churn Distribution')

# 3. Geography distribution
ax = axes[1, 0]
geo = df_clean['Geography'].value_counts()
ax.bar(geo.index, geo.values, color=['#19e6ce', '#7c3aed', '#10b981'])
ax.set_title('Customers by Geography')
for i, v in enumerate(geo.values):
    ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

# 4. Cleaned dataset summary
ax = axes[1, 1]
ax.axis('off')
summary = [
    ['Original rows', f'{len(df):,}'],
    ['Cleaned rows', f'{len(df_clean):,}'],
    ['Rows removed', f'{len(df) - len(df_clean):,}'],
    ['Features', f'{df_clean.shape[1]}'],
    ['Missing values', '0'],
    ['Churn rate', f'{df_clean["Exited"].mean()*100:.2f}%'],
    ['Churned customers', f'{df_clean["Exited"].sum():,}'],
    ['Retained customers', f'{(df_clean["Exited"]==0).sum():,}'],
]
table = ax.table(cellText=summary, colLabels=['Metric', 'Value'],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)
ax.set_title('Dataset Summary')

plt.tight_layout()
plt.savefig('images/phase1/phase1_summary_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase1_summary_report.png")

# ── PHASE 2.1A — UNIVARIATE NUMERICAL ───────────────────────
numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts']
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('PHASE 2.1.A: UNIVARIATE ANALYSIS — NUMERICAL DISTRIBUTIONS', fontsize=18, fontweight='bold')

for i, col in enumerate(numerical_cols):
    ax_hist = axes[i // 2][i % 2 * 2]
    ax_box  = axes[i // 2][i % 2 * 2 + 1]
    sns.histplot(df_clean[col], ax=ax_hist, kde=True, color='#19e6ce')
    ax_hist.set_title(f'{col} — Distribution')
    sns.boxplot(y=df_clean[col], ax=ax_box, color='#7c3aed')
    ax_box.set_title(f'{col} — Boxplot')

plt.tight_layout()
plt.savefig('images/phase2/univariate/phase2_univariate_a.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_univariate_a.png")

# ── PHASE 2.1B — UNIVARIATE CATEGORICAL + AGE GROUPS ────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('PHASE 2.1.B: UNIVARIATE ANALYSIS — CATEGORICAL & DISTRIBUTION INSIGHTS', fontsize=16, fontweight='bold')

cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
for i, col in enumerate(cat_cols):
    ax = axes[i // 3][i % 3]
    vc = df_clean[col].value_counts()
    ax.bar([str(x) for x in vc.index], vc.values, color='#19e6ce')
    ax.set_title(f'{col}')
    for j, v in enumerate(vc.values):
        ax.text(j, v + 30, f'{v:,}\n({v/len(df_clean)*100:.1f}%)', ha='center', fontsize=9)

# Age groups
ax = axes[1][2]
df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0,30,40,50,60,100],
                               labels=['18-30','31-40','41-50','51-60','60+'])
ag = df_clean['AgeGroup'].value_counts().sort_index()
ax.bar(ag.index, ag.values, color='#10b981')
ax.set_title('Age Groups')
for j, v in enumerate(ag.values):
    ax.text(j, v + 30, f'{v:,}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/phase2/univariate/phase2_univariate_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_univariate_b.png")

# ── PHASE 2.2A — BIVARIATE CATEGORICAL vs CHURN ─────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHASE 2.2.A: BIVARIATE ANALYSIS — CHURN BY CATEGORICAL FEATURES', fontsize=16, fontweight='bold')

# Geography
ax = axes[0, 0]
geo_churn = df_clean.groupby('Geography')['Exited'].mean() * 100
bars = ax.bar(geo_churn.index, geo_churn.values, color=['#19e6ce','#7c3aed','#10b981'])
ax.set_title('Churn Rate by Geography (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, geo_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Gender
ax = axes[0, 1]
gen_churn = df_clean.groupby('Gender')['Exited'].mean() * 100
bars = ax.bar(gen_churn.index, gen_churn.values, color=['#19e6ce','#7c3aed'])
ax.set_title('Churn Rate by Gender (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, gen_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Active Member
ax = axes[1, 0]
act_churn = df_clean.groupby('IsActiveMember')['Exited'].mean() * 100
bars = ax.bar(['Inactive', 'Active'], act_churn.values, color=['#ef4444','#10b981'])
ax.set_title('Churn Rate by Member Activity (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, act_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Num of Products
ax = axes[1, 1]
prod_churn = df_clean.groupby('NumOfProducts')['Exited'].mean() * 100
bars = ax.bar([str(x) for x in prod_churn.index], prod_churn.values, color='#f59e0b')
ax.set_title('Churn Rate by Number of Products (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, prod_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/phase2/bivariate/phase2_bivariate_a_churn_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_bivariate_a_churn_by_category.png")

# ── PHASE 2.2B — BIVARIATE NUMERICAL vs CHURN ───────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHASE 2.2.B: BIVARIATE ANALYSIS — NUMERICAL FEATURES vs CHURN', fontsize=16, fontweight='bold')

num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
for i, col in enumerate(num_cols):
    ax = axes[i // 2][i % 2]
    churned  = df_clean[df_clean['Exited']==1][col]
    retained = df_clean[df_clean['Exited']==0][col]
    ax.hist(retained, bins=40, alpha=0.6, label=f'Retained (μ={retained.mean():.0f})', color='#19e6ce')
    ax.hist(churned,  bins=40, alpha=0.6, label=f'Churned (μ={churned.mean():.0f})',  color='#7c3aed')
    ax.set_title(f'{col} — Churned vs Retained')
    ax.legend()

plt.tight_layout()
plt.savefig('images/phase2/bivariate/phase2_bivariate_b_churn_by_numeric.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_bivariate_b_churn_by_numeric.png")

# ── PHASE 2.2C — AGE GROUPS & BALANCE vs CHURN ──────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('PHASE 2.2.C: BIVARIATE ANALYSIS — AGE GROUPS & BALANCE vs CHURN', fontsize=16, fontweight='bold')

# Age groups churn rate
ax = axes[0]
age_churn = df_clean.groupby('AgeGroup', observed=True)['Exited'].mean() * 100
bars = ax.bar(age_churn.index, age_churn.values,
              color=['#10b981','#19e6ce','#f59e0b','#ef4444','#7c3aed'])
ax.set_title('Churn Rate by Age Group (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, age_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Balance vs Churn (zero vs non-zero)
ax = axes[1]
df_clean['BalanceBucket'] = df_clean['Balance'].apply(lambda x: 'Zero Balance' if x == 0 else 'Has Balance')
bal_churn = df_clean.groupby('BalanceBucket')['Exited'].mean() * 100
bars = ax.bar(bal_churn.index, bal_churn.values, color=['#19e6ce','#7c3aed'])
ax.set_title('Churn Rate by Balance Status (%)')
ax.set_ylabel('Churn Rate (%)')
for bar, v in zip(bars, bal_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/phase2/bivariate/phase2_bivariate_c_age_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_bivariate_c_age_balance.png")

# ── PHASE 2.3 — CORRELATION MATRIX ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('PHASE 2.3: CORRELATION & STATISTICAL ANALYSIS', fontsize=18, fontweight='bold')

# Correlation heatmap
ax = axes[0]
num_df = df_clean[['CreditScore','Age','Tenure','Balance','NumOfProducts',
                   'HasCrCard','IsActiveMember','EstimatedSalary','Exited']]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Matrix')

# Churn correlation bar chart
ax = axes[1]
churn_corr = corr['Exited'].drop('Exited').sort_values()
colors = ['#ef4444' if v > 0 else '#10b981' for v in churn_corr.values]
ax.barh(churn_corr.index, churn_corr.values, color=colors)
ax.axvline(0, color='white', linewidth=0.8)
ax.set_title('Feature Correlation with Churn')
ax.set_xlabel('Pearson Correlation')
for i, (idx, v) in enumerate(churn_corr.items()):
    ax.text(v + (0.002 if v >= 0 else -0.002), i, f'{v:.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('images/phase2/correlation/phase2_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase2_correlation_matrix.png")

print("\nAll figures generated successfully!")
