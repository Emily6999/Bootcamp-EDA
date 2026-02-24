# Bootcamp-EDA 
Data bootcamp's EDA Activity

## Introduction
The objective of this analysis is to determine whether a client is likely to have payment difficulties and to identify the characteristics that increase default risk.

Specifically, we aim to:

1. Examine consumer attributes (e.g., age, income, employment type)

2. Analyze credit-related features (e.g., external credit score, loan amount)

3. Investigate historical behavior (e.g., previous application outcomes)

4. Identify strong driver variables associated with higher default probability

5. Detect meaningful interaction effects between key variables

## Baseline Default Rate
Before analyzing risk factors, we first establish the overall benchmark default rate of the portfolio.
```python
# Overall default rate
default_rate = application['TARGET'].mean()
# Total observations
total_count = application.shape[0]
# Default count
default_count = application['TARGET'].sum()
```
Output:

```
Total applications: 307511
Number of defaults: 24825
Overall default rate: 8.0729%
```
## 1. Gender Analysis
```python
gender_df = application[application['CODE_GENDER'] != 'XNA']
gender_analysis = (
    gender_df
    .groupby('CODE_GENDER')
    .agg(
        count=('TARGET', 'count'),
        default_rate=('TARGET', 'mean')
    )
    .reset_index()
)
sns.barplot(
    data=gender_analysis,
    x='CODE_GENDER',
    y='default_rate'
)

plt.axhline(
    application['TARGET'].mean(),
    color='red',
    linestyle='--',
    label='Overall Default Rate'
)

plt.title('Default Rate by Gender')
plt.ylabel('Default Rate')
```

<img width="1184" height="792" alt="image" src="https://github.com/user-attachments/assets/ead23507-8eb3-4ac3-a690-5a48667b0b79" />
Interpretation: The gender-based analysis shows a clear difference in default behavior. Female applicants account for 202,448 observations with a default rate of 6.999%, which is below the overall portfolio benchmark of 8.07%. In contrast, male applicants (105,059 observations) exhibit a higher default rate of 10.142%, exceeding the benchmark. This indicates that male applicants carry a moderately higher default risk compared to female applicants.

## 2. Income Type Risk Analysis
Motivation： Income type may reflect employment stability and economic background. However, categorical variables often suffer from severe class imbalance, which may distort default rate estimation.
Therefore, we will analyze both the sample size and default rate.
```python
income_stats = (
    application
    .groupby('NAME_INCOME_TYPE')['TARGET']
    .agg(['count', 'mean'])
    .reset_index()
)

income_stats = income_stats.sort_values('count', ascending=False)
fig, ax1 = plt.subplots()
# 左轴：样本量
sns.barplot(
    data=income_stats,
    x='count',
    y='NAME_INCOME_TYPE',
    color='lightgray',
    ax=ax1
)
ax1.set_xlabel('Count')

# 右轴：违约率
ax2 = ax1.twiny()
sns.scatterplot(
    data=income_stats,
    x='mean',
    y='NAME_INCOME_TYPE',
    color='red',
    s=100,
    ax=ax2
)
ax2.set_xlabel('Default Rate')
```
<img width="1558" height="948" alt="image" src="https://github.com/user-attachments/assets/69b2dde6-ae30-4af2-ae62-0a708b6085ab" />
This visualization presents both the sample size and the default rate for each income category. The gray horizontal bars represent the number of observations (count) within each income type, while the red dots indicate the corresponding default rate (mean of TARGET). Plotting both metrics together is essential because categorical variables often suffer from severe class imbalance. A category with only a handful of observations may exhibit an extremely high default rate simply due to random variation rather than genuine risk. For example, if a group contains only five borrowers and two of them default, the default rate would appear to be 40%, which may not be statistically reliable. By comparing the length of the gray bars with the position of the red dots, we can assess whether a high default rate is supported by sufficient sample size. In this case, large categories such as “Working,” “Commercial associate,” and “Pensioner” provide stable and interpretable estimates, whereas very small groups like “Maternity leave” or “Student” show volatile default rates that are likely driven by statistical noise rather than true underlying risk.

## Interaction Effect: Age × External Risk Score
```python
application['AGE'] = -application['DAYS_BIRTH'] / 365
application['age_bin'] = pd.qcut(application['AGE'],5,duplicates='drop')
application['ext2_bin'] = pd.qcut(application['EXT_SOURCE_2'],5,duplicates='drop')

pivot_age_ext2 = application.pivot_table(values='TARGET',index='age_bin',columns='ext2_bin',aggfunc='mean'）
sns.heatmap(pivot_age_ext2, annot=True, fmt=".3f")
```
<img width="1344" height="1134" alt="image" src="https://github.com/user-attachments/assets/07044424-ce86-4ca1-9f8d-cce1b2433ef2" />
The interaction between Age and External Risk Score (EXT_SOURCE) reveals a clear amplification effect in default risk. While both age and credit score are individually strong predictors, their combined impact is not merely additive. Younger applicants already exhibit higher baseline risk, and low external credit scores independently indicate elevated default probability. However, when these two risk factors occur together — young age combined with low EXT score — the default rate increases disproportionately, reaching nearly 18–19%, far above the portfolio benchmark. Conversely, older applicants with high external scores form an extremely low-risk segment, with default rates dropping to approximately 2–3%. This interaction suggests that effective credit risk segmentation must consider multiple variables jointly rather than relying on single-factor analysis. 
