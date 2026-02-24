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
## Gender Analysis
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

### Interpretation
The gender-based analysis shows a clear difference in default behavior. Female applicants account for 202,448 observations with a default rate of 6.999%, which is below the overall portfolio benchmark of 8.07%. In contrast, male applicants (105,059 observations) exhibit a higher default rate of 10.142%, exceeding the benchmark. This indicates that male applicants carry a moderately higher default risk compared to female applicants.

## Age Analysis
```python
age_default = application.groupby(pd.cut(application['AGE'], 5))['TARGET'].mean().reset_index()
sns.barplot(x='AGE', y='TARGET', data=age_default)
```
<img width="1304" height="878" alt="image" src="https://github.com/user-attachments/assets/a4cd2f16-c68a-417f-bd9a-623c619b50a4" />

### Interpretation
The age-based analysis shows a clear downward trend in default risk as applicants get older. Applicants in the youngest age band (roughly 20–30) have the highest default rate (11–12%), and the rate steadily decreases across subsequent age groups—around 9–10% for 30–40, 7–8% for 40–50, 6% for 50–60, and 5% for 60–70. Overall, this suggests that younger borrowers carry materially higher default risk, while older applicants tend to be more reliable in repayment.

## Education Degree
```python
edu_default = application.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
sns.barplot(x='TARGET', y='NAME_EDUCATION_TYPE', data=edu_default)
```
<img width="1590" height="864" alt="image" src="https://github.com/user-attachments/assets/10c68316-c939-49fd-befe-8acd82eb5165" />

### Interpretation
The chart indicates that applicants with lower education levels tend to have higher default rates, while default rates decrease as education level increases.
## Credit Amount
```python
credit_tmp = application[["AMT_CREDIT", "TARGET"]].dropna().copy()
credit_tmp["CREDIT_BIN"] = pd.qcut(credit_tmp["AMT_CREDIT"], q=10, duplicates="drop")
credit_default = credit_tmp.groupby("CREDIT_BIN")["TARGET"].agg(["count", "mean"]).reset_index()
credit_default = credit_default.rename(columns={"count": "customers", "mean": "default_rate"})
credit_default["default_rate"] = credit_default["default_rate"] * 100
display(credit_default)

sns.lineplot(data=credit_default, x=credit_default.index, y="default_rate", marker="o")
plt.title("Default Rate by AMT_CREDIT (Binned)")
plt.ylabel("Default Rate (%)")
plt.xlabel("Credit Amount Bin (low → high)")
```
<img width="1204" height="926" alt="image" src="https://github.com/user-attachments/assets/f7138ab4-d577-416c-b86e-105e357ca690" />

### Interpretation
The binned credit-amount analysis shows a non-linear relationship: default rates rise from low to mid-range loan amounts (peaking around the middle bins) and then decline for the highest credit amounts, suggesting mid-sized loans carry higher default risk than very small or very large loans.
##  EXT_SOURCE_3 (Credit Score Analysis)
```python
sns.kdeplot(data=application, x='EXT_SOURCE_3', hue='TARGET', fill=True)
plt.title("Distribution of EXT_SOURCE_3 by TARGET")
```
<img width="1220" height="908" alt="image" src="https://github.com/user-attachments/assets/6737fabe-eb3a-48aa-a5eb-cac86c0df75c" />

The figure below shows the distribution of `EXT_SOURCE_3` separated by default status (`TARGET`).

- Blue curve (TARGET = 0): Non-default clients  
- Orange curve (TARGET = 1): Default clients  
- X-axis: External credit score  
- Y-axis: Probability density  

### Key Findings

There is a clear separation between the two distributions:

- Defaulted applicants are concentrated at **lower EXT_SOURCE_3 values**
- Non-defaulted applicants are concentrated at **higher EXT_SOURCE_3 values**

### Interpretation

The limited overlap between the two curves suggests that `EXT_SOURCE_3` has strong discriminatory power. As the credit score increases, the probability of default declines significantly.

Compared to demographic variables such as gender, `EXT_SOURCE_3` demonstrates substantially stronger predictive signal and is likely one of the primary drivers of default behavior in this dataset.

## Income Level
```python
income_tmp = application[["AMT_INCOME_TOTAL", "TARGET"]].dropna().copy()
income_tmp["INCOME_BIN"] = pd.qcut(income_tmp["AMT_INCOME_TOTAL"], q=10, duplicates="drop")

income_default_num = income_tmp.groupby("INCOME_BIN")["TARGET"].agg(["count", "mean"]).reset_index()
income_default_num = income_default_num.rename(columns={"count": "customers", "mean": "default_rate"})
income_default_num["default_rate"] = income_default_num["default_rate"] * 100

display(income_default_num)

sns.lineplot(data=income_default_num, x=income_default_num.index, y="default_rate", marker="o")
plt.title("Default Rate by AMT_INCOME_TOTAL (Binned)")
plt.ylabel("Default Rate (%)")
plt.xlabel("Income Bin (low → high)")
```
```
INCOME_BIN	               customers	default_rate
0	(25649.999, 81000.0]	33391	8.190830
1	(81000.0, 99000.0]	    30280	8.223250
2	(99000.0, 112500.0]	    36907	8.719213
3	(112500.0, 135000.0]	48849	8.489427
4	(135000.0, 147150.0]	4333	9.000692
5	(147150.0, 162000.0]	31120	8.640746
6	(162000.0, 180000.0]	30704	8.422355
7	(180000.0, 225000.0]	44809	7.806467
8	(225000.0, 270000.0]	19957	7.040136
9	(270000.0, 117000000.0]	27161	6.137477
```
<img width="1208" height="904" alt="image" src="https://github.com/user-attachments/assets/733ed8cb-613f-4c78-8a47-61877a4325c2" />

### Interpretation
The binned income analysis shows that default rates generally decrease as total income increases, with the highest default risk concentrated in the lower-to-mid income bins and the lowest default rates observed in the highest income bins.
## Previous application status
```python
refused_only = previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]

refused_count = refused_only.groupby("SK_ID_CURR").size().reset_index(name="PREV_REFUSED_COUNT")

app_prev = application.merge(refused_count, on="SK_ID_CURR", how="left")
app_prev["PREV_REFUSED_COUNT"] = app_prev["PREV_REFUSED_COUNT"].fillna(0)
refused_tmp = app_prev[["PREV_REFUSED_COUNT", "TARGET"]].dropna().copy()
refused_tmp["REFUSED_BIN"] = pd.qcut(refused_tmp["PREV_REFUSED_COUNT"], q=6, duplicates="drop")

refused_default = refused_tmp.groupby("REFUSED_BIN")["TARGET"].agg(["count", "mean"]).reset_index()
refused_default = refused_default.rename(columns={"count": "customers", "mean": "default_rate"})
refused_default["default_rate"] = refused_default["default_rate"] * 100

display(refused_default)

sns.lineplot(data=refused_default, x=refused_default.index, y="default_rate", marker="o")
plt.title("Default Rate by Previous Refused Count (Binned)")
plt.ylabel("Default Rate (%)")
plt.xlabel("Previous Refused Count Bin")
```
```
REFUSED_BIN	        customers	default_rate
0	(-0.001, 2.0]	  276490	  7.563384
1	(2.0, 68.0]	      31021	      12.614036
```
<img width="1156" height="904" alt="image" src="https://github.com/user-attachments/assets/87f86f2d-ffc2-4730-92c7-be6f8ca0673e" />

### Interpretation
The previous-application analysis shows that applicants with a higher count of previously refused applications have a higher default rate, indicating that prior refusals are associated with high default risk.
## Correlation with TARGET (Linear Relationship)
```python
target_hist_corr = hist_corr_matrix[["TARGET"]].sort_values("TARGET", ascending=False)

plt.figure(figsize=(6, 5))
sns.heatmap(target_hist_corr,annot=True,cmap="BuGn",center=0,fmt=".2f")
plt.title("Correlation with TARGET (Including Historical Behavior)")
```
<img width="1356" height="904" alt="image" src="https://github.com/user-attachments/assets/d0c905cf-994b-4b19-940c-f41f55a3cb5e" />

### Interpretation
We computed correlation coefficients between selected variables and TARGET.
The strongest linear relationship is observed for EXT_SOURCE_2 (-0.16), confirming that external credit scores are important predictors of default risk.

However, most other variables show weak linear correlation. This suggests that default risk is driven more by nonlinear patterns and interactions rather than simple linear relationships.

Therefore, correlation analysis is used as a supplementary check, while binning analysis and interaction heatmaps provide more meaningful business insights.


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

### Interpretation
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

### Interpretation
This figure is a heatmap showing default rates across combinations of: Age bins (rows) and EXT_SOURCE_2 bins (columns)
   
Each cell represents the default rate for that subgroup.

The interaction between Age and External Risk Score (EXT_SOURCE) reveals a clear amplification effect in default risk. While both age and credit score are individually strong predictors, their combined impact is not merely additive. Younger applicants already exhibit higher baseline risk, and low external credit scores independently indicate elevated default probability. However, when these two risk factors occur together — young age combined with low EXT score — the default rate increases disproportionately, reaching nearly 18–19%, far above the portfolio benchmark. Conversely, older applicants with high external scores form an extremely low-risk segment, with default rates dropping to approximately 2–3%. This interaction suggests that effective credit risk segmentation must consider multiple variables jointly rather than relying on single-factor analysis. 
