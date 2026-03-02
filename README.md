# Bootcamp-EDA 
Data bootcamp's EDA Activity
- Siye Li sl10962
- Yuxin Bai yb2464
- Lylian Li yl11924

## Introduction
The objective of this analysis is to determine whether a client is likely to have payment difficulties and to identify the characteristics that increase default risk.

Specifically, we aim to:

1. Examine consumer attributes (e.g., age, income, employment type)

2. Analyze credit-related features (e.g., external credit score, loan amount)

3. Investigate historical behavior (e.g., previous application outcomes)

4. Identify strong driver variables associated with higher default probability

5. Detect meaningful interaction effects between key variables

## Brief Explanation of Variable Name
Since some variable names from the Excel are a bit confusing, here are some brief explanation of mentioned variable names

1. TARGET -> whether the applicant defaulted before (1=default / payment difficulty, 0=non-default)
2. NAME_EDUCATION_TYPE-> education level
3. AMT_INCOME_TOTAL -> reveal each applicant's income condition (total income)
4. AMT_CREDIT-> requested loan/credit amount
5. EXT_SOURCE 1 /EXT_SOURCE 2 / EXT_SOURcE 3 -> external risk score-like variables
6. SK_ID_CURR -> current customer ID (used to connect to application data)

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

<img width="1138" height="898" alt="image" src="https://github.com/user-attachments/assets/0e638f10-851d-40ee-b057-dce4f1c652f9" />

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

## Loan Contract Type
```python
contract_tmp = application[["NAME_CONTRACT_TYPE", "TARGET"]].dropna().copy()
contract_default = (
    contract_tmp.groupby("NAME_CONTRACT_TYPE")["TARGET"]
    .agg(["count", "mean"])
    .reset_index()
)
contract_default = contract_default.rename(
    columns={"count": "customers", "mean": "default_rate"}
)
contract_default["default_rate"] = contract_default["default_rate"] * 100
display(contract_default)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=contract_default,
    x="NAME_CONTRACT_TYPE",
    y="default_rate",
    palette="Blues_r"
)

plt.title("Default Rate by Loan Contract Type")
plt.xlabel("Loan Contract Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=25)
```
```
NAME_CONTRACT_TYPE	customers	default_rate
0	Cash loans	      278232	8.345913
1	Revolving loans	   29279	5.478329
```
<img width="1366" height="956" alt="image" src="https://github.com/user-attachments/assets/a235f852-a291-46ba-9446-3d873e9a22d5" />


### Intepretattion
Cash loans exhibit a higher default rate (8.35%) compared to revolving loans (5.48%).

Although revolving loans represent a smaller portion of applications, their borrowers demonstrate lower default risk.

Given that cash loans account for the majority of the portfolio, they contribute most significantly to overall credit risk exposure.

## Occupation Type
To avoid instability from very small groups, occupations with fewer than 1,000 customers are excluded.
We then rank occupations by default rate.
```python
# --- Default Rate by Occupation Type ---
occ_tmp = application[["OCCUPATION_TYPE", "TARGET"]].dropna().copy()
occ_default = (
    occ_tmp.groupby("OCCUPATION_TYPE")["TARGET"]
    .agg(["count", "mean"])
    .reset_index()
)

occ_default = occ_default.rename(
    columns={"count": "customers", "mean": "default_rate"}
)
occ_default["default_rate"] = occ_default["default_rate"] * 100
occ_default = occ_default[occ_default["customers"] > 1000]
occ_default = occ_default.sort_values("default_rate", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=occ_default,
    x="OCCUPATION_TYPE",
    y="default_rate",
    palette="Reds_r"
)

plt.title("Default Rate by Occupation Type")
plt.xlabel("Occupation Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=60, ha="right")
```
<img width="2092" height="1314" alt="image" src="https://github.com/user-attachments/assets/f1ce474f-6eac-4bf8-b418-58920f639aab" />

### Key Findings
- Low-skill laborers exhibit the highest default rate (~17%).

- Drivers, waiters/barmen staff, and security staff also show elevated default risk (around 10–11%).

- In contrast, Accountants, IT staff, Managers, and Core staff have significantly lower default rates (around 5–6%).

- There is a clear gradient from lower-skilled/manual occupations to more professional/technical occupations.
  
### Interpretation
Occupation type appears to be a strong socioeconomic risk driver.

Higher default rates are concentrated among:
- Lower-income or physically intensive occupations
- Jobs with potentially unstable income patterns

Lower default rates are observed among:
- Professional roles
- Technical staff
- Managerial occupations

From a risk modeling perspective, occupation type provides meaningful segmentation power and may help improve risk discrimination when combined with income and credit score variables.


## Family status
```python
# --- Default Rate by Family Status ---
family_tmp = application[["NAME_FAMILY_STATUS", "TARGET"]].dropna().copy()
family_default = (
    family_tmp.groupby("NAME_FAMILY_STATUS")["TARGET"]
    .agg(["count", "mean"])
    .reset_index()
)

family_default = family_default.rename(
    columns={"count": "customers", "mean": "default_rate"}
)
family_default["default_rate"] = family_default["default_rate"] * 100
family_default = family_default.sort_values("default_rate", ascending=False)

display(family_default)


plt.figure(figsize=(10, 5))
sns.barplot(
    data=family_default,
    x="NAME_FAMILY_STATUS",
    y="default_rate",
    palette="Reds"
)

plt.title("Default Rate by Family Status")
plt.xlabel("Family Status")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")
```
<img width="1756" height="1146" alt="image" src="https://github.com/user-attachments/assets/973109f7-1212-4829-9159-8f7a391eb167" />

#### Key Findings:

- **Civil marriage** and **Single / not married** applicants exhibit the highest default rates (around 10%).
- **Separated** applicants also show elevated risk (~8%).
- **Married** applicants have a comparatively lower default rate (~7–8%).
- **Widowed** applicants display the lowest observed default rate (below 6%).

#### Interpretation:

The results suggest that marital stability may be associated with repayment behavior. Married applicants may benefit from dual income support or stronger financial stability, reducing credit risk. 

In contrast, single or civil marriage applicants may face higher financial volatility, potentially increasing repayment risk.

This highlights family status as a potentially informative segmentation variable in credit risk modeling.

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

## Previous Application Count
```python
prev_count_tmp = app_hist[["PREV_APP_COUNT", "TARGET"]].dropna().copy()
prev_count_tmp["PREV_APP_COUNT_BIN"] = pd.qcut(prev_count_tmp["PREV_APP_COUNT"], q=6, duplicates="drop")

prev_count_default = prev_count_tmp.groupby("PREV_APP_COUNT_BIN")["TARGET"].agg(["count", "mean"]).reset_index()
prev_count_default = prev_count_default.rename(columns={"count": "customers", "mean": "default_rate"})
prev_count_default["default_rate"] = prev_count_default["default_rate"] * 100

display(prev_count_default)

sns.lineplot(data=prev_count_default, x=prev_count_default.index, y="default_rate", marker="o")
plt.title("Default Rate by Previous Application Count (Binned)")
plt.xlabel("PREV_APP_COUNT Bin (low → high)")
plt.ylabel("Default Rate (%)")
```
```
PREV_APP_COUNT_BIN	customers	default_rate
0	(0.999, 2.0]	98332	     8.134687
1	(2.0, 4.0]	    72936	     7.699901
2	(4.0, 5.0]	    26638	     7.875967
3	(5.0, 8.0]	    50242	     8.134628
4	(8.0, 73.0]	    42909	     9.426927
```
<img width="1228" height="900" alt="image" src="https://github.com/user-attachments/assets/648ede99-b299-4c73-89c2-ed51562a1141" />

### Interpretation
We analyzed the relationship between the number of previous loan applications and current default risk.
While the overall correlation is weak, customers in the highest bin of previous applications show a noticeably higher default rate (~9.4%). This suggests that repeated loan-seeking behavior may indicate financial stress or credit dependence. However, the relationship is not strictly linear. 

## Correlation with TARGET (Linear Relationship)
```python
target_hist_corr = hist_corr_matrix[["TARGET"]].sort_values("TARGET", ascending=False)

plt.figure(figsize=(6, 5))
sns.heatmap(target_hist_corr,annot=True,cmap="BuGn",center=0,fmt=".2f")
plt.title("Correlation with TARGET (Including Historical Behavior)")
```

<img width="1326" height="400" alt="image" src="https://github.com/user-attachments/assets/12edbc22-c989-4986-b3ed-9a352bcf763d" />

<img width="1548" height="1360" alt="image" src="https://github.com/user-attachments/assets/8993d578-8b5a-4940-8604-80da82257754" />

### Interpretation
We computed correlation coefficients between selected variables and TARGET.
The strongest linear relationship is observed for EXT_SOURCE_1 2 3, confirming that external credit scores are important predictors of default risk.

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
sns.barplot(
    data=income_stats,
    x='count',
    y='NAME_INCOME_TYPE',
    color='lightgray',
    ax=ax1
)
ax1.set_xlabel('Count')
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
