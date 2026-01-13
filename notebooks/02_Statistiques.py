import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Charger les données
df = pd.read_csv('../data/raw/cleaned_us_accidents.csv')

# --- Question 1 : Week-end vs semaine (Mann-Whitney) ---
weekday_sev = df[df['is_weekend'] == False]['Severity']
weekend_sev = df[df['is_weekend'] == True]['Severity']
stat1, p1 = stats.mannwhitneyu(weekend_sev, weekday_sev, alternative='greater')
print(f"Q1 - Mann-Whitney: p = {p1:.2e}")

# --- Question 2 : Visibilité vs Gravité (Spearman) ---
vis_data = df[['Severity', 'Visibility(mi)']].dropna()
rho, p2 = stats.spearmanr(vis_data['Visibility(mi)'], vis_data['Severity'])
print(f"Q2 - Spearman: ρ = {rho:.3f}, p = {p2:.2e}")

# --- Question 3 : Météo vs Gravité (Kruskal-Wallis) ---
top_weather = df['Weather_Condition'].value_counts().nlargest(5).index
df['Weather_Group'] = df['Weather_Condition'].where(df['Weather_Condition'].isin(top_weather), 'Other')
groups = [group.values for name, group in df.groupby('Weather_Group')['Severity'] if len(group) > 100]
stat3, p3 = stats.kruskal(*groups)
print(f"Q3 - Kruskal-Wallis: p = {p3:.2e}")

# --- Question 4 : Précipitations vs Accident grave (Chi-2) ---
df['Severe'] = (df['Severity'] >= 3).astype(int)
df['Has_Precip'] = (df['Precipitation(in)'] > 0).astype(int)
contingency = pd.crosstab(df['Has_Precip'], df['Severe'])
chi2, p4, _, _ = stats.chi2_contingency(contingency)
print(f"Q4 - Chi-2: p = {p4:.2e}")

# --- Question 5 : Régression logistique ---
model_data = df[['Severity', 'Visibility(mi)', 'Precipitation(in)', 'is_weekend', 'hour']].dropna()
model_data['Severe'] = (model_data['Severity'] >= 3).astype(int)
model_data['hour_centered'] = model_data['hour'] - 12
X = model_data[['Visibility(mi)', 'Precipitation(in)', 'is_weekend', 'hour_centered']]
X = sm.add_constant(X)
y = model_data['Severe']
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)
print("\nQ5 - Régression logistique (Odds Ratios):")
print(np.exp(result.params))