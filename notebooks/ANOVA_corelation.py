# ============================================
# STATISTIQUES AVANCÉES
# CORRÉLATION & ANOVA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------------------------------
# 1. Chargement des données
# --------------------------------------------
df = pd.read_csv('../data/raw/cleaned_us_accidents.csv')

# ============================================
# PARTIE 1 — CORRÉLATIONS
# ============================================

# Sélection des variables numériques
corr_vars = [
    'Severity',
    'Visibility(mi)',
    'Precipitation(in)',
    'Temperature(F)',
    'Wind_Speed(mph)',
    'Humidity(%)'
]

corr_df = df[corr_vars].dropna()

# Matrice de corrélation de Spearman
corr_matrix = corr_df.corr(method='spearman')

# Heatmap des corrélations
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Matrice de corrélation de Spearman")
plt.tight_layout()
plt.show()

# Test de corrélation : Sévérité vs Visibilité
rho_vis, p_vis = stats.spearmanr(
    corr_df['Severity'],
    corr_df['Visibility(mi)']
)
print(f"Spearman Severity vs Visibility : rho={rho_vis:.3f}, p={p_vis:.2e}")

# Test de corrélation : Sévérité vs Précipitations
rho_prec, p_prec = stats.spearmanr(
    corr_df['Severity'],
    corr_df['Precipitation(in)']
)
print(f"Spearman Severity vs Precipitation : rho={rho_prec:.3f}, p={p_prec:.2e}")

# Test de corrélation : Sévérité vs Température
rho_temp, p_temp = stats.spearmanr(
    corr_df['Severity'],
    corr_df['Temperature(F)']
)
print(f"Spearman Severity vs Temperature : rho={rho_temp:.3f}, p={p_temp:.2e}")

# ============================================
# PARTIE 2 — ANOVA / KRUSKAL-WALLIS
# ============================================

# --------------------------------------------
# ANOVA selon la luminosité (jour / nuit)
# --------------------------------------------
anova_light_df = df[['Severity', 'Sunrise_Sunset']].dropna()

groups_light = [
    group['Severity'].values
    for name, group in anova_light_df.groupby('Sunrise_Sunset')
    if len(group) > 100
]

# ANOVA classique
f_light, p_light = stats.f_oneway(*groups_light)
print(f"\nANOVA Luminosité : F={f_light:.2f}, p={p_light:.2e}")

# Test de Levene (homogénéité des variances)
stat_lev, p_lev = stats.levene(*groups_light)
print(f"Levene test p={p_lev:.2e}")

# Test de Shapiro (normalité sur échantillon)
sample = anova_light_df['Severity'].sample(5000, random_state=42)
stat_shap, p_shap = stats.shapiro(sample)
print(f"Shapiro test p={p_shap:.2e}")

# Alternative robuste : Kruskal-Wallis
h_light, p_kw_light = stats.kruskal(*groups_light)
print(f"Kruskal-Wallis Luminosité : H={h_light:.2f}, p={p_kw_light:.2e}")

# Visualisation
plt.figure(figsize=(6, 4))
sns.boxplot(
    x='Sunrise_Sunset',
    y='Severity',
    data=anova_light_df
)
plt.title("Gravité des accidents selon la luminosité")
plt.tight_layout()
plt.show()

# --------------------------------------------
# ANOVA selon la météo
# --------------------------------------------
top_weather = df['Weather_Condition'].value_counts().nlargest(5).index
df['Weather_Group'] = df['Weather_Condition'].where(
    df['Weather_Condition'].isin(top_weather),
    'Other'
)

anova_weather_df = df[['Severity', 'Weather_Group']].dropna()

groups_weather = [
    group['Severity'].values
    for name, group in anova_weather_df.groupby('Weather_Group')
    if len(group) > 100
]

# ANOVA météo
f_weather, p_weather = stats.f_oneway(*groups_weather)
print(f"\nANOVA Météo : F={f_weather:.2f}, p={p_weather:.2e}")

# Kruskal-Wallis météo (robuste)
h_weather, p_kw_weather = stats.kruskal(*groups_weather)
print(f"Kruskal-Wallis Météo : H={h_weather:.2f}, p={p_kw_weather:.2e}")

# Visualisation météo
plt.figure(figsize=(10, 5))
sns.boxplot(
    x='Weather_Group',
    y='Severity',
    data=anova_weather_df
)
plt.xticks(rotation=45)
plt.title("Gravité des accidents selon les conditions météo")
plt.tight_layout()
plt.show()