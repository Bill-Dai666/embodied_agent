"""
Statistical Analysis Script for Chatbot Study
Part 1: t-test analysis on empathy scores
Part 2: Interaction plots (PCQ vs TRUST, PCQ vs EMP)
Part 3: Two-way ANOVA (Personality × Condition)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STATISTICAL ANALYSIS FOR CHATBOT STUDY")
print("=" * 80)

# ============================================================================
# Step 1: Data Preparation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Loading and Preparing Data")
print("=" * 80)

# Read data
df = pd.read_csv('../survey_response.csv')
print(f"Loaded {len(df)} participants")

# Define sections
sections = {
    'PCQ': [f'PCQ_{i}' for i in range(1, 15)],
    'EMP': [f'EMP_{i}' for i in range(1, 6)],
    'TRUST': [f'TRUST_{i}' for i in range(1, 6)],
    'G_ANTHRO': [f'G_ANTHRO_{i}' for i in range(1, 6)],
    'G_ANIMACY': [f'G_ANIMACY_{i}' for i in range(1, 7)],
    'G_LIKE': [f'G_LIKE_{i}' for i in range(1, 6)],
    'G_INTEL': [f'G_INTEL_{i}' for i in range(1, 6)],
    'G_SAFETY': [f'G_SAFETY_{i}' for i in range(1, 4)]
}

# Calculate composite scores
print("\nCalculating composite scores...")
for section_name, columns in sections.items():
    # Check which columns actually exist in the dataframe
    existing_cols = [col for col in columns if col in df.columns]
    if existing_cols:
        df[f'{section_name}_composite'] = df[existing_cols].mean(axis=1)
        print(f"  {section_name}: {len(existing_cols)} items")

# ============================================================================
# Step 2: Personality Classification (Method 3)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Personality Classification")
print("=" * 80)

# Calculate extraversion score
df['extraversion_score'] = (
    df['P3_do_not_mind_centre_of_attention'] + 
    df['P10_make_friends_easily'] - 
    df['P14_keep_in_the_background'] - 
    df['P18_avoid_contact_with_others']
) / 4

# Calculate empathy tendency score
df['empathy_tendency'] = (
    df['P6_believe_others_have_good_intentions'] + 
    df['P11_feel_comfortable_with_myself'] - 
    df['P9_cut_others_to_pieces']
) / 3

# Classify into 4 personality categories based on median
extraversion_median = df['extraversion_score'].median()
empathy_median = df['empathy_tendency'].median()

print(f"\nExtraversion median: {extraversion_median:.3f}")
print(f"Empathy tendency median: {empathy_median:.3f}")

def classify_personality(row):
    extraversion = row['extraversion_score']
    empathy = row['empathy_tendency']
    
    if extraversion >= extraversion_median and empathy >= empathy_median:
        return '人格1_外向高共情'
    elif extraversion >= extraversion_median and empathy < empathy_median:
        return '人格2_外向低共情'
    elif extraversion < extraversion_median and empathy >= empathy_median:
        return '人格3_内向高共情'
    else:
        return '人格4_内向低共情'

df['personality_category'] = df.apply(classify_personality, axis=1)

print("\nPersonality category distribution:")
print(df['personality_category'].value_counts().sort_index())

# ============================================================================
# Step 3: Part 1 - t-test Analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: T-TEST ANALYSIS ON EMPATHY SCORES")
print("=" * 80)

results_ttest = []

# Function to perform t-test and calculate effect size
def perform_ttest(group1, group2, group_name):
    """Perform t-test and return results"""
    if len(group1) < 2 or len(group2) < 2:
        return None
    
    # Check if variances are zero
    if group1.std() == 0 and group2.std() == 0:
        if group1.mean() == group2.mean():
            t_stat, p_value = 0.0, 1.0
            cohens_d = 0.0
        else:
            t_stat, p_value = np.inf, 0.0
            cohens_d = np.inf
    else:
        t_stat, p_value = ttest_ind(group1, group2)
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std()**2 + 
                             (len(group2) - 1) * group2.std()**2) / 
                            (len(group1) + len(group2) - 2))
        if pooled_std > 0:
            cohens_d = (group1.mean() - group2.mean()) / pooled_std
        else:
            cohens_d = 0.0
    
    return {
        'Category': group_name,
        'Avatar_Mean': group1.mean(),
        'Avatar_SD': group1.std(),
        'Avatar_N': len(group1),
        'Text_Mean': group2.mean(),
        'Text_SD': group2.std(),
        'Text_N': len(group2),
        'T_statistic': t_stat,
        'P_value': p_value,
        "Cohen's_d": cohens_d,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    }

# 3.1 Overall t-test
print("\n3.1 Overall t-test (Avatar vs Text)")
avatar_overall = df[df['condition'] == 'avatar']['EMP_composite']
text_overall = df[df['condition'] == 'text']['EMP_composite']
result = perform_ttest(avatar_overall, text_overall, 'All')
if result:
    results_ttest.append(result)
    print(f"  Avatar Mean: {result['Avatar_Mean']:.3f} (SD: {result['Avatar_SD']:.3f}, N={result['Avatar_N']})")
    print(f"  Text Mean: {result['Text_Mean']:.3f} (SD: {result['Text_SD']:.3f}, N={result['Text_N']})")
    print(f"  t = {result['T_statistic']:.3f}, p = {result['P_value']:.4f}, Cohen's d = {result["Cohen's_d"]:.3f}")

# 3.2 t-test by Personality Category
print("\n3.2 t-test by Personality Category")
results_ttest.append({
    'Category': 'Personality',
    'Avatar_Mean': np.nan, 'Avatar_SD': np.nan, 'Avatar_N': np.nan,
    'Text_Mean': np.nan, 'Text_SD': np.nan, 'Text_N': np.nan,
    'T_statistic': np.nan, 'P_value': np.nan, "Cohen's_d": np.nan,
    'Significant': ''
})

for personality in sorted(df['personality_category'].unique()):
    personality_data = df[df['personality_category'] == personality]
    avatar_personality = personality_data[personality_data['condition'] == 'avatar']['EMP_composite']
    text_personality = personality_data[personality_data['condition'] == 'text']['EMP_composite']
    
    result = perform_ttest(avatar_personality, text_personality, personality)
    if result:
        results_ttest.append(result)
        print(f"\n  {personality}:")
        print(f"    Avatar Mean: {result['Avatar_Mean']:.3f} (SD: {result['Avatar_SD']:.3f}, N={result['Avatar_N']})")
        print(f"    Text Mean: {result['Text_Mean']:.3f} (SD: {result['Text_SD']:.3f}, N={result['Text_N']})")
        print(f"    t = {result['T_statistic']:.3f}, p = {result['P_value']:.4f}, Cohen's d = {result["Cohen's_d"]:.3f}")

# 3.3 t-test by Ethnicity
print("\n3.3 t-test by Ethnicity")
results_ttest.append({
    'Category': 'Ethnicity',
    'Avatar_Mean': np.nan, 'Avatar_SD': np.nan, 'Avatar_N': np.nan,
    'Text_Mean': np.nan, 'Text_SD': np.nan, 'Text_N': np.nan,
    'T_statistic': np.nan, 'P_value': np.nan, "Cohen's_d": np.nan,
    'Significant': ''
})

# Filter out missing ethnicity
df_with_ethnicity = df[df['Ethnicity simplified'].notna()]

for ethnicity in sorted(df_with_ethnicity['Ethnicity simplified'].unique()):
    ethnicity_data = df_with_ethnicity[df_with_ethnicity['Ethnicity simplified'] == ethnicity]
    avatar_ethnicity = ethnicity_data[ethnicity_data['condition'] == 'avatar']['EMP_composite']
    text_ethnicity = ethnicity_data[ethnicity_data['condition'] == 'text']['EMP_composite']
    
    result = perform_ttest(avatar_ethnicity, text_ethnicity, ethnicity)
    if result:
        results_ttest.append(result)
        print(f"\n  {ethnicity}:")
        print(f"    Avatar Mean: {result['Avatar_Mean']:.3f} (SD: {result['Avatar_SD']:.3f}, N={result['Avatar_N']})")
        print(f"    Text Mean: {result['Text_Mean']:.3f} (SD: {result['Text_SD']:.3f}, N={result['Text_N']})")
        print(f"    t = {result['T_statistic']:.3f}, p = {result['P_value']:.4f}, Cohen's d = {result["Cohen's_d"]:.3f}")

# Save t-test results
ttest_df = pd.DataFrame(results_ttest)
ttest_df.to_csv('t_test_results_empathy.csv', index=False, encoding='utf-8-sig')
print("\n✓ Saved t-test results to: t_test_results_empathy.csv")

# ============================================================================
# Step 4: Part 2 - Interaction Plots
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: INTERACTION PLOTS")
print("=" * 80)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: PCQ vs TRUST
ax1 = axes[0]
text_data = df[df['condition'] == 'text']
avatar_data = df[df['condition'] == 'avatar']

# Scatter plot
ax1.scatter(text_data['TRUST_composite'], text_data['PCQ_composite'], 
           alpha=0.6, label='Text-based Chatbot', color='#3498db', s=50)
ax1.scatter(avatar_data['TRUST_composite'], avatar_data['PCQ_composite'], 
           alpha=0.6, label='Avatar-based Chatbot', color='#e74c3c', s=50)

# Regression lines
z_text = np.polyfit(text_data['TRUST_composite'], text_data['PCQ_composite'], 1)
p_text = np.poly1d(z_text)
x_line = np.linspace(df['TRUST_composite'].min(), df['TRUST_composite'].max(), 100)
ax1.plot(x_line, p_text(x_line), "--", alpha=0.7, color='#3498db', linewidth=2)

z_avatar = np.polyfit(avatar_data['TRUST_composite'], avatar_data['PCQ_composite'], 1)
p_avatar = np.poly1d(z_avatar)
ax1.plot(x_line, p_avatar(x_line), "--", alpha=0.7, color='#e74c3c', linewidth=2)

# Calculate correlation
corr_text, p_text_corr = stats.pearsonr(text_data['TRUST_composite'], text_data['PCQ_composite'])
corr_avatar, p_avatar_corr = stats.pearsonr(avatar_data['TRUST_composite'], avatar_data['PCQ_composite'])

ax1.set_xlabel('TRUST Composite Score', fontsize=12)
ax1.set_ylabel('PCQ Composite Score', fontsize=12)
ax1.set_title('PCQ vs TRUST by Condition', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add correlation info
text_str = f'Text: r={corr_text:.3f}, p={p_text_corr:.4f}'
avatar_str = f'Avatar: r={corr_avatar:.3f}, p={p_avatar_corr:.4f}'
ax1.text(0.05, 0.95, f'{text_str}\n{avatar_str}', 
        transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: PCQ vs EMP
ax2 = axes[1]

# Scatter plot
ax2.scatter(text_data['EMP_composite'], text_data['PCQ_composite'], 
           alpha=0.6, label='Text-based Chatbot', color='#3498db', s=50)
ax2.scatter(avatar_data['EMP_composite'], avatar_data['PCQ_composite'], 
           alpha=0.6, label='Avatar-based Chatbot', color='#e74c3c', s=50)

# Regression lines
z_text_emp = np.polyfit(text_data['EMP_composite'], text_data['PCQ_composite'], 1)
p_text_emp = np.poly1d(z_text_emp)
x_line_emp = np.linspace(df['EMP_composite'].min(), df['EMP_composite'].max(), 100)
ax2.plot(x_line_emp, p_text_emp(x_line_emp), "--", alpha=0.7, color='#3498db', linewidth=2)

z_avatar_emp = np.polyfit(avatar_data['EMP_composite'], avatar_data['PCQ_composite'], 1)
p_avatar_emp = np.poly1d(z_avatar_emp)
ax2.plot(x_line_emp, p_avatar_emp(x_line_emp), "--", alpha=0.7, color='#e74c3c', linewidth=2)

# Calculate correlation
corr_text_emp, p_text_emp_corr = stats.pearsonr(text_data['EMP_composite'], text_data['PCQ_composite'])
corr_avatar_emp, p_avatar_emp_corr = stats.pearsonr(avatar_data['EMP_composite'], avatar_data['PCQ_composite'])

ax2.set_xlabel('EMP Composite Score', fontsize=12)
ax2.set_ylabel('PCQ Composite Score', fontsize=12)
ax2.set_title('PCQ vs EMP by Condition', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add correlation info
text_str_emp = f'Text: r={corr_text_emp:.3f}, p={p_text_emp_corr:.4f}'
avatar_str_emp = f'Avatar: r={corr_avatar_emp:.3f}, p={p_avatar_emp_corr:.4f}'
ax2.text(0.05, 0.95, f'{text_str_emp}\n{avatar_str_emp}', 
        transform=ax2.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('interaction_plots_pcq.png', dpi=300, bbox_inches='tight')
print("✓ Saved interaction plots to: interaction_plots_pcq.png")
plt.close()

# ============================================================================
# Step 5: Part 3 - Two-way ANOVA
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: TWO-WAY ANOVA (Personality × Condition)")
print("=" * 80)

results_anova = []

# Dependent variables to analyze
dependent_vars = ['EMP_composite', 'PCQ_composite', 'TRUST_composite', 
                  'G_ANTHRO_composite', 'G_ANIMACY_composite', 'G_LIKE_composite',
                  'G_INTEL_composite', 'G_SAFETY_composite']

for dv in dependent_vars:
    if dv not in df.columns:
        print(f"\n⚠ Skipping {dv} - column not found")
        continue
    
    print(f"\nAnalyzing {dv}...")
    
    # Prepare data (remove missing values)
    anova_data = df[[dv, 'personality_category', 'condition']].dropna()
    
    if len(anova_data) < 10:
        print(f"  ⚠ Insufficient data for {dv}")
        continue
    
    # Fit Two-way ANOVA model
    try:
        model = ols(f'{dv} ~ C(personality_category) + C(condition) + C(personality_category):C(condition)', 
                   data=anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract results
        personality_f = anova_table.loc['C(personality_category)', 'F']
        personality_p = anova_table.loc['C(personality_category)', 'PR(>F)']
        
        condition_f = anova_table.loc['C(condition)', 'F']
        condition_p = anova_table.loc['C(condition)', 'PR(>F)']
        
        interaction_f = anova_table.loc['C(personality_category):C(condition)', 'F']
        interaction_p = anova_table.loc['C(personality_category):C(condition)', 'PR(>F)']
        
        results_anova.append({
            'Dependent_Variable': dv,
            'Personality_F': personality_f,
            'Personality_P': personality_p,
            'Personality_Sig': 'Yes' if personality_p < 0.05 else 'No',
            'Condition_F': condition_f,
            'Condition_P': condition_p,
            'Condition_Sig': 'Yes' if condition_p < 0.05 else 'No',
            'Interaction_F': interaction_f,
            'Interaction_P': interaction_p,
            'Interaction_Sig': 'Yes' if interaction_p < 0.05 else 'No'
        })
        
        print(f"  Personality: F = {personality_f:.3f}, p = {personality_p:.4f} {'***' if personality_p < 0.001 else '**' if personality_p < 0.01 else '*' if personality_p < 0.05 else ''}")
        print(f"  Condition: F = {condition_f:.3f}, p = {condition_p:.4f} {'***' if condition_p < 0.001 else '**' if condition_p < 0.01 else '*' if condition_p < 0.05 else ''}")
        print(f"  Interaction: F = {interaction_f:.3f}, p = {interaction_p:.4f} {'***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else ''}")
        
    except Exception as e:
        print(f"  ⚠ Error analyzing {dv}: {e}")
        continue

# Save ANOVA results
anova_df = pd.DataFrame(results_anova)
anova_df.to_csv('anova_results_personality_condition.csv', index=False, encoding='utf-8-sig')
print("\n✓ Saved ANOVA results to: anova_results_personality_condition.csv")

# ============================================================================
# Step 6: Save processed data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Saving Processed Data")
print("=" * 80)

df.to_csv('survey_response_with_personality.csv', index=False, encoding='utf-8-sig')
print("✓ Saved processed data to: survey_response_with_personality.csv")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. t_test_results_empathy.csv - t-test results")
print("  2. interaction_plots_pcq.png - interaction plots")
print("  3. anova_results_personality_condition.csv - ANOVA results")
print("  4. survey_response_with_personality.csv - processed data with personality categories")
print("=" * 80)

