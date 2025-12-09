# Updated wine quality analysis
# - Loads red and white wine datasets
# - Creates categorical labels (poor/medium/good) according to: quality <=5 -> poor, 6-7 -> medium, >=8 -> good
# - Saves separate CSVs and plots for each category and for each wine color
# - Trains a regression model to predict numeric quality from physico-chemical features
# - Trains a classifier to predict wine color (red/white)
# - Generates synthetic samples by varying alcohol over a progression while keeping other features at their median
#   and predicts quality and type for each alcohol level
# - Saves a predictions table (CSV) sorted by predicted quality (descending) and plots alcohol vs predicted quality
# - Extensive comments and a final summary printed to console

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# Create outputs directory
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Read datasets (CSV files are expected at repository root)
RED_CSV = "winequality-red.csv"
WHITE_CSV = "winequality-white.csv"

# Helper to read semicolon-delimited files with potential quoted header
def read_wine_csv(path):
    # use pandas read_csv with ';' separator
    return pd.read_csv(path, sep=';', decimal='.')

print("Loading datasets...")
red = read_wine_csv(RED_CSV)
white = read_wine_csv(WHITE_CSV)

# Add a 'type' column to identify red vs white
red['type'] = 'red'
white['type'] = 'white'

# Concatenate datasets for unified processing
df = pd.concat([red, white], ignore_index=True)

# Standardize column names if necessary (strip quotes/spaces)
df.columns = [c.strip().strip('"') for c in df.columns]

# Quality categorization (as requested):
# - 5 and below -> 'poor' (user noted 5 is poor)
# - 6-7 -> 'medium'
# - 8 and above -> 'good'
# This creates an absolute partition of all wines into three categories.
def quality_label(q):
    if q <= 5:
        return 'poor'
    elif 6 <= q <= 7:
        return 'medium'
    else:
        return 'good'

df['quality_label'] = df['quality'].apply(quality_label)

# Save separate tables per category and create per-category plots
for label in ['poor', 'medium', 'good']:
    subset = df[df['quality_label'] == label]
    csv_out = os.path.join(OUT_DIR, f"wine_{label}.csv")
    subset.to_csv(csv_out, index=False)
    print(f"Saved {csv_out} with {len(subset)} rows")

    # Simple histogram of alcohol for this category
    plt.figure(figsize=(8,4))
    sns.histplot(subset['alcohol'], kde=True, bins=30)
    plt.title(f"Alcohol distribution for {label} wines (n={len(subset)})")
    plt.xlabel('alcohol')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"alcohol_hist_{label}.png"))
    plt.close()

# Additionally, save plots per (type x quality_label) combination
for wtype in ['red', 'white']:
    for label in ['poor','medium','good']:
        sub = df[(df['type']==wtype) & (df['quality_label']==label)]
        if len(sub)==0:
            continue
        plt.figure(figsize=(8,4))
        sns.boxplot(x='quality_label', y='alcohol', data=sub)
        plt.title(f"Alcohol in {wtype} wines with label {label} (n={len(sub)})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"alcohol_box_{wtype}_{label}.png"))
        plt.close()

# ---------------------------------------------------------------------------------
# Build predictive models
# - Regression model: predict numeric 'quality' from physico-chemical features
# - Classification model: predict 'type' (red/white) from physico-chemical features
# For simplicity and interpretability we use Random Forests with default settings.
# ---------------------------------------------------------------------------------
FEATURE_COLS = [c for c in df.columns if c not in ['quality','quality_label','type']]
print('Feature columns used for modeling:', FEATURE_COLS)

# Prepare training data
X = df[FEATURE_COLS].copy()
y_reg = df['quality'].values

# Encode type target for classifier
le_type = LabelEncoder()
y_type = le_type.fit_transform(df['type'])  # 0/1 where mapping is stored in le_type

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_type_train, y_type_test = train_test_split(
    X, y_reg, y_type, test_size=0.2, random_state=42)

# RandomForestRegressor for numeric quality
reg = RandomForestRegressor(random_state=42, n_estimators=200)
reg.fit(X_train, y_reg_train)
reg_preds = reg.predict(X_test)
rmse = mean_squared_error(y_reg_test, reg_preds, squared=False)
print(f"Regression RMSE on test set: {rmse:.3f}")

# RandomForestClassifier for type prediction
clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf.fit(X_train, y_type_train)
clf_preds = clf.predict(X_test)
acc = accuracy_score(y_type_test, clf_preds)
print(f"Type classification accuracy on test set: {acc:.3f}")

# Save simple model diagnostics plots
plt.figure(figsize=(6,4))
plt.scatter(y_reg_test, reg_preds, alpha=0.3)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel('True quality')
plt.ylabel('Predicted quality')
plt.title('Regression: true vs predicted quality')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'reg_true_vs_pred.png'))
plt.close()

# ---------------------------------------------------------------------------------
# Alcohol progression experiment (core request):
# - Take a single parameter: alcohol
# - Vary alcohol over a progression (linearly from min-1 to max+1)
# - For each alcohol value, set other features to median values (across whole dataset)
# - Use the trained regression model to predict numeric quality for each synthetic sample
# - Use the trained classifier to predict the probability of being 'white' (and thus predicted type)
# - Produce a table and plots. Sort table by predicted quality descending.
# ---------------------------------------------------------------------------------

# Determine median baseline for other features
baseline = X.median()

# Generate alcohol progression range
al_min, al_max = float(df['alcohol'].min()), float(df['alcohol'].max())
al_values = np.linspace(max(0, al_min-1.0), al_max+1.0, 200)

synthetic_rows = []
for a in al_values:
    row = baseline.copy()
    row['alcohol'] = a
    synthetic_rows.append(row)

X_synth = pd.DataFrame(synthetic_rows)

# Predict numeric quality and predicted type probabilities
pred_quality = reg.predict(X_synth[FEATURE_COLS])
pred_type_proba = clf.predict_proba(X_synth[FEATURE_COLS])

# clf.classes_ gives the encoded classes order corresponding to le_type.transform
# map probabilities to 'white' probability
# find index of 'white' in le_type classes
white_index = list(le_type.classes_).index('white') if 'white' in le_type.classes_ else None
if white_index is None:
    # fall back: second column is often 'white'
    white_index = 1 if pred_type_proba.shape[1] > 1 else 0

white_proba = pred_type_proba[:, white_index]
pred_type_label = np.where(white_proba >= 0.5, 'white', 'red')

synth_df = X_synth.copy()
synth_df['pred_quality'] = pred_quality
synth_df['pred_white_proba'] = white_proba
synth_df['pred_type'] = pred_type_label
synth_df['alcohol'] = al_values

# Create a table with alcohol, predicted quality, predicted type, sorted by predicted quality desc
out_table = synth_df[['alcohol','pred_quality','pred_type','pred_white_proba']].copy()
out_table_sorted = out_table.sort_values('pred_quality', ascending=False).reset_index(drop=True)

predictions_csv = os.path.join(OUT_DIR, 'alcohol_vs_predicted_quality.csv')
out_table_sorted.to_csv(predictions_csv, index=False)
print(f"Saved predictions table to {predictions_csv}")

# Show top 10 predicted qualities
print('\nTop 10 alcohol levels by predicted quality:')
print(out_table_sorted.head(10))

# Plot alcohol vs predicted quality (colored by predicted type)
plt.figure(figsize=(10,6))
# scatter with color mapping
palette = {'white':'orange','red':'purple'}
plt.scatter(synth_df['alcohol'], synth_df['pred_quality'], c=[palette[t] for t in synth_df['pred_type']], s=15, alpha=0.7)
# overlay a rolling mean for smoother trend
window = max(3, len(synth_df)//40)
synth_df['pred_quality_smooth'] = pd.Series(synth_df['pred_quality']).rolling(window=window, min_periods=1, center=True).mean()
plt.plot(synth_df['alcohol'], synth_df['pred_quality_smooth'], color='black', linewidth=1.2, label='smoothed')
plt.xlabel('Alcohol')
plt.ylabel('Predicted quality (numeric)')
plt.title('Effect of alcohol (varied) on predicted wine quality\n(points colored by predicted type)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'alcohol_vs_pred_quality.png'))
plt.close()

# Additionally, plot predicted probability of 'white' as alcohol increases
plt.figure(figsize=(10,5))
plt.plot(synth_df['alcohol'], synth_df['pred_white_proba'], color='tab:blue')
plt.xlabel('Alcohol')
plt.ylabel('Predicted probability of being white')
plt.title('Predicted probability of wine being white as alcohol varies')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'alcohol_vs_white_proba.png'))
plt.close()

# Save the top predicted table as CSV with more readable columns
top_csv = os.path.join(OUT_DIR, 'alcohol_quality_type_top_sorted.csv')
out_table_sorted.to_csv(top_csv, index=False)
print(f"Saved sorted predictions to {top_csv}")

# ---------------------------------------------------------------------------------
# Write a short textual summary to outputs/summary.txt explaining results and business value
# ---------------------------------------------------------------------------------
summary_lines = []
summary_lines.append('Summary of updated analysis')
summary_lines.append('-----------------------------------')
summary_lines.append(f'Total wines analyzed: {len(df)} (red: {len(red)}, white: {len(white)})')
summary_lines.append(f'Regression RMSE (quality): {rmse:.3f}')
summary_lines.append(f'Classifier accuracy (type): {acc:.3f}')
summary_lines.append('Quality labeling rule: <=5 -> poor, 6-7 -> medium, >=8 -> good')
summary_lines.append('Business insight: The alcohol progression experiment shows how predicted numeric quality
changes when alcohol is altered while other physico-chemical properties are kept at median values.')
summary_lines.append('A saved CSV `alcohol_vs_predicted_quality.csv` contains the full progression table
and can be inspected or used to select target alcohol percentages for product development.')

with open(os.path.join(OUT_DIR,'summary.txt'),'w',encoding='utf8') as f:
    f.write('\n'.join(summary_lines))

print('\nDone. Output files are in the outputs/ directory in this repository.\n')

# End of updated script
