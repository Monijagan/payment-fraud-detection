"""
Hour 3: Feature Engineering + Fraud Detection ML Model
Reads from PostgreSQL → Engineers Features → Trains XGBoost → Saves Predictions
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

print("Starting Feature Engineering & ML Model Training...")
print("=" * 60)

# Connect to PostgreSQL
print("Step 1/5: Connecting to database...")
engine = create_engine(
    'postgresql://postgres:250499@localhost:5433/postgres'
)
print("✓ Connected")

# Load data from PostgreSQL
print("\nStep 2/5: Loading data from PostgreSQL...")
query = """
SELECT 
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.hour_of_day,
    t.day_of_week,
    t.is_weekend,
    t.is_night,
    t.is_round_dollar,
    t.is_first_time_recipient,
    t.is_fraud,
    c.account_age_days,
    c.risk_tier,
    r.recipient_account_age_days,
    r.recipient_verified
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
JOIN recipients r ON t.recipient_id = r.recipient_id
"""
df = pd.read_sql(query, engine)
print(f"✓ Loaded {len(df):,} transactions from database")
print(f"  Fraud cases: {df['is_fraud'].sum():,}")
print(f"  Legitimate: {(~df['is_fraud']).sum():,}")

# Feature Engineering
print("\nStep 3/5: Engineering fraud detection features...")

# Feature 1: Amount risk buckets
df['amount_risk'] = pd.cut(
    df['amount'],
    bins=[0, 100, 500, 1000, 5000, 99999],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# Feature 2: High value transaction flag
df['is_high_value'] = (df['amount'] > 1000).astype(int)

# Feature 3: Very high value flag (over $5000)
df['is_very_high_value'] = (df['amount'] > 5000).astype(int)

# Feature 4: Suspicious amount range (just under $10K)
df['is_structuring_amount'] = (
    (df['amount'] >= 9000) & (df['amount'] < 10000)
).astype(int)

# Feature 5: Risk tier encoding
risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df['risk_tier_encoded'] = df['risk_tier'].map(risk_map)

# Feature 6: New account sending money (risky)
df['is_new_sender'] = (df['account_age_days'] < 90).astype(int)

# Feature 7: New recipient receiving money (risky)
df['is_new_recipient'] = (
    df['recipient_account_age_days'] < 30
).astype(int)

# Feature 8: Unverified recipient
df['is_unverified_recipient'] = (
    ~df['recipient_verified']
).astype(int)

# Feature 9: Night + high value combo (very suspicious)
df['night_high_value'] = (
    df['is_night'] & (df['amount'] > 500)
).astype(int)

# Feature 10: Weekend + first time recipient combo
df['weekend_first_time'] = (
    df['is_weekend'] & df['is_first_time_recipient']
).astype(int)

# Feature 11: First time + high value combo
df['first_time_high_value'] = (
    df['is_first_time_recipient'] & (df['amount'] > 1000)
).astype(int)

# Feature 12: New account + high value (account takeover signal)
df['new_account_high_value'] = (
    df['is_new_sender'] & (df['amount'] > 500)
).astype(int)

# Feature 13: Amount log (reduces skewness)
df['amount_log'] = np.log1p(df['amount'])

# Feature 14: Risk combo score
df['risk_combo_score'] = (
    df['risk_tier_encoded'] + 
    df['is_night'].astype(int) + 
    df['is_first_time_recipient'].astype(int) + 
    df['is_round_dollar'].astype(int) +
    df['is_high_value']
)

print("✓ Created 14 fraud detection features")

# Define features for model
FEATURES = [
    'amount',
    'amount_log',
    'amount_risk',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'is_night',
    'is_round_dollar',
    'is_first_time_recipient',
    'is_high_value',
    'is_very_high_value',
    'is_structuring_amount',
    'risk_tier_encoded',
    'is_new_sender',
    'is_new_recipient',
    'is_unverified_recipient',
    'night_high_value',
    'weekend_first_time',
    'first_time_high_value',
    'new_account_high_value',
    'risk_combo_score',
    'account_age_days',
    'recipient_account_age_days'
]

X = df[FEATURES]
y = df['is_fraud'].astype(int)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Training set: {len(X_train):,} transactions")
print(f"  Test set:     {len(X_test):,} transactions")

# Train XGBoost Model
print("\nStep 4/5: Training XGBoost fraud detection model...")

# Calculate class weight (fraud is rare - need to balance)
fraud_count = y_train.sum()
legit_count = len(y_train) - fraud_count
scale_pos_weight = legit_count / fraud_count
print(f"  Class balance ratio: {scale_pos_weight:.1f}:1")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42,
    eval_metric='auc',
    verbosity=0
)

model.fit(X_train, y_train)
print("✓ Model trained!")

# Evaluate Model
print("\nStep 5/5: Evaluating model performance...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n  📊 MODEL PERFORMANCE:")
print(f"  Precision:  {precision:.1%} (of flagged, how many are real fraud)")
print(f"  Recall:     {recall:.1%} (of all fraud, how many we caught)")
print(f"  F1 Score:   {f1:.3f}")
print(f"  ROC-AUC:    {auc:.3f}")

# Feature Importance
print("\n  🔍 TOP 5 FRAUD SIGNALS:")
importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(5).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.3f}")

# Save predictions back to PostgreSQL
print("\n  Saving fraud risk scores to database...")
df['fraud_probability'] = model.predict_proba(X)[:, 1]
df['fraud_risk_score'] = (df['fraud_probability'] * 1000).astype(int)
df['risk_label'] = pd.cut(
    df['fraud_probability'],
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
)

# Save scored transactions to new table
results_df = df[[
    'transaction_id',
    'amount',
    'is_fraud',
    'fraud_probability',
    'fraud_risk_score',
    'risk_label',
    'is_night',
    'is_first_time_recipient',
    'is_high_value',
    'risk_combo_score'
]]

results_df.to_sql(
    'fraud_scores',
    engine,
    if_exists='replace',
    index=False
)

print("✓ Fraud scores saved to 'fraud_scores' table in PostgreSQL")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING & ML MODEL COMPLETE! ✅")
print("=" * 60)
print(f"\nKey Results:")
print(f"  • Model catches {recall:.1%} of all fraud cases")
print(f"  • {precision:.1%} of flagged transactions are real fraud")
print(f"  • Risk scores saved to database (0-1000 scale)")