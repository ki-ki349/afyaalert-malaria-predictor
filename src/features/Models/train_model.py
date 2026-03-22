# src/models/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("🤖 Starting Model Training...")
print("="*60)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the processed data
print("📂 Loading feature engineered dataset...")
df = pd.read_csv('data/processed/feature_engineered_dataset.csv')
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features and target
print("\n🔍 Preparing features and target...")

# Define which columns to use as features
# Exclude identifier columns, date, and target
feature_columns = [col for col in df.columns if col not in [
    'county', 'date', 'confirmed_cases', 'risk_level', 'season'
]]

X = df[feature_columns]
y = df['risk_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

print(f"📊 Features used: {len(feature_columns)}")
print(f"🎯 Target distribution:")
print(y.value_counts().sort_index().rename({0:'Low', 1:'Medium', 2:'High'}))

# Split data (temporal split - use older data for training, recent for testing)
print("\n📅 Performing temporal train-test split...")

# Add a date column for sorting
df['date_sort'] = pd.to_datetime(df['date'])
df = df.sort_values('date_sort')

# Use 80% for training, 20% for testing
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"   Training set: {len(X_train)} samples ({split_idx/len(df)*100:.1f}%)")
print(f"   Testing set: {len(X_test)} samples ({100-split_idx/len(df)*100:.1f}%)")

# Initialize models
print("\n🤖 Initializing machine learning models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate each model
print("\n" + "="*60)
print("🏋️ TRAINING MODELS")
print("="*60)

results = {}
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    print("-"*40)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"   ✅ Accuracy:  {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1-Score:  {f1:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"   📊 Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save if this is the best model so far
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name
        # Save the model
        filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, filename)
        print(f"   💾 Saved {name} model (best so far)")

# Display results comparison
print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).T
print(results_df.round(4))

# Save results to CSV
results_df.to_csv('models/model_performance.csv')
print("\n💾 Saved model performance to: models/model_performance.csv")

# Feature importance (for tree-based models)
print("\n" + "="*60)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print("-"*40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("\n💾 Saved feature importance to: models/feature_importance.csv")

# Confusion matrix for best model
print("\n" + "="*60)
print(f"🎯 DETAILED RESULTS FOR BEST MODEL: {best_name}")
print("="*60)

y_pred_best = best_model.predict(X_test)
print("\n📊 Classification Report:")
print("-"*40)
print(classification_report(y_test, y_pred_best, 
                          target_names=['Low', 'Medium', 'High']))

# Save the best model info
with open('models/best_model_info.txt', 'w') as f:
    f.write(f"Best Model: {best_name}\n")
    f.write(f"F1-Score: {best_score:.4f}\n")
    f.write(f"Features used: {len(feature_columns)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")

print("\n" + "="*60)
print("="*60)
print(f"\n🎯 Best model: {best_name} with F1-Score: {best_score:.4f}")
print("📁 All models saved in the 'models' folder")