import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def train_xgboost(X_train, y_train, random_state=42):
    xgb_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': random_state
    }
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model_name, y_true, y_pred, target_names=['No Damage', 'Moderate', 'Severe']):
    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    actual_target_names = [target_names[i] for i in unique_labels] if len(unique_labels) <= len(target_names) else None
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=actual_target_names))
    print("Macro F1 Score:", f1_score(y_true, y_pred, average='macro'))

def plot_feature_importances(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features, y=top_features.index, palette="viridis")
    plt.title("XGBoost Feature Importances (Hazard Predictor)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show(block=False)
