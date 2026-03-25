# Getting the libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


# Defining model training function
def train_model(X, y):
    """
    Train an XGBoost model using Stratified K-Fold Cross Validation.

    WHY THIS FUNCTION EXISTS:
    - Avoids overfitting from single train-test split
    - Ensures robust and consistent AUC performance
    - Selects best-performing model across folds
    """

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,        # preserve churn ratio
        random_state=42
    )

    # StratifiedKFold preserves class distribution in each fold
    # This is critical because churn datasets are imbalanced
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []      # Store AUC for each fold
    best_model = None    # Keep best performing model
    best_auc = 0         # Track highest AUC

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):

        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Handle class imbalance
        # WHY: More non-churn than churn → model may ignore minority class
        # scale_pos_weight balances positive vs negative class impact
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        # Initialize XGBoost model
        model = XGBClassifier(
            n_estimators=300,        # Number of trees → higher = better learning (but slower)
            max_depth=5,             # Controls model complexity (avoid overfitting)
            learning_rate=0.05,      # Smaller step size → better generalization
            subsample=0.8,           # Use 80% data per tree → reduces overfitting
            colsample_bytree=0.8,    # Use 80% features per tree → adds randomness
            scale_pos_weight= scale_pos_weight,  # Handle imbalance
            eval_metric='logloss',   # Required for XGBoost stability
            random_state=42
        )

        # Train Model
        model.fit(X_train, y_train)

        # Predict probabilities (NOT Class labels)
        # WHY: AUC depends on probabilities, not classification
        y_proba = model.predict_proba(X_val)[:, 1]

        # Compute ROC-AUC Score
        auc = roc_auc_score(y_val, y_proba)

        print(f"[Fold {fold + 1}] AUC: {auc: .4f}")
        auc_scores.append(auc)

        # Keep best model (based on validation AUC)
        if auc > best_auc:
            best_auc = auc
            best_model = model
    
    # Print average performance -> More reliable metric than single split
    print("\nMean AUC: ", np.mean(auc_scores))

    # Save best model for reuse
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    return best_model, X_test, y_test
