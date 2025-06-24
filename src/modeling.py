# src/modeling.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def train_models(X_train, y_train):
    """
    Melatih tiga model: Logistic Regression, Random Forest, dan XGBoost.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}

def evaluate_model(model, X_test, y_test):
    """
    Evaluasi sederhana dengan metrik utama.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    return {
        "classification_report": report,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
