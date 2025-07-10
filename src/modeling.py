# src/modeling.py

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import os

def train_models(X_train, y_train):
    """
    Melatih tiga model: Logistic Regression, Random Forest, dan XGBoost.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

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

def run_training():
    mlflow.set_experiment("fraud-detection")

    # Load data
    data = pd.read_csv("./data/creditcard_scaled.csv")
    X = data.drop(columns=['Time', 'Class', 'Hour', 'Amount'])
    y = data['Class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Melatih model
    models = train_models(X_train_res, y_train_res)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Logging parameters (contoh: hanya untuk Random Forest dan XGBoost)
            if model_name == "Random Forest":
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)
            elif model_name == "XGBoost":
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)
            elif model_name == "Logistic Regression":
                mlflow.log_param("max_iter", model.max_iter)

            # Predict
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Evaluasi lanjutan
            eval_metrics = evaluate_model(model, X_test, y_test)
            roc_auc = eval_metrics["roc_auc"]
            pr_auc = eval_metrics["pr_auc"]

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)

            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)

            # Log model ke MLflow
            mlflow.sklearn.log_model(model, "model")

            # === Visualization Logging ===
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            plt.figure(figsize=(6, 4))
            disp.plot(cmap=plt.cm.Blues, values_format="d")
            plt.title(f"Confusion Matrix - {model_name}")
            cm_filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(cm_filename)
            mlflow.log_artifact(cm_filename)
            plt.close()

            # ROC Curve
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_val:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            roc_filename = f"roc_curve_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(roc_filename)
            mlflow.log_artifact(roc_filename)
            plt.close()

            print(f"{model_name} selesai. Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    run_training()