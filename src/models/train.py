import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

def train(data_path='data/featured_data.csv', model_path='models/fraud_model.pkl'):
    df = pd.read_csv(data_path)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20
        )

        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc       = roc_auc_score(y_test, y_prob)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.xgboost.log_model(model, "model")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(0.4, model_path.replace('fraud_model', 'threshold'))

        print(f"AUC: {auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Model saved to {model_path}")

        return model, auc

if __name__ == "__main__":
    train()