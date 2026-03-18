import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

def evaluate(model_path='models/fraud_model.pkl', data_path='data/featured_data.csv'):
    model = joblib.load(model_path)
    threshold = joblib.load('models/threshold.pkl')

    df = pd.read_csv(data_path)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()