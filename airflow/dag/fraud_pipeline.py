from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/app')

default_args = {
    'owner': 'fraud-detection',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'fraud_detection_pipeline',
    default_args=default_args,
    description='End to end fraud detection pipeline',
    schedule_interval='@daily',
    catchup=False
)

def preprocess_data():
    from src.data.preprocess import load_data, clean_data, encode_categorical
    df = load_data('data/train_transaction.csv', 'data/train_identity.csv')
    df = clean_data(df)
    df = encode_categorical(df)
    df.to_csv('data/cleaned_data.csv', index=False)
    print(f"Preprocessing done — shape: {df.shape}")

def engineer_features():
    import pandas as pd
    from src.data.features import create_features
    df = pd.read_csv('data/cleaned_data.csv')
    df = create_features(df)
    df.to_csv('data/featured_data.csv', index=False)
    print(f"Feature engineering done — shape: {df.shape}")

def train_model():
    from src.models.train import train
    model, auc = train()
    print(f"Training done — AUC: {auc:.4f}")

def evaluate_model():
    from src.models.evaluate import evaluate
    evaluate()
    print("Evaluation done")

# Define tasks
task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

task_features = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

