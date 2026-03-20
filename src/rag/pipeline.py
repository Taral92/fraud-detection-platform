from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid

# Initialize clients
client = QdrantClient(":memory:")
# client = QdrantClient(host="localhost", port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

COLLECTION_NAME = "fraud_transactions"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size

def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"Collection {COLLECTION_NAME} created")

def ingest_transactions(data_path='data/featured_data.csv', batch_size=100):
    df = pd.read_csv(data_path)
    
    # Only ingest fraud transactions for investigation
    fraud_df = df[df['isFraud'] == 1].head(1000)
    
    points = []
    for idx, row in fraud_df.iterrows():
        # Create text description of transaction
        text = f"""
        Fraud transaction:
        Amount: {row.get('TransactionAmt', 0):.2f}
        Hour: {row.get('hour', 0)}
        Day: {row.get('day', 0)}
        Card frequency: {row.get('card1_freq', 0)}
        Is night: {row.get('is_night', 0)}
        C sum: {row.get('C_sum', 0)}
        """
        
        # Generate embedding
        vector = encoder.encode(text).tolist()
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "transaction_amount": float(row.get('TransactionAmt', 0)),
                "hour": int(row.get('hour', 0)),
                "is_night": int(row.get('is_night', 0)),
                "is_fraud": int(row['isFraud'])
            }
        ))
        
        # Upload in batches
        if len(points) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points = []
            print(f"Uploaded batch at index {idx}")
    
    # Upload remaining
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    print("Ingestion complete")

def search_similar(query: str, limit: int = 5):
    query_vector = encoder.encode(query).tolist()
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    return results

def ask_question(question: str):
    # Search similar transactions
    results = search_similar(question)
    
    # Build context
    context = ""
    for r in results:
        context += f"""
        Transaction:
        Amount: {r.payload.get('transaction_amount')}
        Hour: {r.payload.get('hour')}
        Is Night: {r.payload.get('is_night')}
        Score: {r.score:.4f}
        ---
        """
    
    return {
        "question": question,
        "similar_transactions": context,
        "count": len(results)
    }

if __name__ == "__main__":
    create_collection()
    ingest_transactions()
    
    # Test query
    result = ask_question("Show me high value fraud transactions at night")
    print(result)