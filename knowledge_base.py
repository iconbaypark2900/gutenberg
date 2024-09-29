from dotenv import load_dotenv
load_dotenv()

import pinecone
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from utils import error_handler, log_function_call
from config import Config
import asyncio
import aiohttp
from tqdm import tqdm
import concurrent.futures
import hashlib
from cachetools import LRUCache
import time
import random
import psutil
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from dask.distributed import Client, LocalCluster
from dask import delayed
import joblib
from pinecone import Pinecone
import os

print(f"Pinecone API Key from env: {os.environ.get('PINECONE_API_KEY')}")
pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
index = pc.Index(Config.PINECONE_INDEX_NAME)

client = MongoClient(Config.MONGODB_URI)
db = client["reference_assistant"]
metadata_collection = db["document_metadata"]
content_hash_collection = db["content_hash"]
performance_collection = db["performance_metrics"]
user_behavior_collection = db["user_behavior"]

model = SentenceTransformer('all-MiniLM-L6-v2')

CACHE_SIZE = 10000
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2500
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 500
MAX_RETRIES = 3

chunk_cache = LRUCache(maxsize=CACHE_SIZE)
predictive_cache = LRUCache(maxsize=CACHE_SIZE)

# Initialize distributed processing
local_cluster = LocalCluster()
distributed_client = Client(local_cluster)

class ChunkSizePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def train(self, content_lengths, chunk_sizes, processing_times):
        X = np.array(list(zip(content_lengths, chunk_sizes)))
        y = np.array(processing_times)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.trained = True
        score = self.model.score(X_test, y_test)
        print(f"Chunk Size Predictor R² score: {score}")

    def predict(self, content_length):
        if not self.trained:
            return determine_chunk_size(content_length)
        
        predictions = []
        for chunk_size in range(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, 100):
            predicted_time = self.model.predict([[content_length, chunk_size]])
            predictions.append((chunk_size, predicted_time[0]))
        
        return min(predictions, key=lambda x: x[1])[0]

chunk_size_predictor = ChunkSizePredictor()

@error_handler
@log_function_call
def compute_content_hash(content, chunk_size=1024*1024):
    md5 = hashlib.md5()
    for i in range(0, len(content), chunk_size):
        md5.update(content[i:i+chunk_size].encode())
    return md5.hexdigest()

def determine_chunk_size(content):
    content_length = len(content)
    return chunk_size_predictor.predict(content_length)

@delayed
@error_handler
@log_function_call
def process_document(doc_id, content):
    content_hash = compute_content_hash(content['text'])
    existing_hash = content_hash_collection.find_one({"doc_id": doc_id})
    
    if existing_hash and existing_hash['hash'] == content_hash:
        return None, None, 0  # No changes, skip processing
    
    start_time = time.time()
    chunk_size = determine_chunk_size(content['text'])
    full_embedding = model.encode(content['text']).tolist()
    
    chunks = [content['text'][i:i+chunk_size] for i in range(0, len(content['text']), chunk_size)]
    chunk_embeddings = model.encode(chunks).tolist()
    
    batches = [(doc_id, full_embedding, content['metadata'])]
    
    for i, chunk_embedding in enumerate(chunk_embeddings):
        chunk_id = f"{doc_id}_chunk_{i}"
        batches.append((chunk_id, chunk_embedding, {
            'doc_id': doc_id,
            'chunk_index': i,
            'text': chunks[i][:100] + '...'
        }))
    
    metadata_update = {
        "doc_id": doc_id,
        "metadata": {
            **content['metadata'],
            'chunk_count': len(chunks),
            'chunk_size': chunk_size
        }
    }
    
    content_hash_collection.update_one(
        {"doc_id": doc_id},
        {"$set": {"hash": content_hash}},
        upsert=True
    )
    
    processing_time = time.time() - start_time
    performance_collection.insert_one({
        "doc_id": doc_id,
        "content_length": len(content['text']),
        "chunk_size": chunk_size,
        "processing_time": processing_time
    })
    
    return batches, metadata_update, len(chunks)

@delayed
async def upsert_with_retry(batch, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            await asyncio.to_thread(index.upsert, vectors=batch)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.random()
            print(f"Upsert failed. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)

def get_system_load():
    return psutil.cpu_percent() / 100.0

def adaptive_batch_size(current_batch_size, system_load, processing_time):
    target_processing_time = 1.0  # Target processing time in seconds
    load_factor = 1 - system_load
    time_factor = target_processing_time / processing_time if processing_time > 0 else 1

    new_batch_size = int(current_batch_size * load_factor * time_factor)
    return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, new_batch_size))

@error_handler
@log_function_call
async def update_knowledge_base(preprocessed_content):
    total_chunks = 0
    all_batches = []
    all_metadata_updates = []

    # Train chunk size predictor
    performance_data = list(performance_collection.find())
    if performance_data:
        content_lengths = [data['content_length'] for data in performance_data]
        chunk_sizes = [data['chunk_size'] for data in performance_data]
        processing_times = [data['processing_time'] for data in performance_data]
        chunk_size_predictor.train(content_lengths, chunk_sizes, processing_times)

    # Use distributed processing for document processing
    futures = [process_document(doc_id, content) for doc_id, content in preprocessed_content.items()]
    results = distributed_client.compute(futures)
    results = distributed_client.gather(results)

    for result in results:
        if result[0]:  # Only update if there are changes
            all_batches.extend(result[0])
            all_metadata_updates.append(result[1])
            total_chunks += result[2]

    # Initial batch size
    batch_size = min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, len(all_batches) // 10))

    # Batch upsert to Pinecone with retry and adaptive batch size
    upsert_futures = []
    for i in tqdm(range(0, len(all_batches), batch_size), desc="Updating Pinecone"):
        batch = all_batches[i:i+batch_size]
        upsert_futures.append(upsert_with_retry(batch))
    
    await distributed_client.compute(upsert_futures)

    # Batch update MongoDB
    if all_metadata_updates:
        await asyncio.to_thread(metadata_collection.bulk_write, [
            pymongo.UpdateOne(
                {"doc_id": update["doc_id"]},
                {"$set": update["metadata"]},
                upsert=True
            ) for update in all_metadata_updates
        ])

    return len(all_metadata_updates), total_chunks

@error_handler
@log_function_call
async def get_relevant_chunks(query_embedding, top_k=5):
    # Check predictive cache first
    cached_results = predictive_cache.get(tuple(query_embedding))
    if cached_results:
        return cached_results

    results = await asyncio.to_thread(index.query, vector=query_embedding, top_k=top_k, include_metadata=True)
    relevant_chunks = []
    for match in results['matches']:
        chunk_id = match['id']
        if chunk_id in chunk_cache:
            chunk = chunk_cache[chunk_id]
        else:
            chunk = {
                'id': chunk_id,
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'doc_id': match['metadata'].get('doc_id', '')
            }
            chunk_cache[chunk_id] = chunk
        relevant_chunks.append(chunk)
    
    # Update predictive cache
    predictive_cache[tuple(query_embedding)] = relevant_chunks

    return relevant_chunks

@error_handler
@log_function_call
def update_user_behavior(user_id, query_embedding, selected_chunks):
    user_behavior_collection.insert_one({
        "user_id": user_id,
        "query_embedding": query_embedding,
        "selected_chunks": selected_chunks,
        "timestamp": time.time()
    })

@error_handler
@log_function_call
def train_predictive_cache_model():
    user_behaviors = list(user_behavior_collection.find())
    if len(user_behaviors) < 100:  # Not enough data to train
        return

    X = np.array([behavior['query_embedding'] for behavior in user_behaviors])
    y = np.array([behavior['selected_chunks'] for behavior in user_behaviors])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'predictive_cache_model.joblib')

@error_handler
@log_function_call
def predict_relevant_chunks(query_embedding):
    try:
        model = joblib.load('predictive_cache_model.joblib')
        predicted_chunks = model.predict([query_embedding])[0]
        return predicted_chunks
    except:
        return None