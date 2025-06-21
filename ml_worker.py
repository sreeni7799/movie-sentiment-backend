# ml_worker.py
import redis
from rq import Worker, Queue
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your ML service function
from app import process_sentiment  # Replace 'paste-2' with your actual ML service filename

if __name__ == '__main__':
    print("Starting ML Worker...")
    
    # Connect to Redis
    redis_conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Create worker
    worker = Worker(['sentiment_analysis'], connection=redis_conn)
    
    print("Worker started, waiting for jobs...")
    worker.work()