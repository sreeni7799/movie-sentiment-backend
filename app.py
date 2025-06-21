from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import joblib
import numpy as np
from preprocessing import clean_text
from rq import Worker, Connection
import redis
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

try:
    redis_conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_conn.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_conn = None

try:
    pipeline = joblib.load('model/sentiment_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    print("Model and label encoder loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    pipeline = None
    label_encoder = None

# ADD THIS WORKER FUNCTION
def process_sentiment(review_data):
    """
    Worker function to process sentiment analysis jobs from the queue
    This function will be called by RQ workers
    """
    try:
        if pipeline is None or label_encoder is None:
            raise Exception("Model not loaded")
        
        # Extract data
        review_text = review_data.get('text', '')
        movie_name = review_data.get('movie_name', 'Unknown Movie')
        
        logger.info(f"Processing sentiment for movie: {movie_name}")
        
        # Clean text
        cleaned_text = clean_text(review_text)
        
        if not cleaned_text.strip():
            raise Exception("Text becomes empty after cleaning")
        
        # Make prediction
        prediction = pipeline.predict([cleaned_text])[0]
        prediction_proba = pipeline.predict_proba([cleaned_text])[0]
        confidence = float(np.max(prediction_proba))
        sentiment_label = label_encoder.inverse_transform([prediction])[0]
        
        # Prepare result
        result = {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "movie_name": movie_name,
            "original_text": review_text,
            "cleaned_text": cleaned_text,
            "processed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        logger.info(f"Completed sentiment analysis for {movie_name}: {sentiment_label}")
        return result
        
    except Exception as e:
        logger.error(f"Error in process_sentiment: {str(e)}")
        return {
            "error": str(e),
            "status": "failed",
            "processed_at": datetime.now().isoformat()
        }
    
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is working!"})

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if pipeline is not None else "not_loaded"
    return jsonify({
        "status": "ML service is running",
        "model_status": model_status,
        "service": "sentiment_analysis"
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        if pipeline is None or label_encoder is None:
            return jsonify({
                "error": "Model not loaded."
            }), 500
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "JSON data not provided"}), 400
        
        if 'text' not in data:
            return jsonify({"error": "Missing text field in request"}), 400
        
        review_text = data['text']
        movie_name = data.get('movie_name', 'Unknown Movie')
        
        cleaned_text = clean_text(review_text)
        
        if not cleaned_text.strip():
            return jsonify({
                "error": "Text becomes empty after cleaning"
            }), 400
        
        prediction = pipeline.predict([cleaned_text])[0]
        
        prediction_proba = pipeline.predict_proba([cleaned_text])[0]
        confidence = float(np.max(prediction_proba))
        
        sentiment_label = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            "sentiment": sentiment_label,
            "confidence": confidence,
            "movie_name": movie_name,
            "original_text": review_text,
            "cleaned_text": cleaned_text
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

''''''''''
@app.route('/model-info', methods=['GET'])
def model_info():
    if pipeline is None or label_encoder is None:
        return jsonify({
            "model_loaded": False,
            "message": "Model not loaded"
        })
    
    try:
        classes = label_encoder.classes_.tolist()
        return jsonify({
            "model_loaded": True,
            "model_type": "Logistic Regression with TF-IDF",
            "available_classes": classes,
            "message": "Model ready for predictions"
        })
    except Exception as e:
        return jsonify({
            "error": f"Error getting model info: {str(e)}"
        }), 500
'''''''''''
@app.route('/worker/start', methods=['POST'])
def start_worker():
    """Start a worker process (for development/testing)"""
    try:
        if redis_conn is None:
            return jsonify({"error": "Redis not connected"}), 500
            
        return jsonify({
            "message": "Use 'python ml_worker.py' to start worker",
            "info": "This endpoint is for reference only"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/worker/status', methods=['GET'])
def worker_status():
    """Check worker and queue status"""
    try:
        if redis_conn is None:
            return jsonify({"error": "Redis not connected"}), 500
            
        # Get queue info
        from rq import Queue
        queue = Queue('sentiment_analysis', connection=redis_conn)
        
        return jsonify({
            "redis_connected": True,
            "queue_length": len(queue),
            "failed_jobs": len(queue.failed_job_registry),
            "model_loaded": pipeline is not None,
            "service": "ml_worker"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask ML Service...")
    print("Model loaded:", pipeline is not None)
    print("Label encoder loaded:", label_encoder is not None)
    print("Redis connected:", redis_conn is not None)
    print("\nTo start worker: python ml_worker.py")
    print("To test API: curl http://localhost:8000/health")
    app.run(debug=True, port=8000, host='0.0.0.0')