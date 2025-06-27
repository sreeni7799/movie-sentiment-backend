# Add this to your ml-service to properly load the model
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        self.is_fitted = False
        
    def train_model(self, csv_path='data/reviews_labeled.csv'):
        """Train the model if not already trained"""
        try:
            # Load labeled data
            df = pd.read_csv(csv_path)
            
            # Clean data - remove rows where label is 'Unknown'
            df_clean = df[df['label'] != 'Unknown'].dropna(subset=['review', 'label'])
            
            # Extract features and labels
            X = df_clean['review'].astype(str)
            y = df_clean['label']
            
            # Convert labels to binary (positive=1, negative=0)
            y_binary = (y == 'positive').astype(int)
            
            # Fit the vectorizer on all training data
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Train the model
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y_binary, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Save the trained model and vectorizer
            joblib.dump(self.model, 'model/sentiment_model.pkl')
            joblib.dump(self.vectorizer, 'model/tfidf_vectorizer.pkl')
            
            print(f"Model trained successfully on {len(df_clean)} samples")
            print(f"Test accuracy: {self.model.score(X_test, y_test):.3f}")
            
        except Exception as e:
            print(f"Error training model: {e}")
            # Fallback: create a basic model
            self._create_fallback_model()
    
    def load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists('model/sentiment_model.pkl') and os.path.exists('model/tfidf_vectorizer.pkl'):
                self.model = joblib.load('model/sentiment_model.pkl')
                self.vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
                self.is_fitted = True
                print("Pre-trained model loaded successfully")
            else:
                print("No pre-trained model found, training new model...")
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model for testing"""
        # Sample data for fallback
        sample_reviews = [
            "This movie was amazing and fantastic",
            "Terrible movie, waste of time",
            "Great acting and storyline",
            "Boring and poorly made"
        ]
        sample_labels = [1, 0, 1, 0]  # 1=positive, 0=negative
        
        X_sample = self.vectorizer.fit_transform(sample_reviews)
        self.model.fit(X_sample, sample_labels)
        self.is_fitted = True
        print("Fallback model created")
    
    def predict_sentiment(self, text):
        """Predict sentiment for given text"""
        if not self.is_fitted:
            self.load_model()
        
        try:
            # Vectorize the input text
            text_vectorized = self.vectorizer.transform([str(text)])
            
            # Get prediction
            prediction = self.model.predict(text_vectorized)[0]
            confidence = self.model.predict_proba(text_vectorized)[0].max()
            
            sentiment = "positive" if prediction == 1 else "negative"
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "status": "error",
                "error": str(e)
            }


# Initialize the model when the service starts
sentiment_analyzer = SentimentModel()
sentiment_analyzer.load_model()  # This will train if no model exists