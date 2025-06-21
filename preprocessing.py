# preprocessing.py
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def preprocess_dataframe(df, is_labeled=True):
    df = df.copy()
    df['review'] = df['review'].apply(clean_text)
    if is_labeled:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        return df, label_encoder
    return df
