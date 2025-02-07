import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier_model = joblib.load('models/logistic_model.pkl')


def classify_with_bert(sentences):
    message_embeddings = transformer_model.encode(sentences).reshape(1,-1)
    predicted = classifier_model.predict_proba(message_embeddings)[0]
    if max(predicted) < 0.5:
        return "Unclassifed"
    predicted_label = classifier_model.predict(message_embeddings)[0]
    return predicted_label




