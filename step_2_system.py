import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_PATH = "smart_home_model.model"
DATASET_PATH = "TR_Commands_Expanded.xlsx"
THRESHOLD = 0.75  # Confidence threshold

try:
    model = Word2Vec.load(MODEL_PATH)
    df = pd.read_excel(DATASET_PATH)
    corpus = df['Sentence'].tolist()
    labels = df['Label'].tolist()
    print(f"Dataset loaded ({len(corpus)} commands)")

    # Train TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus)
    # Create a dictionary {word: weight}
    word2weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

except Exception as e:
    print(f"Error loading resources: {e}")
    exit()
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()


# Mean of Word Vectors
def get_mean_vector(sentence, w2v_model):
    words = preprocess(sentence)
    valid_vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not valid_vectors: return np.zeros(w2v_model.vector_size)
    return np.mean(valid_vectors, axis=0)


# Weighted Mean (TF-IDF)
def get_weighted_vector(sentence, w2v_model, tfidf_weights):
    words = preprocess(sentence)
    valid_vectors = []
    weights = []

    for word in words:
        if word in w2v_model.wv:
            valid_vectors.append(w2v_model.wv[word])
            # Use TF-IDF weight if available, else 1.0
            weights.append(tfidf_weights.get(word, 1.0))

    if not valid_vectors: return np.zeros(w2v_model.vector_size)
    return np.average(valid_vectors, axis=0, weights=weights)


target_vectors_mean = [get_mean_vector(s, model) for s in corpus]
target_vectors_weighted = [get_weighted_vector(s, model, word2weight) for s in corpus]


def find_best_match(user_input, method="mean"):
    # Convert user input to vector
    if method == "mean":
        input_vec = get_mean_vector(user_input, model)
        target_matrix = target_vectors_mean
    else:
        input_vec = get_weighted_vector(user_input, model, word2weight)
        target_matrix = target_vectors_weighted

    # Calculate Cosine Similarity against ALL commands
    similarities = cosine_similarity([input_vec], target_matrix)[0]

    # Find the winner
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    # Check Threshold
    if best_score >= THRESHOLD:
        return labels[best_idx], best_score
    else:
        return "NOT RECOGNIZED", best_score


print("\nType a command (e.g., 'ışıkları aç', 'tv kapat'). Type 'exit' to stop.")

while True:
    user_text = input("\nYour Command: ")
    if user_text.lower() == "exit": break

    match_mean, score_mean = find_best_match(user_text, method="mean")
    match_tfidf, score_tfidf = find_best_match(user_text, method="weighted")

    print(f"Method 1 (Mean):    {match_mean} (Score: {score_mean:.2f})")
    print(f"Method 2 (TF-IDF):  {match_tfidf} (Score: {score_tfidf:.2f})")