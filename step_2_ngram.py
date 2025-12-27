import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_PATH = "smart_home_model_ngram.model"
DATASET_PATH = "TR_Commands_Expanded.xlsx"
THRESHOLD = 0.75

def preprocess(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()

try:
    model = Word2Vec.load(MODEL_PATH)
    df = pd.read_excel(DATASET_PATH)
    corpus = df['Sentence'].tolist()
    labels = df['Label'].tolist()
    print(f"Dataset loaded ({len(corpus)} commands)")

    # Build bigram transformer
    train_sentences = [re.sub(r'[^\w\s]', '', str(s).lower()).split() for s in corpus]
    phrases = Phrases(train_sentences, min_count=1, threshold=0.01, scoring='npmi', delimiter='_')
    bigram = Phraser(phrases)

    # Transform corpus to n-grams
    corpus_ngrams = [' '.join(bigram[preprocess(s)]) for s in corpus]

    # Train TF-IDF on n-grams
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus_ngrams)
    word2weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

    print(f"N-gram model loaded with {len(model.wv)} phrases")

except Exception as e:
    print(f"Error loading resources: {e}")
    exit()


def get_mean_vector(sentence, w2v_model):
    words = preprocess(sentence)
    # Apply bigram transformation
    words = bigram[words]
    valid_vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not valid_vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(valid_vectors, axis=0)


def get_weighted_vector(sentence, w2v_model, tfidf_weights):
    words = preprocess(sentence)
    # Apply bigram transformation
    words = bigram[words]
    valid_vectors = []
    weights = []

    for word in words:
        if word in w2v_model.wv:
            valid_vectors.append(w2v_model.wv[word])
            weights.append(tfidf_weights.get(word, 1.0))

    if not valid_vectors:
        return np.zeros(w2v_model.vector_size)
    return np.average(valid_vectors, axis=0, weights=weights)


# Pre-compute target vectors (with n-grams)
target_vectors_mean = [get_mean_vector(s, model) for s in corpus]
target_vectors_weighted = [get_weighted_vector(s, model, word2weight) for s in corpus]


def find_best_match(user_input, method="mean"):
    if method == "mean":
        input_vec = get_mean_vector(user_input, model)
        target_matrix = target_vectors_mean
    else:
        input_vec = get_weighted_vector(user_input, model, word2weight)
        target_matrix = target_vectors_weighted

    similarities = cosine_similarity([input_vec], target_matrix)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= THRESHOLD:
        return labels[best_idx], best_score
    else:
        return "NOT RECOGNIZED", best_score

print("\nType a command (e.g., 'ışıkları aç', 'tv kapat'). Type 'exit' to stop.")

while True:
    user_text = input("\nYour Command: ")
    if user_text.lower() == "exit": break

    transformed = bigram[preprocess(user_text)]
    print(f"Transformed to: {transformed}")

    match_mean, score_mean = find_best_match(user_text, method="mean")
    match_tfidf, score_tfidf = find_best_match(user_text, method="weighted")

    print(f"  Method 1 (Mean):    {match_mean} (Score: {score_mean:.2f})")
    print(f"  Method 2 (TF-IDF):  {match_tfidf} (Score: {score_tfidf:.2f})")