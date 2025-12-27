import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import re

THRESHOLD = 0.75


def preprocess(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()


def get_mean_vector(sentence, model):
    words = preprocess(sentence)
    valid_vectors = [model.wv[w] for w in words if w in model.wv]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)


def get_weighted_vector(sentence, model, tfidf_weights):
    words = preprocess(sentence)
    valid_vectors = []
    weights = []

    for word in words:
        if word in model.wv:
            valid_vectors.append(model.wv[word])
            weights.append(tfidf_weights.get(word, 1.0))

    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.average(valid_vectors, axis=0, weights=weights)


def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def evaluate_model(model_path, method_name, use_tfidf=False, use_ngrams=False):
    print(f"EVALUATING: {method_name}")
    model = Word2Vec.load(model_path)
    train_df = pd.read_excel("TR_Commands_Expanded.xlsx")
    test_df = pd.read_excel("test_commands.xlsx")

    corpus = train_df['Sentence'].tolist()
    labels = train_df['Label'].tolist()

    bigram_transformer = None
    if use_ngrams:
        from gensim.models.phrases import Phrases, Phraser
        train_sentences = [preprocess(s) for s in corpus]
        phrases = Phrases(train_sentences, min_count=1, threshold=0.01, scoring='npmi', delimiter='_')
        bigram_transformer = Phraser(phrases)

    # Setup vectorization
    tfidf_weights = {}
    if use_ngrams:
        # First transform corpus to n-grams
        corpus_ngrams = [' '.join(bigram_transformer[preprocess(s)]) for s in corpus]

        if use_tfidf:
            # Train TF-IDF on n-gram transformed corpus
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectorizer.fit(corpus_ngrams)
            tfidf_weights = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
            target_vectors = [get_weighted_vector(s, model, tfidf_weights) for s in corpus_ngrams]
        else:
            # Mean vectors on n-grams
            target_vectors = [get_mean_vector(s, model) for s in corpus_ngrams]
    else:
        # Word-level (no n-grams)
        if use_tfidf:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectorizer.fit(corpus)
            tfidf_weights = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
            target_vectors = [get_weighted_vector(s, model, tfidf_weights) for s in corpus]
        else:
            target_vectors = [get_mean_vector(s, model) for s in corpus]

    # Evaluate
    predictions = []
    ground_truth = []
    mismatches = []

    for idx, row in test_df.iterrows():
        user_input = row['Command']
        is_correct = row['IsCorrect']

        # Apply n-grams to test input if needed
        if use_ngrams and bigram_transformer:
            processed_input = ' '.join(bigram_transformer[preprocess(user_input)])
        else:
            processed_input = user_input

        # Get input vector
        if use_tfidf:
            input_vec = get_weighted_vector(processed_input, model, tfidf_weights)
        else:
            input_vec = get_mean_vector(processed_input, model)

        # Calculate similarities
        similarities = [cosine_similarity(input_vec, tv) for tv in target_vectors]
        best_score = max(similarities) if similarities else 0.0
        best_idx = similarities.index(best_score) if similarities else -1

        # Decision
        predicted = 1 if best_score >= THRESHOLD else 0
        predictions.append(predicted)
        ground_truth.append(is_correct)

        # Track mismatches
        if predicted != is_correct:
            matched_label = labels[best_idx] if best_idx >= 0 else "NONE"
            mismatches.append({
                'command': user_input,
                'predicted': predicted,
                'actual': is_correct,
                'score': best_score,
                'matched': matched_label
            })

        # Show mismatches
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)} errors):")
        for m in mismatches:
            pred_str = "ACCEPTED" if m['predicted'] == 1 else "REJECTED"
            actual_str = "SHOULD ACCEPT" if m['actual'] == 1 else "SHOULD REJECT"
            print(f"  '{m['command']}'")
            print(f"   {pred_str} (score: {m['score']:.3f}) | {actual_str} | Matched: {m['matched']}")
    else:
        print("\nNo mismatches")

    # Calculate metrics
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions)

    print(f"\nMETRICS:")
    print(f"  Precision: {precision:.3f} (of accepted commands, {precision * 100:.1f}% were correct)")
    print(f"  Recall:    {recall:.3f} (caught {recall * 100:.1f}% of valid commands)")
    print(f"  F1-Score:  {f1:.3f}")

    print(f"\nCONFUSION MATRIX:")
    print(f"                     Predicted NO   Predicted YES")
    print(f"  Actual NO  (TN/FP)      {cm[0][0]:4d}           {cm[0][1]:4d}")
    print(f"  Actual YES (FN/TP)      {cm[1][0]:4d}           {cm[1][1]:4d}")
    print(f"\n  True Negatives (TN):  {cm[0][0]:4d} - Correctly rejected invalid commands")
    print(f"  False Positives (FP): {cm[0][1]:4d} - Incorrectly accepted invalid commands")
    print(f"  False Negatives (FN): {cm[1][0]:4d} - Incorrectly rejected valid commands")
    print(f"  True Positives (TP):  {cm[1][1]:4d} - Correctly accepted valid commands")

    return {"Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Mismatches": len(mismatches),
        "TP": cm[1][1],
        "TN": cm[0][0],
        "FP": cm[0][1],
        "FN": cm[1][0]
    }


results = {}

results["Mean (No N-grams)"] = evaluate_model(
    "smart_home_model.model",
    "Method 1: Mean of Vectors (No N-grams)",
    use_tfidf=False,
    use_ngrams=False
)

results["Mean (With N-grams)"] = evaluate_model(
    "smart_home_model_ngram.model",
    "Method 2: Mean of Vectors (With N-grams)",
    use_tfidf=False,
    use_ngrams=True
)

results["TF-IDF (No N-grams)"] = evaluate_model(
    "smart_home_model.model",
    "Method 3: TF-IDF Weighted (No N-grams)",
    use_tfidf=True,
    use_ngrams=False
)

results["TF-IDF (With N-grams)"] = evaluate_model(
    "smart_home_model_ngram.model",
    "Method 4: TF-IDF Weighted (With N-grams)",
    use_tfidf=True,
    use_ngrams=True
)

print("FINAL COMPARISON TABLE")
print(f"{'Method':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Errors':<10}")
for method, metrics in results.items():
    print(f"{method:<35} {metrics['Precision']:<12.3f} {metrics['Recall']:<12.3f} {metrics['F1']:<12.3f} {metrics['Mismatches']:<10d}")

best_method = max(results.items(), key=lambda x: x[1]['F1'])
print(f"BEST METHOD: {best_method[0]}")
print(f"F1-Score: {best_method[1]['F1']:.3f}")
print(f"Precision: {best_method[1]['Precision']:.3f}")
print(f"Recall: {best_method[1]['Recall']:.3f}")

print(f"{'Method':<35} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
for method, metrics in results.items():
    print(f"{method:<35} {metrics['TP']:<6} {metrics['TN']:<6} {metrics['FP']:<6} {metrics['FN']:<6}")