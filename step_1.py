import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors
import re
import os
from gensim.models.phrases import Phrases, Phraser

INPUT_FILE = "TR_Commands.xlsx"
OUTPUT_DATASET = "TR_Commands_Expanded.xlsx"
PRETRAINED_MODEL = "tvec.model"
REPAIRED_MODEL = "tvec.txt"
FINETUNED_MODEL = "smart_home_model.model"

variations_map = {
    "ışığı yak": ["lambayı aç", "ışıkları aç", "aydınlatmayı çalıştır", "şavkı yak", "odayı aydınlat"],
    "ışığı kapat": ["lambayı söndür", "ışıkları kapat", "aydınlatmayı durdur", "karanlık yap"],
    "televizyonu aç": ["tv'yi aç", "ekranı aç", "televizyonu çalıştır", "üniteyi aç"],
    "televizyonu kapat": ["tv'yi kapat", "ekranı karart", "televizyonu söndür"],
    "klimayı aç": ["klimayı çalıştır", "soğutmayı aç", "serinlet", "iklimlendirmeyi aç"],
    "klimayı kapat": ["klimayı durdur", "soğutmayı kapat", "iklimlendirmeyi durdur"],
    "sesi aç": ["sesi yükselt", "volümü artır", "daha yüksek ses", "sesi çoğalt"],
    "sesi kıs": ["sesi alçalt", "volümü düşür", "daha az ses", "sessize yaklaş"],
    "kapıyı kilitle": ["kapıyı kitle", "kilidi kapat", "evi kilitle"],
    "kapıyı aç": ["kilidi aç", "kapı kilidini aç", "misafir geldi"],
    "antrenman zamanı": ["spora başla", "egzersiz modu", "antrenmanı başlat"],
    "senaryo başlat": ["modu aç", "rutini başlat", "senaryoyu çalıştır"]
}

try:
    df = pd.read_excel(INPUT_FILE, header=None)
    target_col = 1 if (str(df.iloc[0, 0]).isdigit() or "#" in str(df.iloc[0, 0])) else 0
    raw_data = df.iloc[:, target_col].dropna().tolist()

    training_data = []
    for cmd in raw_data:
        cmd_str = str(cmd).strip().lower()
        if not cmd_str or cmd_str.isdigit() or cmd_str == "#": continue
        training_data.append({"Label": cmd_str, "Sentence": cmd_str})
        if cmd_str in variations_map:
            for v in variations_map[cmd_str]:
                training_data.append({"Label": cmd_str, "Sentence": v})

    full_corpus_df = pd.DataFrame(training_data)
    full_corpus_df.to_excel(OUTPUT_DATASET, index=False)
    print(f"Dataset saved: {len(full_corpus_df)} sentences")
except Exception as e:
    print(f"Error processing data: {e}")
    exit()


def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()


train_sentences = [clean_text(row) for row in full_corpus_df['Sentence']]

phrases = Phrases(train_sentences, min_count=1, threshold=0.01, scoring='npmi', delimiter='_')
bigram_transformer = Phraser(phrases)
train_sentences_ngram = [bigram_transformer[sent] for sent in train_sentences]

print(f"\nOriginal:  {train_sentences[0]}")
print(f"N-grams:   {train_sentences_ngram[0]}")
print(f"\nExample transformations:")
for i in range(min(5, len(train_sentences))):
    if train_sentences[i] != train_sentences_ngram[i]:
        print(f"  {train_sentences[i]} → {train_sentences_ngram[i]}")
        break
else:
    print(f"No bigrams in first 5 sentences")

print(f"\nBigrams detected: {len([k for k in phrases.vocab.keys() if '_' in str(k)])}")

if not os.path.exists(REPAIRED_MODEL):
    print(f"\nConverting '{PRETRAINED_MODEL}' to text format...")
    if not os.path.exists(PRETRAINED_MODEL):
        if os.path.exists("trmodel"):
            PRETRAINED_MODEL = "trmodel"
        else:
            print(f"Could not find pre-trained model")
            exit()

    try:
        kv = KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True, unicode_errors='ignore')
        kv.save_word2vec_format(REPAIRED_MODEL, binary=False)
        print(f"Saved as '{REPAIRED_MODEL}'")
    except Exception as e:
        print(f"Error: {e}")
        exit()
else:
    print(f"\nFound '{REPAIRED_MODEL}'")

print(f"\nLoading pre-trained vectors...")
try:
    pretrained_kv = KeyedVectors.load_word2vec_format(REPAIRED_MODEL, binary=False)
    vector_dim = pretrained_kv.vector_size
    print(f"Dimension: {vector_dim}")

    print("Training Model 1: Without N-grams")
    model = Word2Vec(vector_size=vector_dim, min_count=1, window=5)
    model.build_vocab(train_sentences)

    count = 0
    for word in model.wv.index_to_key:
        if word in pretrained_kv:
            model.wv[word] = pretrained_kv[word]
            count += 1

    print(f"Transferred vectors: {count} words")
    model.train(train_sentences, total_examples=len(train_sentences), epochs=20)
    model.save(FINETUNED_MODEL)
    print(f"Saved: {FINETUNED_MODEL}")

    print("Training Model 2: With N-grams")
    model_ngram = Word2Vec(vector_size=vector_dim, min_count=1, window=5)
    model_ngram.build_vocab(train_sentences_ngram)

    count_ngram = 0
    for word in model_ngram.wv.index_to_key:
        if word in pretrained_kv:
            model_ngram.wv[word] = pretrained_kv[word]
            count_ngram += 1

    print(f"Transferred vectors: {count_ngram} words")
    model_ngram.train(train_sentences_ngram, total_examples=len(train_sentences_ngram), epochs=20)

    FINETUNED_MODEL_NGRAM = "smart_home_model_ngram.model"
    model_ngram.save(FINETUNED_MODEL_NGRAM)
    print(f"Saved: {FINETUNED_MODEL_NGRAM}")

except Exception as e:
    print(f"Error during training: {e}")