import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors
import re
import os
import numpy as np

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
    # checks if col A is an ID number
    target_col = 1 if (str(df.iloc[0, 0]).isdigit() or "#" in str(df.iloc[0, 0])) else 0
    raw_data = df.iloc[:, target_col].dropna().tolist()

    training_data = []
    for cmd in raw_data:
        cmd_str = str(cmd).strip().lower()
        if not cmd_str or cmd_str.isdigit() or cmd_str == "#": continue
        training_data.append({"Label": cmd_str, "Sentence": cmd_str})
        if cmd_str in variations_map:
            for v in variations_map[cmd_str]: training_data.append({"Label": cmd_str, "Sentence": v})

    full_corpus_df = pd.DataFrame(training_data)
    full_corpus_df.to_excel(OUTPUT_DATASET, index=False)
    print(f"Dataset saved to '{OUTPUT_DATASET}' with {len(full_corpus_df)} sentences.")
except Exception as e:
    print(f"Error processing data: {e}")
    exit()


def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()


train_sentences = [clean_text(row) for row in full_corpus_df['Sentence']]


if not os.path.exists(REPAIRED_MODEL):
    print(f"Clean file '{REPAIRED_MODEL}' not found. Creating it from '{PRETRAINED_MODEL}'...")

    if not os.path.exists(PRETRAINED_MODEL):
        # Fallback check
        if os.path.exists("trmodel"):
            PRETRAINED_MODEL = "trmodel"
        else:
            print(f"Could not find '{PRETRAINED_MODEL}'")
            exit()

    try:
        print("Loading binary file...")
        kv = KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True, unicode_errors='ignore')

        # Save it as clean text
        print(f"Saving as text to '{REPAIRED_MODEL}'")
        kv.save_word2vec_format(REPAIRED_MODEL, binary=False)

    except Exception as e:
        print(f"Error converting file: {e}")
        exit()
else:
    print(f"Found '{REPAIRED_MODEL}', skipping conversion.")

print(f"Loading '{REPAIRED_MODEL}' into memory...")
try:
    # Load the clean text model
    pretrained_kv = KeyedVectors.load_word2vec_format(REPAIRED_MODEL, binary=False)
    vector_dim = pretrained_kv.vector_size
    print(f"Loaded pre-trained vectors. Dimension: {vector_dim}")

    # Initialize Smart Home Model
    model = Word2Vec(vector_size=vector_dim, min_count=1, window=5)
    model.build_vocab(train_sentences)

    # Manual Injection
    count = 0
    for word in model.wv.index_to_key:
        if word in pretrained_kv:
            model.wv[word] = pretrained_kv[word]
            count += 1

    print(f"Transferred vectors for {count} words.")

    # Fine-Tune
    model.train(train_sentences, total_examples=len(train_sentences), epochs=20)

    model.save(FINETUNED_MODEL)
    print(f"Model saved as: {FINETUNED_MODEL}")

except Exception as e:
    print(f"Error during training: {e}")