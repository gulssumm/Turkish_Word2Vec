import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors
import re
import os
from gensim.models.phrases import Phrases, Phraser
import unicodedata

INPUT_FILE = "TR_Commands.xlsx"
OUTPUT_DATASET = "TR_Commands_Expanded.xlsx"
PRETRAINED_MODEL = "tvec.model"
REPAIRED_MODEL = "tvec.txt"
FINETUNED_MODEL = "smart_home_model.model"

# Variations map for ALL 71 commands
# Generated using semantic understanding of Turkish smart home commands
variations_map = {
    # Light commands (1-15)
    "ışığı aç": ["lambayı aç", "ışıkları aç", "aydınlatmayı çalıştır", "ışıkları yak"],
    "ışığı yak": ["lambayı yak", "aydınlatmayı yak", "ışıkları çalıştır"],
    "ışığı söndür": ["lambayı söndür", "ışıkları söndür", "aydınlatmayı kapat"],
    "ışığı kapa": ["lambayı kapat", "ışıkları kapat", "karanlık yap"],
    "ışığı arttır": ["lambayı arttır", "daha parlak yap", "ışığı yükselt"],
    "ışığı kıs": ["lambayı kıs", "ışığı düşür", "loş yap"],
    "ışığı azalt": ["lambayı azalt", "ışığı alçalt", "daha az ışık"],
    "parlaklığı arttır": ["parlaklık yükselt", "daha parlak", "parlaklığı çoğalt"],
    "parlaklığı azalt": ["parlaklık düşür", "daha az parlak", "parlaklığı kıs"],
    "aydınlatmayı arttır": ["aydınlatma seviyesini yükselt", "daha aydınlık"],
    "aydınlatmayı azalt": ["aydınlatma seviyesini düşür", "daha az aydınlık"],
    "lambayı aç": ["lambayı çalıştır", "lambayı yak", "lamba ver"],
    "lambayı yak": ["lambayı aç", "lamba çalıştır"],
    "lambayı söndür": ["lambayı kapat", "lambayı durdur"],
    "lambayı kapat": ["lambayı söndür", "lambayı kıs"],

    # Color commands (16-18)
    "kırmızı": ["rengi kırmızı yap", "kırmızıya çevir", "kırmızı renk"],
    "mavi": ["rengi mavi yap", "maviye çevir", "mavi renk"],
    "yeşil": ["rengi yeşil yap", "yeşile çevir", "yeşil renk"],

    # Socket commands (19-20)
    "prizi aç": ["prize güç ver", "prizi çalıştır", "elektriği aç"],
    "prizi kapat": ["prize gücü kes", "prizi söndür", "elektriği kapat"],

    # Climate/AC commands (21-44)
    "klimayı aç": ["klimayı çalıştır", "soğutmayı aç", "klima ver"],
    "klimayı kapa": ["klimayı durdur", "soğutmayı kapat", "klimayı söndür"],
    "iklimlendirmeyi aç": ["iklimlendirme çalıştır", "havalandırmayı aç"],
    "iklimlendirmeyi kapa": ["iklimlendirme durdur", "havalandırmayı kapat"],
    "ısıtmayı aç": ["ısıtıcıyı çalıştır", "ısınmayı başlat"],
    "ısıtmayı kapa": ["ısıtıcıyı durdur", "ısınmayı kapat"],
    "ısıt": ["ısınma başlasın", "evi ısıt", "sıcak yap"],
    "soğut": ["soğutma başlasın", "evi soğut", "serin yap"],
    "sıcaklığı arttır": ["ısıyı yükselt", "daha sıcak yap", "derece arttır"],
    "sıcaklığı düşür": ["ısıyı düşür", "daha soğuk yap", "derece azalt"],
    "evi ısıt": ["ev sıcak olsun", "evi sıcak yap", "ısınma başlasın"],
    "evi soğut": ["ev serin olsun", "evi serin yap", "soğutma başlasın"],
    "odayı ısıt": ["oda sıcak olsun", "odayı sıcak yap"],
    "odayı soğut": ["oda serin olsun", "odayı serin yap"],
    "kombiyi aç": ["kombiyi çalıştır", "kombiye başla"],
    "kombiyi kapa": ["kombiyi durdur", "kombiyi söndür"],
    "oda kaç derece": ["oda sıcaklığı kaç", "odanın ısısı ne", "oda ısısı"],
    "içerisi kaç derece": ["içerinin sıcaklığı kaç", "ev kaç derece"],
    "hava kaç derece": ["dış sıcaklık kaç", "dışarısı kaç derece"],
    "modu değiştir": ["mod çevir", "başka moda geç", "farklı mod"],
    "fanı aç": ["fanı çalıştır", "pervaneyi aç", "havalandırma aç"],
    "fanı kapa": ["fanı durdur", "pervaneyi kapat", "havalandırma kapat"],
    "fanı arttır": ["fan hızını yükselt", "daha hızlı fan"],
    "fanı düşür": ["fan hızını düşür", "daha yavaş fan"],

    # TV/Media commands (45-52)
    "tv aç": ["televizyonu aç", "tv'yi çalıştır", "ekranı aç"],
    "tv kapa": ["televizyonu kapat", "tv'yi kapat", "ekranı kapat"],
    "televizyonu aç": ["tv'yi aç", "televizyonu çalıştır"],
    "televizyonu kapa": ["tv'yi kapat", "televizyonu söndür"],
    "multimedyayı aç": ["medya sistemini aç", "eğlence sistemini aç"],
    "multimedyayı kapa": ["medya sistemini kapat", "eğlence sistemini kapat"],
    "müzik aç": ["müziği çal", "şarkı aç", "müzik çalıştır"],
    "müzik kapa": ["müziği durdur", "şarkıyı kapat", "müziği söndür"],

    # Curtain/Blind commands (53-56)
    "panjuru aç": ["panjurları çek", "panjurları aç", "güneş girsin"],
    "panjuru kapa": ["panjurları kapat", "panjurları çek", "dışarıyı kapat"],
    "perdeyi aç": ["perdeleri aç", "perdeyi çek", "pencereyi aç"],
    "perdeyi kapa": ["perdeleri kapat", "perdeyi çek", "karanlık yap"],

    # Alarm (57)
    "alarmı kur": ["alarm ayarla", "alarmı çalıştır", "uyandır beni"],

    # Yes/No (58-59)
    "evet": ["tamam", "olur", "kabul"],
    "hayır": ["olmaz", "istemiyorum", "red"],

    # Scene/Mode commands (60-71)
    "parti zamanı": ["parti modu", "eğlence zamanı", "parti başlasın"],
    "dinlenme zamanı": ["dinlenme modu", "rahatlama zamanı", "rahatlama modu"],
    "uyku zamanı": ["uyku modu", "yatma zamanı", "uyuma zamanı"],
    "eve geldim": ["evdeyim", "eve vardım", "geldim"],
    "evden çıkıyorum": ["evden gidiyorum", "ayrılıyorum", "çıkıyorum"],
    "günaydın": ["günaydın modu", "sabah oldu", "sabah zamanı"],
    "iyi geceler": ["gece modu", "iyi uykular", "uyku zamanı"],
    "film zamanı": ["film modu", "sinema modu", "film izleme zamanı"],
    "çalışma zamanı": ["çalışma modu", "iş zamanı", "odaklanma zamanı"],
    "spor zamanı": ["egzersiz zamanı", "spor modu", "antrenman zamanı"],
    "antrenman zamanı": ["spora başla", "egzersiz modu", "antrenmanı başlat"],
    "senaryo başlat": ["modu aç", "rutini başlat", "senaryoyu çalıştır"]
}

print("DATASET CREATION")
print("\nReading core commands from Excel...")

# Normalize variations_map keys to handle Turkish character encoding
normalized_variations_map = {}
for key, variations in variations_map.items():
    normalized_key = key.lower().replace('i̇', 'i').replace('İ', 'i')
    normalized_variations_map[normalized_key] = variations

try:
    df = pd.read_excel(INPUT_FILE, header=None)
    target_col = 1 if (str(df.iloc[0, 0]).isdigit() or "#" in str(df.iloc[0, 0])) else 0
    raw_data = df.iloc[:, target_col].dropna().tolist()

    training_data = []
    commands_with_variations = 0
    commands_without_variations = []

    for cmd in raw_data:
        cmd_str = str(cmd).strip().lower()
        cmd_str = cmd_str.replace('i̇', 'i').replace('İ', 'i')

        if not cmd_str or cmd_str.isdigit() or cmd_str == "#" or cmd_str == "nan":
            continue

        # Add core command
        training_data.append({"Label": cmd_str, "Sentence": cmd_str})

        # Add variations if available
        if cmd_str in normalized_variations_map:
            commands_with_variations += 1
            for v in normalized_variations_map[cmd_str]:
                training_data.append({"Label": cmd_str, "Sentence": v})
        else:
            commands_without_variations.append(cmd_str)

    full_corpus_df = pd.DataFrame(training_data)
    full_corpus_df.to_excel(OUTPUT_DATASET, index=False)

    print(f"Dataset created successfully!")
    print(f"Core commands: {full_corpus_df['Label'].nunique()}")
    print(f"Commands with variations: {commands_with_variations}")
    print(f"Total variations generated: {len(full_corpus_df) - full_corpus_df['Label'].nunique()}")
    print(f"Total sentences: {len(full_corpus_df)}")
    print(f"Saved to: {OUTPUT_DATASET}")

    if commands_without_variations:
        print(f"\n{len(commands_without_variations)} commands without variations:")
        for cmd in commands_without_variations[:5]:
            print(f"     - {cmd}")
        if len(commands_without_variations) > 5:
            print(f"     ... and {len(commands_without_variations) - 5} more")

except Exception as e:
    print(f"Error processing data: {e}")
    exit()

print("Text Preprocessing")

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()


train_sentences = [clean_text(row) for row in full_corpus_df['Sentence']]
print(f"Preprocessed {len(train_sentences)} sentences")

print("Building N-gram Model")

phrases = Phrases(train_sentences, min_count=1, threshold=0.01, scoring='npmi', delimiter='_')
bigram_transformer = Phraser(phrases)
train_sentences_ngram = [bigram_transformer[sent] for sent in train_sentences]

print("N-gram transformation examples:")
count = 0
for i in range(len(train_sentences)):
    original = ' '.join(train_sentences[i])
    transformed = ' '.join(train_sentences_ngram[i])
    if original != transformed:
        print(f"  '{original}' → '{transformed}'")
        count += 1
        if count >= 5:
            break

if count == 0:
    print("(No significant bigrams in first sentences)")

bigram_count = len([k for k in phrases.vocab.keys() if '_' in str(k)])
print(f"Total bigrams detected: {bigram_count}")

print("Loading Pre-trained Turkish Word2Vec Model")

if not os.path.exists(REPAIRED_MODEL):
    print(f"Converting '{PRETRAINED_MODEL}' to text format...")
    if not os.path.exists(PRETRAINED_MODEL):
        if os.path.exists("trmodel"):
            PRETRAINED_MODEL = "trmodel"
        else:
            print(f"Pre-trained model not found!")
            exit()

    try:
        kv = KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True, unicode_errors='ignore')
        kv.save_word2vec_format(REPAIRED_MODEL, binary=False)
        print(f"Converted and saved as '{REPAIRED_MODEL}'")
    except Exception as e:
        print(f"Error: {e}")
        exit()
else:
    print(f"Found existing '{REPAIRED_MODEL}'")

print("Training Word2Vec Models")

try:
    pretrained_kv = KeyedVectors.load_word2vec_format(REPAIRED_MODEL, binary=False)
    vector_dim = pretrained_kv.vector_size
    print(f"Pre-trained vector dimension: {vector_dim}")

    print("\nModel 1: Word-level embeddings (without n-grams)")
    model = Word2Vec(vector_size=vector_dim, min_count=1, window=5, sg=1)
    model.build_vocab(train_sentences)

    count = 0
    for word in model.wv.index_to_key:
        if word in pretrained_kv:
            model.wv[word] = pretrained_kv[word]
            count += 1

    print(f"Vocabulary size: {len(model.wv)} words")
    print(f"Transferred pre-trained vectors: {count} ({count / len(model.wv) * 100:.1f}%)")
    print(f"Training for 20 epochs...")

    model.train(train_sentences, total_examples=len(train_sentences), epochs=20)
    model.save(FINETUNED_MODEL)
    print(f"Saved: {FINETUNED_MODEL}")

    print("\nModel 2: N-gram embeddings (with bigrams)")
    model_ngram = Word2Vec(vector_size=vector_dim, min_count=1, window=5, sg=1)
    model_ngram.build_vocab(train_sentences_ngram)

    count_ngram = 0
    for word in model_ngram.wv.index_to_key:
        if word in pretrained_kv:
            model_ngram.wv[word] = pretrained_kv[word]
            count_ngram += 1

    print(f"Vocabulary size: {len(model_ngram.wv)} tokens")
    print(f"Transferred pre-trained vectors: {count_ngram} ({count_ngram / len(model_ngram.wv) * 100:.1f}%)")
    print(f"Training for 20 epochs...")

    model_ngram.train(train_sentences_ngram, total_examples=len(train_sentences_ngram), epochs=20)

    FINETUNED_MODEL_NGRAM = "smart_home_model_ngram.model"
    model_ngram.save(FINETUNED_MODEL_NGRAM)
    print(f"Saved: {FINETUNED_MODEL_NGRAM}")
    print(f"\nGenerated files:")
    print(f"{OUTPUT_DATASET} ({len(full_corpus_df)} sentences)")
    print(f"{FINETUNED_MODEL} (word-level)")
    print(f"{FINETUNED_MODEL_NGRAM} (n-gram)")

except Exception as e:
    print(f"Error during training: {e}")