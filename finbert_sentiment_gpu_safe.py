import sys
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
import os
import time

# === Enable unbuffered stdout for real-time logging ===
sys.stdout.reconfigure(line_buffering=True)

# === File paths and config ===
input_path = "/home/u762545/Thesis/Data/FNS_Dataset_cleaned_deduped_title_article.csv"
output_path = "/home/u762545/Thesis/Data/FNS_with_sentiment.csv"
log_path = "/home/u762545/Thesis/Data/finbert_chunks_done.txt"
chunk_size = 10000  # ✅ Balanced size for GPU

# === OPTIONAL: Reset to start fresh ===
if os.path.exists(output_path):
    os.remove(output_path)
if os.path.exists(log_path):
    os.remove(log_path)

# === Use GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# === Load FinBERT model ===
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# === Sentiment label map ===
label_map = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

# === Batch classification function ===
def classify_texts(texts, batch_size=64):
    all_labels = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = softmax(outputs.logits, dim=1).tolist()
        labels = [label_map[torch.argmax(torch.tensor(p)).item()] for p in probs]

        all_labels.extend(labels)
        all_probs.extend(probs)

    return all_labels, all_probs

# === Setup chunking ===
completed_chunks = set()
first_chunk = True
reader = pd.read_csv(input_path, chunksize=chunk_size)

# === Start processing ===
for i, chunk in enumerate(reader):
    print(f"\n🧩 Processing chunk {i+1}...")
    start_time = time.time()

    # Drop rows with missing values
    chunk = chunk.dropna(subset=["Article", "Article_title"])

    # ✂️ Truncate long texts
    chunk["Article"] = chunk["Article"].str.slice(0, 512)
    chunk["Article_title"] = chunk["Article_title"].str.slice(0, 512)

    # --- Step 1: FinBERT on articles ---
    article_texts = chunk["Article"].tolist()
    article_labels, article_probs = classify_texts(article_texts)

    chunk['sentiment_article_label'] = article_labels
    chunk['sentiment_article_positive'] = [p[1] for p in article_probs]
    chunk['sentiment_article_neutral'] = [p[0] for p in article_probs]
    chunk['sentiment_article_negative'] = [p[2] for p in article_probs]

    # --- Step 2: FinBERT on titles ---
    title_texts = chunk["Article_title"].tolist()
    title_labels, title_probs = classify_texts(title_texts)

    chunk['sentiment_title_label'] = title_labels
    chunk['sentiment_title_positive'] = [p[1] for p in title_probs]
    chunk['sentiment_title_neutral'] = [p[0] for p in title_probs]
    chunk['sentiment_title_negative'] = [p[2] for p in title_probs]

    # --- Save chunk ---
    if first_chunk:
        chunk.to_csv(output_path, index=False, mode='w')
        first_chunk = False
    else:
        chunk.to_csv(output_path, index=False, mode='a', header=False)

    # ✅ Log progress
    with open(log_path, "a") as log:
        log.write(f"{i}\n")

    elapsed_time = time.time() - start_time
    print(f"✅ Chunk {i+1} processed and saved in {elapsed_time / 60:.2f} minutes.")

print("\n🎉 All chunks completed successfully!")
print(f"📄 Output saved to: {output_path}")
