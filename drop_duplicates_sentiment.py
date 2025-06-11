
import pandas as pd

# === File paths ===
input_file = "/home/u762545/Thesis/Data/FNS_with_sentiment.csv"
output_file = "/home/u762545/Thesis/Data/FNS_with_sentiment_deduplicated.csv"

# === Load CSV ===
print("📂 Loading file...")
df = pd.read_csv(input_file)

# === Drop duplicates ===
before = len(df)
df = df.drop_duplicates(subset=["Date", "Article_title", "Article"])
after = len(df)

# === Save cleaned file ===
df.to_csv(output_file, index=False)
print(f"✅ Duplicates removed: {before - after}")
print(f"📄 Cleaned file saved to: {output_file}")
