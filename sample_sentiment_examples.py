import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# === Enable unbuffered stdout for real-time logging ===
sys.stdout.reconfigure(line_buffering=True)

# === Load dataset ===
file_path = "../Data/FNS_with_sentiment_deduplicated.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# === Ensure date is datetime and sort ===
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

# === Filter and convert sentiment columns to numeric ===
sentiment_cols = [
    "sentiment_article_positive", 
    "sentiment_article_neutral", 
    "sentiment_article_negative"
]

# Coerce to numeric and drop problematic rows
df[sentiment_cols] = df[sentiment_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=sentiment_cols)

# === Group by date ===
daily_sentiment = df.groupby("Date")[sentiment_cols].mean()
daily_sentiment = daily_sentiment.dropna()

print("✅ Aggregated sentiment (first 5 rows):")
print(daily_sentiment.head())

# === Plot ===
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

# ✅ Sanity check: confirm values are floats
print("\n✅ Column types before plotting:")
print(daily_sentiment.dtypes)

daily_sentiment.plot(title="Average Article Sentiment Over Time", figsize=(14, 7))
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.tight_layout()

output_path = "/home/u762545/Thesis/Plots/article_sentiment_time_series.png"
plt.savefig(output_path)
print(f"\n✅ Plot saved successfully to {output_path}")
