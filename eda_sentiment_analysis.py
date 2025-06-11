
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# === Enable unbuffered stdout for real-time logging ===
sys.stdout.reconfigure(line_buffering=True)

# === Load data ===
file_path = "../Data/FNS_with_sentiment_deduplicated.csv"
df = pd.read_csv(file_path)

# === Basic Info ===
print("📊 Dataset Info:")
print(df.info())
print("\n🧼 Missing Values:")
print(df.isnull().sum())
print("\n🧩 Duplicate Rows:", df.duplicated().sum())

# === Sentiment Distribution ===
print("\n📈 Sentiment Label Distribution (Articles):")
print(df['sentiment_article_label'].value_counts())
print("\n📈 Sentiment Label Distribution (Titles):")
print(df['sentiment_title_label'].value_counts())

# === Plot Distributions ===
plt.figure(figsize=(12, 5))
sns.histplot(df['sentiment_article_positive'], color="green", label="Positive", kde=True, bins=30)
sns.histplot(df['sentiment_article_neutral'], color="blue", label="Neutral", kde=True, bins=30)
sns.histplot(df['sentiment_article_negative'], color="red", label="Negative", kde=True, bins=30)
plt.legend()
plt.title("Article Sentiment Score Distributions")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("article_sentiment_distributions.png")
plt.close()

# === Correlation Heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(df[['sentiment_article_positive', 'sentiment_article_neutral', 'sentiment_article_negative',
                'sentiment_title_positive', 'sentiment_title_neutral', 'sentiment_title_negative']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("📌 Correlation Between Sentiment Probabilities")
plt.tight_layout()
plt.savefig("sentiment_correlation_heatmap.png")
plt.close()

# === Time Series Volume ===
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['month'] = df['Date'].dt.to_period("M")
monthly_counts = df['month'].value_counts().sort_index()

plt.figure(figsize=(14, 5))
monthly_counts.plot(kind='bar', color="skyblue")
plt.title("🕒 Number of Articles per Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("monthly_article_volume.png")
plt.close()

print("\n✅ EDA completed. Plots saved as PNG files.")
