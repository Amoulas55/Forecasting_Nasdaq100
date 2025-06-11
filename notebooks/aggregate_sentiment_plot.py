import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
file_path = "/home/u762545/Thesis/Data/FNS_with_sentiment_deduplicated.csv"
df = pd.read_csv(file_path)

# === Convert date column to datetime ===
df["Date"] = pd.to_datetime(df["Date"])

# === Group by date and aggregate mean sentiment probabilities ===
daily_sentiment = df.groupby("Date").agg({
    "sentiment_article_positive": "mean",
    "sentiment_article_neutral": "mean",
    "sentiment_article_negative": "mean",
    "sentiment_title_positive": "mean",
    "sentiment_title_neutral": "mean",
    "sentiment_title_negative": "mean"
}).reset_index()

# === Save the aggregated sentiment to CSV for ARIMAX prep later ===
daily_sentiment.to_csv("/home/u762545/Thesis/Data/daily_sentiment_aggregated.csv", index=False)

# === Plot sentiment over time ===
plt.figure(figsize=(15, 6))
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_article_positive", label="Article Positive")
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_article_negative", label="Article Negative")
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_article_neutral", label="Article Neutral")
plt.title("Daily Average Article Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.tight_layout()
plt.savefig("/home/u762545/Thesis/Plots/article_sentiment_time_series.png")

# === Plot title sentiment as well ===
plt.figure(figsize=(15, 6))
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_title_positive", label="Title Positive")
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_title_negative", label="Title Negative")
sns.lineplot(data=daily_sentiment, x="Date", y="sentiment_title_neutral", label="Title Neutral")
plt.title("Daily Average Title Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.tight_layout()
plt.savefig("/home/u762545/Thesis/Plots/title_sentiment_time_series.png")

# Optional preview
print(daily_sentiment.tail())
