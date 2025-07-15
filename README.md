# Forecasting_Nasdaq100

This repository contains the code and documentation for my Master's thesis project at Tilburg University. The project explores the use of sentiment analysis and deep learning models to forecast trends in the Nasdaq-100 index.

## 📌 Project Summary

- 🎯 **Objective:** Predict the daily closing price and trend direction (up/down) of the Nasdaq-100 index using historical stock data and sentiment extracted from financial news.
- 🧠 **Models:** Informer (Transformer-based), LSTM
- 🗞️ **Sentiment Engine:** FinBERT (HuggingFace Transformers)
- 🧪 **Tech Stack:** PyTorch, Optuna, Pandas, Matplotlib, Scikit-learn

## 📂 Project Structure

```
📁 data/                     → Input datasets (preprocessed news + stock prices)
📁 models/                   → Saved model checkpoints
📁 notebooks/                → Jupyter notebooks with EDA, training, evaluation
📁 figures/                  → Plots (loss curves, predictions, evaluation)
📄 README.md                 → This file
📄 requirements.txt          → List of Python dependencies
📄 main.py                   → Training script for Informer model
```

## 📊 Results (Highlights)

| Model                    | MAE     | MSE     | R²     |
|--------------------------|---------|---------|--------|
| LSTM (no sentiment)      | 15.56   | 288.68  | 0.11   |
| Informer (no sentiment)  | 7.96    | 95.19   | 0.71   |
| LSTM + Sentiment         | 17.24   | ~477.25 | 0.525  |
| Informer + Sentiment     | **5.85** | **50.62** | **0.9497** |

> Informer with sentiment input significantly outperformed other models in both rising and declining market conditions.
> 
## 🗃️ Datasets Used

- 📉 Nasdaq-100 daily closing prices from Yahoo Finance
- 📰 Financial news from FNSPID (Financial News and Stock Price Integration Dataset)
- 📊 Sentiment scores generated with FinBERT on headlines + full-text

## ⚙️ How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/Amoulas55/Forecasting_Nasdaq100.git
   cd Forecasting_Nasdaq100
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main model script (optional):
   ```bash
   python main.py
   ```

4. Or follow the steps in `notebooks/` to go from preprocessing to prediction.

## 📚 Citation

If you use or build on this work, please cite:

> Moulas, A. (2025). *Forecasting Nasdaq-100 Trends Using Sentiment Analysis and Transformer-Based Models*. MSc Thesis, Tilburg University.

## 🪪 License

This project is open-source under the MIT License.

---

