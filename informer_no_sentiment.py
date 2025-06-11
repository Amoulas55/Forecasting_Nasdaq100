# informer_model.py — FINAL INFORMER-ONLY VERSION

import sys
sys.path.append('/home/u762545/Thesis')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = '/home/u762545/Thesis/Data/merged_stock_sentiment.csv'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = '/home/u762545/Thesis/informer/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset
class StockDataset(Dataset):
    def __init__(self, data, seq_length, target_col='Close'):
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col
    def __len__(self):
        return len(self.data) - self.seq_length
    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_length].values
        y = self.data.iloc[idx+self.seq_length][self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
historical_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return']
df = df.dropna(subset=historical_features)[historical_features]
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=historical_features)

train_size = int(0.7 * len(df_scaled))
val_size = int(0.15 * len(df_scaled))
train_data = df_scaled.iloc[:train_size]
val_data = df_scaled.iloc[train_size:train_size + val_size]
test_data = df_scaled.iloc[train_size + val_size:]

train_loader = DataLoader(StockDataset(train_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)
val_loader = DataLoader(StockDataset(val_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)
test_loader = DataLoader(StockDataset(test_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)

# Informer model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(queries.size(-1), dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(attn), values)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU6(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x))
        return self.norm2(x + self.ff(x))

class Informer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        d_model = 128
        e_layers = 2
        dropout = 0.03076210562572996
        self.enc_embedding = nn.Sequential(nn.Linear(input_dim, d_model), PositionalEncoding(d_model))
        self.encoder = nn.Sequential(*[EncoderLayer(d_model, dropout) for _ in range(e_layers)])
        self.projection = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.enc_embedding(x)
        x = self.encoder(x)
        return self.projection(x[:, -1, :]).squeeze()

# Train function
def train_model(model, train_loader, val_loader, optimizer, criterion, model_name):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_losses.append(criterion(model(x), y).item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={np.mean(train_losses):.4f}, Val Loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'best_{model_name}.pt'))
            print("✅ Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping.")
                break

# Train Informer
input_dim = len(historical_features)
informer = Informer(input_dim).to(DEVICE)
optimizer_inf = torch.optim.Adam(informer.parameters(), lr=0.0001864196232761308, weight_decay=5e-4)
train_model(informer, train_loader, val_loader, optimizer_inf, nn.MSELoss(), 'informer')

# Evaluate
informer.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_informer.pt')))
def evaluate(model):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y_hat = model(x).cpu().numpy()
            pred.append(y_hat)
            true.append(y.numpy())
    return np.concatenate(true), np.concatenate(pred)

true_norm, inf_pred_norm = evaluate(informer)
zeros = np.zeros((len(true_norm), len(historical_features)))
zeros[:, historical_features.index('Close')] = true_norm
true_close = scaler.inverse_transform(zeros)[:, historical_features.index('Close')]
zeros[:, historical_features.index('Close')] = inf_pred_norm
inf_close = scaler.inverse_transform(zeros)[:, historical_features.index('Close')]

full_df = pd.read_csv(DATA_PATH)
dates = full_df.dropna(subset=historical_features).reset_index(drop=True).iloc[train_size + val_size + SEQUENCE_LENGTH:]['date'].reset_index(drop=True)
preds_df = pd.DataFrame({'Date': dates, 'True': true_close, 'Informer': inf_close})
preds_df.to_csv(os.path.join(SAVE_DIR, 'predictions_informer.csv'), index=False)

# Save metrics
mae_inf = mean_absolute_error(true_close, inf_close)
mse_inf = mean_squared_error(true_close, inf_close)
r2_inf = r2_score(true_close, inf_close)
with open(os.path.join(SAVE_DIR, 'metrics_informer.txt'), 'w') as f:
    f.write(f"MAE: {mae_inf:.4f}\nMSE: {mse_inf:.4f}\nR2: {r2_inf:.4f}\n")

# Plot true vs predicted
plt.figure(figsize=(12, 6))
plt.plot(dates, true_close, label='True Close Price', color='black')
plt.plot(dates, inf_close, label='Informer Prediction', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Price (€)')
plt.title('Informer Predictions vs True Prices')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'plot_informer_predictions.png'))
plt.close()

plt.ylabel('Price (€)')
plt.title('Informer Predictions vs True Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'plot_informer_predictions.png'))
plt.close()

print(f"\n✅ Final Informer Metrics:\nMAE: {mae_inf:.4f}\nMSE: {mse_inf:.4f}\nR2: {r2_inf:.4f}\n")
print("📊 Informer-only evaluation complete.")
