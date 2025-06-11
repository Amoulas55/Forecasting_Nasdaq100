# lstm_no_sentiment.py — NAIVE LSTM WITH EARLY STOPPING AND DROPOUT (TUNED FOR 0–0.5 R² TARGET)

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

# --- Configuration ---
DATA_PATH = '/home/u762545/Thesis/Data/merged_stock_sentiment.csv'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = '/home/u762545/Thesis/lstm_naive/'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Dataset Definition ---
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

# --- Data Loading & Preprocessing ---
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return']
df = pd.read_csv(DATA_PATH).dropna(subset=cols)[cols]
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)

train_size = int(0.7 * len(df_scaled))
val_size = int(0.15 * len(df_scaled))
train_data = df_scaled.iloc[:train_size]
val_data = df_scaled.iloc[train_size:train_size + val_size]
test_data = df_scaled.iloc[train_size + val_size:]

train_loader = DataLoader(StockDataset(train_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)
val_loader = DataLoader(StockDataset(val_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)
test_loader = DataLoader(StockDataset(test_data, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)

# --- Naive LSTM Model ---
class NaiveLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size=16, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze()

# --- Training Function ---
def train(model, train_loader, val_loader, optimizer, criterion):
    best_loss = float('inf')
    counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_naive_lstm.pt'))
            print("✅ Saved best model.")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("🛑 Early stopping.")
                break

# --- Training Run ---
input_dim = len(cols)
model = NaiveLSTM(input_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=0.025)
train(model, train_loader, val_loader, opt, nn.MSELoss())

# --- Evaluation ---
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_naive_lstm.pt')))
def evaluate(model):
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds.append(model(x).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)

y_true_norm, y_pred_norm = evaluate(model)

zeros = np.zeros((len(y_true_norm), len(cols)))
zeros[:, cols.index('Close')] = y_true_norm
y_true = scaler.inverse_transform(zeros)[:, cols.index('Close')]
zeros[:, cols.index('Close')] = y_pred_norm
y_pred = scaler.inverse_transform(zeros)[:, cols.index('Close')]

# --- Save Predictions ---
full_df = pd.read_csv(DATA_PATH)
dates = full_df.dropna(subset=cols).reset_index(drop=True).iloc[train_size + val_size + SEQUENCE_LENGTH:]['date'].reset_index(drop=True)
preds_df = pd.DataFrame({'Date': dates, 'True': y_true, 'Naive_LSTM': y_pred})
preds_df.to_csv(os.path.join(SAVE_DIR, 'naive_lstm_predictions.csv'), index=False)

# --- Save Metrics ---
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
with open(os.path.join(SAVE_DIR, 'naive_lstm_metrics.txt'), 'w') as f:
    f.write(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR2: {r2:.4f}\n")
print(f"\n✅ Final Naive LSTM Metrics:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nR2: {r2:.4f}\n")

# --- Plot Predictions ---
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Naive LSTM Prediction')
plt.legend()
plt.title('Naive LSTM: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price (€)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'naive_lstm_plot.png'))
plt.show()
