# INFORMER FULL MODEL WITH ROBUST GENERALIZATION

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random

# --- Set seeds for reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# --- Load and prepare data ---
data_path = "/home/u762545/Thesis/Data/merged_stock_sentiment.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.reset_index(drop=True)

features = [
    'High', 'Open', 'Low', 'Volume',
    'sentiment_title_positive',
    'sentiment_title_neutral',
    'sentiment_title_negative',
    'Close'
]
target = 'Close'

# --- Temporal split ---
train_size = int(len(df) * 0.8)
df_train_full = df.iloc[:train_size].reset_index(drop=True)
df_test = df.iloc[train_size:].reset_index(drop=True)

val_size = int(len(df_train_full) * 0.2)
df_val = df_train_full.iloc[-val_size:].reset_index(drop=True)
df_train = df_train_full.iloc[:-val_size].reset_index(drop=True)

# --- Scale using MinMaxScaler ---
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(df_train[features])
scaled_val = scaler.transform(df_val[features])
scaled_test = scaler.transform(df_test[features])

df_train_scaled = pd.DataFrame(scaled_train, columns=features)
df_val_scaled = pd.DataFrame(scaled_val, columns=features)
df_test_scaled = pd.DataFrame(scaled_test, columns=features)
df_train_scaled['date'] = df_train['date']
df_val_scaled['date'] = df_val['date']
df_test_scaled['date'] = df_test['date']

# --- Sequence Creator ---
def create_sequences(data, window=24, target_col='Close'):
    X, y, dates = [], [], []
    for i in range(len(data) - window):
        X.append(data[features].iloc[i:i+window].values)
        y.append(data[target_col].iloc[i+window])
        dates.append(data['date'].iloc[i+window])
    return np.array(X), np.array(y), dates

X_train, y_train, _ = create_sequences(df_train_scaled)
X_val, y_val, _ = create_sequences(df_val_scaled)
X_test, y_test, test_dates = create_sequences(df_test_scaled)

# --- Tensors and Dataloaders ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
# [Preserve existing data prep blocks as-is]
# [Now use DataLoaders instead of full batch training]

BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class InformerFull(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            PositionalEncoding(d_model),
            nn.Dropout(dropout)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x[:, -1, :])

# --- Training Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InformerFull(input_dim=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_loss = float('inf')
patience = 15
wait = 0
EPOCHS = 150

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_pred = model(xb)
            val_loss = criterion(val_pred, yb)
            val_losses.append(val_loss.item())

    scheduler.step()
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

    if avg_val < best_loss:
        best_loss = avg_val
        wait = 0
        torch.save(model.state_dict(), "best_informer_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# --- Evaluation ---
model.load_state_dict(torch.load("best_informer_model.pt"))
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
        actuals.append(yb.numpy())

preds = np.concatenate(preds).squeeze()
actuals = np.concatenate(actuals).squeeze()

# --- Sanity Check Block ---
# Confirm scaling and inverse logic with spot-checks
print("🔍 Sanity Check (first 5 samples):")
print("Scaled y_test_tensor:", y_test_tensor[:5].squeeze().numpy())
print("Actual values (inverse transformed):", actuals[:5])

close_idx = features.index('Close')
def inverse_close(arr):
    dummy = np.zeros((len(arr), len(features)))
    dummy[:, close_idx] = arr
    return scaler.inverse_transform(dummy)[:, close_idx]

pred_inv = inverse_close(preds)
actual_inv = inverse_close(actuals)

# Direct match check with raw test data
print("df_test raw Close:", df_test['Close'].iloc[24:29].values)
print("actual_inv[:5]:", actual_inv[:5])

# Optional: export debug CSV for manual Excel review
debug_df = pd.DataFrame({
    'date': test_dates,
    'scaled_target': actuals,
    'actual_close': actual_inv,
    'predicted_close': pred_inv
})
debug_df.to_csv("debug_check.csv", index=False)

# --- Inverse transform and plot ---
results_dir = "/home/u762545/Thesis/Results"
os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame({
    'date': pd.to_datetime(test_dates),
    'actual_close': actual_inv,
    'predicted_close': pred_inv
})
results_df.to_csv(os.path.join(results_dir, "informer_predictions.csv"), index=False)

rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
mae = mean_absolute_error(actual_inv, pred_inv)
r2 = r2_score(actual_inv, pred_inv)

with open(os.path.join(results_dir, "informer_metrics.txt"), "w") as f:
    f.write("Informer Evaluation Metrics")
    f.write("-----------------------------")
    f.write(f"RMSE: {rmse:.4f}")
    f.write(f"MAE: {mae:.4f}")
    f.write(f"R²: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['actual_close'], label='Actual', linewidth=2)
plt.plot(results_df['date'], results_df['predicted_close'], label='Predicted', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Informer: Actual vs Predicted Close')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "informer_prediction_plot.png"))
plt.close()

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")
print("✅ Informer full model trained and evaluated.")
