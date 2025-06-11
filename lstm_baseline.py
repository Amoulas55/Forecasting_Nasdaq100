import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates

# === Load and clean data ===
data_path = "/home/u762545/Thesis/Data/merged_stock_sentiment.csv"
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
initial_shape = df.shape
df.dropna(inplace=True)
print(f"Dropped missing values: {initial_shape[0] - df.shape[0]} rows removed.")
print(f"Remaining rows: {df.shape[0]}")

features = [
    'High', 'Open', 'Low', 'Volume',
    'sentiment_title_positive',
    'sentiment_title_neutral',
    'sentiment_title_negative',
    'Close'
]

# === Temporal split ===
train_size = int(len(df) * 0.8)
df_train_full = df.iloc[:train_size].reset_index(drop=True)
df_test = df.iloc[train_size:].reset_index(drop=True)

# Split validation from training (last 20% of training)
val_size = int(len(df_train_full) * 0.2)
df_val = df_train_full.iloc[-val_size:].reset_index(drop=True)
df_train = df_train_full.iloc[:-val_size].reset_index(drop=True)

# === Scale each partition separately ===
scaler = StandardScaler()
scaled_train = scaler.fit_transform(df_train[features])
scaled_val = scaler.transform(df_val[features])
scaled_test = scaler.transform(df_test[features])

df_train_scaled = pd.DataFrame(scaled_train, columns=features)
df_val_scaled = pd.DataFrame(scaled_val, columns=features)
df_test_scaled = pd.DataFrame(scaled_test, columns=features)
df_val_scaled['date'] = df_val['date']
df_test_scaled['date'] = df_test['date']

# === Create sequences ===
def create_sequences(data, window_size=24, target_col='Close'):
    X, y = [], []
    values = data[features].values
    for i in range(len(data) - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size][features.index(target_col)])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(df_train_scaled)
X_val, y_val = create_sequences(df_val_scaled)
X_test, y_test = create_sequences(df_test_scaled)
test_dates = df_test_scaled['date'].iloc[24:].reset_index(drop=True)

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print("\nFinal tensor shapes:")
print(f"Train: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
print(f"Val:   X={X_val_tensor.shape}, y={y_val_tensor.shape}")
print(f"Test:  X={X_test_tensor.shape}, y={y_test_tensor.shape}")

# === LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

input_dim = X_train_tensor.shape[2]
model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Training with early stopping ===
epochs = 100
patience = 10
min_val_loss = np.inf
wait = 0
best_state = None

print("\nTraining started...")
for epoch in range(1, epochs + 1):
    model.train()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = loss_fn(val_output, y_val_tensor)

    print(f"Epoch {epoch}: Train Loss = {loss.item():.4f} | Val Loss = {val_loss.item():.4f}")

    if val_loss.item() < min_val_loss:
        min_val_loss = val_loss.item()
        wait = 0
        best_state = model.state_dict()
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

if best_state:
    model.load_state_dict(best_state)

# === Make predictions ===
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()
    actuals = y_test_tensor.squeeze().numpy()

# === Inverse scale only the Close values ===
close_idx = features.index('Close')
def inverse_close(arr):
    dummy = np.zeros((len(arr), len(features)))
    dummy[:, close_idx] = arr
    return scaler.inverse_transform(dummy)[:, close_idx]

pred_inv = inverse_close(predictions)
actual_inv = inverse_close(actuals)

# === Save predictions ===
results_dir = "/home/u762545/Thesis/Results"
os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame({
    'date': pd.to_datetime(test_dates),
    'actual_close': actual_inv,
    'predicted_close': pred_inv
})
results_df.to_csv(os.path.join(results_dir, "lstm_predictions.csv"), index=False)

# === Save metrics ===
rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
mae = mean_absolute_error(actual_inv, pred_inv)
r2 = r2_score(actual_inv, pred_inv)

with open(os.path.join(results_dir, "lstm_metrics.txt"), "w") as f:
    f.write("LSTM Evaluation Metrics (Final Baseline)\n")
    f.write("-------------------------------------------------\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# === Plot results ===
plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['actual_close'], label='Actual', linewidth=2)
plt.plot(results_df['date'], results_df['predicted_close'], label='Predicted', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('LSTM Baseline: Actual vs Predicted Close')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lstm_prediction_plot.png"))
plt.close()

# === Final output ===
print("\nModel training complete.")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")
print("Predictions, metrics, and plot saved.")
