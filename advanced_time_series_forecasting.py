"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms
----------------------------------------------------------------------------

End-to-end project script:
1. Synthetic multivariate data generation
2. Train/val/test split and sequence creation
3. Transformer-style self-attention model (PyTorch)
4. SARIMAX baseline (statsmodels)
5. XGBoost baseline with lag features
6. Hyperparameter tuning hooks
7. Metrics & comparative evaluation
8. Attention weight visualization

Before running, install dependencies (in terminal / notebook):

pip install numpy pandas matplotlib scikit-learn torch statsmodels xgboost

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import statsmodels.api as sm
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# 1. Synthetic multivariate time-series dataset
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_steps: int = 6000,
    n_features: int = 5,
    freq: str = "H",  # hourly data
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a multivariate time series with:
    - clear upward trend
    - daily and weekly seasonality
    - Gaussian noise
    - correlated features
    """
    rng = np.random.default_rng(seed)
    time_idx = pd.date_range(start="2020-01-01", periods=n_steps, freq=freq)
    t = np.arange(n_steps)

    # Base components
    trend = 0.0005 * t                      # slow trend
    daily = 0.8 * np.sin(2 * np.pi * t / 24)        # daily seasonality
    weekly = 0.4 * np.sin(2 * np.pi * t / (24 * 7)) # weekly seasonality
    noise_base = 0.1 * rng.standard_normal(n_steps)

    data = {}
    base_signal = trend + daily + weekly + noise_base

    # Create multiple correlated features
    for i in range(n_features):
        coeff_trend = 1 + 0.2 * i
        coeff_daily = 1 + 0.1 * i
        coeff_weekly = 1 - 0.05 * i
        noise = 0.1 * rng.standard_normal(n_steps)

        feature = (
            coeff_trend * trend
            + coeff_daily * daily
            + coeff_weekly * weekly
            + noise
        )
        data[f"feature_{i+1}"] = feature

    # Target is a nonlinear combination of features + base signal
    weights = rng.uniform(0.5, 1.5, size=n_features)
    features_matrix = np.column_stack([data[f"feature_{i+1}"] for i in range(n_features)])
    target = (features_matrix @ weights) / weights.sum() + 0.5 * base_signal
    target += 0.1 * rng.standard_normal(n_steps)  # extra noise
    data["target"] = target

    df = pd.DataFrame(data, index=time_idx)
    return df

# ---------------------------------------------------------------------------
# 2. Dataset, sequence creation, and DataLoader
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    horizon: int
):
    """
    Turn a multivariate series into (input_seq, future_targets) pairs.

    data:   shape (N, num_features)
    target: shape (N,)
    """
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(target[i + seq_len : i + seq_len + horizon])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# 3. Transformer-style model with self attention
# ---------------------------------------------------------------------------

class SelfAttentionEncoder(nn.Module):
    """
    Simple encoder with Multihead Self-Attention + FFN + residuals + LN.
    Stores attention weights for interpretability.
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None  # to store weights

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, d_model)
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        attn_out, attn_weights = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        self.last_attn_weights = attn_weights  # shape: (batch, n_heads, seq, seq)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=64,
        n_heads=4,
        d_ff=128,
        seq_len=48,
        horizon=24
    ):
        super().__init__()
        self.encoder = SelfAttentionEncoder(
            input_dim=n_features,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
        )
        # take representation of last time step and map to future horizon
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        """
        enc_out = self.encoder(x)             # (batch, seq_len, d_model)
        last_step = enc_out[:, -1, :]         # (batch, d_model)
        out = self.fc_out(last_step)          # (batch, horizon)
        return out

    def get_attention_weights(self):
        """
        Returns attention weights of shape:
            (batch, n_heads, seq_len, seq_len)
        from last forward pass.
        """
        return self.encoder.last_attn_weights

# ---------------------------------------------------------------------------
# 4. Training utilities
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    seq_len: int = 48
    horizon: int = 24
    batch_size: int = 64
    lr: float = 1e-3
    n_epochs: int = 15
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_transformer_model(
    X_train, y_train, X_val, y_val, n_features, cfg: TrainConfig
):
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TimeSeriesTransformer(
        n_features=n_features,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
    ).to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = np.inf
    best_state = None

    for epoch in range(cfg.n_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
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
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{cfg.n_epochs} - Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def evaluate_model_predictions(y_true, y_pred):
    """
    y_true, y_pred: (num_samples, horizon)
    Returns MAE, RMSE, MAPE for the whole horizon.
    """
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0
    return mae, rmse, mape

# ---------------------------------------------------------------------------
# 5. Baseline models
# ---------------------------------------------------------------------------

def sarimax_forecast(train_series, test_steps, order=(1,1,1), seasonal_order=(1,1,1,24)):
    """
    Fit SARIMAX on training series and forecast test_steps ahead iteratively.
    To keep it simpler, we re-fit once and take direct forecast.
    """
    model = sm.tsa.statespace.SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.forecast(steps=test_steps)
    return np.array(forecast)

def build_lag_features(series: pd.Series, lags: int = 24):
    """
    Create lag features for XGBoost.
    """
    df = pd.DataFrame({"target": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["target"].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y

# ---------------------------------------------------------------------------
# 6. Attention visualization
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn_weights, seq_len, time_index=None, title="Attention Heatmap"):
    """
    attn_weights: (batch, n_heads, seq_len, seq_len)
    We average over heads and take the first sample in the batch.
    """
    if attn_weights is None:
        print("No attention weights stored.")
        return

    attn = attn_weights[0].mean(dim=0).detach().cpu().numpy()  # (seq_len, seq_len)

    plt.figure(figsize=(6, 5))
    plt.imshow(attn, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.title(title)
    plt.xlabel("Key / Time step")
    plt.ylabel("Query / Time step")

    if time_index is not None and len(time_index) >= seq_len:
        ticks = np.linspace(0, seq_len - 1, num=min(8, seq_len), dtype=int)
        labels = [str(time_index[i]) for i in ticks]
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ------------------- Data Generation -------------------
    df = generate_synthetic_dataset()
    print("Dataset head:")
    print(df.head())
    print("Shape:", df.shape)

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    target_col = "target"

    # ------------------- Train/Val/Test Split -------------------
    # Use chronological split (no shuffling)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Standardize features (fit on train, apply to all)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].values)
    val_features = scaler.transform(val_df[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    train_target = train_df[target_col].values
    val_target = val_df[target_col].values
    test_target = test_df[target_col].values

    # ------------------- Hyperparameters -------------------
    cfg = TrainConfig(
        seq_len=48,
        horizon=24,
        batch_size=64,
        lr=1e-3,
        n_epochs=15,
        d_model=64,
        n_heads=4,
        d_ff=128,
    )
    print("\nUsing configuration:", cfg)

    # ------------------- Sequence creation -------------------
    X_train, y_train = create_sequences(train_features, train_target, cfg.seq_len, cfg.horizon)
    X_val, y_val = create_sequences(val_features, val_target, cfg.seq_len, cfg.horizon)
    X_test, y_test = create_sequences(test_features, test_target, cfg.seq_len, cfg.horizon)

    print("Train sequences:", X_train.shape, y_train.shape)
    print("Val sequences  :", X_val.shape, y_val.shape)
    print("Test sequences :", X_test.shape, y_test.shape)

    # ------------------- Train Transformer Model -------------------
    n_features = len(feature_cols)
    transformer_model = train_transformer_model(
        X_train, y_train, X_val, y_val, n_features, cfg
    )

    # ------------------- Evaluate Transformer on Test -------------------
    transformer_model.eval()
    device = cfg.device
    test_ds = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = transformer_model(xb).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred_trans = np.concatenate(all_preds, axis=0)
    y_true_trans = np.concatenate(all_true, axis=0)
    mae_t, rmse_t, mape_t = evaluate_model_predictions(y_true_trans, y_pred_trans)

    print("\n=== Transformer Model Metrics (full horizon) ===")
    print(f"MAE  : {mae_t:.4f}")
    print(f"RMSE : {rmse_t:.4f}")
    print(f"MAPE : {mape_t:.2f}%")

    # ------------------- Baseline 1: SARIMAX -------------------
    # For SARIMAX, we forecast the target directly (univariate)
    # Forecast only length of test set, horizon=1 step each
    train_series = train_df[target_col]
    val_series = val_df[target_col]
    test_series = test_df[target_col]

    # We combine train+val for fitting SARIMAX
    sarimax_train = pd.concat([train_series, val_series])
    sarimax_test = test_series

    sarimax_forecasts = sarimax_forecast(
        train_series=sarimax_train,
        test_steps=len(sarimax_test),
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 24),
    )

    mae_s = mean_absolute_error(sarimax_test.values, sarimax_forecasts)
    rmse_s = np.sqrt(mean_squared_error(sarimax_test.values, sarimax_forecasts))
    mape_s = np.mean(
        np.abs((sarimax_test.values - sarimax_forecasts) / (sarimax_test.values + 1e-8))
    ) * 100.0

    print("\n=== SARIMAX Baseline Metrics (1-step forecast over test) ===")
    print(f"MAE  : {mae_s:.4f}")
    print(f"RMSE : {rmse_s:.4f}")
    print(f"MAPE : {mape_s:.2f}%")

    # ------------------- Baseline 2: XGBoost with lag features -------------------
    # Use train+val for fitting, test for evaluation
    full_baseline_train = pd.concat([train_series, val_series])
    X_lag, y_lag = build_lag_features(full_baseline_train, lags=24)

    # Align test segment for evaluation
    # The last len(test_series) points of (full_train + test) should be predicted.
    combined = pd.concat([full_baseline_train, test_series])
    X_all, y_all = build_lag_features(combined, lags=24)

    # Indices for test region in the lagged matrix
    offset = len(X_all) - len(test_series)
    X_test_lag = X_all[offset:]
    y_test_lag = y_all[offset:]

    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb_model.fit(X_lag, y_lag)
    xgb_preds = xgb_model.predict(X_test_lag)

    mae_x = mean_absolute_error(y_test_lag, xgb_preds)
    rmse_x = np.sqrt(mean_squared_error(y_test_lag, xgb_preds))
    mape_x = np.mean(np.abs((y_test_lag - xgb_preds) / (y_test_lag + 1e-8))) * 100.0

    print("\n=== XGBoost Baseline Metrics (1-step with lags) ===")
    print(f"MAE  : {mae_x:.4f}")
    print(f"RMSE : {rmse_x:.4f}")
    print(f"MAPE : {mape_x:.2f}%")

    # ------------------- Comparative summary -------------------
    print("\n================ Comparative Summary (Test Set) ================")
    print("Model        | MAE      | RMSE     | MAPE")
    print("-----------------------------------------------")
    print(f"Transformer  | {mae_t:8.4f} | {rmse_t:8.4f} | {mape_t:6.2f}%")
    print(f"SARIMAX      | {mae_s:8.4f} | {rmse_s:8.4f} | {mape_s:6.2f}%")
    print(f"XGBoost      | {mae_x:8.4f} | {rmse_x:8.4f} | {mape_x:6.2f}%")

    # ------------------- Attention Analysis -------------------
    # Take one batch from test loader to visualize attention
    batch_x, _ = next(iter(test_loader))
    batch_x = batch_x.to(device)
    with torch.no_grad():
        _ = transformer_model(batch_x)
        attn_weights = transformer_model.get_attention_weights()

    # Use the corresponding time index for the first sequence:
    # It starts at val_end + seq_len
    start_idx = val_end + cfg.seq_len
    time_index_seq = df.index[start_idx : start_idx + cfg.seq_len]

    plot_attention_heatmap(
        attn_weights, seq_len=cfg.seq_len, time_index=time_index_seq,
        title="Transformer Self-Attention (first test sample)"
    )

    print("\nDone. Use printed metrics and attention plot for your report.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
