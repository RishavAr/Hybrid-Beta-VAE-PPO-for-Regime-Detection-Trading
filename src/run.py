import pandas as pd
import torch
from vae import BetaVAE
from regimes import fit_gmm
from metrics import compute_metrics
from config import CONFIG

def main():
    # Load data
    df = pd.read_csv(CONFIG["DATA_CSV_PATH"])
    print("Loaded dataset:", df.head())

    # Extract features (close log returns, volumes z-score, etc.)
    df["log_return"] = df.groupby("tic")["close"].apply(lambda x: x.pct_change().apply(lambda y: np.log(1+y)))
    df["vol_zscore"] = df.groupby("tic")["volume"].apply(lambda x: (x - x.mean()) / x.std())
    features = df[["log_return", "vol_zscore"]].dropna().values

    # Train VAE
    vae = BetaVAE(input_dim=features.shape[1], latent_dim=CONFIG["LATENT_DIM"])
    optimizer = torch.optim.Adam(vae.parameters(), lr=CONFIG["LR"])
    X = torch.tensor(features, dtype=torch.float32)

    for epoch in range(CONFIG["VAE_EPOCHS"]):
        optimizer.zero_grad()
        recon, mu, logvar = vae(X)
        loss = vae.loss_function(recon, X, mu, logvar)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    # Get latent reps
    with torch.no_grad():
        mu, logvar = vae.encode(X)
        latents = mu.numpy()

    # Fit regimes
    gmm = fit_gmm(latents, CONFIG["K_MIN"], CONFIG["K_MAX"])
    df["regime"] = gmm.predict(latents)

    # Example backtest (equal-weight baseline)
    eq_curve = (1 + df["log_return"].fillna(0)).cumprod()
    metrics = compute_metrics(eq_curve)
    print("Baseline metrics:\n", metrics)

if __name__ == "__main__":
    main()
