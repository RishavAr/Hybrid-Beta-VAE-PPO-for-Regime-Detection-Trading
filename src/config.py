CONFIG = {
    "SEED": 42,
    "ROLL_WIN": 10,
    "LATENT_DIM": 16,
    "BETA": 6.0,
    "VAE_EPOCHS": 500,
    "VAE_BATCH_SIZE": 128,
    "LR": 1e-3,
    "TRANSACTION_COST_BPS": 10,   # 0.10%
    "SLIPPAGE_BPS": 5,           # 0.05%
    "RL_TRAIN_TIMESTEPS": 200_000,
    "DATA_CSV_PATH": "data/S&P500_all_companies.csv",
    "RESULTS_FILE": "final_results_no_lookahead.csv",
    "EWMA_ALPHA": 0.90,
    "HARD_ENTER_TH": 0.58,
    "HARD_EXIT_TH": 0.42,
    "DWELL_DAYS": 5,
    "K_MIN": 3,
    "K_MAX": 6,
}
