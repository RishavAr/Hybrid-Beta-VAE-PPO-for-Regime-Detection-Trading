import numpy as np
from sklearn.mixture import GaussianMixture

def fit_gmm(latents, k_min=3, k_max=6):
    """Fit GMM with BIC selection."""
    best_gmm, best_bic = None, np.inf
    for k in range(k_min, k_max+1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(latents)
        bic = gmm.bic(latents)
        if bic < best_bic:
            best_gmm, best_bic = gmm, bic
    return best_gmm
