"""PCA projection to 2D for scatter visualizations (iris, breast_cancer, blobs,
moons), mirroring how the frontend plots standardized features on two PCs."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def project_2d(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Standardize then project to 2 principal components.

    Returns the (n, 2) projection and the explained-variance ratio (%) of the
    two components, which the frontend uses as axis labels.
    """
    x = np.asarray(x, dtype=float)
    if x.shape[1] < 2:
        # Degenerate: pad so we always return 2 columns.
        pad = np.zeros((x.shape[0], 2 - x.shape[1]))
        return np.hstack([x, pad]), [100.0, 0.0]
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    variance = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    return coords, variance
