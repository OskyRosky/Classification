from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs

@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    name: str

def synthetic_classification(n_samples: int = 2000, n_features: int = 20,
                             weights=(0.8, 0.2), random_state: int = 42) -> DatasetBundle:
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               weights=list(weights), random_state=random_state)
    return DatasetBundle(X=X, y=y, name="make_classification")

def moons(n_samples: int = 1500, noise: float = 0.2, random_state: int = 42) -> DatasetBundle:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return DatasetBundle(X=X, y=y, name="moons")

def circles(n_samples: int = 1500, noise: float = 0.1, factor: float = 0.5, random_state: int = 42) -> DatasetBundle:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return DatasetBundle(X=X, y=y, name="circles")

def blobs(n_samples: int = 1500, centers: int = 2, cluster_std: float = 1.0, random_state: int = 42) -> DatasetBundle:
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return DatasetBundle(X=X, y=y, name="blobs")
