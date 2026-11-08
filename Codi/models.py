"""Models de predicció. Cada model té fit(D) i predict(usuaris, restaurants)."""
import numpy as np


class MitjanesUsuariRestaurant:
    """Predicció = w * mitjana_usuari + (1 - w) * mitjana_restaurant."""

    def __init__(self, pes_usuari=0.5):
        self.pes_usuari = pes_usuari

    def fit(self, D):
        conegut = ~np.isnan(D)

        suma_u = np.nansum(D, axis=1, dtype=np.float64)
        suma_i = np.nansum(D, axis=0, dtype=np.float64)
        n_u = conegut.sum(axis=1)
        n_i = conegut.sum(axis=0)

        self.mitjana_global = float(suma_u.sum() / n_u.sum())

        # Si un usuari/restaurant no té cap vot, fem servir la mitjana global
        self.mitjana_usuari = np.where(
            n_u > 0, suma_u / np.maximum(n_u, 1), self.mitjana_global
        ).astype(np.float32)
        self.mitjana_restaurant = np.where(
            n_i > 0, suma_i / np.maximum(n_i, 1), self.mitjana_global
        ).astype(np.float32)
        return self

    def predict(self, usuaris, restaurants):
        p = (
            self.pes_usuari * self.mitjana_usuari[usuaris]
            + (1.0 - self.pes_usuari) * self.mitjana_restaurant[restaurants]
        )
        return np.clip(p, -10.0, 10.0)