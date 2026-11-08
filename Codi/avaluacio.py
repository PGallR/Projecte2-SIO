"""Mètriques d'avaluació."""
import numpy as np


def mae(reals, prediccions):
    return float(np.mean(np.abs(reals - prediccions)))