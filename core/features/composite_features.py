# src/features/composite_features.py

import numpy as np

def competitor_composite(last4, last8, last10, w4, w8, w10):
    co_pressure = (
        last4.sum() +
        0.5 * last8.sum() +
        0.2 * last10.sum()
    )

    return (
        0.6 * np.dot(w4, last4) +
        0.3 * np.dot(w8, last8) +
        0.1 * np.dot(w10, last10) +
        0.2 * co_pressure
    )
