# src/features/rolling_features.py

import numpy as np
from src.util.helpers import exp_weights

def rolling_stats(series, idx, win):
    slice_ = series.iloc[max(0, idx-win): idx]
    padded = np.pad(slice_.values, (win - len(slice_), 0), constant_values=0)
    return slice_, padded
