# src/models/temperature.py

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """
    Temperature Scaling for binary logistic regression outputs.
    Calibrates logits: sigmoid(logit / T)
    """
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, y):
        """
        Fit T using LOOCV logits + true labels.
        """

        def objective(T):
            T = float(T)
            p = _sigmoid(logits / T)
            return log_loss(y, p, labels=[0, 1])

        res = minimize(objective, x0=[1.0], bounds=[(0.01, 10)])
        self.T = float(res.x[0])
        return self

    def predict_proba(self, logits):
        return _sigmoid(logits / self.T)
