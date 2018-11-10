from src.Classificador import Classificador
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressor(Classificador):

    def __init__(self, path, applicationmethod):
        Classificador.__init__(self, path, applicationmethod)
        self.classificador = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    def train(self, x_t, y_t):
        self.classificador.fit(x_t, y_t)

    def predict(self, x_v):
        return self.classificador.predict(x_v)

    def score(self, x_v, y_v):
        return self.classificador.score(x_v, y_v)

    def calculateError(self, y_v, y_pred):
        return np.mean(y_v == y_pred).astype('float32')

