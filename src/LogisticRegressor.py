from src.Classificador import Classificador
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressor(Classificador):

    def __init__(self, path, applicationmethod):
        Classificador.__init__(self, path, applicationmethod)
        self.classificador = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    def train(self):
        self.classificador.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_pred = self.classificador.predict(self.x_val)
