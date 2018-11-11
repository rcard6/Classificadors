import numpy as np
from sklearn.metrics import recall_score


class Classificador(object):
    def __init__(self, path, applicationmethod):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        x = data
        # Triatge output (parkinsons = status(-7), parkinsons_updrs = motor_UPDRS(4) o total_UPDRS(5))
        self.classificadorName = ""
        self.y = data[:, 4].astype('int32')
        self.x = np.delete(x, 4, axis=1)

        self.x_train = 0
        self.x_val = 0
        self.y_train = 0
        self.y_val = 0
        self.y_pred = 0

        self.method = applicationmethod

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def setx_train(self, x_train):
        self.x_train = x_train

    def setx_val(self, x_val):
        self.x_val = x_val

    def sety_train(self, y_train):
        self.y_train = y_train

    def sety_val(self, y_val):
        self.y_val = y_val

    def split_data(self, train_ratio=0.8):
        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)
        n_train = int(np.floor(self.x.shape[0] * train_ratio))
        indices_train = indices[:n_train]
        indices_val = indices[n_train:]
        self.x_train = self.x[indices_train, :]
        self.y_train = self.y[indices_train]
        self.x_val = self.x[indices_val, :]
        self.y_val = self.y[indices_val]

    def train(self):
        pass

    def score(self):
        return self.classificador.score(self.x_val, self.y_val)

    def calculate_error(self):
        return np.mean(self.y_val == self.y_pred).astype('float32')

    def predict(self):
        pass

    def process(self):
        self.method.process(self)

    def recall_score(self):
        return recall_score(self.y_val, self.y_pred, average='micro')
