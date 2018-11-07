import numpy as np


class Classificador(object):

    def __init__(self, path, ApplicationMethod):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        x = data
        # Triatge output (parkinsons = status(-7), parkinsons_updrs = motor_UPDRS(4) o total_UPDRS(5))
        self.y = data[:, 4].astype('int32')
        self.x = np.delete(x, 4, axis=1)
        self.method = ApplicationMethod

    def split_data(self, train_ratio=0.8):
        print(train_ratio)
        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)
        n_train = int(np.floor(self.x.shape[0] * train_ratio))
        indices_train = indices[:n_train]
        indices_val = indices[n_train:]
        x_train = self.x[indices_train, :]
        y_train = self.y[indices_train]
        x_val = self.x[indices_val, :]
        y_val = self.y[indices_val]
        return x_train, y_train, x_val, y_val

    def train(self,x_t,y_t):
        pass

    def calculateError(self):
        pass

    def process(self):
        self.method.process(self)
