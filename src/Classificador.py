import numpy as np
import sklearn as svm
from sklearn.svm import SVC


class Classificador(object):

    def load_dataset(self, path):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        x = data.astype('int32')
        # Triatge output (parkinsons = status(-7), parkinsons_updrs = motor_UPDRS(4) o total_UPDRS(5))
        y = data[:, 4].astype('int32')
        x = np.delete(x, 4, axis=1)
        return x, y

    # < img src = "images/table_1.png" width = "80%" > # Ens recomanen fer una taula amb els valors obtinguts

    def split_data(self, x, y, train_ratio=0.8):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        n_train = int(np.floor(x.shape[0] * train_ratio))
        indices_train = indices[:n_train]
        indices_val = indices[-n_train:]
        x_train = x[indices_train, :]
        y_train = y[indices_train]
        x_val = x[indices_val, :]
        y_val = y[indices_val]
        return x_train, y_train, x_val, y_val

    def train_svm(self, x, y, kernel='linear', C=0.01, gamma=0.001, probability=True):
        if kernel == 'linear':
            svclin = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
        if kernel == 'poly':
            svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
        if kernel == 'rbf':
            svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)  # l'entrenem
        return svclin.fit(x, y)
