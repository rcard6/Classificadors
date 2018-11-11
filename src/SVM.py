from sklearn.svm import SVC
from src.Classificador import Classificador


class SVM(Classificador):

    def __init__(self, path, applicationmethod, kernel='linear', c=0.01, gamma=0.001, probability=True):
        Classificador.__init__(self, path, applicationmethod)
        self.kernel = kernel
        self.C = c
        self.gamma = gamma
        self.probability = probability
        self.model = 0

    def train(self):
        svclin = SVC(self.C, self.kernel, self.gamma, self.probability)
        self.model = svclin.fit(self.x.astype(float), self.y.astype(int))

    def predict(self):
        self.model.predict_proba(self)

