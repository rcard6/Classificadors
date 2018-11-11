from sklearn.svm import SVC
from src.Classificador import Classificador


class SVM(Classificador):

    def __init__(self, path, applicationmethod, kernel='linear', c=0.01, gamma=0.001, probability=True):
        Classificador.__init__(self, path, applicationmethod)
        self.svclin = SVC(C=c, kernel=kernel, gamma=gamma, probability=probability)
        self.probs = 0
        self.model = 0

    def train(self):
        self.model = self.svclin.fit(self.x.astype(float), self.y.astype(int))

    def predict(self):
        self.probs = self.model.predict_proba(self.x_val)
        print(self.probs)
