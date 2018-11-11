from sklearn.svm import SVC
from Classificador import Classificador
from sklearn.metrics import recall_score


class SVM(Classificador):

    def __init__(self, path, applicationmethod, kernel='linear', c=0.01, gamma=0.001, probability=True):
        Classificador.__init__(self, path, applicationmethod)
        self.classificadorName = "SVM"
        self.classificador = SVC(C=c, kernel=kernel, gamma=gamma, probability=probability)
        self.model = 0

    def train(self):
        self.model = self.classificador.fit(self.x.astype(float), self.y.astype(int))

    def predict(self):
        self.y_pred = self.model.predict_proba(self.x_val)

    def recall_score(self):
        return recall_score(self.y_val, self.y_pred, average='binary')
