from sklearn.svm import SVC
import src.Classificador as Classificador

class SVM(Classificador):

    def __init__(self, path):
        Classificador.__init__(self, path)

    def train(self, kernel='linear', C=0.01, gamma=0.001, probability=True):
        svclin = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
        return svclin.fit(self.x.astype(float), self.y.astype(int))