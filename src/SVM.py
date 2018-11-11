from sklearn.svm import SVC
from Classificador import Classificador
from sklearn.metrics import recall_score


class SVM(Classificador):

    def __init__(self, path, applicationmethod, kernel='linear', c=0.01, gamma=0.001, probability=True):
        Classificador.__init__(self, path, applicationmethod)
        self.classificadorName = "SVM"
        print "Kernel: "+kernel
        self.classificador = SVC(C=c, kernel=kernel, gamma=gamma, probability=probability)

    def train(self):
        self.classificador.fit(self.x.astype(float), self.y.astype(int))



