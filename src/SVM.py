from sklearn.svm import SVC
import src.Classificador as Classificador

class SVM(Classificador):

    def train_svm(self, x, y, kernel='linear', C=0.01, gamma=0.001, probability=True):
        svclin = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
        return svclin.fit(x.astype(float), y.astype(int))