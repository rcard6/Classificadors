from LogisticRegressor import LogisticRegressor
from Basic import Basic
from KFold import KFold
from SVM import SVM
from OneLeave import OneLeave

if __name__ == '__main__':

    DATA = "parkinsons_updrs.data"
    # #   Manera de classificar: 'Basic'
    # #   Classificador: Regressor logistic
    classificador = LogisticRegressor(DATA, Basic())
    classificador.process()
    # #   Classificador: SVM
    # #   Tipus Kernel: linear
    classificador = SVM(DATA, Basic(), 'linear')
    classificador.process()
    # # #   Tipus Kernel: Polinomial
    # classificador = SVM(DATA, Basic(), 'poly')
    # classificador.process()
    # # #   Tipus Kernel: Gaussia
    # classificador = SVM(DATA, Basic(), 'rbf')
    # classificador.process()
    # ----------------------------------------------------------------------
    #   Manera de classificar: 'K-fold validation'
    #   Classificador: Regressor logistic
    # classificador = LogisticRegressor(DATA, KFold(4))
    # classificador.process()
    # Classificador: SVM
    # Tipus Kernel: linear
    # classificador = SVM(DATA, KFold(2), 'linear')
    # classificador.process()
    # # Tipus Kernel: Polinomial
    # classificador = SVM(DATA, KFold(4), 'poly')
    # classificador.process()
    # # Tipus Kernel: Gaussia
    # classificador = SVM(DATA, KFold(4), 'rbf')
    # classificador.process()
    # ----------------------------------------------------------------------
    # Manera de classificar: 'Leave-One-Out-Cross-Validation'
    # Classificador: Regressor logistic
    # classifcador = LogisticRegressor(DATA, OneLeave())
    # classifcador.process()
    # Classificador: SVM
    # Tipus Kernel: linear
    # Tipus Kernel: Polinomial
    # Tipus Kernel: Gaussia

