from src.LogisticRegressor import LogisticRegressor
from src.Basic import Basic
from src.KFold import KFold
from src.SVM import SVM

if __name__ == '__main__':

    DATA = "parkinsons_updrs.data"
    # Manera de classificar: 'Basic'
    # Classificador: Regressor logistic
    classificador = LogisticRegressor(DATA, Basic())
    classificador.process()

    # Classificador: SVM
    # Tipus Kernel: linear
    classificador = SVM(DATA, Basic(), 'linear')
    classificador.process()

    # Tipus Kernel: Polinomial
    # Tipus Kernel: Gaussià

    # Manera de classificar: 'K-fold validation'
    # Classificador: Regressor logistic
    classificador = LogisticRegressor(DATA, KFold(2))
    classificador.process()

    # Classificador: SVM
    # Tipus Kernel: linear
    # Tipus Kernel: Polinomial
    # Tipus Kernel: Gaussià

    # Manera de classificar: 'Leave-One-Out-Cross-Validation'
    # Classificador: Regressor logistic

    # Classificador: SVM
    # Tipus Kernel: linear
    # Tipus Kernel: Polinomial
    # Tipus Kernel: Gaussià

