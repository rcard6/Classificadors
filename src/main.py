from src.LogisticRegressor import LogisticRegressor
from src.Basic import Basic
from src.KFold import KFold

if __name__ == '__main__':

    classificador = LogisticRegressor("parkinsons_updrs.data", Basic())
    classificador.process()

    # classificador = SVM("parkinsons_updrs.data", KFold(2))
    # classificador.process()

    # x, y = cl.load_dataset()
    #
    # print(x.shape)
    # print(y.shape)
    #
    # tipus = ''
    #
    # # k-Fold Validation
    # tipus = 'K-fold validation'
    # # Logistic Regression
    #
    # # SVM - Tipus kernel:
    # # Linear
    # # Polinomial
    # # Gaussia
    #
    # # Leave-One-Out-Cross-Validation
    # tipus = 'Leave-One-Out-Cross-Validation'
    # # Logistic Regression
    #
    # # SVM - Tipus kernel:
    # # Linear
    # # Polinomial
    # # Gaussia

