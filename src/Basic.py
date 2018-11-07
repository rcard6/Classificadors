import src.RegressorLog


class Basic(object):
    regLog = src.RegressorLog()

    # Logistic Regression
    particions = [0.7]  # Split de 0.7 implica 70% d'entrenament i 30% per validació


    # SVM - Tipus kernel:
    # Linear
    # kernel = 'linear'
    # Polinomial
    # kernel = 'poly' - Descomentar quan estigui fet el switch
    # Gaussia
    # kernel = 'rbf' - Descomentar quan estigui fet el switch