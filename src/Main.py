import src.Classificador
import numpy as np
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    cl = src.Classificador.Classificador()
    x, y = cl.load_dataset('parkinsons_updrs.data')

    # x = x.astype('float32')
    # y = y.astype('float32')

    print(x.shape);
    print(y.shape)

    tipus = ''

    # Bàsica
    tipus = 'Bàsica'
    # Logistic Regression
    particions = [0.5, 0.7, 0.8]

    # SVM - Tipus kernel:
    # Linear
    kernel = 'linear'
    # Polinomial
    # kernel = 'poly' - Descomentar quan estigui fet el switch
    # Gaussia
    # kernel = 'rbf' - Descomentar quan estigui fet el switch

    # k-Fold Validation
    tipus = 'K-fold validation'
    # Logistic Regression

    # SVM - Tipus kernel:
    # Linear
    # Polinomial
    # Gaussia

    # Leave-One-Out-Cross-Validation
    tipus = 'Leave-One-Out-Cross-Validation'
    # Logistic Regression

    # SVM - Tipus kernel:
    # Linear
    # Polinomial
    # Gaussia


    # ---------------------------------------------------------------------------------------------------------
    # Codi regressor logistic
    for part in particions:
        x_t, y_t, x_v, y_v = cl.split_data(x, y, part)
        # Creem el regresor logístic
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
        # l'entrenem
        logireg.fit(x_t, y_t)

        print("Correct classification Logistic ", part, "%: ", logireg.score(x_v, y_v))
        y_pred = logireg.predict(x_v)
        percent_correct_log = np.mean(y_v == y_pred).astype('float32')
        print("Correct classification Logistic ", part, "%: ", percent_correct_log, "\n")

    # Crida a codi SVM
    print(cl.train_svm(x, y, kernel=kernel, C=0.01, gamma=0.001, probability=True))

