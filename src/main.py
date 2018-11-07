from LogisticRegressor import LogisticRegressor
from Basic import Basic
from KFold import KFold

if __name__ == '__main__':

    classificador = LogisticRegressor("parkinsons_updrs.data", Basic())
    classificador.process()

    classificador = LogisticRegressor("parkinsons_updrs.data", KFold(2))
    classificador.process()

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
    #
    #
    #
    # # ---------------------------------------------------------------------------------------------------------
    # # Codi regressor logistic
    # # for part in particions:
    # #     x_t, y_t, x_v, y_v = cl.split_data(x, y, part)
    # #     # Creem el regresor logistic
    # #     logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    # #     # l'entrenem
    # #     logireg.fit(x_t, y_t)
    # #
    # #     print("Correct classification Logistic ", part*100, "%: ", logireg.score(x_v, y_v))
    # #     y_pred = logireg.predict(x_v)
    # #     percent_correct_log = np.mean(y_v == y_pred).astype('float32')
    # #     print("Correct classification Logistic ", part, "%: ", percent_correct_log, "\n")
    #
    # # Crida a codi SVM
    # # print(cl.train_svm(x, y, kernel=kernel, C=0.01, gamma=0.001, probability=True))

