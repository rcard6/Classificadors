import src.Classificador as Classificador
import numpy as np
from sklearn.linear_model import LogisticRegression


class RegressorLog(Classificador):

    def regressor_logistic(self, x, y, particions, cl):
        for part in particions:
            x_t, y_t, x_v, y_v = self.split_data(x, y, part)
            # Creem el regresor logístic
            logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
            # l'entrenem
            logireg.fit(x_t, y_t)

            print("Correct classification Logistic ", part * 100, "%: ", logireg.score(x_v, y_v))
            y_pred = logireg.predict(x_v)
            percent_correct_log = np.mean(y_v == y_pred).astype('float32')
            print("Correct classification Logistic ", part, "%: ", percent_correct_log, "\n")