from sklearn.model_selection import LeaveOneOut
from ApplicationMethod import ApplicationMethod
import numpy as np


class OneLeave(ApplicationMethod):
    def __init__(self):
        ApplicationMethod.__init__(self)
        self.splitter = LeaveOneOut()

    def process(self, classificador):
        x = classificador.getx()
        n = x.shape[0]
        y = classificador.gety()
        score = np.array(n)
        error = np.array(n)
        for i in range(n):
            for train_index, test_index in self.splitter.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classificador.setx_train(X_train)
                classificador.setx_val(X_test)
                classificador.sety_train(y_train)
                classificador.sety_val(y_test)
                classificador.train()
                classificador.predict()
                np.append(score, classificador.score())
                np.append(error, classificador.calculate_error())
        print "Mitja score (One Leave Out): " + str(np.mean(score))
        print "Mitja error (One Leave Out): " + str(np.mean(error))
        print "Recall:"
        classificador.recall_score()


    def split_data(self, classificador, i):
        pass

