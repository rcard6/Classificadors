from ApplicationMethod import ApplicationMethod
import numpy as np


class OneLeave(ApplicationMethod):
    def __init__(self):
        ApplicationMethod.__init__(self)

    def process(self, classificador):
        n = classificador.getx().shape[0]
        n_little = n-1
        score = np.zeros(n)
        error = np.array(n)
        for i in range(n):
            self.split_data(classificador, i)
            classificador.train()
            classificador.predict()
            np.append(score, classificador.score())
            np.append(error, classificador.calculate_error())
        print(np.mean(error))
        print(np.mean(score))

    def split_data(self, classificador, i):
        x = classificador.getx()
        y = classificador.gety()
        # Assignem X
        classificador.setx_val(x[i])
        x_t = np.delete(x, i, axis=1)
        classificador.setx_train(x_t)
        # Assignem Y
        classificador.sety_val(y[i])
        # y_t = np.delete(y, i, axis=1)
        classificador.sety_train(y)
