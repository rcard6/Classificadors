from src.ApplicationMethod import ApplicationMethod
import numpy as np


class OneLeave(ApplicationMethod):
    def __init__(self):
        ApplicationMethod.__init__(self)

    def process(self, classificador):
        n = classificador.getx().shape[0]
        x = classificador.getx()
        n_little = n-1
        score = np.array()
        error = np.array()
        for i in range(n):
            x_t, y_t, x_v, y_v = classificador.split_data(n_little/n)

            classificador.train(x_t, y_t)
            classificador.predict(x_v)
            np.append(score, classificador.score(x_v, y_v))
            np.append(error, classificador.calculate_error(y_v, self.y_pred))
        print(np.mean(error))
        print(np.mean(score))
