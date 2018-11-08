from ApplicationMethod import ApplicationMethod

class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k
    def process(self, classificador):
        for i in range(self.k):
            n = classificador.getX().shape[0]
            n_k = n/self.k
            x_t, y_t, x_v, y_v = classificador.split_data(0.2)
            classificador.train(x_t, y_t)
            y_pred = classificador.predict(x_v)
            print classificador.score(x_v, y_v)
            print classificador.calculateError(y_v, y_pred)

