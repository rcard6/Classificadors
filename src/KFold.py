from src.ApplicationMethod import ApplicationMethod


class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k

    def process(self, classificador):
        score = 0
        error = 0
        k_aux = 1
        for i in range(self.k):
            k_aux += i
            n = classificador.getx().shape[0]
            n_k = (n/k_aux)/10000  # Per obtenir un train ratio valid cal dividir per 10k
            classificador.split_data(n_k)
            classificador.train()
            classificador.predict()
            # Cal acumular l'error en una variable i dividir entre k, es podria fer amb np.array i fer el mean
            score += classificador.score()
            error += classificador.calculate_error()
        print("Correct classification Logistic(K-fold): ", (score / self.k)*100)
        print("Correct classification Logistic(K-fold): ", (error / self.k)*100)

