from ApplicationMethod import ApplicationMethod


class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k

    def process(self, classificador):
        score = 0
        error = 0
        for i in range(self.k):
            classificador.split_data(1/self.k)
            classificador.train()
            classificador.predict()
            # Cal acumular l'error en una variable i dividir entre k, es podria fer amb np.array i fer el mean
            score += classificador.score()
            error += classificador.calculate_error()
        if classificador.score() != 0:
            print("Correct classification (K-fold): ", (score / self.k)*100)
            print("Correct classification (K-fold): ", (error / self.k)*100)

