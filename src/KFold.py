from src.ApplicationMethod import ApplicationMethod

class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k

    def process(self, classificador):
        part = 0.2
        score = 0
        error = 0
        for i in range(self.k):
            n = classificador.getx().shape[0]
            n_k = n/self.k

            classificador.split_data(part)
            classificador.train()
            classificador.predict()
            # Cal acumular l'error en una variable i dividir entre k
            score += classificador.score()
            error += classificador.calculateError()
        print("Correct classification Logistic(K-fold) ", part * 100, "%: ", (score / self.k)*100)
        print("Correct classification Logistic(K-fold) ", part * 100, "%: ", (error / self.k)*100)

