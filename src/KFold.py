from src.ApplicationMethod import ApplicationMethod


class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k

    def process(self, classificador):
        # part = 0.2
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
            # ATENCIÓ: L'score, l'error i n_k s'han de multiplicar per 100?
            print("Correct classification Logistic(K-fold) ", n_k, "%: ", (score / self.k))
            print("Correct classification Logistic(K-fold) ", n_k, "%: ", (error / self.k))
            # Cal acumular l'error en una variable i dividir entre k
            score += classificador.score()
            error += classificador.calculate_error()

        print("Correct classification Logistic(K-fold): ", (score / self.k)*100)
        print("Correct classification Logistic(K-fold): ", (error / self.k)*100)

