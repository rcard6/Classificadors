from ApplicationMethod import ApplicationMethod

class KFold(ApplicationMethod):
    def __init__(self, k):
        ApplicationMethod.__init__(self)
        self.k = k
    def process(self, classificador):
        for i in range(self.k):
            n = classificador.getX().shape[0]
            n_k = n/self.k

