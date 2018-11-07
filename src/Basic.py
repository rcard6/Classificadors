from ApplicationMethod import ApplicationMethod


class Basic(ApplicationMethod):

    def __init__(self):
        pass


    def process(self,classificador):
        x_t,y_t,x_v,y_v = classificador.split_data(0.7)
        classificador.train(x_t,y_t)
        y_pred = classificador.predict(x_v)
        print classificador.score(x_v, y_v)
        print classificador.calculateError(y_v, y_pred)
