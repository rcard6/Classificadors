from src.ApplicationMethod import ApplicationMethod


class Basic(ApplicationMethod):

    def __init__(self):
        pass

    def process(self,classificador):

        part = 0.7
        x_t, y_t, x_v, y_v = classificador.split_data(part)
        classificador.train(x_t, y_t)
        y_pred = classificador.predict(x_v)

        print("Correct classification Logistic(Basic) ", part * 100, "%: ", classificador.score(x_v, y_v))
        print("Correct classification Logistic(Basic) ", part * 100, "%: ", classificador.calculateError(y_v, y_pred))
