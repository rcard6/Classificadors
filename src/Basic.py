from src.ApplicationMethod import ApplicationMethod


class Basic(ApplicationMethod):

    def __init__(self):
        pass

    def process(self, classificador):

        part = 0.7
        classificador.split_data(part)
        classificador.train()
        classificador.predict()
        print("Correct classification Logistic(Basic) ", part * 100, "%: ", classificador.score() * 100)
        print("Correct classification Logistic(Basic) ", part * 100, "%: ", classificador.calculate_error() * 100)
        print("Recall: ", classificador.recall_score())
