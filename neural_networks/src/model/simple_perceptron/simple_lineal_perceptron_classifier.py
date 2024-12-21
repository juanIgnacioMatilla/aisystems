from neural_networks.src.model.simple_perceptron.simple_lineal_perceptron import SimpleLinearPerceptron


class SimplePerceptronClassifier(SimpleLinearPerceptron):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate, activation_function=lambda x: 1 if x >= 0 else 0)

