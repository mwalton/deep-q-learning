import logging
import numpy as np
log = logging.getLogger(__name__)

class QNetwork:
    def __init__(self):
        pass

    def _createLayers(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def load_weigths(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError


class ConvNet(QNetwork):
    def __init__(self,shape):
        pass

    def _createLayers(self):
        raise NotImplementedError

    def train(self, minibatch):
        pass

    def predict(self, state):
        return np.random.rand(6)

    def load_weigths(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError


class ConvAutoEncoder(QNetwork):
    def __init__(self,shape):
        pass

    def _createLayers(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def load_weigths(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError
