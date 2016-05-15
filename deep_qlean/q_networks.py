import logging
import numpy as np
log = logging.getLogger(__name__)

class QNetwork:
    def __init__(self):
        pass

    def _createLayers(self):
        raise NotImplementedError

    def train(self, minibatch):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def load_weight(self, filename):
        raise NotImplementedError

    def save_weights(self, filename):
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

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass


class ConvAutoEncoder(QNetwork):
    def __init__(self,shape):
        pass

    def _createLayers(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError
