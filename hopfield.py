#!/usr/bin/env python
import numpy as np
__author__ = "Sujit Maharjan"
__copyright__ = "Copyright 2015, Neural Network"
__credits__ = ["Shasidhar Ram Joshi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Sujit Maharjan"
__email__ = "sujitmhj@gmail.com"
__status__ = "Testing"


class Hopefield(object):
    """docstring for Hopefield"""
    def __init__(self, n=3):
        super(Hopefield, self).__init__()
        self.n = n
        self.w = np.zeros(shape=(self.n, self.n))

    def train(self, inputs):
        M = inputs.shape[0]
        for i in range(M):
            self.w = self.w + np.dot(inputs[i], inputs[i].T)
        self.w = self.w - M * np.identity(self.n)

    def output(self, _input):
        out = np.dot(self.w, _input.T)
        return np.sign(out)

if __name__ == "__main__":
    # Testing the Hopfield network using 3 neurons
    hp = Hopefield(3)
    train_data = np.array([[1, 1, 1], [-1, -1, -1]])
    hp.train(train_data)
    print hp.output(np.array([-1, 1, -1]))
