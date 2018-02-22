import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):

    @abstractmethod
    def eval(self, x):
        '''Evaluation on a particular point x'''


class PolynomialBF(BaseFunction):

    def __init__(self, grade):
        self.grade = grade

    def eval(self, x):
        return np.power(x, self.grade)
