from abc import ABC, abstractmethod


class BaseFunction(ABC):

    @abstractmethod
    def eval(self, x):
        '''Evaluation on a particular point x'''
