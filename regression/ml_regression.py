import numpy as np


class MLRegression():

    def __init__(self, base_functions, training_x=None, training_t=None):
        self.base_functions = base_functions
        if training_x is not None and training_t is not None:
            self.train(training_x, training_t)

    def _get_design_mat(self, x):
        design_mat = np.zeros((len(x), len(self.base_functions)))
        for i in range(0, len(x)):
            for j in range(0, len(self.base_functions)):
                design_mat[i][j] = self.base_functions[j].eval(x[i])
        return design_mat

    def train(self, training_x, training_t):
        design_mat = self._get_design_mat(training_x)
        design_mat_t = design_mat.T
        design_mat_2_inv = np.linalg.inv(design_mat_t.dot(design_mat))
        self.w_ml = design_mat_2_inv.dot(design_mat_t.dot(training_t))

        aux = training_t - design_mat.dot(self.w_ml)
        self.beta_ml_inv = (1 / len(training_x)) * (aux.T.dot(aux))

    def predict(self, new_x):
        design_mat = self._get_design_mat(new_x)
        return design_mat.dot(self.w_ml)
