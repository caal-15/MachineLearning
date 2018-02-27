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


class BayesMLRegression(MLRegression):

    def train(self, training_x, training_t, max_iterations=100):
        alpha = np.random.uniform(0.1, 0.5, (1, 1))
        beta = np.random.uniform(0.1, 0.5, (1, 1))

        design_mat = self._get_design_mat(training_x)
        design_mat_t = design_mat.T
        design_mat_2 = design_mat_t.dot(design_mat)

        beta_design_mat_2 = beta * design_mat_2
        cov_inv = (alpha * np.eye(len(design_mat_2))) + beta_design_mat_2
        mean = beta * (
            np.linalg.inv(cov_inv).dot(design_mat_t.dot(training_t)))

        lambda_i, _ = np.linalg.eig(beta_design_mat_2)
        alpha_lambda_inv = 1 / (alpha + lambda_i)

        for i in range(0, max_iterations):
            gamma = alpha_lambda_inv.dot(lambda_i)
            alpha = gamma / mean.dot(mean.T)
            beta = (len(training_x) - gamma) / np.linalg.norm(
                training_t - design_mat.dot(mean.T))

            beta_design_mat_2 = beta * design_mat_2
            cov_inv = (alpha * np.eye(len(design_mat_2))) + beta_design_mat_2
            mean = beta * (
                np.linalg.inv(cov_inv).dot(design_mat_t.dot(training_t)))

            lambda_i, _ = np.linalg.eig(beta_design_mat_2)
            alpha_lambda_inv = 1 / (alpha + lambda_i)

        self.w_ml = mean
