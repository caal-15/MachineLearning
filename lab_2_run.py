import numpy as np
from matplotlib import pyplot as plot
from regression.ml_regression import BayesMLRegression
from regression.base_function import PolynomialBF

start = 0
end = 1
num_samples = 100
full_spectrum = np.arange(start, end, (end - start) / num_samples)
full_sine = np.sin(2 * np.pi * full_spectrum)

sigma = 0.01
full_sine_noise = full_sine + np.random.normal(0, sigma, num_samples)

plot.plot(full_spectrum, full_sine)

training_samples = 20
all_indexes = np.random.permutation(num_samples)
x_training = full_spectrum[all_indexes[0:training_samples]]
t_training = full_sine_noise[all_indexes[0:training_samples]]

plot.plot(x_training, t_training, 'rx')

M = 6
base_functions = []
for i in range(0, M):
    base_functions.append(PolynomialBF(i))

ml_reg = BayesMLRegression(base_functions, x_training, t_training)
results = ml_reg.predict(full_spectrum)

plot.plot(full_spectrum, results)

plot.show()
