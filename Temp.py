import my_adam
import numpy as np


def loss_function(m):
    return m ** 2 - 2 * m + 1


def grad_function(m):
    return 2 * m - 2


def check_convergence(theta0, theta1):
    return all(theta0 == theta1)


###################################################################

theta = np.random.rand(5)
num_iter = 1
converged = False
adam = my_adam.adam_optimizer(theta)
d_theta = np.zeros_like(theta)
old_theta = np.zeros_like(theta)
while not converged:
    for i, theta_i in enumerate(theta):
        d_theta[i] = grad_function(theta_i)
        old_theta = theta_i
    theta = adam.step(num_iter, theta, d_theta)
    if check_convergence(theta, old_theta):
        print('number of iterations until ADAM converged: ' + str(num_iter))
        break
    else:
        print('iteration number ' + str(num_iter) + ' : weight = ' + str(theta))
        num_iter += 1

##########################################
# loss(w,b) = (w ** 2 - 2 * w + 1) + (b ** 2 - 2 * b + 1)
# w = b = 1
#
# loss(theta) = sum(theta_i ** 2 - 2 * theta_i + 1)
