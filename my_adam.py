import numpy as np


class adam_optimizer:
    def __init__(self, theta, step_size=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.mean_d_theta, self.u_variance_d_theta = np.zeros_like(theta), np.zeros_like(theta)
        self.beta1 = beta1  # β₁ is the exponential decay of the rate for the first moment estimates
        self.beta2 = beta2  # β₂ is the exponential decay rate for the second-moment estimates
        self.epsilon = epsilon  # prevent zero-division
        self.step_size = step_size

    def step(self, num_iter, theta, d_theta):
        mean_d_theta_corr, u_variance_d_theta_corr = np.zeros_like(theta), np.zeros_like(theta)
        for i, theta_i in enumerate(theta):
            self.mean_d_theta[i] = self.beta1 * self.mean_d_theta[i] + (1 - self.beta1) * d_theta[i]
            self.u_variance_d_theta[i] = self.beta2 * self.u_variance_d_theta[i] + (1 - self.beta2) * (d_theta[i] ** 2)

            # the bias correction step
            mean_d_theta_corr[i] = self.mean_d_theta[i] / (1 - self.beta1 ** num_iter)
            u_variance_d_theta_corr[i] = self.u_variance_d_theta[i] / (1 - self.beta2 ** num_iter)

            # update weights and biases
            theta[i] = theta[i] - self.step_size * (mean_d_theta_corr[i] / (np.sqrt(u_variance_d_theta_corr[i])
                                                                            + self.epsilon))
        return theta


