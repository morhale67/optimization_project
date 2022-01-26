import torch
import math
import numpy as np


class adam_optimizer(torch.optim.Optimizer):
    def __init__(self, params, step_size=0.01, lr=1e-3, weight_decay=0.0, epsilon=1e-6, correct_bias=True, betas=(0.9, 0.999)):  # params type is torch.Tensor
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >=0.0".format(lr))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >=0.0".format(epsilon))
        self.defaults = dict(lr=lr, betas=betas, epsilon=epsilon, weight_decay=weight_decay, correct_bias=correct_bias)
        self.num_iter = 0
        self.mean_d_theta, self.u_variance_d_theta = np.zeros_like(params), np.zeros_like(params)
        self.beta1 = betas[0]  # β₁ is the exponential decay of the rate for the first moment estimates
        self.beta2 = betas[1]  # β₂ is the exponential decay rate for the second-moment estimates
        self.epsilon = epsilon  # prevent zero-division
        self.step_size = step_size

        super(adam_optimizer, self).__init__(params, self.defaults)

    def step(self):
        self.num_iter += 1
        theta = self.defaults['params']
        mean_d_theta_corr, u_variance_d_theta_corr = np.zeros_like(theta), np.zeros_like(theta)
        for i, theta_i in enumerate(theta):
            grad_theta = theta_i.grad.data
            self.mean_d_theta[i] = self.beta1 * self.mean_d_theta[i] + (1 - self.beta1) * grad_theta
            self.u_variance_d_theta[i] = self.beta2 * self.u_variance_d_theta[i] + (1 - self.beta2) * (grad_theta ** 2)

            # the bias correction step
            mean_d_theta_corr[i] = self.mean_d_theta[i] / (1 - self.beta1 ** self.num_iter)
            u_variance_d_theta_corr[i] = self.u_variance_d_theta[i] / (1 - self.beta2 ** self.num_iter)

            # update weights and biases
            theta[i] = theta[i] - self.step_size * (mean_d_theta_corr[i] / (np.sqrt(u_variance_d_theta_corr[i])
                                                                            + self.epsilon))


                                    # for p in self.defaults['params']:
                                    #     if p.grad is None:
                                    #         continue
                                    #     grad = p.grad.data
                                    #     # if grad.is_sparse:
                                    #     #     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                                    #     state = self.state[p]  # dict, the optimizer that holds the curr configuration of all params.
                                    #     if len(state) == 0:
                                    #         state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(p.data), torch.zeros_like(p.data)
                                    #         exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                    #         beta1, beta2 = self.defaults['betas']
                                    #         state['step'] = state['step'] + 1
                                    #
                                    #     # Decay the first and second moment running average coefficient
                                    #     # In-place operations to update the averages at the same time
                                    #     exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                                    #     exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                                    #     denom = exp_avg_sq.sqrt().add_(self.defaults['eps'])
                                    #
                                    #     step_size = self.defaults['lr']
                                    #     if self.defaults['correct_bias']:  # No bias correction for Bert
                                    #         bias_correction1 = 1.0 - beta1 ** state['step']
                                    #         bias_correction2 = 1.0 - beta2 ** state['step']
                                    #         step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                                    #
                                    #     p.data.addcdiv_(-step_size, exp_avg, denom)
                                    #
                                    #     if self.defaults['weight_decay'] > 0.0:
                                    #         p.data.add_(-self.defaults['lr'] * self.defaults['weight_decay'], p.data)

