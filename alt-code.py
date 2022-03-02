# Run this script to generate the figures in 
# "Infinitely Divisible Noise in the Low Privacy Regime"
# by Rasmus Pagh and Nina Mesing Stausholm

import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import code
from scipy import signal
import pickle

stepsize = 0.001 # width/stepsize should be integer
width = 8


# Density functions

def gamma_density(x, alpha, beta):
    if x < 1e-10: # Ignore probability mass close to zero for numerical stability
        return 0
    else:
        return (beta**alpha) * (x**(alpha - 1)) * np.exp(- beta * x) / math.gamma(alpha)
    
def laplace_density(x, gamma):
    return gamma * np.exp(-gamma * np.abs(x)) / 2

def normal_density(x, sigma = 1):
    return np.exp(- (x/sigma)**2 / 2) / (sigma * np.sqrt(2 * np.pi))

def compute_arete(alpha, theta, lamb):
    eval_range = np.arange(-width, width + stepsize/2, stepsize)
    gamma_val = [ gamma_density(x, alpha, 1/theta) * stepsize for x in eval_range ]
    gamma_val[len(gamma_val)//2] = 1 - sum(gamma_val)
    laplace_val = [ laplace_density(x, 1/lamb) * stepsize for x in eval_range ]
    gamma_minus_gamma = signal.convolve(gamma_val, np.flip(gamma_val))
    arete_val = signal.convolve(gamma_minus_gamma, laplace_val)
    arete_val = arete_val[len(arete_val)//2 - len(eval_range)//2 : len(arete_val)//2 + (len(eval_range)+1) // 2]
    return arete_val


# Compare CDFs for Arete, Laplace, and Staircase distributions

def compute_cdfs(epsilon):
    alpha = np.exp(- epsilon / 4)
    theta = 4 / epsilon
    lamb = np.exp(- epsilon / 4)
    n = 100000
    arete = np.random.gamma(alpha, scale = theta, size = n) - np.random.gamma(alpha, scale = theta, size = n) + np.random.exponential(scale = lamb, size = n) - np.random.exponential(scale = lamb, size = n)
    laplace_param = 1./epsilon
    laplace = np.random.exponential(scale = laplace_param, size = n) - np.random.exponential(scale = laplace_param, size = n)
    staircase_epsilon = epsilon / 2
    staircase_width_param = 1 / (1 + np.exp(staircase_epsilon / 2))
    staircase_height_param = 0.5 * (1 - np.exp(-staircase_epsilon)) / (staircase_width_param + np.exp(-staircase_epsilon) * (1 - staircase_width_param))
    arete.sort()
    laplace.sort()
    x_min, x_max = laplace[n//100], laplace[-n//100]
    plt.clf()
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1)
    plt.plot(arete,np.linspace(0, 1, n), label=f'Arete({round(alpha,3)},{round(theta, 3)},{round(lamb, 3)})')
    plt.plot(laplace,np.linspace(0, 1, n), label=f'Laplace({round(laplace_param, 3)})')
    plt.plot([-staircase_width_param - 1, -staircase_width_param, staircase_width_param, staircase_width_param + 1], [0.5 - staircase_width_param * staircase_height_param - staircase_height_param / np.exp(staircase_epsilon), 0.5 - staircase_width_param * staircase_height_param, 0.5 + staircase_width_param * staircase_height_param, 0.5 + staircase_width_param * staircase_height_param + staircase_height_param / np.exp(staircase_epsilon)], label=f"Staircase(epsilon/2)" )
    plt.legend()
    plt.title(f"Empirical CDF ($n$={n}), " + r"$\varepsilon$=" + f"{epsilon}")
    plt.savefig(f"arete_cdf_eps_{epsilon}.pdf")
    
for epsilon in [5, 10, 15]:
    compute_cdfs(epsilon)


# Privacy loss of staircase and Arete mechanisms

def estimate_epsilon(alpha, theta, lamb):
    eval_range = np.arange(-width, width + stepsize/2, stepsize)
    arete_val = compute_arete(alpha, theta, lamb)
    num_steps = int(1/stepsize)
    ratios = arete_val[num_steps:] / arete_val[:-num_steps]
    observed_epsilon = np.log(max(ratios))
    observed_error = sum([abs(eval_range[i]) * arete_val[i] for i in range(len(eval_range))])
    return observed_epsilon, observed_error

def plot_privacy_loss(alpha, theta, lamb, plot_range = [0, 4], y_range = [0, 4], filename = None):
    observed_epsilon, observed_error = estimate_epsilon(alpha, theta, lamb)
    eval_range = np.arange(-width, width + stepsize/2, stepsize)
    arete_val = compute_arete(alpha, theta, lamb)
    density_at_zero = arete_val[len(arete_val)//2]
    plt.clf()
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(eval_range, - np.log(arete_val / density_at_zero) / observed_epsilon)
    plt.xlim((min(plot_range), max(plot_range)))
    plt.ylim((min(y_range), max(y_range)))
    plt.title(r'Worst case privacy loss ($\varepsilon$ =' + str(round(observed_epsilon)) + ')')
    plt.xlabel(r'$|q(x)-q(y)|/\Delta$')
    plt.ylabel(r'Privacy loss / $\varepsilon$')
    if filename is not None:
        plt.savefig(filename)

def plot_staircase_privacy_loss(plot_range = [0, 4], y_range = [0, 4], filename = None):
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.clf()
    plt.step(range(min(plot_range), max(plot_range)+1), range(min(plot_range), max(plot_range)+1))
    plt.xlim((min(plot_range), max(plot_range)))
    plt.ylim((min(y_range), max(y_range)))
    plt.title(r'Worst case privacy loss')
    plt.xlabel(r'$|q(x)-q(y)|/\Delta$')
    plt.ylabel(r'Privacy loss / $\varepsilon$')
    if filename is not None:
        plt.savefig(filename)


epsilon_to_arete_parameters = {} # Empirically found parameter combinations
epsilon_to_arete_parameters[6] = (0.07559007048913552, 0.6295929433519876, 0.05453349704921005)
epsilon_to_arete_parameters[7] = (0.03438865645893487, 0.7753538691868341, 0.03916449536622209)
epsilon_to_arete_parameters[8] = (0.01922231819807612, 0.8312962061581013, 0.02547890283133737)

plot_staircase_privacy_loss(filename="SC_PrivacyLoss.pdf")
for eps in [6]:
    alpha, theta, lamb = epsilon_to_arete_parameters[eps]
    plot_privacy_loss(alpha, theta, lamb, filename=f"arete_privacy_loss_eps_{eps}.pdf")


# Arete versus Laplace density functions

def plot_arete(alpha, theta, lamb, epsilon, plot_range = [-1, 1], filename = None):
    eval_range = np.arange(-width, width + stepsize/2, stepsize)
    arete_val = compute_arete(alpha, theta, lamb)
    epsilon_laplace_val = [ laplace_density(x, epsilon) for x in eval_range ]
    epsilon_laplace_val = epsilon_laplace_val / np.sum(epsilon_laplace_val)
    plt.clf()
    plt.plot(eval_range, arete_val, label=f'Arete({round(alpha,3)},{round(theta, 3)},{round(lamb, 3)})')
    plt.plot(eval_range, epsilon_laplace_val, label=f'Laplace({round(1/epsilon, 3)})')
    plt.xlim((min(plot_range), max(plot_range)))
    plt.legend()
    if filename is not None:
        plt.savefig(filename)

for eps in [6, 8]:
    alpha, theta, lamb = epsilon_to_arete_parameters[eps]
    observed_epsilon, observed_error = estimate_epsilon(alpha, theta, lamb)
    plot_arete(alpha, theta, lamb, observed_epsilon, [-0.5,0.5], f"arete_density_eps_{round(observed_epsilon)}.pdf")
