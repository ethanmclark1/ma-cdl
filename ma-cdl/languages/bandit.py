import numpy as np

from math import inf
from scipy.optimize import minimize
from languages.utils.cdl import CDL
from Signal8 import get_problem_list
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

MAX_ARMS = 7
N_ROUNDS = 2500
COST_THRESH = 20

"""Infinitely Armed Bandit"""
# TODO: Entire thing
class Bandit(CDL):
    def __init__(self, agent_radius, obstacle_radius):
        super().__init__(agent_radius, obstacle_radius)