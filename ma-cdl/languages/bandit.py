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
class Bandit(CDL):
    def __init__(self, agent_radius, num_obstacles, obstacle_radius, dynamic_obstacles):
        super().__init__(agent_radius, num_obstacles, obstacle_radius, dynamic_obstacles)
        self.arms = []
        self.coeffs = []
        problem_scenarios = get_problem_list()
        self.X = {scenario: [np.empty((0, 12)) for _ in range(MAX_ARMS)] for scenario in problem_scenarios}
        self.y = {scenario: [np.empty((0,)) for _ in range(MAX_ARMS)] for scenario in problem_scenarios}
        self._create_arm()
    
    # Add a new arm to the pool
    def _create_arm(self):
        kernel = C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
        self.arms.append(gp)
        self.coeffs.append([])
    
    # Choose arm with highest reward uncertainty to explore (UCB algorithm)
    def _select_arm(self):
        ucb_values = []
        exploration_factor = 0.2
        
        for i, gp in enumerate(self.arms):
            if len(self.coeffs[i]) == 0:
                return i
            input_array = np.array(self.coeffs[i])
            mean, std = gp.predict(input_array, return_std=True)
            mean = np.mean(mean)
            std = np.mean(std)
            n_pulls = len(self.coeffs[i])
            ucb = mean + exploration_factor * (std / np.sqrt(n_pulls))
            ucb_values.append(ucb)
        
        return np.argmax(ucb_values)
    
    # Choose coefficients from the selected arm
    def _select_coeffs_from_arm(self, arm_idx, encoded_scenario):
        x0 = self.rng.uniform(-1, 1, size=3)
        bounds = [(-1, 1)] * 3
        
        def cost_function(coeffs, encoded_scenario):
            input_array = np.hstack((coeffs, encoded_scenario, arm_idx))
            return self.arms[arm_idx].predict([input_array])[0]
        
        res = minimize(cost_function, x0, bounds=bounds, args=(encoded_scenario,))
        return res.x
    
    # Select arm highest UCB value, then select coefficients from that arm
    def _select_coeffs(self):
        arm_idx = self._select_arm()
        coeffs = self._select_coeffs_from_arm(arm_idx)
        self.coeffs[arm_idx].append(coeffs)
        return coeffs, arm_idx
    
    def _generate_optimal_coeffs(self, scenario):
        total_cost = []
        for episode in range(N_ROUNDS):
            counter = 0
            total_coeffs = []
            
            while True:
                counter += 1
                coeffs, arm_idx = self._select_coeffs()
                total_coeffs.append(coeffs)
                input_array = np.hstack((coeffs))
                cost, _ = self._optimizer(total_coeffs)
                
                total_cost.append(cost)
                avg_cost = np.mean(total_cost[-100:])

                # Add new arm if cost is not optimal, all arms explored, and max arms not reached
                if cost > COST_THRESH and counter == len(self.arms) and len(self.arms) < MAX_ARMS:
                    self._create_arm()
                    
                X_new = input_array.reshape(1, -1)
                self.X[scenario][arm_idx] = np.vstack((self.X[scenario][arm_idx], X_new))
                self.y[scenario][arm_idx] = np.append(self.y[scenario][arm_idx], cost)
                self.arms[arm_idx].fit(self.X[scenario][arm_idx], self.y[scenario][arm_idx])
                
                # If cost is optimal or all lines have been explored, stop
                if cost <= COST_THRESH or counter == MAX_ARMS:
                    break
                
            print(f'Episode: {episode}\nAverage Penalty: {avg_cost}\n')
            
            
"""  
Infinitely Armed Bandit  w/ Context 
Context: Scenario and arm index
"""
class ContextualBandit(Bandit):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
