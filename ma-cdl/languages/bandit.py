import os
import pickle
import numpy as np

from scipy.optimize import minimize
from languages.utils.cdl import CDL
from sklearn.preprocessing import OneHotEncoder
from environment.utils.problems import problem_scenarios
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

MAX_ARMS = 10
N_ROUNDS = 2500
COST_THRESH = 20

""" Infinitely Armed Bandit w/o Context """
class InfinitelyArmedBandit(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles, n_arms=1):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self.encoder = OneHotEncoder()
        self.rng = np.random.default_rng()
        self.n_scenarios = len(problem_scenarios)
        scenarios = np.array(list(problem_scenarios.keys())).reshape(-1, 1)
        self.encoded_scenarios = self.encoder.fit_transform(scenarios).toarray()
        
        self.arms = []
        self.coeffs = []
        self.X = [np.empty((0, 3)) for _ in range(len(self.arms))]
        self.y = [np.empty((0,)) for _ in range(len(self.arms))]
        self.coeffs_by_arm = [[] for _ in range(n_arms)]
        self.rewards_by_arm = [[] for _ in range(n_arms)]
        self._create_arm()
        
    def _save(self):
        class_name = self.__class__.__name__
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(self.language, file)
    
    def _load(self):
        class_name = self.__class__.__name__
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        self.language = language
    
    # Add a new arm to the pool
    def _create_arm(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
        self.arms.append(gp)
        self.coeffs.append([])
        self.coeffs_by_arm.append([])
        
        self.X.append(np.empty((0, 3)))
        self.y.append(np.empty((0,)))
    
    # Choose arm with highest reward uncertainty to explore (UCB algorithm)
    def _select_arm(self):
        ucb_values = []
        exploration_factor = 0.2
        
        for i, gp in enumerate(self.arms):
            if len(self.coeffs_by_arm[i]) == 0:
                return i
            input_array = np.array(self.coeffs_by_arm[i])
            mean, std = gp.predict(input_array, return_std=True)
            n_pulls = len(self.coeffs_by_arm[i])
            ucb = mean + exploration_factor * (std / np.sqrt(n_pulls))
            ucb_values.append(ucb)
        
        return np.argmax(ucb_values)
    
    # Choose coefficients from the selected arm
    def _select_coeffs_from_arm(self, arm_idx):
        x0 = np.zeros(3)+0.0001
        bounds = [(-1, 1)] * 3
        res = minimize(lambda x: self.arms[arm_idx].predict([x])[0], x0, bounds=bounds)
        return res.x
    
    # Select arm highest UCB value, then select coefficients from that arm
    def _select_coeffs(self):
        arm_idx = self._select_arm()
        coeffs = self._select_coeffs_from_arm(arm_idx)
        self.coeffs[arm_idx].append(coeffs)
        self.coeffs_by_arm[arm_idx].append(coeffs)
        return coeffs, arm_idx
    
    def _optimizer(self, coeffs, scenario):
        weights = np.array([1.75, 2, 1.5, 2, 2])
        criterion, regions = super()._optimizer(coeffs, scenario)
        problem_cost = np.sum(weights * criterion)
        return problem_cost, regions
    
    def _train_model(self):
        for _ in range(N_ROUNDS):
            counter = 0
            total_coeffs = []
            scenario = self.rng.choice(list(problem_scenarios.keys()))    
            
            while True:
                counter += 1
                coeffs, arm_idx = self._select_coeffs()
                total_coeffs.append(coeffs)
                cost, _ = self._optimizer(total_coeffs, scenario)

                # Add new arm if cost is not optimal, all arms explored, and max arms not reached
                if cost > COST_THRESH and counter == len(self.arms) and len(self.arms) < MAX_ARMS:
                    self._create_arm()

                X_new = coeffs.reshape(1, 3)
                self.X[arm_idx] = np.vstack((self.X[arm_idx], X_new))
                self.y[arm_idx] = np.append(self.y[arm_idx], cost)
                self.arms[arm_idx].fit(self.X[arm_idx], self.y[arm_idx])
                
                if cost <= COST_THRESH or counter == MAX_ARMS:
                    break

        

class ContextualBandit(InfinitelyArmedBandit):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)

    def _get_contexts(self):
        contexts = []
        for _ in range(self.n_rounds):
            bandit_number = self.rng.integers(0, self.n_bandits)
            encoded_scenario = self.rng.choice(self.encoded_scenarios, size=1).flatten().tolist()
            context = [bandit_number] + encoded_scenario
            context_2d = np.atleast_2d(context)
            contexts.append(context_2d)
        
        return contexts