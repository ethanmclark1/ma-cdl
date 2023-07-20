import time
import wandb
import warnings
import numpy as np

from scipy.optimize import minimize
from languages.utils.cdl import CDL
from languages.utils.replay_buffer import ReplayBufferMAB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

warnings.filterwarnings('ignore')

""" Infinitely Armed Bandit using GP-UCB """
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()
        self.memory = ReplayBufferMAB(self.action_dim, self.mem_len)
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.alpha = 1
        self.n_restarts = 100
        self.num_episodes = 750
        self.mem_len = round(self.num_episodes * 0.10)
        self.record_freq = self.num_episodes // num_records
        
        self.matern_nu = 1.5
        self.matern_length_scale = 1
        self.matern_length_scale_bounds = (1e-4, 1e4)
        
        self.constant_length_scale = 1
        self.constant_length_scale_bounds = (1e-4, 1e4)
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='Bandit')
        config = wandb.config
        config.weights = self.weights
        config.mem_len = self.mem_len
        config.num_episodes = self.num_episodes
        
        config.matern_nu = self.matern_nu
        config.matern_length_scale = self.matern_length_scale
        config.matern_length_scale_bounds = self.matern_length_scale_bounds
        
        config.constant_length_scale = self.constant_length_scale
        config.constant_length_scale_bounds = self.constant_length_scale_bounds
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
    
    def _create_model(self, problem_instance):
        self.valid_lines = set()
        self.coeffs = np.empty((0, self.action_dim))
        self.rewards = np.empty((0, 1))
        
        kernel = C(self.constant_length_scale, self.constant_length_scale_bounds) \
            * Matern(self.matern_length_scale, self.matern_length_scale_bounds, self.matern_nu)
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts, alpha=self.alpha)
        
        coeffs = self.rng.uniform(-1, 1, self.action_dim)
        reward, _, _ = self._step(problem_instance, coeffs, num_lines=1)
        self._update_model(coeffs, reward, num_lines=1)
                
    # Overlay lines in the environment
    def _step(self, problem_instance, coeffs, num_lines):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
        
        line = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_lines == self.max_lines:
            done = True
            reward = super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
        
        return reward, done, regions
    
    # Choose coefficients using UCB approach
    def _select_coeffs(self):
        x0 = self.rng.uniform(-1, 1, self.action_dim)
        bounds = [(-1, 1)] * 3
        
        def objective(x):
            mean, std = self.gpr.predict([x], return_std=True)
            ucb = mean  + std
            return -ucb[0] 
        
        res = minimize(objective, x0, bounds=bounds)
        return res.x
    
    # Learn from experience
    def _update_model(self, coeffs, reward, num_lines):        
        distributed_reward = reward / num_lines
        rewards = np.full((num_lines, 1), distributed_reward)     
        
        self.memory.add(coeffs, rewards)
        
        actions, rewards = self.memory.get_data()
        self.gpr.fit(actions, rewards)    
    
    # Train model on a given problem_instance
    def _train(self, problem_instance):
        returns = []
        
        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            coeffs_lst = []
            while not done:
                num_lines += 1
                coeffs = self._select_coeffs()
                reward, done, regions = self._step(problem_instance, coeffs, num_lines)
                coeffs_lst.append(coeffs)
                
            self._update_model(coeffs_lst, reward, num_lines)
            
            returns.append(reward)
            avg_returns = np.mean(returns[-50:])
            wandb.log({'avg_returns': avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
                     
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        start_time = time.time()
        
        self._create_model(problem_instance)
        self._train(problem_instance)
        
        done = False
        num_lines = 1
        optim_coeffs = []
        while not done:
            action = self._select_coeffs()
            optim_coeffs.append(action)
            reward, done, regions = self._step(problem_instance, action, num_lines)
            num_lines += 1
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        
        self._log_regions(problem_instance, 'Final', regions, reward)
        
        wandb.log({"Final Reward": -reward})
        wandb.log({"Best Coeffs": optim_coeffs})
        wandb.log({"Best Num Lines": len(optim_coeffs)})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()  

        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        return optim_coeffs