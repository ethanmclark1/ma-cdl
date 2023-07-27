import time
import wandb
import warnings
import numpy as np

from scipy.optimize import minimize
from languages.utils.cdl import CDL
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

warnings.filterwarnings('ignore')

""" Infinitely Armed Bandit using GP-UCB """
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()
        
    def _init_hyperparams(self):
        self.alpha = 5e-2
        self.n_restarts = 350
        self.num_episodes = 500
        self.record_freq = self.num_episodes // 10
        
        self.epsilon_start = 1
        self.epsilon_decay = 0.99
        
        self.memory_length = 100
        
        self.matern_nu = 0.5
        self.matern_length_scale = 0.5
        self.matern_length_scale_bounds = (1e-5, 1e5)
        
        self.constant_length_scale = 1.25
        self.constant_length_scale_bounds = (1e-5, 1e5)
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'Bandit/{problem_instance.capitalize()}')
        config = wandb.config
        config.num_episodes = self.num_episodes
        
        config.alpha = self.alpha
        config.n_restarts = self.n_restarts
        config.epsilon = self.epsilon_start
        config.epsilon_decay = self.epsilon_decay
        
        config.memory_length = self.memory_length
        
        config.matern_nu = self.matern_nu
        config.matern_length_scale = self.matern_length_scale
        config.matern_length_scale_bounds = self.matern_length_scale_bounds
        
        config.constant_length_scale = self.constant_length_scale
        config.constant_length_scale_bounds = self.constant_length_scale_bounds
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
    
    def _init_gp(self, problem_instance):
        self.valid_lines = set()
        
        self.ptr = 0
        self.epsilon = self.epsilon_start
        self.coeffs = np.empty((0, self.action_dim))
        self.rewards = np.empty((0, 1))
        
        
        kernel = C(self.constant_length_scale, self.constant_length_scale_bounds) \
            * Matern(self.matern_length_scale, self.matern_length_scale_bounds, self.matern_nu)
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts, alpha=self.alpha)
        
        coeffs = self.rng.uniform(-0.1, 0.1, self.action_dim)
        mapped_coeffs = CDL.get_mapped_coeffs(coeffs)
        reward, _, _ = self._step(problem_instance, mapped_coeffs, num_lines=1, init_gp=True)
        self._learn(coeffs, reward, num_lines=1)
                
    # Overlay lines in the environment
    def _step(self, problem_instance, mapped_coeffs, num_lines, init_gp=False):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
        
        line = CDL.get_lines_from_coeffs(mapped_coeffs)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if init_gp or len(self.valid_lines) == prev_num_lines or num_lines == self.max_lines:
            done = True
            reward = super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
        
        return reward, done, regions
    
    # Choose coefficients using UCB approach and match them to possible coeffs
    def _select_coeffs(self):
        x0 = self.rng.uniform(-0.1, 0.1, self.action_dim)
        bounds = [(-0.1, 0.1)] * self.action_dim
        
        def objective(x):
            mean, std = self.gpr.predict([x], return_std=True)
            ucb = mean  + self.epsilon * std
            return -ucb[0] 
        
        res = minimize(objective, x0, bounds=bounds)
        
        mapped_coeffs = CDL.get_mapped_coeffs(res.x)
        return res.x, mapped_coeffs
    
    # Add coefficients and rewards to memory
    def _add_to_memory(self, coeffs, rewards):
        coeffs = np.array(coeffs).reshape(-1, self.action_dim)
        for coeff, reward in zip(coeffs, rewards):
            if self.ptr < self.memory_length:
                self.coeffs = np.vstack((self.coeffs, coeff))
                self.rewards = np.vstack((self.rewards, reward))
            else:
                self.coeffs[self.ptr % self.memory_length] = coeff
                self.rewards[self.ptr % self.memory_length] = reward
            self.ptr += 1
    
    # Learn from memory
    def _learn(self, coeffs, reward, num_lines):        
        distributed_reward = reward / num_lines
        rewards = np.full((num_lines, 1), distributed_reward)     
        self._add_to_memory(coeffs, rewards)
        
        self.gpr.fit(self.coeffs, self.rewards)  
    
    # Train model on a given problem_instance
    def _train(self, problem_instance):
        returns = []
        best_coeffs = None
        best_regions = None
        best_reward = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            coeffs_lst = []
            mapped_coeffs_lst = []
            while not done:
                num_lines += 1
                coeffs, mapped_coeffs = self._select_coeffs()
                reward, done, regions = self._step(problem_instance, mapped_coeffs, num_lines)
                coeffs_lst.append(coeffs)
                mapped_coeffs_lst.append(mapped_coeffs)
            
            self._learn(coeffs_lst, reward, num_lines)
            self.epsilon *= self.epsilon_decay
            
            returns.append(reward)
            avg_returns = np.mean(returns[-50:])
            wandb.log({'Average Returns': avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
            
            if reward > best_reward:
                best_coeffs = mapped_coeffs_lst
                best_regions = regions
                best_reward = reward
        
        return best_coeffs, best_regions, best_reward
                     
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb(problem_instance)
        
        start_time = time.time()
        
        self._init_gp(problem_instance)
        best_coeffs, best_regions, best_reward = self._train(problem_instance)
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        
        self._log_regions(problem_instance, 'Final', best_regions, best_reward)
        
        wandb.log({"Final Reward": best_reward})
        wandb.log({"Best Coeffs": best_coeffs})
        wandb.log({"Best Num Lines": len(best_coeffs)})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()  

        optim_coeffs = np.array(best_coeffs).reshape(-1, self.action_dim)
        return optim_coeffs