import time
import wandb
import warnings
import numpy as np

from languages.utils.cdl import CDL
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

warnings.filterwarnings("ignore")

""" Infinitely Armed Bandit using GP-UCB """
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()
        
        self.action_dim = 3
        action_grid = np.arange(-1, 1 + self.resolution, self.resolution)
        self.action_space = np.array(np.meshgrid(action_grid, action_grid, action_grid)).T.reshape(-1,3)

    def _init_hyperparams(self):
        num_records = 10
        
        self.alpha = 2
        self.n_restarts = 50
        self.num_episodes = 600
        self.record_freq = self.num_episodes // num_records

        self.matern_nu = 0.75
        self.matern_length_scale = 1
        self.matern_length_scale_bounds = (1e-6, 1e6)
        
        self.constant_length_scale = 1
        self.constant_length_scale_bounds = (1e-6, 1e6)
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='Bandit')
        config = wandb.config
        config.weights = self.weights
        config.resolution = self.resolution
        config.configs_to_consider = self.configs_to_consider
        
        config.alpha = self.alpha
        config.n_restarts = self.n_restarts
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
        self.action = np.empty((0, self.action_dim))
        self.rewards = np.empty((0, 1))
        
        constant_kernel = C(self.constant_length_scale, self.constant_length_scale_bounds)
        matern = Matern(self.matern_length_scale, self.matern_length_scale_bounds, self.matern_nu)
        kernel = constant_kernel * matern
            
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts, alpha=self.alpha)
        
        action = self.rng.choice(self.action_space)
        reward, _, _ = self._step(problem_instance, action, num_lines=1, init_gpr=True)
        self._update_model(action, reward, num_lines=1)
                
    # Overlay lines in the environment
    def _step(self, problem_instance, action, num_lines, init_gpr=False):
        reward = 0
        done = False
        prev_num_valid = max(len(self.valid_lines), 4)
        
        line = CDL.get_lines_from_coeffs(action)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if init_gpr or len(self.valid_lines) == prev_num_valid or num_lines == self.max_lines:
            done = True
            reward = -super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
        
        return reward, done, regions
    
    # Choose coefficients using UCB approach
    def _select_action(self):
        means, stds = self.gpr.predict(self.action_space, return_std=True)
        ucbs = means + stds
        return self.action_space[np.argmax(ucbs)]
    
    # Learn from experience
    def _update_model(self, actions, reward, num_lines):                
        distributed_reward = reward / num_lines
        rewards = np.full((num_lines, 1), distributed_reward)     
           
        self.action = np.vstack((self.action, actions))
        self.rewards = np.vstack((self.rewards, rewards))
        self.gpr.fit(self.action, self.rewards)    
    
    # Train model on a given problem_instance
    def _train(self, problem_instance):
        returns = []
        
        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            action_lst = []
            while not done:
                num_lines += 1
                action = self._select_action()
                reward, done, regions = self._step(problem_instance, action, num_lines)
                action_lst.append(action)
                
            self._update_model(action_lst, reward, num_lines)
            self._decrement_epsilon()
            
            returns.append(reward)
            avg_returns = np.mean(returns[-50:])
            wandb.log({'Average Returns': avg_returns})
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
            action = self._select_action()
            optim_coeffs.append(action)
            reward, done, regions = self._step(problem_instance, action, num_lines)
            num_lines += 1
        
        self._log_regions(problem_instance, 'Final', regions, reward)
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        
        wandb.log({"Final Reward": reward})
        wandb.log({"Best Coeffs": optim_coeffs})
        wandb.log({"Best Num Lines": len(optim_coeffs)})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()
        
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        return optim_coeffs