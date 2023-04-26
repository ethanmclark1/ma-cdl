"""
This code is based on the following repository:

Author: Donal Byrne
Repository: TD3
URL: https://github.com/djbyrne/TD3
Version: 21d162f
License: MIT License
"""

import os
import io
import copy
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.optim import Adam
from torch.nn.functional import mse_loss
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer
from sklearn.preprocessing import OneHotEncoder
from environment.utils.problems import problem_scenarios

wandb.init(project='td3', entity='ethanmclark1')

config = wandb.config
config.tau = TAU = 0.005
config.batch_size = BATCH_SIZE = 64
config.num_dummy = NUM_DUMMY = 250000
config.num_episodes = NUM_EPISODES = 75000
config.penalty_thres = PENALTY_THRES = -6
config.num_iterations = NUM_ITERATIONS = 100
config.policy_update_freq = POLICY_UPDATE_FREQ = 2
config.weights = WEIGHTS = np.array([1, 1, 1.25, 1, 1])
config.replay_buffer_size = REPLAY_BUFFER_SIZE = 1000000

""" Twin Delayed Deep Deterministic Policy Gradient (TD3) """
class TD3(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self.encoder = OneHotEncoder()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer(size=REPLAY_BUFFER_SIZE)
        scenarios = np.array(list(problem_scenarios.keys())).reshape(-1, 1)
        self.encoded_scenarios = self.encoder.fit_transform(scenarios).toarray()
        
        self.max_action = 1
        self.action_dim = 18
        state_dim = self.encoded_scenarios.shape[1] + len(self.square.exterior.coords) * len(self.square.exterior.coords[0])
        self.actor = Actor(state_dim, self.action_dim, self.max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters())   
        
    def _save(self):
        directory = 'ma-cdl/languages/history/saved_models'
        filepaths = [f'{directory}/actor.pth', f'{directory}/critic.pth']
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), filepaths[0])
        torch.save(self.critic.state_dict(), filepaths[1])

    def _load(self):
        directory = 'ma-cdl/languages/history/saved_models'
        self.actor.load_state_dict(torch.load(f'{directory}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/critic.pth')) 
    
    # Upload regions to Weights and Biases
    def _log_regions(self, scenario, episode, regions, penalty):
        _, ax = plt.subplots()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Scenario: {scenario} \nEpisode: {episode}, \nPenalty: {penalty}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)

        wandb.log({"image": wandb.Image(pil_image)})

    # Get state from the environment
    def _get_state(self, scenario=None):
        start_regions = np.array(self.square.exterior.coords).reshape(1, -1)

        if scenario is None:
            encoded_scenario = self.rng.choice(self.encoded_scenarios, size=1)
            scenario = self.encoder.inverse_transform(encoded_scenario.reshape(1, -1)).item()
        else:
            encoded_scenario = self.encoder.transform([[scenario]]).toarray()
                    
        state = torch.FloatTensor(np.concatenate((encoded_scenario, start_regions), axis=-1))
        return state, scenario, encoded_scenario
    
    def _get_feedback(self, coeffs, scenario):
        criterion, regions = super()._optimizer(coeffs, scenario)
        penalty = -np.sum(criterion * WEIGHTS)
        return penalty, regions
            
    # Populate the replay buffer with dummy transitions
    def _populate_buffer(self):
        while len(self.replay_buffer) < NUM_DUMMY:
            state, scenario, _ = self._get_state()
            coeffs = self.rng.uniform(-self.max_action, self.max_action, size=self.action_dim)
            penalty, _ = self._get_feedback(coeffs, scenario)
            
            self.replay_buffer.add(state, coeffs, penalty)
        
    # Select an action (ceefficients of a linear line)
    def _select_coeffs(self, state, noise=0.1):
        coeffs = self.actor(state).data.numpy().flatten()
        if noise != 0:
            coeffs = (coeffs + np.random.normal(0, noise, size=self.action_dim))
            
        return coeffs
            
    # Learn from the replay buffer
    def _learn(self):
        for it in range(NUM_ITERATIONS):
            state, action, penalty = self.replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            penalty = torch.FloatTensor(penalty)
            
            # Next state is terminal, therefore target_Q is the penalty
            target_Q = penalty.view(len(penalty), 1, 1)
            
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = mse_loss(current_Q1, target_Q) + mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if it % POLICY_UPDATE_FREQ == 0:
                actor_loss = -self.critic.get_Q1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    
    def _train_model(self):
        penalties = []
        best_avg = -200
        
        self._populate_buffer()
        for episode in range(NUM_EPISODES):
            state, scenario, _ = self._get_state()
            coeffs = self._select_coeffs(state)
            penalty, regions = self._get_feedback(coeffs, scenario)
            self.replay_buffer.add(state, coeffs, penalty)
            
            penalties.append(penalty)
            avg_penalty = np.mean(penalties[-100:])
            
            if best_avg < avg_penalty:
                best_avg = avg_penalty
                print("Saving best model....\n")
                self._save()
            
            print(f'Episode: {episode}\nPenalty: {penalty}\nAverage Penalty: {avg_penalty}\n', end="")
            
            wandb.log({"penalty": penalty, "avg_penalty": avg_penalty})
            if episode % 50 == 0:
                self._log_regions(scenario, episode, regions, penalty)
            
            if avg_penalty >= PENALTY_THRES:
                break
                
            self._learn()
    
    # Set the model to generate the language
    def setup(self):
        try:
            self._load()
        except:
            print('No existing model found.')
            print('Training new TD3 model...')
            self._train_model()
            self._save()
            
    def get_language(self, scenario):
        state, _, _ = self._get_state(scenario)
        coeffs = self._select_coeffs(state, noise=0)
        lines = self._get_lines_from_coeffs(coeffs)
        regions = self._create_regions(lines)
        return regions