import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.cdl import CDL, SQUARE
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicTD3(CDL):
    def __init__(self, scenario, world):
        super(BasicTD3, self).__init__(scenario, world)
        self._init_hyperparams()
        
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.buffer = None
        
        self.action_dims = 3
        self.autoencoder = AE(self.state_dims, self.max_action, self.rng)
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.005
        self.gamma = 0.99
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.batch_size = 256
        self.policy_noise = 0.2
        self.actor_alpha = 0.0003
        self.critic_alpha = 0.0004
        self.memory_size = 50000
        self.num_episodes = 20000
        self.exploration_noise_start = 0.1
        self.exploration_noise_decay = 1
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.gamma = self.gamma 
        config.batch_size = self.batch_size
        config.noise_clip = self.noise_clip
        config.memory_size = self.memory_size
        config.policy_freq = self.policy_freq
        config.actor_alpha = self.actor_alpha
        config.critic_alpha = self.critic_alpha
        config.policy_noise = self.policy_noise
        config.num_episodes = self.num_episodes
        config.dummy_episodes = self.dummy_episodes
        config.exploration_noise = self.exploration_noise_start
        config.exploration_noise_decay = self.exploration_noise_decay  
          
    def _select_action(self, state, noise):
        with torch.no_grad():
            action = self.actor(torch.tensor(state))
            
            noise = self.rng.normal(0, self.exploration_noise, size=self.action_dim)
            action = (action.detach().numpy() + noise).clip(-1, 1)
                
        return action
    
    # TODO: Validate this function
    def _update(self, episode, state, actor_loss, critic_loss):
        self.critic.optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic.optim.step()
        
        if episode % self.policy_freq == 0:
            actor_loss = -self.critic.get_Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor.optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor.optim.step()
            
            actor_loss = actor_loss.item()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    # TODO: Complete this function
    def _learn(self):
        actor_loss, critic_loss = 0, 0
        if self.buffer.real_size < self.batch_size:
            return actor_loss, critic_loss, None
        
        indices = self.buffer.sample(self.batch_size)
        
        states = self.buffer.states[indices]
        actions = self.buffer.actions[indices]
        rewards = self.buffer.rewards[indices]
        next_states = self.buffer.next_states[indices]
        dones = self.buffer.dones[indices]
                
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        # TODO: Specifically fix this area
        current_Q1, current_Q2 = self.critic(states, actions)
        current_Q = torch.min(current_Q1, current_Q2)
        td_error = torch.abs(target_Q - current_Q).detach()
        
        weighted_loss_Q1 = (F.mse_loss(current_Q1, target_Q, reduction='none') * weights).mean()
        weighted_loss_Q2 = (F.mse_loss(current_Q2, target_Q, reduction='none') * weights).mean()
        critic_loss = weighted_loss_Q1 + weighted_loss_Q2
        
        return actor_loss, critic_loss.item()
    
    def _train(self, problem_instance):
        actor_losses = []
        critic_losses = []
        rewards = []
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            state, regions = self._generate_state()
            while not done:
                num_action += 1
                action = self._select_action(state, noise=True)
                reward, done, next_regions, next_state = self._step(problem_instance, regions, action, num_action)
                self.buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                regions = next_regions
                episode_reward += reward
                
            actor_loss, critic_loss, _ = self._learn()           
            actor_loss, critic_loss = self._update(episode, actor_loss, critic_loss)
            self.epsilon *= self.epsilon_decay

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            rewards.append(episode_reward)
            avg_actor_losses = np.mean(actor_losses[-self.sma_window:])
            avg_critic_losses = np.mean(critic_losses[-self.sma_window:])
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Actor Loss": avg_actor_losses})
            wandb.log({"Average Critic Loss": avg_critic_losses})
            wandb.log({"Average Reward": avg_rewards})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)
    
    def _get_final_lines(self, problem_instance):
        done = False
        language = []
        num_action = 0
        regions = [SQUARE]
        # TODO: Turn off exploration
        episode_reward = 0
        state = self.autoencoder.get_state(regions)
        while not done:
            num_action += 1
            action = self._select_action(state)
            reward, done, next_regions, next_state = self._step(problem_instance, regions, action, num_action)
            
            state = next_state
            regions = next_regions
            language += [action]
            episode_reward += reward
            
        language = np.array(language).reshape(-1,3)
        return language

    def _generate_language(self, problem_instance):
        self.exploration_noise = self.exploration_noise_start
        
        self.actor = Actor(self.state_dims, self.action_dims)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.state_dims, self.action_dims)
        self.critic_target = copy.deepcopy(self.critic)
        self.buffer = ReplayBuffer(self.state_dims, self.action_dims, self.memory_size)

        # self._init_wandb(problem_instance)
        self._train(problem_instance)
        language, reward = self._get_final_lines(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', language, reward)
        wandb.log({"Language": language})
        wandb.log({"Final Reward": reward})
        wandb.finish()  
        
        return language
    

class CommutativeTD3(BasicTD3):
    def __init__(self, scenario, world):
        super(CommutativeTD3, self).__init__(scenario, world)
        
    def _learn(self):
        pass
    
    def _train(self):
        pass
    
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}
        self.buffer = CommutativeReplayBuffer(self.state_dims, self.action_dims, self.memory_size, self.max_action)
        
        return super()._generate_language(problem_instance, self.buffer)