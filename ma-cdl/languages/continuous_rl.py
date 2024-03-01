import os
import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, Actor, Critic
from languages.utils.buffers import ReplayBuffer, CommutativeReplayBuffer, RewardBuffer, CommutativeRewardBuffer

torch.manual_seed(42)


class BasicTD3(CDL):
    def __init__(self, scenario, world, random_state, reward_prediction_type):
        super(BasicTD3, self).__init__(scenario, world, random_state)
        self._init_hyperparams()
        
        self.name = self.__class__.__name__
        self.output_dir = f'ma-cdl/languages/history/estimator/{self.name.lower()}'  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.replay_buffer = None
        self.reward_estimator = None
        self.target_reward_estimator = None
        self.reward_prediction_type = reward_prediction_type
        
        self.action_dims = 3
        self.state_dims = self.max_action * self.action_dims
        
    def _init_hyperparams(self):        
        # Reward Estimator
        self.dropout_rate = 0.4
        self.estimator_tau = 0.01
        self.estimator_alpha = 0.008
        self.model_save_interval = 2500
        
        # TD3
        self.tau = 0.005
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.batch_size = 256
        self.sma_window = 500
        self.policy_noise = 0.2
        self.memory_size = 100000
        self.actor_alpha = 0.0003
        self.critic_alpha = 0.0003
        self.max_timesteps = 250000
        self.start_timesteps = 25000
        self.start_timesteps = 1000
        self.exploration_noise = 0.1
        
        # Evaluation Settings (timesteps)
        self.eval_window = 10
        self.eval_freq = 5000
        self.eval_configs = 15
        self.eval_episodes = 5
        self.eval_obstacles = 10
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.noise_clip = self.noise_clip
        config.eval_window = self.eval_window
        config.memory_size = self.memory_size
        config.action_cost = self.action_cost
        config.policy_freq = self.policy_freq
        config.actor_alpha = self.actor_alpha
        config.critic_alpha = self.critic_alpha
        config.random_state = self.random_state
        config.dropout_rate = self.dropout_rate
        config.policy_noise = self.policy_noise
        config.eval_configs = self.eval_configs
        config.max_timesteps = self.max_timesteps
        config.estimator_tau = self.estimator_tau
        config.eval_episodes = self.eval_episodes
        config.eval_obstacles = self.eval_obstacles
        config.util_multiplier = self.util_multiplier
        config.start_timesteps = self.start_timesteps
        config.estimator_alpha = self.estimator_alpha
        config.reward_estimator = self.reward_estimator
        config.exploration_noise = self.exploration_noise
        config.model_save_interval = self.model_save_interval
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        config.num_large_obstacles = len(self.world.large_obstacles)
        
    def _is_terminating_action(self, action):
        threshold = 0.20
        return (abs(action) < threshold).all()
          
    def _select_action(self, state, timestep=None, is_noisy=True):
        if timestep and timestep < self.start_timesteps:
            action = self.rng.uniform(-1, 1, size=3)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state, dtype=torch.float32)
                action = self.actor(state)
                
                if is_noisy:
                    action += torch.normal(mean=0, std=self.exploration_noise, size=action.size()).clamp(-1, 1)
                    
        return action
    
    def _add_transition(self, state, action, reward, next_state, prev_state, prev_action, prev_reward):
        self.reward_buffer.add(state, action, reward, next_state)
        
        if prev_state is not None:   
            commutative_state = self._get_next_state(prev_state, action)
            self.commutative_reward_buffer.add(prev_state, action, prev_reward, commutative_state, prev_action, reward, next_state)
                        
    def _learn(self, timestep, losses):        
        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices].view(-1, 1)
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices].view(-1, 1)
        
        with torch.no_grad():
            if self.reward_prediction_type == 'approximate':
                steps = torch.cat([state, action, next_state], dim=-1)
                reward = self.target_reward_estimator(steps).detach()      
                    
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ~done * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        
        traditional_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic.optim.zero_grad(set_to_none=True)
        traditional_critic_loss.backward()
        self.critic.optim.step()
        
        losses['traditional_critic_loss'] += traditional_critic_loss.item()
        
        if timestep % self.policy_freq == 0:
            traditional_actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            losses['traditional_actor_loss'] += traditional_actor_loss.item()
            
            self.actor.optim.zero_grad(set_to_none=True)
            traditional_actor_loss.backward()
            self.actor.optim.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return indices, losses
    
    def _update_estimator(self, losses, traditional_update=True):
        if self.reward_buffer.real_size < self.batch_size:
            return losses
        
        indices = self.reward_buffer.sample(self.batch_size)
        steps = self.reward_buffer.transition[indices]
        rewards = self.reward_buffer.reward[indices].view(-1, 1)
        # Predict r_1 from actual (s_1,b,s')
        self.reward_estimator.optim.zero_grad(set_to_none=True)
        r_pred = self.reward_estimator(steps)
        step_loss = F.mse_loss(r_pred, rewards)
        step_loss.backward()
        self.reward_estimator.optim.step()
        
        if traditional_update:
            for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
                target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
                
        losses['step_loss'] += step_loss.item()
        
        return losses
    
    def _eval_policy(self, problem_instance):        
        rewards = []

        best_regions = None
        best_language = None
        best_reward = -np.inf

        empty_action = np.array([0.] * 3)
        
        training_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        new_obstacles = self.eval_obstacles - len(self.world.large_obstacles)
        self.scenario.add_large_obstacles(self.world, new_obstacles)
        for _ in range(self.eval_episodes):
            done = False
            language = []
            episode_reward = 0
            regions, adaptations = self._generate_fixed_state()
            state = np.concatenate(sorted(list(adaptations), key=np.sum) + (self.max_action - len(adaptations)) * [empty_action])
            num_action = len(adaptations)
            
            while not done:
                num_action += 1
                action = self._select_action(state, is_noisy=False)
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action=action, line=None, num_action=num_action)
                
                state = next_state
                regions = next_regions
                episode_reward += reward
                language += [action]
            
            rewards.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_language = language
                best_regions = regions
        
        self.configs_to_consider = training_configs
        self.world.large_obstacles = self.world.large_obstacles[:-new_obstacles]
        return np.mean(rewards), best_reward, best_language, best_regions
    
    def _train(self, problem_instance):
        rewards = []
        traditional_actor_loss = []
        traditional_critic_losses = []
        step_losses = []
        trace_losses = []
        
        episode_num = 0 
        best_reward = -np.inf
                
        empty_action = np.array([0.] * 3)
        regions, adaptations = self._generate_init_state()
        state = np.concatenate(sorted(list(adaptations), key=np.sum) + (self.max_action - len(adaptations)) * [empty_action])
        num_action = len(adaptations)
        language = []
        
        prev_state = None
        prev_action = None
        prev_reward = None
        losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        
        for timestep in range(self.max_timesteps):
            num_action += 1
            action = self._select_action(state, timestep)    
            reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action=action, line=None, num_action=num_action)
            
            if self.reward_prediction_type == 'approximate':
                self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
            
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            prev_state = state
            prev_action = action
            prev_reward = reward
            
            state = next_state
            regions = next_regions
            language += [action]
            
            if timestep >= self.start_timesteps:
                _, losses = self._learn(timestep, losses)
            
            if (timestep + 1) % self.eval_freq == 0:
                eval_rewards, best_eval_reward, best_eval_language, best_eval_regions = self._eval_policy(problem_instance)
                rewards.append(eval_rewards)
                avg_rewards = np.mean(rewards[-self.eval_window:])
                wandb.log({"Average Reward": avg_rewards}, step=episode_num) 
                
                if best_eval_reward > best_reward:
                    best_reward = best_eval_reward
                    best_language = best_eval_language
                    best_regions = best_eval_regions
            
            if done:
                if self.reward_prediction_type == 'approximate':
                    losses = self._update_estimator(losses)
            
                if timestep >= self.start_timesteps:
                    traditional_actor_loss.append(losses['traditional_actor_loss'])
                    traditional_critic_losses.append(losses['traditional_critic_loss'])
                    step_losses.append(losses['step_loss'] / (num_action - len(adaptations)))
                    trace_losses.append(losses['trace_loss'] / (num_action - len(adaptations)))
                    
                    avg_traditional_actor_losses = np.mean(traditional_actor_loss[-self.sma_window:])
                    avg_traditional_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
                    avg_step_losses = np.mean(step_losses[-self.sma_window:])
                    avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
                
                    wandb.log({
                        "Average Traditional Actor Loss": avg_traditional_actor_losses,
                        "Average Traditional Critic Loss": avg_traditional_critic_losses,
                        'Average Step Loss': avg_step_losses,
                        'Average Trace Loss': avg_trace_losses,
                        }, step=episode_num)
                
                # Reset environment
                regions, adaptations = self._generate_init_state()
                state = np.concatenate(sorted(list(adaptations), key=np.sum) + (self.max_action - len(adaptations)) * [empty_action])
                num_action = len(adaptations)   
                language = [] 
                
                losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
                           
                prev_state = None
                prev_action = None
                prev_reward = None   
                
                episode_num += 1
                
        best_language = np.array(best_language).reshape(-1,3)
        return best_reward, best_language, best_regions

    def _generate_language(self, problem_instance, replay_buffer=None):
        step_dims = self.max_action * self.action_dims * 2 + self.action_dims
        
        self.reward_buffer = RewardBuffer(self.batch_size, step_dims, self.rng)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, step_dims, self.rng)
        self.reward_estimator = RewardEstimator(step_dims, self.estimator_alpha, self.dropout_rate)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)
        
        self.reward_estimator.train()
        self.target_reward_estimator.eval()
                
        self.actor = Actor(self.state_dims, self.action_dims, self.actor_alpha)
        self.critic = Critic(self.state_dims, self.action_dims, self.critic_alpha)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(self.state_dims, self.action_dims, self.memory_size, self.rng)
            
        self._init_wandb(problem_instance)
        
        best_reward, best_language, best_regions = self._train(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Language": best_language, 'Final Reward': best_reward})
        wandb.finish()  
        
        return best_language
    

class CommutativeTD3(BasicTD3):
    def __init__(self, scenario, world, random_state, reward_prediction_type):
        super(CommutativeTD3, self).__init__(scenario, world, random_state, reward_prediction_type)
        
    def _update_estimator(self, losses):
        losses = super()._update_estimator(losses, traditional_update=False)
        
        if self.commutative_reward_buffer.real_size < self.batch_size:
            return losses
        
        self.reward_estimator.optim.zero_grad(set_to_none=True)
        commutative_indices = self.commutative_reward_buffer.sample(self.batch_size)
        commutative_steps = self.commutative_reward_buffer.transition[commutative_indices]
        commutative_rewards = self.commutative_reward_buffer.reward[commutative_indices]
        # Approximate r_2 from (s,b,s_2) and r_3 from (s_2,a,s')
        # MSE Loss: r_2 + r_3 = r_0 + r_1
        summed_r0r1 = torch.sum(commutative_rewards, axis=1).view(-1, 1)
        # Predict r_2 and r_3 from (s,b,s_2) and (s_2,a,s') respectively
        r2_pred = self.reward_estimator(commutative_steps[:, 0])
        r3_pred = self.reward_estimator(commutative_steps[:, 1])
        # Calculate loss with respect to r_2
        trace_loss_r2 = F.mse_loss(r2_pred + r3_pred.detach(), summed_r0r1)
        # Calculate loss with respect to r_3
        trace_loss_r3 = F.mse_loss(r2_pred.detach() + r3_pred, summed_r0r1)
        combined_loss = trace_loss_r2 + trace_loss_r3
        combined_loss.backward()
        
        self.reward_estimator.optim.step()
        
        for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
            
        losses['trace_loss'] += trace_loss_r2.item()
        
        return losses
        
    def _learn(self, episode, losses):
        indices, losses = super()._learn(episode, losses)
        
        if indices is None:
            return losses
        
        r3_pred = None
        has_previous = self.replay_buffer.has_previous[indices]
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        s = self.replay_buffer.prev_state[indices][valid_indices]
        b = self.replay_buffer.action[indices][valid_indices]
        
        s_2 = self._get_next_state(s, b)
        a = self.replay_buffer.prev_action[indices][valid_indices]
        s_prime = self.replay_buffer.next_state[indices][valid_indices]
        done = self.replay_buffer.done[indices][valid_indices].view(-1, 1)        
        
        with torch.no_grad():        
            r3_step = torch.cat([s_2, a, s_prime], dim=-1)
            r3_pred = self.target_reward_estimator(r3_step).detach()
            
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s_prime) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(s_prime, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r3_pred + ~done * target_Q
        
        current_Q1, current_Q2 = self.critic(s_2, a)
        commutative_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic.optim.zero_grad(set_to_none=True)
        commutative_critic_loss.backward()
        self.critic.optim.step()
        
        losses['commutative_critic_loss'] += commutative_critic_loss.item()
        
        if episode % self.policy_freq == 0:
            commutative_actor_loss = -self.critic.Q1(s_2, self.actor(s_2)).mean()
            
            losses['commutative_actor_loss'] += commutative_actor_loss.item()
            
            self.actor.optim.zero_grad(set_to_none=True)
            commutative_actor_loss.backward()
            self.actor.optim.step()
                        
        return losses
    
    def _train(self, problem_instance):
        rewards = []
        traditional_actor_loss = []
        traditional_critic_losses = []
        commutative_actor_losses = []
        commutative_critic_losses = []
        step_losses = []
        trace_losses = []
        
        episode_num = 0 
        best_reward = -np.inf
                
        empty_action = np.array([0.] * 3)
        regions, adaptations = self._generate_init_state()
        state = np.concatenate(sorted(list(adaptations), key=np.sum) + (self.max_action - len(adaptations)) * [empty_action])
        num_action = len(adaptations)
        language = []
        
        prev_state = None
        prev_action = None
        prev_reward = None
        losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'commutative_actor_loss': 0, 'commutative_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
        
        for timestep in range(self.max_timesteps):
            num_action += 1
            action = self._select_action(state, timestep)    
            reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action=action, line=None, num_action=num_action)

            if self.reward_prediction_type == 'approximate':
                self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
                
            self.replay_buffer.add(state, action, reward, next_state, done, prev_state, prev_action, prev_reward)
                
            prev_state = state
            prev_action = action
            prev_reward = reward
            
            state = next_state
            regions = next_regions
            language += [action]
            
            if timestep >= self.start_timesteps:
                losses = self._learn(timestep, losses)
            
            if (timestep + 1) % self.eval_freq == 0:
                eval_rewards, best_eval_reward, best_eval_language, best_eval_regions = self._eval_policy(problem_instance)
                rewards.append(eval_rewards)
                avg_rewards = np.mean(rewards[-self.eval_window:])
                wandb.log({"Average Reward": avg_rewards}, step=episode_num) 
                
                if best_eval_reward > best_reward:
                    best_reward = best_eval_reward
                    best_language = best_eval_language
                    best_regions = best_eval_regions

            if done:
                if self.reward_prediction_type == 'approximate':
                    losses = self._update_estimator(losses)
            
                if timestep >= self.start_timesteps:
                    traditional_actor_loss.append(losses['traditional_actor_loss'])
                    traditional_critic_losses.append(losses['traditional_critic_loss'])
                    commutative_actor_losses.append(losses['commutative_actor_loss'])
                    commutative_critic_losses.append(losses['commutative_critic_loss'])
                    step_losses.append(losses['step_loss'] / (num_action - len(adaptations)))
                    trace_losses.append(losses['trace_loss'] / (num_action - len(adaptations)))
                    
                    avg_traditional_actor_losses = np.mean(traditional_actor_loss[-self.sma_window:])
                    avg_traditional_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
                    avg_commutative_actor_losses = np.mean(commutative_actor_losses[-self.sma_window:])
                    avg_commutative_critic_losses = np.mean(commutative_critic_losses[-self.sma_window:])
                    avg_step_losses = np.mean(step_losses[-self.sma_window:])
                    avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
                
                    wandb.log({
                        "Average Traditional Actor Loss": avg_traditional_actor_losses,
                        "Average Traditional Critic Loss": avg_traditional_critic_losses,
                        'Average Commutative Actor Loss': avg_commutative_actor_losses,
                        'Average Commutative Critic Loss': avg_commutative_critic_losses,
                        'Average Step Loss': avg_step_losses,
                        'Average Trace Loss': avg_trace_losses,
                        }, step=episode_num)
                
                # Reset environment
                regions, adaptations = self._generate_init_state()
                state = np.concatenate(sorted(list(adaptations), key=np.sum) + (self.max_action - len(adaptations)) * [empty_action])
                num_action = len(adaptations)   
                language = [] 
                
                losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0, 'commutative_actor_loss': 0, 'commutative_critic_loss': 0, 'step_loss': 0, 'trace_loss': 0}
                           
                prev_state = None
                prev_action = None
                prev_reward = None   
                
                episode_num += 1
                
        best_language = np.array(best_language).reshape(-1,3)
        return best_reward, best_language, best_regions
    
    def _generate_language(self, problem_instance):        
        self.replay_buffer = CommutativeReplayBuffer(self.state_dims, self.action_dims, self.memory_size, self.max_action, self.rng)
        
        return super()._generate_language(problem_instance, self.replay_buffer)