import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicTD3(CDL):
    def __init__(self, scenario, world, random_state):
        super(BasicTD3, self).__init__(scenario, world, random_state)
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
        
        self.tau = 0.001
        self.policy_freq = 2
        self.noise_clip = 0.25
        self.batch_size = 16
        self.sma_window = 100
        self.lr_actor = 0.0004
        self.lr_critic = 0.0004
        self.policy_noise = 0.1
        self.memory_size = 25000
        self.num_episodes = 5000
        self.estimator_tau = 0.01
        self.estimator_lr = 0.001
        self.exploration_noise = 0.05
        self.exploration_decay = 0.999
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.lr_actor = self.lr_actor
        config.lr_critic = self.lr_critic
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.noise_clip = self.noise_clip
        config.memory_size = self.memory_size
        config.action_cost = self.action_cost
        config.policy_freq = self.policy_freq
        config.policy_noise = self.policy_noise
        config.num_episodes = self.num_episodes
        config.estimator_lr = self.estimator_lr
        config.estimator_tau = self.estimator_tau
        config.exploration_noise = self.exploration_noise
        config.exploration_decay = self.exploration_decay
        config.training_configs = self.training_configs
          
    def _select_action(self, state, is_noisy=True):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state))
            
            if is_noisy:
                noise = self.rng.normal(0, self.exploration_noise, size=self.action_dims).astype(np.float32)
                action = (action.detach().numpy() + noise).clip(-1, 1)
                
        return action
    
    def _is_terminating_action(self, action):
        threshold = 0.02
        return (abs(action) < threshold).all()
    
    def _update(self, actor_loss, critic_loss, traditional_update=True, update_target=True):
        if not isinstance(actor_loss, int):
            self.actor.optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor.optim.step()
            
            if update_target:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        if not isinstance(critic_loss, int):
            self.critic.optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic.optim.step()
                        
    def _learn(self, episode, stored_losses):
        actor_loss, critic_loss = 0, 0
        if self.buffer.real_size < self.batch_size:
            return actor_loss, critic_loss, None, stored_losses
        
        indices = self.buffer.sample(self.batch_size)
        
        states = self.buffer.state[indices]
        actions = self.buffer.action[indices]
        rewards = self.buffer.reward[indices]
        next_states = self.buffer.next_state[indices]
        dones = self.buffer.done[indices]
                
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        
        with torch.no_grad():
            noise = torch.normal(mean=0, std=self.policy_noise, size=actions.size()).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + ~dones * target_Q
        
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        stored_losses['traditional_critic_loss'] += critic_loss.item()
        
        if episode % self.policy_freq == 0:
            actor_loss = -self.critic.get_Q1(states, self.actor(states)).mean()
            stored_losses['traditional_actor_loss'] += actor_loss.item()
        
        return actor_loss, critic_loss, indices, stored_losses
    
    def _train(self, problem_instance):
        rewards = []
        traditional_actor_loss = []
        traditional_critic_losses = []
        
        best_regions = None
        best_language = None
        best_rewards = -np.inf
        
        empty_action = np.array([0.] * 3)
        for episode in range(self.num_episodes):
            done = False
            language = []
            episode_reward = 0
            regions, lines = self._generate_init_state()
            num_action = len(lines)
            _state = sorted(list(lines))
            stored_losses = {'traditional_actor_loss': 0, 'traditional_critic_loss': 0}
            while not done:
                num_action += 1
                
                state = np.concatenate(_state + (self.max_action - len(_state)) * [empty_action])
                action = self._select_action(state)
                reward, done, next_regions, _ = self._step(problem_instance, regions, action, num_action)
                
                _state = sorted(_state + [action], key=np.sum)
                next_state = np.concatenate(_state + (self.max_action - len(_state)) * [empty_action])
                
                self.buffer.add(state, action, reward, next_state, done)
                
                regions = next_regions
                language += [action]
                episode_reward += reward
            
            actor_loss, critic_loss, _, stored_losses = self._learn(episode, stored_losses)           
            self._update(actor_loss, critic_loss)

            rewards.append(episode_reward)
            traditional_actor_loss.append(actor_loss)
            traditional_critic_losses.append(critic_loss)
            
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_traditional_actor_losses = np.mean(traditional_actor_loss[-self.sma_window:])
            avg_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
            
            wandb.log({
                "Average Reward": avg_rewards,
                "Average Actor Loss": avg_traditional_actor_losses,
                "Average Critic Loss": avg_critic_losses
                })
            
            if episode_reward > best_rewards:
                best_rewards = episode_reward
                best_language = language
                best_regions = regions
            
        best_language = np.array(best_language).reshape(-1,3)
        return best_language, best_regions, best_rewards

    def _generate_language(self, problem_instance, actor=None, critic=None, buffer=None):
        state_dims = self.max_action * self.action_dims
        
        if actor is None and critic is None and buffer is None:
            self.actor = Actor(state_dims, self.action_dims, self.lr_actor)
            self.critic = Critic(self.state_dims, self.action_dims, self.lr_critic)
            self.buffer = ReplayBuffer(state_dims, self.action_dims, self.memory_size, self.rng)
            
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # self._init_wandb(problem_instance)
        best_language, best_regions, best_rewards = self._train(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_rewards)
        wandb.log({"Language": best_language})
        wandb.log({"Final Reward": best_rewards})
        wandb.finish()  
        
        return best_language
    

class CommutativeTD3(BasicTD3):
    def __init__(self, scenario, world, random_state):
        super(CommutativeTD3, self).__init__(scenario, world, random_state)
        
    def _truncate(self, action, digits=3):
        truncated_action = np.empty(action.shape, dtype=np.float32)
        format_string = f".{digits}f"
        for i, num in enumerate(action):
            truncated_action[i] = float(format(num, format_string))
        return truncated_action
        
    def _learn(self, episode, stored_losses):
        traditional_actor_loss, traditional_critic_loss, indices, stored_losses = super()._learn(episode, stored_losses)
        
        if indices is None:
            return traditional_actor_loss, traditional_critic_loss, stored_losses
        
        self._update(traditional_actor_loss, traditional_critic_loss)
        
        # Commutative Update
        r3_pred = None
        has_previous = self.buffer.has_previous[indices]
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        s = self.buffer.prev_state_proxy[indices][valid_indices]
        b = self.buffer.action[indices][valid_indices]
        
        if self.reward_prediction_type == 'lookup':
            key_tuples = [(tuple(_s.tolist()), _b.item()) for _s, _b in zip(s, b)]
            # Check whether key exists in ptr_lst
            valid_mask = torch.tensor([self.ptr_lst.get(key, (None, None))[0] is not None for key in key_tuples])
        
            if torch.any(valid_mask):
                # Previous samples
                s = s[valid_mask]
                a = self.buffer.prev_action[indices][valid_indices][valid_mask]
                r_0 = self.buffer.prev_reward[indices][valid_indices][valid_mask]
                
                # Current samples
                b = b[valid_mask]
                r_1 = self.buffer.reward[indices][valid_indices][valid_mask]
                s_prime = self.buffer.next_state[indices][valid_indices][valid_mask]
                done = self.buffer.done[indices][valid_indices][valid_mask]
                
                r_2 = np.array([self.ptr_lst[key][0] for i, key in enumerate(key_tuples) if valid_mask[i]])
                r_2 = torch.from_numpy(r_2).type(torch.float)
                s_2 = np.stack([self.ptr_lst[key][1] for i, key in enumerate(key_tuples) if valid_mask[i]])
                s_2 = torch.from_numpy(s_2).type(torch.float)
                
                r3_pred = r_0 + r_2 - r_1
        else:
            a = self.buffer.prev_action[indices][valid_indices]
            r_0 = self.buffer.prev_reward[indices][valid_indices]
            
            s_1 = self.buffer.state[indices][valid_indices]
            r_1 = self.buffer.reward[indices][valid_indices]
            s_prime = self.buffer.next_state[indices][valid_indices]
            done = self.buffer.done[indices][valid_indices]
            
            tmp_tensor = s.clone()
            tmp_tensor[tmp_tensor == 0] = self.action_dims + 1
            tmp_tensor = torch.cat([tmp_tensor, b], dim=-1)
            tmp_tensor = torch.sort(tmp_tensor, dim=-1).values
            tmp_tensor[tmp_tensor == self.action_dims + 1] = 0
            s_2 = tmp_tensor[:, :-1]
            
            traces = [[s, a, s_1], [s_1, b, s_prime], [s, b, s_2], [s_2, a, s_prime]]
            
            r3_step, step_loss, trace_loss = self._update_estimator(traces, r_0, r_1)
            r3_pred = self.reward_estimator(r3_step).flatten().detach()
            
            stored_losses['step_loss'] += step_loss
            stored_losses['trace_loss'] += trace_loss
        
        # TODO: Implement update
        if r3_pred is not None:
            a=3
        
        return stored_losses
    
    def _train(self, problem_instance):
        rewards = []
        traditional_actor_loss = []
        traditional_critic_losses = []
        commutative_actor_losses = []
        commutative_critic_losses = []
        step_losses = []
        trace_losses = []
        
        empty_action = np.array([0.] * 3)
        for episode in range(self.num_episodes):
            done = False
            language = []
            num_action = 0
            episode_reward = 0
            
            prev_state = []
            _prev_state = []
            prev_action = None
            prev_reward = None
            regions, lines = self._generate_init_state()
            num_action = len(lines)
            _state = sorted(list(lines))
            
            stored_losses = {
                'traditional_actor_loss': 0, 
                'traditional_critic_loss': 0, 
                'commutative_actor_loss': 0, 
                'commutative_critic_loss': 0,
                'step_loss': 0,
                'trace_loss': 0
                }
            
            while not done:
                num_action += 1
                
                state = np.concatenate(_state + (self.max_action - len(_state)) * [empty_action])
                action = self._select_action(state)
                reward, done, next_regions, _ = self._step(problem_instance, regions, action, num_action)
                
                tmp_state = sorted(_state + [action], key=np.sum)
                next_state = np.concatenate(tmp_state + (self.max_action - len(tmp_state)) * [empty_action])
                                
                prev_state = np.concatenate(_prev_state + (self.max_action - len(_prev_state)) * [empty_action])
                self.buffer.add(state, action, reward, next_state, done, prev_state, prev_action, prev_reward)
                self.ptr_lst[(tuple(state), tuple(action))] = (reward, next_state)
                
                _prev_state = _state.copy()
                prev_action = action.copy()
                prev_reward = reward
                                
                _state = tmp_state.copy()
                
                regions = next_regions
                language += [action]
                episode_reward += reward
                
            stored_losses = self._learn(episode, stored_losses)           

            rewards.append(episode_reward)
            step_losses.append(stored_losses['step_loss'] / (num_action - len(lines)))
            trace_losses.append(stored_losses['trace_loss'] / (num_action - len(lines)))
            traditional_actor_loss.append(stored_losses['traditional_actor_loss'])
            traditional_critic_losses.append(stored_losses['traditional_critic_loss'])
            commutative_actor_losses.append(stored_losses['commutative_actor_loss'])
            commutative_critic_losses.append(stored_losses['commutative_critic_loss'])
            
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_step_losses = np.mean(step_losses[-self.sma_window:])
            avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
            avg_traditional_actor_losses = np.mean(traditional_actor_loss[-self.sma_window:])
            avg_traditional_critic_losses = np.mean(traditional_critic_losses[-self.sma_window:])
            avg_commutative_actor_losses = np.mean(commutative_actor_losses[-self.sma_window:])
            avg_commutative_critic_losses = np.mean(commutative_critic_losses[-self.sma_window:])
            
            wandb.log({
                "Average Reward": avg_rewards,
                'Average Step Loss': avg_step_losses,
                'Average Trace Loss': avg_trace_losses,
                "Average Traditional Actor Loss": avg_traditional_actor_losses,
                "Average Traditional Critic Loss": avg_traditional_critic_losses,
                "Average Commutative Actor Loss": avg_commutative_actor_losses,
                "Average Commutative Critic Loss": avg_commutative_critic_losses,
                })
    
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}        
        
        self.reward_estimator = RewardEstimator(self.max_action*2 + 1, 1, self.estimator_lr)

        state_dims = self.max_action * self.action_dims
        self.actor = Actor(state_dims, self.action_dims, self.lr_actor)
        self.critic = Critic(state_dims, self.action_dims, self.lr_critic)
        self.buffer = CommutativeReplayBuffer(state_dims, self.action_dims, self.memory_size, self.max_action, self.rng)
        
        return super()._generate_language(problem_instance, self.actor, self.critic, self.buffer)