import os
import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.buffers import ReplayBuffer, RewardBuffer, CommutativeReplayBuffer, CommutativeRewardBuffer


class BasicDQN(CDL):
    def __init__(self, scenario, world, random_state, reward_prediction_type):
        super(BasicDQN, self).__init__(scenario, world, random_state)     
        self._init_hyperparams()         
        
        self.name = self.__class__.__name__
        self.output_dir = f'ma-cdl/languages/history/{self.name.lower()}'  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)     
        
        self.dqn = None
        self.target_dqn = None
        self.replay_buffer = None
        self.reward_estimator = None
        self.target_reward_estimator = None
        self.reward_prediction_type = reward_prediction_type
        
        self._create_candidate_set_of_lines()
        
        self.action_dims = len(self.candidate_lines)

    def _init_hyperparams(self):        
        # Reward Estimator
        self.dropout_rate = 0.40
        self.estimator_tau = 0.095
        self.estimator_alpha = 0.008
        
        # DQN
        self.tau = 0.0005
        self.alpha = 0.0002
        self.batch_size = 256
        self.sma_window = 500
        self.granularity = 0.20
        self.min_epsilon = 0.10
        self.epsilon_start = 1.0
        self.memory_size = 75000
        self.num_episodes = 20000
        self.epsilon_decay = 0.0005
        
        # Evaluation Settings (episodes)
        self.eval_freq = 40
        self.eval_window = 10
        self.eval_configs = 15
        self.eval_episodes = 10
        self.eval_obstacles = 10
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.eval_window = self.eval_window
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.eval_configs = self.eval_configs
        config.dropout_rate = self.dropout_rate
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.eval_episodes = self.eval_episodes
        config.estimator_tau = self.estimator_tau
        config.epsilon_decay = self.epsilon_decay
        config.eval_obstacles = self.eval_obstacles
        config.util_multiplier = self.util_multiplier
        config.estimator_alpha = self.estimator_alpha
        config.reward_estimator = self.reward_estimator
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        config.num_large_obstacles = len(self.world.large_obstacles)
        
    def _create_candidate_set_of_lines(self):
        self.candidate_lines = []
        granularity = int(self.granularity * 100)
        
        # Termination Line
        self.candidate_lines += [(0, 0, 0)]
        
        # Vertical/Horizontal lines
        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)] # vertical lines
            self.candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # Diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]
            self.candidate_lines += [(-0.1, 0.1, i)]
            
    def _decrement_epsilon(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state, is_train=True):
        if is_train and self.rng.random() < self.epsilon:
            action_index = self.rng.choice(len(self.candidate_lines))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state) if isinstance(state, list) else state.float()
                action_index = self.dqn(state).argmax().item()
                
        return action_index
                
    def _add_transition(self, state, action, reward, next_state, prev_state, prev_action, prev_reward):
        self.reward_buffer.add(state, action, reward, next_state)
        
        if prev_state is not None:   
            commutative_state = self._get_next_state(prev_state, action)
            self.commutative_reward_buffer.add(prev_state, action, prev_reward, commutative_state, prev_action, reward, next_state)

    def _learn(self, losses):
        if self.replay_buffer.real_size < self.batch_size:
            return None, losses

        indices = self.replay_buffer.sample(self.batch_size)
        
        state = self.replay_buffer.state[indices]
        action = self.replay_buffer.action[indices]
        reward = self.replay_buffer.reward[indices]
        next_state = self.replay_buffer.next_state[indices]
        done = self.replay_buffer.done[indices]
        
        if self.reward_prediction_type == 'approximate':
            with torch.no_grad():
                steps = torch.cat([state, action, next_state], dim=-1)
                reward = self.target_reward_estimator(steps).flatten()
        
        q_values = self.dqn(state)
        selected_q_values = torch.gather(q_values, 1, action).squeeze(-1)
        next_q_values = self.target_dqn(next_state)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values
        traditional_loss = F.mse_loss(selected_q_values, target_q_values)  
        
        self.dqn.optim.zero_grad(set_to_none=True)
        traditional_loss.backward()
        self.dqn.optim.step()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
        losses['traditional_loss'] += traditional_loss.item()
        
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
        rewards  = []
        
        best_regions = None
        best_language = None
        best_reward = -np.inf
        
        training_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        new_obstacles = self.eval_obstacles - len(self.world.large_obstacles)
        self.scenario.add_large_obstacles(self.world, new_obstacles)
        for _ in range(self.eval_episodes):
            done = False
            language = []
            episode_reward = 0
            regions, adaptations = self._generate_init_state()
            state = sorted(list(adaptations)) + (self.max_action - len(adaptations)) * [0]
            num_action = len(adaptations)
            
            while not done:
                num_action += 1
                action = self._select_action(state, is_train=False)
                line = self.candidate_lines[action]
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                
                state = next_state
                regions = next_regions
                episode_reward += reward
                language += [line]
                
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
        traditional_losses = []
        step_losses = []
        trace_losses = []
        
        best_reward = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            regions, adaptations = self._generate_init_state()
            state = sorted(list(adaptations)) + (self.max_action - len(adaptations)) * [0]
            num_action = len(adaptations)
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:                
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                                
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                if self.reward_prediction_type == 'approximate':
                    self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
                
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                regions = next_regions
                
            self._decrement_epsilon()
            
            if self.reward_prediction_type == 'approximate':
                losses = self._update_estimator(losses)

            _, losses = self._learn(losses)
            
            if episode % self.eval_freq == 0:
                eval_rewards, best_eval_reward, best_eval_language, best_eval_regions = self._eval_policy(problem_instance)
                rewards.append(eval_rewards)
                avg_rewards = np.mean(rewards[-self.eval_window:])
                wandb.log({'Average Reward': avg_rewards}, step=episode)
                
                if best_eval_reward > best_reward:
                    best_reward = best_eval_reward
                    best_language = best_eval_language
                    best_regions = best_eval_regions
            
            traditional_losses.append(losses['traditional_loss'])            
            step_losses.append(losses['step_loss'] / (num_action - len(adaptations)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(adaptations)))
            
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_step_losses = np.mean(step_losses[-self.sma_window:])
            avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                'Average Traditional Loss': avg_traditional_losses,
                'Average Step Loss': avg_step_losses,
                'Average Trace Loss': avg_trace_losses
                }, step=episode)

        best_language = np.array(best_language).reshape(-1,3)
        return best_reward, best_language, best_regions

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance, replay_buffer=None):
        self.epsilon = self.epsilon_start
        
        step_dims = 2*self.max_action + 1
        
        self.reward_buffer = RewardBuffer(self.batch_size, step_dims, self.rng)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, step_dims, self.rng)
        self.reward_estimator = RewardEstimator(step_dims, self.estimator_alpha, self.dropout_rate)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)
        
        self.reward_estimator.train()
        self.target_reward_estimator.eval()
        
        self.dqn = DQN(self.max_action, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(self.max_action, 1, self.memory_size, self.rng)
        
        self._init_wandb(problem_instance)
        
        best_reward, best_language, best_regions = self._train(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Language": best_language, "Final Reward": best_reward})
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, scenario, world, random_state, reward_prediction_type=None):
        super(CommutativeDQN, self).__init__(scenario, world, random_state, reward_prediction_type)
        
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
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self, losses):
        indices, losses = super()._learn(losses)
        
        if indices is None:
            return losses
                        
        r3_pred = None
        has_previous = self.replay_buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        s = self.replay_buffer.prev_state[indices][valid_indices]
        b = self.replay_buffer.action[indices][valid_indices]
        
        if self.reward_prediction_type == 'lookup':        
            key_tuples = [(tuple(_s.tolist()), _b.item()) for _s, _b in zip(s, b)]
            # Check whether key exists in ptr_lst
            valid_mask = torch.tensor([self.ptr_lst.get(key, (None, None))[0] is not None for key in key_tuples])
            
            if torch.any(valid_mask):    
                # Previous samples
                s = s[valid_mask]
                a = self.replay_buffer.prev_action[indices][valid_indices][valid_mask]
                r_0 = self.replay_buffer.prev_reward[indices][valid_indices][valid_mask]
                
                # Current samples
                b = b[valid_mask]
                r_1 = self.replay_buffer.reward[indices][valid_indices][valid_mask]
                s_prime = self.replay_buffer.next_state[indices][valid_indices][valid_mask]
                done = self.replay_buffer.done[indices][valid_indices][valid_mask]
                
                # Lookup table
                r_2 = np.array([self.ptr_lst[key][0] for i, key in enumerate(key_tuples) if valid_mask[i]])
                r_2 = torch.from_numpy(r_2).type(torch.float)
                s_2 = np.stack([self.ptr_lst[key][1] for i, key in enumerate(key_tuples) if valid_mask[i]])
                s_2 = torch.from_numpy(s_2).type(torch.float)
                
                r3_pred = r_0 + r_1 - r_2
        else:
            s_2 = self._get_next_state(s, b)
            a = self.replay_buffer.prev_action[indices][valid_indices]
            s_prime = self.replay_buffer.next_state[indices][valid_indices]
            
            r3_step = torch.cat([s_2, a, s_prime], dim=-1)
            r3_pred = self.target_reward_estimator(r3_step).flatten().detach()
            
            done = self.replay_buffer.done[indices][valid_indices]            
            
        if r3_pred is not None:
            q_values = self.dqn(s_2)
            selected_q_values = torch.gather(q_values, 1, a).squeeze(-1)
            next_q_values = self.target_dqn(s_prime)
            target_q_values = r3_pred + ~done * torch.max(next_q_values, dim=1).values
            commutative_loss = F.mse_loss(selected_q_values, target_q_values)
            
            self.dqn.optim.zero_grad(set_to_none=True)
            commutative_loss.backward()
            self.dqn.optim.step()
            
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)       
                     
            losses['commutative_loss'] += commutative_loss.item()
        
        return losses

    def _train(self, problem_instance):
        rewards = []
        traditional_losses = []
        commutative_losses = []
        step_losses = []
        trace_losses = []
        
        best_reward = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            language = []
            regions, adaptations = self._generate_init_state()
            state = sorted(list(adaptations)) + (self.max_action - len(adaptations)) * [0]
            num_action = len(adaptations)

            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, next_state, next_regions, done = self._step(problem_instance, state, regions, action, line, num_action)
                                
                self.replay_buffer.add(state, action, reward, next_state, done, prev_state, prev_action, prev_reward)
                
                if self.reward_prediction_type == 'lookup':
                    self.ptr_lst[(tuple(state), action)] = (reward, next_state)
                elif self.reward_prediction_type == 'approximate':
                    self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
                
                prev_state = state
                prev_action = action
                prev_reward = reward

                state = next_state
                regions = next_regions
                language += [line]
            
            self._decrement_epsilon()
            
            if self.reward_prediction_type == 'approximate':
                losses = self._update_estimator(losses)
                
            losses = self._learn(losses)
            
            if episode % self.eval_freq == 0:
                eval_rewards, best_eval_reward, best_eval_language, best_eval_regions = self._eval_policy(problem_instance)
                rewards.append(eval_rewards)
                avg_rewards = np.mean(rewards[-self.eval_window:])
                wandb.log({"Average Reward": avg_rewards}, step=episode) 
                
                if best_eval_reward > best_reward:
                    best_reward = best_eval_reward
                    best_language = best_eval_language
                    best_regions = best_eval_regions
                    
            traditional_losses.append(losses['traditional_loss'])
            commutative_losses.append(losses['commutative_loss'])
            step_losses.append(losses['step_loss'] / (num_action - len(adaptations)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(adaptations)))
            
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
            avg_step_losses = np.mean(step_losses[-self.sma_window:])
            avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                'Average Traditional Loss': avg_traditional_losses,
                'Average Commutative Loss': avg_commutative_losses,
                'Average Step Loss': avg_step_losses,
                'Average Trace Loss': avg_trace_losses,
                }, step=episode)
        
        best_language = np.array(best_language).reshape(-1,3)
        return best_reward, best_language, best_regions
                        
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}
        self.replay_buffer = CommutativeReplayBuffer(self.max_action, 1, self.memory_size, self.max_action, self.rng)

        return super()._generate_language(problem_instance, self.replay_buffer)