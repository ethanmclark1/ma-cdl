import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicDQN(CDL):
    def __init__(self, scenario, world, random_state, reward_prediction_type=None):
        super(BasicDQN, self).__init__(scenario, world, random_state)     
        self._init_hyperparams()                
        
        self.dqn = None
        self.target_dqn = None
        self.buffer = None
        self.reward_prediction_type = reward_prediction_type
        
        self._create_candidate_set_of_lines()
        
        self.action_dims = len(self.candidate_lines)

    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.0005
        self.batch_size = 128
        self.sma_window = 1000
        self.granularity = 0.20
        self.min_epsilon = 0.10
        self.epsilon_start = 1.0
        self.memory_size = 100000
        self.num_episodes = 15000
        self.estimator_tau = 0.01
        self.estimator_lr = 0.001
        self.traditional_lr = 0.0003
        self.commutative_lr = 0.0003
        self.epsilon_decay = 0.0005 if self.random_state else 0.000175
        
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.granularity = self.granularity 
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.estimator_lr = self.estimator_lr
        config.estimator_tau = self.estimator_tau
        config.epsilon_decay = self.epsilon_decay
        config.traditional_lr = self.traditional_lr
        config.commutative_lr = self.commutative_lr
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        
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
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action_index = self.rng.choice(len(self.candidate_lines))
        else:
            with torch.no_grad():
                action_index = self.dqn(torch.FloatTensor(state)).argmax().item()
                
        return action_index
    
    def _update(self, loss, traditional_update=True, update_target=True):
        if not isinstance(loss, int):    
            if traditional_update:
                self.dqn.traditional_optim.zero_grad()
                loss.backward()
                self.dqn.traditional_optim.step()
            else:
                self.dqn.commutative_optim.zero_grad()
                loss.backward()
                self.dqn.commutative_optim.step()

        if update_target:
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def _learn(self, stored_losses):
        if self.buffer.real_size < self.batch_size:
            return 0, None, stored_losses

        indices = self.buffer.sample(self.batch_size)
        
        state = self.buffer.state[indices]
        action = self.buffer.action[indices]
        reward = self.buffer.reward[indices]
        next_state = self.buffer.next_state[indices]
        done = self.buffer.done[indices]
        
        if self.reward_prediction_type == 'approximate':
            steps = torch.cat([state, action, next_state], dim=-1)
            reward = self.target_reward_estimator(steps).detach().flatten()
        
        q_values = self.dqn(state)
        selected_q_values = torch.gather(q_values, 1, action).squeeze(-1)
        next_q_values = self.target_dqn(next_state)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values
        loss = F.mse_loss(selected_q_values, target_q_values)  
        stored_losses['traditional_loss'] += loss.item()   
        
        return loss, indices, stored_losses
    
    def _train(self, problem_instance):
        rewards = []        
        traditional_losses = []
        
        best_regions = None
        best_language = None
        best_rewards = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            language = []
            episode_reward = 0
            regions, actions = self._generate_init_state()
            num_action = len(actions)
            _state = sorted(list(actions))
            stored_losses = {'traditional_loss': 0}
            while not done:
                num_action += 1
                
                state = _state + (self.max_action - len(_state)) * [0]
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions = self._step(problem_instance, regions, line, num_action)
                
                _state = sorted(_state + [action])
                next_state = _state + (self.max_action - len(_state)) * [0]
                
                self.buffer.add(state, action, reward, next_state, done)

                regions = next_regions
                language += [line]
                episode_reward += reward
                
            self._decrement_epsilon()
            
            loss, _, stored_losses = self._learn(stored_losses)
            self._update(loss)

            rewards.append(episode_reward)
            traditional_losses.append(stored_losses['traditional_loss'])
            
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            
            wandb.log({
                'Average Reward': avg_rewards,
                'Average Traditional Loss': avg_traditional_losses,
                })
            
            if episode_reward > best_rewards:
                best_rewards = episode_reward
                best_language = language
                best_regions = regions
            
        best_language = np.array(best_language).reshape(-1,3)
        return best_language, best_regions, best_rewards

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance, dqn=None, buffer=None):
        self.epsilon = self.epsilon_start
        
        if dqn is None and buffer is None:
            self.dqn = DQN(self.max_action, self.action_dims, self.traditional_lr)
            self.buffer = ReplayBuffer(self.max_action, 1, self.memory_size, self.rng)
            
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self._init_wandb(problem_instance)
        best_language, best_regions, best_reward = self._train(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Language": best_language, "Final Reward": best_reward})
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, scenario, world, random_state, reward_prediction_type=None):
        super(CommutativeDQN, self).__init__(scenario, world, random_state, reward_prediction_type)

    def _update_estimator(self, traces, r_0, r_1):
        traces = torch.stack([torch.cat(rows, dim=1) for rows in traces])
        
        r0r1_pred = self.reward_estimator(traces[:2])
        self.reward_estimator.optim.zero_grad()
        stacked_r0r1 = torch.cat([r_0, r_1], dim=-1).view(2, -1, 1)
        step_loss = F.mse_loss(stacked_r0r1, r0r1_pred)
        step_loss.backward(retain_graph=True)
        self.reward_estimator.optim.step()
        
        r2r3_pred = self.reward_estimator(traces[2:])
        self.reward_estimator.optim.zero_grad()
        summed_r0r1 = (r_0 + r_1).view(-1, 1)
        trace_loss_r2 = F.mse_loss(summed_r0r1, r2r3_pred[0] + r2r3_pred[1].detach())
        trace_loss_r3 = F.mse_loss(summed_r0r1, r2r3_pred[0].detach() + r2r3_pred[1])
        combined_loss = trace_loss_r2 + trace_loss_r3
        combined_loss.backward()
        self.reward_estimator.optim.step()
        
        for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
        
        return traces[3], step_loss.item(), trace_loss_r2.item()
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self, stored_losses):
        traditional_loss, indices, stored_losses = super()._learn(stored_losses)
        
        if indices is None:
            return stored_losses
        
        self._update(traditional_loss)
                
        # Commutative Q-Update
        r3_pred = None
        has_previous = self.buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        s = self.buffer.prev_state[indices][valid_indices]
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
                
                # Lookup table
                r_2 = np.array([self.ptr_lst[key][0] for i, key in enumerate(key_tuples) if valid_mask[i]])
                r_2 = torch.from_numpy(r_2).type(torch.float)
                s_2 = np.stack([self.ptr_lst[key][1] for i, key in enumerate(key_tuples) if valid_mask[i]])
                s_2 = torch.from_numpy(s_2).type(torch.float)
                
                r3_pred = r_0 + r_1 - r_2
        else:
            a = self.buffer.prev_action[indices][valid_indices]
            r_0 = self.buffer.prev_reward[indices][valid_indices]
                        
            s_1 = self.buffer.state[indices][valid_indices]
            r_1 = self.buffer.reward[indices][valid_indices]
            s_prime = self.buffer.next_state[indices][valid_indices]
            done = self.buffer.done[indices][valid_indices]
            
            # Concatenate states together
            tmp_tensor = s.clone()
            tmp_tensor[tmp_tensor == 0] = self.action_dims + 1
            tmp_tensor = torch.cat([tmp_tensor, b], dim=-1)
            tmp_tensor = torch.sort(tmp_tensor, dim=-1).values
            tmp_tensor[tmp_tensor == self.action_dims + 1] = 0
            s_2 = tmp_tensor[:, :-1]
            
            traces = [[s, a, s_1], [s_1, b, s_prime], [s, b, s_2], [s_2, a, s_prime]]
            
            r3_step, step_loss, trace_loss = self._update_estimator(traces, r_0, r_1)
            r3_pred = self.target_reward_estimator(r3_step).flatten().detach()
            
            stored_losses['step_loss'] += step_loss
            stored_losses['trace_loss'] += trace_loss
                        
        if r3_pred is not None:
            q_values = self.dqn(s_2)
            selected_q_values = torch.gather(q_values, 1, a).squeeze(-1)
            next_q_values = self.target_dqn(s_prime)
            target_q_values = r3_pred + ~done * torch.max(next_q_values, dim=1).values
            commutative_loss = F.mse_loss(selected_q_values, target_q_values)
            self._update(commutative_loss, traditional_update=False)
            
            stored_losses['commutative_loss'] += commutative_loss.item()
        
        return stored_losses

    def _train(self, problem_instance):
        rewards = []
        step_losses = []
        trace_losses = []
        traditional_losses = []
        commutative_losses = []
        
        best_regions = None
        best_language = None
        best_rewards = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            language = []
            episode_reward = 0

            prev_action = None
            prev_reward = None
            _prev_state = []
            regions, adaptations = self._generate_init_state()
            num_action = len(adaptations)
            _state = sorted(list(adaptations))
            stored_losses = {'traditional_loss': 0, 'commutative_loss': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:
                num_action += 1
                prev_state = _prev_state + (self.max_action - len(_prev_state)) * [0]
                state = _state + (self.max_action - len(_state)) * [0]
                
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions = self._step(problem_instance, regions, line, num_action)
                
                _prev_state = _state
                _state = sorted(_state + [action])
                next_state = _state + (self.max_action - len(_state)) * [0]
                
                self.buffer.add(state, action, reward, next_state, done, prev_state, prev_action, prev_reward)
                self.ptr_lst[(tuple(state), action)] = (reward, next_state)
                
                prev_action = action
                prev_reward = reward

                language += [line]
                regions = next_regions
                episode_reward += reward
            
            self._decrement_epsilon()
            
            stored_losses = self._learn(stored_losses)
            
            rewards.append(episode_reward)
            step_losses.append(stored_losses['step_loss'] / (num_action - len(adaptations)))
            trace_losses.append(stored_losses['trace_loss'] / (num_action - len(adaptations)))
            traditional_losses.append(stored_losses['traditional_loss'])
            commutative_losses.append(stored_losses['commutative_loss'])
            
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_step_losses = np.mean(step_losses[-self.sma_window:])
            avg_trace_losses = np.mean(trace_losses[-self.sma_window:])
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
            
            wandb.log({
                'Average Reward': avg_rewards,
                'Average Step Loss': avg_step_losses,
                'Average Trace Loss': avg_trace_losses,
                'Average Traditional Loss': avg_traditional_losses,
                'Average Commutative Loss': avg_commutative_losses
                })
            
            if episode_reward > best_rewards:
                best_rewards = episode_reward
                best_language = language
                best_regions = regions
            
        best_language = np.array(best_language).reshape(-1,3)
        return best_language, best_regions, best_rewards
            
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}
        
        self.reward_estimator = RewardEstimator(self.max_action*2 + 1, self.estimator_lr)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)

        self.dqn = DQN(self.max_action, self.action_dims, self.traditional_lr, self.commutative_lr)
        self.buffer = CommutativeReplayBuffer(self.max_action, 1, self.memory_size, self.max_action, self.rng)
        
        return super()._generate_language(problem_instance, self.dqn, self.buffer)