import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.networks import DQN
from languages.utils.cdl import CDL, SQUARE
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicDQN(CDL):
    def __init__(self, scenario, world):
        super(BasicDQN, self).__init__(scenario, world)     
        self._init_hyperparams()                
        
        self.dqn = None
        self.target_dqn = None
        self.buffer = None
        
        self._create_candidate_set_of_lines()
        
        self.action_dims = len(self.candidate_lines)
        self.autoencoder = AE(self.state_dims, self.max_action, self.rng, self.candidate_lines)

    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.0005
        self.alpha = 0.0002
        self.batch_size = 128
        self.sma_window = 100
        self.granularity = 0.20
        self.epsilon_start = 1.0
        self.memory_size = 25000
        self.num_episodes = 7500
        self.epsilon_decay = 0.999
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.granularity = self.granularity 
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.efficiency_factor = self.efficiency_factor
        config.configs_to_consider = self.configs_to_consider
        
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
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action_index = self.rng.choice(len(self.candidate_lines))
        else:
            with torch.no_grad():
                action_index = self.dqn(torch.tensor(state)).argmax().item()
        return action_index
    
    def _update(self, loss, update_target=True):
        if not isinstance(loss, int):    
            self.dqn.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.dqn.optim.step()
            loss = loss.item()
        
        if update_target:
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                
        return loss
            
    def _learn(self):
        if self.buffer.real_size < self.batch_size:
            return 0, None

        indices = self.buffer.sample(self.batch_size)
        
        state = self.buffer.state[indices]
        action = self.buffer.action[indices]
        reward = self.buffer.reward[indices]
        next_state = self.buffer.next_state[indices]
        done = self.buffer.done[indices]
        
        q_values = self.dqn(state)
        selected_q_values = torch.gather(q_values, 1, action).squeeze(-1)
        next_q_values = self.target_dqn(next_state)
        target_q_values = reward + ~done * torch.max(next_q_values, dim=1).values
        loss = F.mse_loss(selected_q_values, target_q_values)       
        
        return loss, indices
    
    def _train(self, problem_instance):
        losses = []
        rewards = []
        
        best_regions = None
        best_language = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            language = []
            num_action = 0
            episode_reward = 0
            state, regions, _ = self._generate_state()
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                self.buffer.add(state, action, reward, next_state, done)

                language += [line]
                state = next_state
                regions = next_regions
                episode_reward += reward
                
            loss, _ = self._learn()
            loss = self._update(loss)
            self.epsilon *= self.epsilon_decay

            losses.append(loss)
            rewards.append(episode_reward)
            avg_losses = np.mean(losses[-self.sma_window:])
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Reward": avg_rewards})
            
            if episode_reward > best_reward:
                best_regions = regions
                best_reward = episode_reward
                best_language = np.array(language).reshape(-1, 3)
                
        return best_language, best_regions, best_reward

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance, buffer=None):
        self.epsilon = self.epsilon_start
        self.dqn = DQN(self.state_dims, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        if buffer is None:
            self.buffer = ReplayBuffer(self.state_dims, 1, self.memory_size)
        
        self._init_wandb(problem_instance)
        language, regions, reward = self._train(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', regions, reward)
        wandb.log({"Language": language})
        wandb.log({"Final Reward": reward})
        wandb.finish()  
        
        return language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, scenario, world):
        super(CommutativeDQN, self).__init__(scenario, world)
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self):
        traditional_loss, indices = super()._learn()
        commutative_loss = 0
        
        if indices is None:
            return traditional_loss
        
        # Step 2: Commutative Q-Update
        has_previous = self.buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        s = self.buffer.prev_state_proxy[indices][valid_indices]
        b = self.buffer.action[indices][valid_indices]
        
        key_tuples = [(tuple(_s.tolist()), _b.item()) for _s, _b in zip(s, b)]
        valid_mask = torch.tensor([self.ptr_lst.get(key, (None, None))[0] is not None for key in key_tuples])
        
        if torch.any(valid_mask):
            s_2 = np.stack([self.ptr_lst[key][0] for i, key in enumerate(key_tuples) if valid_mask[i]])
            r_2 = np.array([self.ptr_lst[key][1] for i, key in enumerate(key_tuples) if valid_mask[i]])
            
            s_2 = torch.from_numpy(s_2).type(torch.float)
            r_2 = torch.from_numpy(r_2).type(torch.float)
        
            a = self.buffer.prev_action[indices][valid_indices][valid_mask]
            r_0 = self.buffer.prev_reward[indices][valid_indices][valid_mask]
            r_1 = self.buffer.reward[indices][valid_indices][valid_mask]
            s_prime = self.buffer.next_state[indices][valid_indices][valid_mask]
            done = self.buffer.done[indices][valid_indices][valid_mask]

            q_values = self.dqn(s_2)
            selected_q_values = torch.gather(q_values, 1, a).squeeze(-1)
            next_q_values = self.target_dqn(s_prime)
            target_q_values = r_0 - r_2 + r_1 + ~done * torch.max(next_q_values, dim=1).values
            commutative_loss = F.mse_loss(selected_q_values, target_q_values)

        if commutative_loss == 0:
            total_loss = traditional_loss
        else:
            total_loss = torch.cat([traditional_loss.unsqueeze(-1), commutative_loss.unsqueeze(-1)])
            total_loss = torch.mean(total_loss)
            
        total_loss = self._update(total_loss)            
        return total_loss

    def _train(self, problem_instance):
        losses = []
        rewards = []
        
        best_regions = None
        best_language = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0

            prev_action = None
            prev_reward = None
            _prev_state_proxy = []
            state, regions, actions = self._generate_state()
            _state_proxy = sorted(list(actions))
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                
                state_proxy = _state_proxy + (2*self.max_action - len(_state_proxy)) * [0]
                prev_state_proxy = _prev_state_proxy + (2*self.max_action - len(_prev_state_proxy)) * [0]
                self.buffer.add((state, action, reward, next_state, done, prev_state_proxy, prev_action, prev_reward))
                # (s, a) -> (s', r)
                self.ptr_lst[(tuple(state_proxy), action)] = (next_state, reward)
                
                prev_action = action
                prev_reward = reward
                _prev_state_proxy = _state_proxy.copy()
                
                _state_proxy += [action]
                _state_proxy = sorted(_state_proxy)

                language += [line]
                state = next_state
                regions = next_regions
                episode_reward += reward
                
            loss = self._learn()
            self.epsilon *= self.epsilon_decay

            losses.append(loss)
            rewards.append(episode_reward)
            avg_losses = np.mean(losses[-self.sma_window:])
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Reward": avg_rewards})
            
            if episode_reward > best_reward:
                best_regions = regions
                best_reward = episode_reward
                best_language = np.array(language).reshape(-1, 3)
                
        return best_language, best_regions, best_reward
            
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}
        self.buffer = CommutativeReplayBuffer(self.state_dims, 1, self.memory_size, self.max_action)
        
        return super()._generate_language(problem_instance, self.buffer)