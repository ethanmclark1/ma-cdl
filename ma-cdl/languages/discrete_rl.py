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
        
        self.tau = 0.005
        self.alpha = 0.001
        self.batch_size = 256
        self.sma_window = 100
        self.granularity = 0.20
        self.epsilon_start = 1.0
        self.num_episodes = 7500
        self.memory_size = 30000
        self.epsilon_decay = 0.999
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.granuarlity = self.granularity 
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
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
    
    def _update(self, loss):
        if not isinstance(loss, int):    
            self.dqn.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.dqn.optim.step()
            loss = loss.item()
        
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                
        return loss
            
    def _learn(self):
        loss = 0
        if self.buffer.real_size < self.batch_size:
            return loss, None

        indices = self.buffer.sample(self.batch_size)
        
        states = self.buffer.state[indices]
        actions = self.buffer.action[indices]
        rewards = self.buffer.reward[indices]
        next_states = self.buffer.next_state[indices]
        dones = self.buffer.done[indices]
        
        q_values = self.dqn(states)
        next_q_values = self.target_dqn(next_states)
        target_q_values = rewards + (1 - dones) * torch.max(next_q_values, dim=1).values
        selected_q_values = torch.gather(q_values, 1, actions).squeeze(-1)
        loss += F.mse_loss(selected_q_values, target_q_values)       
        
        return loss, indices
    
    def _train(self, problem_instance):
        losses = []
        rewards = []
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            state, regions = self._generate_state()
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                self.buffer.add(state, action, reward, next_state, done)

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
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)
    
    def _get_final_lines(self, problem_instance):
        done = False
        language = []
        num_action = 0
        self.epsilon = 0
        regions = [SQUARE]
        episode_reward = 0
        state = self.autoencoder.get_state(regions)
        while not done:
            num_action += 1
            action = self._select_action(state)
            line = self.candidate_lines[action]
            reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
            
            state = next_state
            regions = next_regions
            language += [line]
            episode_reward += reward
        
        language = np.array(language).reshape(-1,3)
        return language, regions, episode_reward

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance, buffer=None):
        self.epsilon = self.epsilon_start
        self.dqn = DQN(self.state_dims, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        if buffer is None:
            self.buffer = ReplayBuffer(self.state_dims, 1, self.memory_size)
        
        self._init_wandb(problem_instance)
        self._train(problem_instance)
        language, regions, reward = self._get_final_lines(problem_instance)
        
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
        loss, indices = super()._learn()
        
        if indices is None:
            return loss
        
        states = self.buffer.state[indices]
        action_seqs = self.buffer.action_seq[indices]
        actions = self.buffer.action[indices]
        rewards = self.buffer.reward[indices]
        next_states = self.buffer.next_state[indices]
        dones = self.buffer.done[indices]
        
        prev_action_seqs = self.buffer.prev_action_seq[indices]
        prev_actions = self.buffer.prev_action[indices]
        prev_rewards = self.buffer.prev_reward[indices]
        
         # Step 2: Commutative Q-Update
        s_2 = torch.empty(self.batch_size, self.state_dims)
        r_2 = torch.empty(self.batch_size)
        
        s, a, r_0 = prev_action_seqs, prev_actions, prev_rewards
        s_1 = states
        b = actions
        r_1 = rewards
        s_prime = next_states
        
        # Retrieve s_2 and r_2 from ptr_lst to get most recent transition for each (s, b)
        valid_indices = []
        for i, (_s, _b) in enumerate(zip(s, b)):
            _s = tuple(_s.tolist())
            _b = _b.item()
            _s_2, _r_2 = self.ptr_lst.get((_s, _b), (None, None))
            if _s_2 is not None:
                valid_indices.append(i)
                s_2[i] = torch.as_tensor(_s_2)
                r_2[i] = torch.as_tensor(_r_2)
        
        if len(valid_indices) != 0:
            s_1 = s_1[valid_indices]
            s_2 = s_2[valid_indices]
            a = a[valid_indices]
            r_0 = r_0[valid_indices]
            r_1 = r_1[valid_indices]
            r_2 = r_2[valid_indices]
            s_prime = s_prime[valid_indices]
            dones = dones[valid_indices]
            action_seqs = action_seqs[valid_indices]
            
            next_q_values = self.target_dqn(s_prime)
            target_q_values = r_0 - r_2 + r_1 + (1 - dones) * torch.max(next_q_values, dim=1).values
            
            q_values = self.dqn(s_2)
            selected_q_values = torch.gather(q_values, 1, a).squeeze(-1)
            
            loss += F.mse_loss(selected_q_values, target_q_values)
            
        return loss

    # action_seq is a proxy for the state
    def _train(self, problem_instance):
        losses = []
        rewards = []
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            
            _action_seq = []
            prev_action = None
            prev_reward = None
            _prev_action_seq = []
            state, regions = self._generate_state()
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                
                action_seq = _action_seq + (self.max_action - len(_action_seq)) * [-1]
                prev_action_seq = _prev_action_seq + (self.max_action - len(_prev_action_seq)) * [-1]
                self.buffer.add((state, action_seq, action, reward, next_state, done, prev_action_seq, prev_action, prev_reward))
                # (s, a) -> (s', r)
                self.ptr_lst[(tuple(action_seq), action)] = (next_state, reward)
                
                prev_action = action
                prev_reward = reward
                _prev_action_seq = _action_seq.copy()
                _action_seq += [action]
                _action_seq = sorted(_action_seq)

                state = next_state
                regions = next_regions
                
                episode_reward += reward
                
            loss = self._learn()
            loss = self._update(loss)
            self.epsilon *= self.epsilon_decay

            losses.append(loss)
            rewards.append(episode_reward)
            avg_losses = np.mean(losses[-self.sma_window:])
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Reward": avg_rewards})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)
            
    def _generate_language(self, problem_instance):
        self.ptr_lst = {}
        self.buffer = CommutativeReplayBuffer(self.state_dims, 1, self.memory_size, self.max_action)
        
        return super()._generate_language(problem_instance, self.buffer)