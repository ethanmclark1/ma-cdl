import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import RewardEstimator, DQN
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicDQN(CDL):
    def __init__(self, scenario, world, random_state):
        super(BasicDQN, self).__init__(scenario, world, random_state)     
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
        self.batch_size = 128
        self.sma_window = 750
        self.granularity = 0.20
        self.epsilon_start = 1.0
        self.memory_size = 75000
        self.num_episodes = 10000
        self.estimator_tau = 0.01
        self.estimator_lr = 0.005
        self.epsilon_decay = 0.999
        self.traditional_lr = 0.0006
        self.commutative_lr = 0.0001
        
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.granularity = self.granularity 
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
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action_index = self.rng.choice(len(self.candidate_lines))
        else:
            with torch.no_grad():
                action_index = self.dqn(torch.tensor(state)).argmax().item()
                
        return action_index
    
    def _update(self, loss, traditional_update=True, update_target=True):
        if not isinstance(loss, int):    
            if traditional_update:
                self.dqn.traditional_optim.zero_grad(set_to_none=True)
                loss.backward()
                self.dqn.traditional_optim.step()
                loss = loss.item()
            else:
                self.dqn.commutative_optim.zero_grad(set_to_none=True)
                loss.backward()
                self.dqn.commutative_optim.step()
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
        rewards = []
        traditional_losses = []
        
        best_regions = None
        best_language = None
        best_rewards = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            language = []
            num_action = 0
            episode_reward = 0
            state, regions, _ = self._get_state()
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                self.buffer.add(state, action, reward, next_state, done)

                state = next_state
                regions = next_regions
                language += [line]
                episode_reward += reward
                
            traditional_loss, _ = self._learn()
            traditional_loss = self._update(traditional_loss)
            self._decrement_epsilon()

            rewards.append(episode_reward)
            traditional_losses.append(traditional_loss)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            
            wandb.log({"Average Reward": avg_rewards, "Average Traditional Loss": avg_traditional_losses})
            
            if episode_reward > best_rewards:
                best_rewards = episode_reward
                best_language = language
                best_regions = regions
            
        best_language = np.array(best_language).reshape(-1,3)
        return best_language, best_regions, best_rewards
    
    def _get_final_language(self, problem_instance):
        best_reward = -np.inf
        best_language = None
        best_regions = None
        
        for _ in range(25):
            done = False
            language = []
            num_action = 0
            self.epsilon = 0
            episode_reward = 0
            state, regions, _ = self._generate_fixed_state()
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                
                state = next_state
                regions = next_regions
                language += [line]
                episode_reward += reward
                
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_language = language
                best_regions = regions
        
        best_language = np.array(best_language).reshape(-1,3)
        return best_language, best_regions, best_reward

    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_language(self, problem_instance, dqn=None, buffer=None):
        self.epsilon = self.epsilon_start
        if dqn is None and buffer is None:
            self.dqn = DQN(self.state_dims, self.action_dims, self.traditional_lr)
            self.buffer = ReplayBuffer(self.state_dims, 1, self.memory_size, self.rng)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self._init_wandb(problem_instance)
        best_language, best_regions, best_reward = self._train(problem_instance)
        
        if self.random_state:
            best_language, best_regions, best_reward = self._get_final_language(problem_instance)
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Language": best_language, "Final Reward": best_reward})
        wandb.finish()  
        
        return best_language
    

class CommutativeDQN(BasicDQN):
    def __init__(self, scenario, world, random_state):
        super(CommutativeDQN, self).__init__(scenario, world, random_state)

    # Estimator is approximating r_3 given r_0, r_1, r_2
    # TODO: Update estimator to be based on only r_0 and r_1
    def _update_estimator(self, r_0, r_1, r_2, r_3):        
        r0_r1 = r_0.unsqueeze(-1) + r_1.unsqueeze(-1)
        r2_r3 = r_2.unsqueeze(-1) + r_3.unsqueeze(-1)
        estimator_loss = F.mse_loss(r0_r1, r2_r3)
        
        self.reward_estimator.optim.zero_grad(set_to_none=True)
        estimator_loss.backward()
        self.reward_estimator.optim.step()
        estimator_loss = estimator_loss.item()
        
        for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
        
        return estimator_loss
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self):
        estimator_loss = 0
        traditional_loss, indices = super()._learn()
        commutative_loss = 0
        
        if indices is None:
            return estimator_loss, traditional_loss, commutative_loss
        
        traditional_loss = self._update(traditional_loss, True)
                
        # Commutative Q-Update
        has_previous = self.buffer.has_previous[indices]        
        valid_indices = torch.nonzero(has_previous, as_tuple=True)[0]
        
        prev_state_proxy = self.buffer.prev_state_proxy[indices][valid_indices]
        action = self.buffer.action[indices][valid_indices]
        
        key_tuples = [(tuple(_s.tolist()), _b.item()) for _s, _b in zip(prev_state_proxy, action)]
        # Determine whether the key exists in the ptr_lst
        valid_mask = torch.tensor([self.ptr_lst.get(key, (None, None))[0] is not None for key in key_tuples])
        
        if torch.any(valid_mask):
            r_2 = np.array([self.ptr_lst[key][0] for i, key in enumerate(key_tuples) if valid_mask[i]])
            s_2 = np.stack([self.ptr_lst[key][1] for i, key in enumerate(key_tuples) if valid_mask[i]])
            
            r_2 = torch.from_numpy(r_2).type(torch.float)
            s_2 = torch.from_numpy(s_2).type(torch.float)
    
            a = self.buffer.prev_action[indices][valid_indices][valid_mask]
            r_0 = self.buffer.prev_reward[indices][valid_indices][valid_mask]

            r_1 = self.buffer.reward[indices][valid_indices][valid_mask]
            s_prime = self.buffer.next_state[indices][valid_indices][valid_mask]
            done = self.buffer.done[indices][valid_indices][valid_mask]
            
            r_3 = r_0 + r_1 - r_2
            r_2, r_3 = self.reward_estimator(r_0, r_1)
            estimator_loss = self._update_estimator(r_0, r_1, r_2, r_3)

            q_values = self.dqn(s_2)
            selected_q_values = torch.gather(q_values, 1, a).squeeze(-1)
            next_q_values = self.target_dqn(s_prime)
            target_q_values = r_3.squeeze(-1).detach() + ~done * torch.max(next_q_values, dim=1).values
            commutative_loss = F.mse_loss(selected_q_values, target_q_values)
            commutative_loss = self._update(commutative_loss, False)
        
        return estimator_loss, traditional_loss, commutative_loss

    def _train(self, problem_instance):
        rewards = []
        estimator_losses = []
        traditional_losses = []
        commutative_losses = []
        
        best_regions = None
        best_language = None
        best_rewards = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            language = []
            num_action = 0
            episode_reward = 0

            _state_proxy = []
            prev_action = None
            prev_reward = None
            _prev_state_proxy = []
            state, regions, actions = self._get_state()
            _state_proxy = sorted(list(actions))
            while not done:
                num_action += 1
                action = self._select_action(state)
                line = self.candidate_lines[action]
                reward, done, next_regions, next_state = self._step(problem_instance, regions, line, num_action)
                
                state_proxy = _state_proxy + (2*self.max_action - len(_state_proxy)) * [0]
                prev_state_proxy = _prev_state_proxy + (2*self.max_action - len(_prev_state_proxy)) * [0]
                self.buffer.add(state, action, reward, next_state, done, prev_state_proxy, prev_action, prev_reward)
                self.ptr_lst[(tuple(state_proxy), action)] = (reward, next_state)
                
                prev_action = action
                prev_reward = reward
                _prev_state_proxy = _state_proxy.copy()
                
                _state_proxy += [action]
                _state_proxy = sorted(_state_proxy)

                language += [line]
                state = next_state
                regions = next_regions
                episode_reward += reward
                
            estimator_loss, traditional_loss, commutative_loss = self._learn()
            self._decrement_epsilon()
            
            rewards.append(episode_reward)
            estimator_losses.append(estimator_loss)
            traditional_losses.append(traditional_loss)
            commutative_losses.append(commutative_loss)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_estimator_losses = np.mean(estimator_losses[-self.sma_window:])
            avg_traditional_losses = np.mean(traditional_losses[-self.sma_window:])
            avg_commutative_losses = np.mean(commutative_losses[-self.sma_window:])
            
            wandb.log({
                'Average Reward': avg_rewards,
                'Average Estimator Loss': avg_estimator_losses,
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
        self.reward_estimator = RewardEstimator(3, 1, self.estimator_lr)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)

        self.dqn = DQN(self.state_dims, self.action_dims, self.traditional_lr, self.commutative_lr)
        self.buffer = CommutativeReplayBuffer(self.state_dims, 1, self.memory_size, self.max_action, self.rng)
        
        return super()._generate_language(problem_instance, self.dqn, self.buffer)