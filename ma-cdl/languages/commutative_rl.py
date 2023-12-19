import io
import copy
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import DQN
from languages.utils.replay_buffer import ReplayBuffer


class CommutativeRL(CDL):
    def __init__(self, scenario, world):
        super(CommutativeRL, self).__init__(scenario, world)     
        self._init_hyperparams()                
        
        self.dqn = None
        self.target_dqn = None
        self.buffer = None
        
        self._create_candidate_set_of_lines()
        
        self.action_dims = len(self.candidate_lines)
        self.autoencoder = AE(self.candidate_lines, self.state_dims, self.max_action, self.rng)

    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.001
        self.alpha = 0.0007
        self.batch_size = 5
        self.granularity = 0.20
        self.action_cost = 0.25
        self.epsilon_start = 1.0
        self.num_episodes = 10000
        self.memory_size = 100000
        self.epsilon_decay = 0.999
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'{self.__class__.__name__}/{problem_instance.capitalize()}')
        config = wandb.config
        config.alpha = self.alpha
        config.batch_size = self.batch_size
        config.granuarlity = self.granularity 
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        
    # Log image of partitioned regions to Weights & Biases
    def _log_regions(self, problem_instance, title_name, title_data, regions, reward):
        _, ax = plt.subplots()
        problem_instance = problem_instance.capitalize()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Problem Instance: {problem_instance}   {title_name.capitalize()}: {title_data}   Reward: {reward:.2f}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        plt.close()
        wandb.log({"image": wandb.Image(pil_image)})
        
    def _create_candidate_set_of_lines(self):
        self.candidate_lines = []
        granularity = int(self.granularity * 100)
        
        # termination line
        self.candidate_lines += [(0, 0, 0)]
        
        # vertical/horizontal lines
        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)] # vertical lines
            self.candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]
            self.candidate_lines += [(-0.1, 0.1, i)]
        
    def _update_target_network(self):
        for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def _generate_state(self):
        num_actions = self.rng.choice(self.max_action)
        actions = self.rng.choice(self.candidate_lines[1:], size=num_actions, replace=False)
        linestrings = CDL.get_shapely_linestring(actions)
        valid_lines = CDL.get_valid_lines(linestrings)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        state = self.autoencoder.get_state(regions)
        
        return state, regions
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            idx = self.rng.choice(len(self.candidate_lines))
            action = self.candidate_lines[idx]
        else:
            with torch.no_grad():
                q_values = self.dqn(torch.tensor(state))
                idx = q_values.argmax().item()
                action = self.candidate_lines[idx]
        return action, idx
    
    def _step(self, problem_instance, regions, action, num_action):
        reward, next_regions, done = super()._step(problem_instance, regions, action, num_action)
        next_state = self.autoencoder.get_state(next_regions)
        
        return reward, next_state, done, next_regions
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _learn(self):
        loss = 0
        
        if self.buffer.real_size < self.batch_size:
            return loss

        sample = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, prev_action_seqs, prev_actions, prev_rewards, action_seqs = sample

        # Step 1: Traditional Q-Update        
        q_values = self.dqn(states)
        next_q_values = self.target_dqn(next_states)
        target_q_values = rewards + (1 - dones) * torch.max(next_q_values, dim=1).values
        selected_q_values = torch.gather(q_values, 1, actions).squeeze()
        loss += F.mse_loss(selected_q_values, target_q_values)
        
        # Step 2: Commutative Update
        s_2 = torch.empty(self.batch_size, self.state_dims)
        r_2 = torch.empty(self.batch_size)
        
        s = prev_action_seqs
        a = prev_actions
        r_0 = prev_rewards
        
        s_1 = states
        b = actions
        r_1 = rewards
        s_prime = next_states
        
        # TODO: Make sure to remove -1 in the indices
        for i, (action_seq, action) in enumerate(zip(s, b)):
            action_seq = tuple(action_seq.tolist())
            action = action.item()
            tmp_s2, tmp_r2 = self.ptr_lst.get((action_seq, action), (None, None))
            # TODO: Remove sample from batch
            if tmp_s2 is None:
                tmp_s2 = torch.zeros(self.state_dims)
                tmp_r2 = torch.zeros(1)
            s_2[i] = torch.as_tensor(tmp_s2)
            r_2[i] = torch.as_tensor(tmp_r2)
        
        q_values = self.dqn(s_1)
        next_q_values = self.target_dqn(s_prime)
        target_q_values = r_0 - r_2 + r_1 + (1 - dones) * torch.max(next_q_values, dim=1).values
        selected_q_values = torch.gather(q_values, 1, a).squeeze()
        loss += F.mse_loss(selected_q_values, target_q_values)
        
        self.ptr_lst[(tuple(s_1), b)] = (s_prime, r_1)
        
        self.dqn.optim.zero_grad()
        loss.backward()
        self.dqn.optim.step()
        
        if self.iterations % 100 == 0:
            self._update_target_network()                
        
        self.iterations += 1
        return loss.item()
    
    # index_seq is a proxy for the state
    def _train(self, problem_instance):
        losses = []
        rewards = []
        best_lines = None
        best_regions = None
        best_reward = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            
            index_seq = []
            action_seq = []
            prev_index = None
            prev_reward = None
            prev_index_seq = []
            state, regions = self._generate_state()
            while not done:
                # TODO: Figure out when to add index to index_seq
                num_action += 1
                action, index = self._select_action(state)
                reward, next_state, done, next_regions = self._step(problem_instance, regions, action, num_action)
                
                index_seq += [index]
                index_seq = sorted(index_seq)
                filled_index_seq = index_seq + (self.max_action - len(index_seq)) * [-1]
                filled_prev_index_seq = prev_index_seq + (self.max_action - len(prev_index_seq)) * [-1]
                
                self.buffer.add((state, index, reward, next_state, done, filled_index_seq, filled_prev_index_seq, prev_index, prev_reward))
                
                self.ptr_lst[(tuple(filled_index_seq), index)] = (next_state, reward)
                prev_index = index
                prev_reward = reward
                prev_index_seq = index_seq.copy()

                state = next_state
                regions = next_regions
                
                action_seq += [action]
                episode_reward += reward
                
            loss = self._learn()
            self.epsilon *= self.epsilon_decay

            losses.append(loss)
            rewards.append(episode_reward)
            avg_losses = np.mean(losses[-100:])
            avg_rewards = np.mean(rewards[-100:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Reward": avg_rewards})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)

            if reward > best_reward:
                best_lines = action_seq
                best_regions = regions
                best_reward = reward

        return best_lines, best_regions, best_reward
    
    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_optimal_lines(self, problem_instance):
        # self._init_wandb(problem_instance)
        
        self.ptr_lst = {}
        self.iterations = 0
        self.epsilon = self.epsilon_start
        
        self.dqn = DQN(self.state_dims, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.buffer = ReplayBuffer(self.state_dims, 1, self.memory_size, self.max_action)
        
        best_lines, best_regions, best_reward = self._train(problem_instance)
        best_lines = list(map(lambda x: self.candidate_lines[x], best_lines))
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Lines": best_lines})
        wandb.log({"Reward": best_reward})
        wandb.finish()  
        
        optim_lines = np.array(best_lines).reshape(-1, 3)   
        return optim_lines  
    
    # rewards regions that defines the language
    def get_language(self, problem_instance):
        approach = self.__class__.__name__
        try:
            language = self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored language for {approach} on the {problem_instance.capitalize()} problem instance.')
            print('Generating new language...\n')
            lines = self._generate_optimal_lines(problem_instance)
            linestrings = CDL.get_shapely_linestring(lines)
            valid_lines = CDL.get_valid_lines(linestrings)
            language = CDL.create_regions(valid_lines)
            self._visualize(approach, problem_instance, language)
            self._save(approach, problem_instance, language)
        
        return language