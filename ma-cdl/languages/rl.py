import copy
import wandb
import torch
import numpy as np

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import BayesianDQN
from languages.utils.replay_buffer import PrioritizedReplayBuffer

"""Using Bayesian DQN with Prioritized Experience Replay"""
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)     
           
        self._init_hyperparams()        
        self.num_actions = self.max_actions = len(self.candidate_lines)
        
        self.autoencoder = AE(self.state_dim, self.rng, self.candidate_lines)

    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 5e-3
        self.alpha = 1e-4
        self.batch_size = 256
        self.action_cost = -0.1
        self.memory_size = 30000
        self.num_episodes = 15000
        self.kl_coefficient = 1e-1
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'{self.__class__.__name__}/{problem_instance.capitalize()}')
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.batch_size = self.batch_size
        config.action_cost = self.action_cost
        config.granularity = self.granularity
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        
    # Log regions to WandB
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
        
    # Select action using Thompson Sampling
    def _select_action(self, onehot_state):
        with torch.no_grad():
            q_values_sample = self.bdqn(torch.FloatTensor(onehot_state))
            action = torch.argmax(q_values_sample).item()
        return action
    
    def _step(self, problem_instance, action, num_lines):
        reward, done, regions = super()._step(problem_instance, action, num_lines)
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
    
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        q_values = self.bdqn(state)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.bdqn(next_state)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).view(-1).detach()
        
        q_hat = reward + (1 - done) * next_q
        
        td_error = torch.abs(q_hat - q).detach()
        
        if weights is None:
            weights = torch.ones_like(q)

        self.bdqn.optim.zero_grad()
        # Multiply by importance sampling weights to correct bias from prioritized replay
        loss = (weights * (q_hat - q) ** 2).mean()
        
        # Form of regularization to prevent posterior collapse
        kl_divergence = 0
        for layer in [self.bdqn.fc1, self.bdqn.fc2, self.bdqn.fc3]:
            w_mu = layer.w_mu
            w_rho = layer.w_rho
            w_sigma = torch.log1p(torch.exp(w_rho))
            posterior = torch.distributions.Normal(w_mu, w_sigma)
            prior = torch.distributions.Normal(0, 1)
            
            kl_divergence += torch.distributions.kl_divergence(posterior, prior).sum()
            
        loss += self.kl_coefficient * kl_divergence
        loss.backward()
        self.bdqn.optim.step()
                
        # Update target network
        for param, target_param in zip(self.bdqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item(), td_error.numpy(), tree_idxs
    
    def _train(self, problem_instance, start_state):
        losses = []
        rewards = []
        best_lines = None
        best_regions = None
        best_reward = -np.inf

        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            action_lst = []
            state = start_state
            # TODO: Think about learning without anything in the buffer
            while not done:
                num_lines += 1
                action = self._select_action(state)
                reward, next_state, done, regions = self._step(problem_instance, action, num_lines)
                self.buffer.add((state, action, reward, next_state, done))
                loss, td_error, tree_idxs = self._learn()
                state = next_state
                action_lst.append(action)

            self.buffer.update_priorities(tree_idxs, td_error)

            losses.append(value_loss)
            rewards.append(reward)
            avg_losses = np.mean(losses[-100:])
            avg_returns = np.mean(rewards[-100:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Reward": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)

            if reward > best_reward:
                best_lines = action_lst
                best_regions = regions
                best_reward = reward

        return best_lines, best_regions, best_reward
    
    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_optimal_lines(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.buffer = PrioritizedReplayBuffer(self.state_dim, 1, self.memory_size)
        self.bdqn = BayesianDQN(self.state_dim, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.bdqn)
        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        best_lines, best_regions, best_reward = self._train(problem_instance, start_state)
        best_lines = list(map(lambda x: self.candidate_lines[x], best_lines))
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Final Reward": best_reward})
        wandb.log({"Final Lines": best_lines})
        wandb.finish()  
        
        optim_lines = np.array(best_lines).reshape(-1, 3)   
        return optim_lines  
    
    # rewards regions that defines the language
    def get_language(self, problem_instance):
        approach = self.__class__.__name__
        try:
            language = self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored {approach} language for {problem_instance.capitalize()} problem instance.')
            print('Generating new language...\n')
            lines = self._generate_optimal_lines(problem_instance)
            linestrings = CDL.get_shapely_linestring(lines)
            valid_lines = CDL.get_valid_lines(linestrings)
            language = CDL.create_regions(valid_lines)
            self._visualize(approach, problem_instance, language)
            self._save(approach, problem_instance, language)
        
        return language