import pdb
import copy
import torch
import wandb
import numpy as np

from environment import simple_path
from arguments import get_arguments
from agents.speaker import Speaker
from agents.listener import Listener

# wandb.init(project='thesis')
# wandb.config = {
#     'batch_size': 32,
#     'message_size': 8,
#     'num_alternate': 1000,
#     'max_episode_len': 128
# }

class Driver():
    def __init__(self, args):
        self._init_hyperparams()
        env_type = args.env_shape
        self.env = simple_path.env(
            num_obstacles=args.num_obstacles,
            max_cycles=self.max_episode_len, 
            render_mode=args.render_mode
            )
        
        # 2 for start (xy), 2 for goal (xy), 2*num_obstacles for obstacles (xy), 1 for env_type
        input_dims = 2 + 2 + 2*args.num_obstacles + 1
        self.speaker = Speaker(input_dims, args.min_symbol, args.max_symbol)
        input_dims = self.speaker.direction_len + self.env.state_space.shape[0]
        output_dims = self.env.action_spaces['agent_0'].n
        self.listener = Listener(input_dims, output_dims)
        
    def _init_hyperparams(self):
        self.epochs = 30000
        self.num_alternate = 1000
        self.max_episode_len = 200
        
    # Alternate between training Speaker and Listener
    def synchronize(self):        
        speaker = False
        speaker_loss, actor_loss, critic_loss = [], [], []            
            
        for epoch in range(self.epochs):
            alternate = epoch % self.num_alternate == 0 and epoch != 0
            speaker = not speaker if alternate else speaker
            
            if speaker:
                batch_path = self.speaker.get_batch(self.env)
                loss = self.speaker.train(batch_path)
                speaker_loss.append(loss)
                # wandb.log({'speaker_loss': np.mean(speaker_loss[-25:])})
            else:
                batch_trajectory = self.rollout()
                loss = self.listener.train(batch_trajectory)
                actor_loss.append(loss[0])
                critic_loss.append(loss[1])
                # wandb.log({'actor_loss': np.mean(actor_loss[-25:]), 'critic_loss': np.mean(critic_loss[-25:])})
                        
    # Gather trajectories to train Listener
    def rollout(self):
        batch_obs, batch_representation, batch_actions = [], [], []
        batch_log_probs, batch_rewards, batch_rtgs = [], [], []
        env_shape = self.env.state_space.shape[0]

        for _ in range(32):
            episodic_return = []
            truncation = termination = False
            # Environment reset is done in self.speaker.search()
            path, obstacles, backup = self.speaker.search(self.env)
            representation_idx = self.speaker.select(path[0], path[-1], obstacles, env_shape)
            directions = self.speaker.communicate(path, obstacles, representation_idx)
            # Reinitialize environment with backup
            self.env.unwrapped.steps = 0
            self.env.unwrapped.world = backup
            
            obs, _, _, _, _ = self.env.last()
            for timestep in range(self.max_episode_len):
                batch_representation.append(directions)
                batch_obs.append(obs)
                
                action, log_probs = self.listener.get_action(obs, directions, setting='train')
                batch_actions.append(action.item())
                batch_log_probs.append(log_probs)
                
                self.env.step(action)
                obs, _, termination, truncation, _ = self.env.last()
                reward = self.speaker.feedback(obs, path[-1], obstacles, directions, representation_idx)
                episodic_return.append(reward)
                
                if termination or truncation:
                    break

            batch_rewards.append(episodic_return)
        
        batch_obs = torch.tensor(np.array(batch_obs))
        batch_representation = torch.tensor(np.array(batch_representation))
        batch_actions = torch.tensor(batch_actions)
        batch_log_probs = torch.tensor(batch_log_probs)
        batch_rtgs = self.listener.compute_rtgs(batch_rewards)
        return [batch_representation, batch_obs, batch_actions, batch_log_probs, batch_rtgs]
        
if __name__ == '__main__':
    args = get_arguments()
    driver = Driver(args)
    driver.synchronize()