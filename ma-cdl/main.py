import pdb
import copy
import torch
import wandb
import numpy as np

from env import simple_path
from arguments import get_arguments
from agent.speaker import Speaker
from agent.listener import Listener

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
        self.env = simple_path.env(
            max_cycles=self.max_episode_len, 
            num_landmarks=args.num_agents,
            render_mode=args.render_mode
            )
        
        self.speaker = Speaker(args.max_symbols)
        self.listener = Listener()
        
    def _init_hyperparams(self):
        self.epochs = 30000
        self.num_landmarks = 3
        self.render_mode = None
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
        batch_obs, batch_directions, batch_actions = [], [], []
        batch_log_probs, batch_rewards, batch_rtgs = [], [], []

        for _ in range(32):
            episodic_return = []
            truncation = termination = False
            # Environment reset is done in self.speaker.search()
            path, obstacles, backup = self.speaker.search(self.env)
            direction_set = self.speaker.communicate(path, obstacles, 2)
            # Reinitialize environment with backup
            self.env.unwrapped.steps = 0
            self.env.unwrapped.world = backup
            
            obs, _, _, _, _ = self.env.last()
            for timestep in range(self.max_episode_len):
                batch_directions.append(direction_set)
                batch_obs.append(obs)

                action, log_probs = self.listener.get_action(obs, direction_set, setting='train')
                batch_actions.append(action.item())
                batch_log_probs.append(log_probs)
                
                self.env.step(action)
                obs, _, termination, truncation, _ = self.env.last()
                reward = self.speaker.feedback(obs, obstacles, path[-1])
                episodic_return.append(reward)
                
                if termination or truncation:
                    break

            batch_rewards.append(episodic_return)
        
        batch_obs = torch.tensor(np.array(batch_obs))
        batch_directions = torch.tensor(np.array(batch_directions))
        batch_actions = torch.tensor(batch_actions)
        batch_log_probs = torch.tensor(batch_log_probs)
        batch_rtgs = self.listener.compute_rtgs(batch_rewards)
        return [batch_directions, batch_obs, batch_actions, batch_log_probs, batch_rtgs]
        
if __name__ == '__main__':
    args = get_arguments()
    driver = Driver(args)
    driver.synchronize()