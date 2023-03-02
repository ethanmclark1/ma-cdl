import copy
import torch
import numpy as np

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language

class MA_CDL():
    def __init__(self, args):
        self._init_hyperparams()
        self.env = simple_path.env(
            max_cycles=self.max_episode_len, 
            num_obstacles=args.num_obstacles,
            obstacle_size=args.obstacle_size,
            render_mode=args.render_mode
            )
        language = Language(args.num_obstacles, args.obstacle_size, args.num_languages).get_langauge()
        self.speaker = Speaker(language, args.num_obstacles, args.num_languages)
        self.listener = Listener(args.num_obstacles, self.env.action_space('agent_0').n)
        
    def _init_hyperparams(self):
        self.epochs = 30000
        self.memory_len = 5
        self.num_alternate = 1000
        self.max_episode_len = 200
    
    def rollout(self):
        a=3
        
    
    
            
    # # Alternate training between training Speaker and Listener
    # def learn(self, language):        
    #     speaker = False
    #     speaker_loss, actor_loss, critic_loss = [], [], []            
            
    #     for epoch in range(self.epochs):
    #         alternate = epoch % self.num_alternate == 0 and epoch != 0
    #         speaker = not speaker if alternate else speaker
            
    #         if speaker:
    #             batch_path = self.speaker.get_batch(self.env)
    #             loss = self.speaker.train(batch_path)
    #             speaker_loss.append(loss)
    #             # wandb.log({'speaker_loss': np.mean(speaker_loss[-25:])})
    #         else:
    #             batch_trajectory = self.rollout(language)
    #             loss = self.listener.train(batch_trajectory)
    #             actor_loss.append(loss[0])
    #             critic_loss.append(loss[1])
    #             # wandb.log({'actor_loss': np.mean(actor_loss[-25:]), 'critic_loss': np.mean(critic_loss[-25:])})
                
    #     return 5
                        
    # # Gather trajectories to train Listener
    # def rollout(self, language):
    #     batch_obs, batch_representation, batch_actions = [], [], []
    #     batch_log_probs, batch_rewards, batch_rtgs = [], [], []
    #     env_shape = self.env.state_space.shape[0]

    #     for _ in range(32):
    #         episodic_return = []
    #         truncation = termination = False
    #         # Environment reset is done in self.speaker.search()
    #         path, obstacles, backup = self.speaker.search(self.env)
    #         directions, polygons = self.speaker.communicate(language, path, obstacles)
    #         # Reinitialize environment with backup
    #         self.env.unwrapped.steps = 0
    #         self.env.unwrapped.world = backup
            
    #         obs, _, _, _, _ = self.env.last()
    #         for timestep in range(self.max_episode_len):
    #             batch_representation.append(directions)
    #             batch_obs.append(obs)
                
    #             action, log_probs = self.listener.get_action(obs, directions, setting='train')
    #             batch_actions.append(action.item())
    #             batch_log_probs.append(log_probs)
                
    #             self.env.step(action)
    #             obs, _, termination, truncation, _ = self.env.last()
    #             reward = self.speaker.feedback(obs, path[-1], obstacles, representation_idx, polygons)
    #             episodic_return.append(reward)
                
    #             if termination or truncation:
    #                 break

    #         batch_rewards.append(episodic_return)
        
    #     batch_obs = torch.tensor(np.array(batch_obs))
    #     batch_representation = torch.tensor(np.array(batch_representation))
    #     batch_actions = torch.tensor(batch_actions)
    #     batch_log_probs = torch.tensor(batch_log_probs)
    #     batch_rtgs = self.listener.compute_rtgs(batch_rewards)
    #     return [batch_representation, batch_obs, batch_actions, batch_log_probs, batch_rtgs]
        
if __name__ == '__main__':
    args = get_arguments()
    ma_cdl = MA_CDL(args)
    ma_cdl.rollout()