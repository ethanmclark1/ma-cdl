import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instance, **kwargs):
    metric_names = list(kwargs.keys())
    metric_data = list(kwargs.values())
    
    directory = f'results/{problem_instance}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for name, data in zip(metric_names, metric_data):
        fig, ax = plt.subplots()
        fig.suptitle(f'{problem_instance} - {name}')
        
        approaches = list(data.keys())
        num_approaches = len(approaches)
        
        ea = data[approaches[0]]
        rl = data[approaches[1]]
        bandit = data[approaches[2]]
        grid_world = data[approaches[3]]
        voronoi_map = data[approaches[4]]
        direct_path = data[approaches[5]]
        
        ax.bar(np.arange(num_approaches) - 0.6, ea, width=0.2, label='EA')
        ax.bar(np.arange(num_approaches) - 0.4, rl, width=0.2, label='RL')
        ax.bar(np.arange(num_approaches) - 0.2, bandit, width=0.2, label='Bandit')
        ax.bar(np.arange(num_approaches), grid_world, width=0.2, label='Grid World')
        ax.bar(np.arange(num_approaches) + 0.2, voronoi_map, width=0.2, label='Voronoi Map')
        ax.bar(np.arange(num_approaches) + 0.4, direct_path, width=0.2, label='Direct Path')
        ax.set_xlabel(problem_instance)
        ax.set_ylabel(name)
        ax.set_xticklabels(approaches[0:3])
        ax.set_ylim(0, 100)
        
        fig.savefig(f'results/{problem_instance}/{name}.png')