import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instances, all_metrics):
    metric_names = list(all_metrics[0].keys())
    num_approaches = 4
    approaches = ['RL', 'Grid World', 'Voronoi Map', 'Direct Path']
    
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'{metric_name}')
        width = 0.15
        
        for idx, problem_instance in enumerate(problem_instances):
            metrics = all_metrics[idx][metric_name]
            rl = metrics[approaches[0]]
            grid_world = metrics[approaches[1]]
            voronoi_map = metrics[approaches[2]]
            direct_path = metrics[approaches[3]]
            
            x_values = np.arange(num_approaches) + (idx * num_approaches * width * 2)
            ax.bar(x_values - width, rl, width=width, label=f'{problem_instance} - RL')
            ax.bar(x_values, grid_world, width=width, label=f'{problem_instance} - Grid World')
            ax.bar(x_values + width, voronoi_map, width=width, label=f'{problem_instance} - Voronoi Map')
            ax.bar(x_values + 2 * width, direct_path, width=width, label=f'{problem_instance} - Direct Path')
        
        ax.set_xlabel('Approaches')
        ax.set_ylabel(metric_name)
        ax.set_xticks(np.arange(num_approaches) + (len(problem_instances) - 1) * num_approaches * width)
        ax.set_xticklabels(approaches)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig(f'results/{metric_name}.png')