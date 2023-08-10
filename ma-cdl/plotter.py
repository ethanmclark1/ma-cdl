import math
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instances, all_metrics):
    metric_names = list(all_metrics[0].keys())
    approaches = ['rl', 'grid_world', 'voronoi_map', 'direct_path']
    num_approaches = len(approaches)

    for metric_name in metric_names:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.suptitle(f'{metric_name}')

        # Loop through problem instances and plot the metric data
        for idx, problem_instance in enumerate(problem_instances):
            ax = axes.flatten()[idx]
            metrics = all_metrics[idx][metric_name]

            rl = metrics[approaches[0]]
            grid_world = metrics[approaches[1]]
            voronoi_map = metrics[approaches[2]]
            direct_path = metrics[approaches[3]]
            
            max_value = max(rl, grid_world, voronoi_map, direct_path)
            y_limit = math.ceil(max_value / 10) * 10
            if y_limit == 0:
                y_limit = 10

            x_values = np.arange(num_approaches)

            ax.bar(x_values[0], rl, width=0.2, label='RL')
            ax.bar(x_values[1], grid_world, width=0.2, label='Grid World')
            ax.bar(x_values[2], voronoi_map, width=0.2, label='Voronoi Map')
            ax.bar(x_values[3], direct_path, width=0.2, label='Direct Path')
            ax.set_xlabel(problem_instance)
            ax.set_ylabel(metric_name)
            ax.set_xticks(x_values)
            ax.set_xticklabels(approaches)
            ax.set_ylim(0, y_limit)
            ax.legend(loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f'results/{metric_name}.png')