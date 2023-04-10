import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


# Baseline 0: Grid World
# Discretize environment into a grid world
# Demonstrates lack of expressitivity

# Baseline 1: Voronoi Maps
# Create a Voronoi map for each problem type based on first observed scenario
# Demonstrates lack of generalization

# Baseline 2: Random Walk
# Introduce a new line for every problem
# Demonstrates lack of efficiency

# Potential Baselines: Window Based Multi-agent Search