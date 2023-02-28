import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from statistics import variance
from scipy.optimize import minimize
from shapely.ops import polygonize, split
from shapely.geometry import Point, LineString, Polygon

class Language:
    def __init__(self, num_obstacles, num_languages):
        self.obs_size = 0.02
        self.num_obstacles = num_obstacles
        self.num_languages = num_languages
        self.points = [Point(1, 1), Point(-1, 1), Point(-1, -1), Point(1, -1)]
        self.lines = [LineString([(1, 1), (-1, 1)]), LineString([(-1, 1), (-1, -1)]), 
                          LineString([(-1, -1), (1, -1)]), LineString([(1, -1), (1, 1)])]
        self.shape = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)])
    
    # Both endpoints must be on an environment boundary to be considered valid
    def _get_line_info(self, lines):
        split_lines = set()
        valid_lines = [*self.lines]
        garbage_lines = set()

        # Get valid lines s.t. both endpoints are on an environment boundary
        for line in lines:
            intersection = self.shape.intersection(line)
            if not intersection.is_empty and np.any(np.abs([*intersection.coords]) == 1, axis=1).all():
                valid_lines.append(intersection)
                plt.plot(*intersection.xy)

        plt.plot(*self.shape.exterior.xy)
        plt.show()

        # TODO: Split lines at intersection points
        for line_0, line_1 in product(valid_lines, repeat=2):
            if line_0 == line_1:
                continue
            result = split(line_0, line_1)
            split_lines.update([*result.geoms])
            # Remove lines that have been split into two lines
            if len(result.geoms) == 2:
                if line_0 in self.lines:
                    garbage_lines.add(line_0)
                elif line_1 in self.lines:
                    garbage_lines.add(line_1)
                elif line_0 not in self.lines and line_1 not in self.lines:
                    garbage_lines.update([line_0, line_1])

        split_lines -= garbage_lines
        return split_lines            
            
    # Create regions from valid lines
    def _create_regions(self, lines):
        segmented_lines = self._get_line_info(lines)
        regions = list(polygonize(segmented_lines))
        print(regions)
        return regions

    def _optimizer(self, lines):
        cost = math.inf
        obs_prob, region_prob = 0, 0
        nonnavigable = []
        # Obstacle(s) constrained to be in top right quadrant
        obs_pos = np.random.rand(self.num_obstacles, 2)
        obs_list = [Point(obs_pos[i]) for i in range(self.num_obstacles)]
        print(lines)
        lines = [LineString([tuple(lines[i:i+2]), tuple(lines[i+2:i+4])])
                 for i in range(0, len(lines), 4)]
        regions = self._create_regions(lines)

        if regions:
            # 1. Probability of colliding into an obstacle
            for obs, region in product(obs_list, regions):
                if region.contains(obs):
                    region_prob += (region.area / self.shape.area)
                    obs_prob += (self.obs_size / region.area)
                    nonnavigable.append(region.area)
            collision_prob = region_prob * obs_prob
            # 2. Variance on region area
            region_var = variance([region.area for region in regions])
            # 3. Amount of navigable space
            navigable_space = self.shape.area - sum(nonnavigable)
            # TODO 4. Variance on navigable space across problems
            navigable_space_var = 3
        
            cost = 0.3*collision_prob + 0.15*region_var + (-0.3*navigable_space) + 0.25*navigable_space_var
            
        return cost

    def _generate_lines(self):
        bounds = (-3, 3)
        optim_val, optim_coeffs = math.inf, None
        for num in range(2, self.num_languages+2):
            # TODO: Add num back into x0 [np.random.rand(num, 4)]
            x0 = (bounds[1] - bounds[0])*np.random.rand(5, 4)+bounds[0]
            res = minimize(self._optimizer, x0, method='nelder-mead',
                           options={'xatol': 1e-8})

            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x

        optim_coeffs = np.reshape(optim_coeffs, (-1, 4))
        return optim_coeffs
    
    def get_langauge(self):
        return self._generate_lines()