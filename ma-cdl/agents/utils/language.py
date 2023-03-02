import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize
from statistics import mean, variance
from shapely.ops import polygonize, split
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point, LineString, Polygon

class Language:
    def __init__(self, num_obstacles, obstacle_radius, num_languages):
        corners = list(product((1, -1), repeat=2))
        self.configs_to_consider = 10
        self.num_obstacles = num_obstacles
        self.num_languages = num_languages
        self.obs_area = (math.pi*obstacle_radius) ** 2
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
    
    # Split lines that intersect with each other
    def split_lines(self, lines, idx=0):
        if idx == len(lines):
            return lines
        
        split_lines, garbage_lines = [], []
        line_0 = lines[idx]
        for line_1 in lines:
            try:
                result = split(line_0, line_1)
                split_lines.extend([*result.geoms])
                if len(result.geoms) == 2:
                    garbage_line = line_0 if line_0 in self.boundaries else line_1
                    garbage_lines.append(garbage_line)
            except ValueError:
                if line_0 == line_1:
                    continue
                
                if line_0.contains(line_1):
                    garbage_lines.append(line_0)
                    difference = line_0.difference(line_1)
                    split_lines.append(difference)
                elif line_1.contains(line_0):
                    garbage_lines.append(line_1)
                    difference = line_1.difference(line_0)
                    split_lines.append(difference)
                # TODO: Lines are parallel, but neither consumes the other
                else:
                    a=3
        
        split_lines = list(dict.fromkeys(split_lines))
        garbage_lines = list(dict.fromkeys(garbage_lines))
        split_lines = [line for line in split_lines if line not in lines]
        lines[idx:idx] = split_lines
        lines = [line for line in lines if line not in garbage_lines]
        idx += 1 if not garbage_lines else 0
        lines = self.split_lines(lines, idx)
        return lines
            
    # Both endpoints must be on an environment boundary to be considered valid
    def _get_line_info(self, lines):
        valid_lines = [*self.boundaries]
        
        # Get valid lines s.t. both endpoints are on an environment boundary
        for line in lines:
            intersection = self.square.intersection(line)
            if not intersection.is_empty and np.any(np.abs([*intersection.coords]) == 1, axis=1).all():
                valid_lines.append(intersection)
                plt.plot(*intersection.xy)

        plt.plot(*self.square.exterior.xy)
        plt.show()
        
        split_lines = self.split_lines(valid_lines)        
        return split_lines        
            
    # Create regions from valid lines
    def _create_regions(self, lines):
        split_lines = self._get_line_info(lines)
        regions = list(polygonize(split_lines))
        print(len(regions))
        return regions

    # Cost function to minimize:
    # 1. Mean and variance of collision probability
    # 2. Mean and variance of non-navigable area
    def _optimizer(self, lines):
        lines = [LineString([tuple(lines[i:i+2]), tuple(lines[i+2:i+4])]) 
                 for i in range(0, len(lines), 4)]
        regions = self._create_regions(lines)
        
        collision_prob, nonnavigable = [], []
        # Obstacle(s) are constrained to be in the top right quadrant
        for _ in range(self.configs_to_consider):
            region_prob, obs_prob, unsafe = 0, 0, 0
            obs_pos = np.random.rand(self.num_obstacles, 2)
            obs_list = [Point(obs_pos[i]) for i in range(self.num_obstacles)]
            
            for obs, region in product(obs_list, regions):
                if region.contains(obs):
                    region_prob += (region.area / self.square.area)
                    obs_prob += (self.obs_area / region.area)
                    unsafe += region.area
            collision_prob.append(obs_prob / region_prob)
            nonnavigable.append(unsafe)
        
        # TODO: Standardize values
        col_mu = mean(collision_prob)
        col_var = variance(collision_prob)
        nav_mu = mean(nonnavigable)
        nav_var = variance(nonnavigable)
        cost = col_mu + col_var + nav_mu + nav_var
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
        lines = self._generate_lines()
        regions = self._create_regions(lines)
        return regions