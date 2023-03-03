import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize
from statistics import mean, variance
from shapely.ops import polygonize, split
from shapely.geometry import Point, LineString, MultiLineString, Polygon

class Language:
    def __init__(self, args):
        corners = list(product((1, -1), repeat=2))
        self.configs_to_consider = 10
        self.num_obstacles = args.num_obstacles
        self.num_languages = args.num_languages
        self.obs_area = (math.pi*args.obstacle_size) ** 2
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
    
    # Split lines that intersect with each other
    def _split_lines(self, lines, idx=0):
        lines = MultiLineString(lines)
        lines = lines.buffer(0.000000000001)
        boundary = lines.convex_hull
        multipolygons = boundary.difference(lines)
        print(len(multipolygons.geoms))
        a=3
        # if idx == len(lines):
        #     return lines

        # split_lines, garbage_lines = [], []
        # line_0 = lines[idx]
        # for line_1 in lines:
        #     try:
        #         result = split(line_0, line_1)
        #         split_lines.extend([*result.geoms])
        #         if len(result.geoms) == 2:
        #             garbage_line = line_0 if line_0 in self.boundaries else line_1
        #             garbage_lines.append(garbage_line)
        #     except ValueError:
        #         if line_0 == line_1:
        #             continue
        #         contains_0 = line_0.contains(line_1)
        #         contains_1 = line_1.contains(line_0)
        #         crosses = line_0.crosses(line_1)
        #         if contains_0 or contains_1:
        #             consumer, consumed = (line_0, line_1) if contains_0 else (line_1, line_0)
        #             difference = consumer.difference(consumed)
        #             split_lines.append(difference)
        #             garbage_lines.append(consumer)
        #         elif not crosses:
        #             intersection = line_0.intersection(line_1)
        #             split_lines.append(intersection)
        #         else:
        #             a=3
        
        # split_lines = list(dict.fromkeys(split_lines))
        # garbage_lines = list(dict.fromkeys(garbage_lines))
        # split_lines = [line for line in split_lines if line not in lines]
        # lines[idx:idx] = split_lines
        # lines = [line for line in lines if line not in garbage_lines]
        # idx += 1 if not garbage_lines else 0
        # lines = self._split_lines(lines, idx)
        # return lines
            
    # Both endpoints must be on an environment boundary to be considered valid
    def _get_valid_lines(self, lines):
        valid_lines = [*self.boundaries]
        
        # Get valid lines s.t. both endpoints are on an environment boundary
        for line in lines:
            intersection = self.square.intersection(line)
            if not intersection.is_empty and np.any(np.abs([*intersection.coords]) == 1, axis=1).all():
                valid_lines.append(intersection)
                plt.plot(*intersection.xy)

        plt.plot(*self.square.exterior.xy)
        plt.show()
        
        return valid_lines        
            
    # Create polygonal regions from lines
    def _create_regions(self, lines):
        valid_lines = self._get_valid_lines(lines)
        split_lines = self._split_lines(valid_lines)
        regions = list(polygonize(split_lines))
        print(len(regions))
        return regions

    """ 
    Cost function to minimize:
        1. Mean and variance of collision probability
        2. Mean and variance of non-navigable area
        3. Variance of region area 
    """
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
        
        collision_mu = mean(collision_prob)
        collision_var = variance(collision_prob)
        navigable_mu = mean(nonnavigable)
        navigable_var = variance(nonnavigable)
        region_var = variance([region.area for region in regions])
        # TODO: Linear combination of cost function
        cost = collision_mu + collision_var + navigable_mu + navigable_var
        return cost

    # Minimizes cost function to generate lines
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
    
    # Returns regions that define the language
    def create(self):
        lines = self._generate_lines()
        regions = self._create_regions(lines)
        return regions