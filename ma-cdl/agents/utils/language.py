import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from statistics import variance
from scipy.spatial import Delaunay
from scipy.optimize import minimize
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
        valid_lines = []
        segmented_lines = [*self.lines]

        # Get valid lines s.t. both endpoints are on an environment boundary
        for line in lines:
            intersection = self.shape.intersection(line)
            if not intersection.is_empty and np.any(np.abs([*intersection.coords]) == 1, axis=1).all():
                valid_lines.append(intersection)
                plt.plot(*intersection.xy)

        plt.plot(*self.shape.exterior.xy)
        plt.show()
        
        # Segment lines based on intersections with environment boundaries
        for valid, bound in product(valid_lines, segmented_lines):
            intersection = valid.intersection(bound)
            if intersection.is_empty or intersection in self.points:
                continue
            line_0 = LineString([bound.coords[0], intersection])
            line_1 = LineString([intersection, bound.coords[1]])
            segmented_lines.remove(bound)
            segmented_lines += [line_0, line_1]
            
        # Segment lines based on intersections with other valid lines
            
        return segmented_lines
    
    def _regionalize(self, line, critical_points, valid_lines):
        a=3
            
            
    # Create regions from valid lines and critical points            
    def _create_regions(self, lines):
        # Counter clockwise sort of starting points of critical lines
        def sort(x):
            atan = math.atan2(x[1], x[0])
            return (atan, x[0]**2+x[1]**2) if atan >= 0 else (2*math.pi + atan, x[0]**2+x[1]**2)

        regions = []
        segmented_lines = self._get_line_info(lines)
        segmented_lines = sorted(segmented_lines, key=lambda x: sort(x.coords[0]))

        # Recursively find boundaries of regions by traversing the perimeter
        for idx, line in enumerate(segmented_lines, start=1):
            regions += [self._regionalize(line, segmented_lines[idx:])]

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