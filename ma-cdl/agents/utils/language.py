import math
import numpy as np
import matplotlib.pyplot as plt

from shapely import plotting
from itertools import product
from statistics import variance
from scipy.optimize import minimize
from shapely.geometry import Point, LineString, MultiLineString, Polygon

class Language:
    def __init__(self, num_obstacles, num_languages):
        self.obs_size = 0.02
        self.num_obstacles = num_obstacles
        self.num_languages = num_languages
        self.boundaries = Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1)])
        
    def _create_regions(self, lines):
        def sort(x):
            atan = math.atan2(x[1], x[0])
            return (atan, x[0]**2+x[1]**2) if atan >= 0 else (2*math.pi + atan, x[0]**2+x[1]**2)

        valid_intersections = []
        regions, region_points = [], {}
        boundaries = [*self.boundaries.exterior.coords]
        intersections = self.boundaries.intersection(lines)

        plt.plot(*self.boundaries.exterior.xy)
        plotting.plot_line(intersections)
        plt.show()

        # Check for intersections
        if intersections:
            if intersections.geom_type == 'MultiLineString':
                intersections = [[*intersections.geoms[i].coords]
                                 for i in range(len(intersections.geoms))]
            else:
                intersections = [[*intersections.coords]]

            # Valid intersection must be on the environment boundary
            for point in intersections:
                if np.any(np.abs(point) == 1, axis=1).all():
                    valid_intersections.append(point)
            
            # TODO: Check for valid lines that intersect

        # TODO: Do not create duplicate regions from different perspectives
        for valid in valid_intersections:
            region_points[0], region_points[1] = set(valid), set(valid)
            for boundary in boundaries:
                # Cross product to find which side of the line the boundary is on
                v1 = (valid[1][0] - valid[0][0], valid[1][1] - valid[0][1])
                v2 = (valid[1][0] - boundary[0], valid[1][1] - boundary[1])
                xp = v1[0]*v2[1] - v1[1]*v2[0]
                idx = 0 if xp > 0 else 1
                region_points[idx].add(boundary)
            regions.extend(([Polygon(sorted(list(region_points[idx]), key=sort))
                           for idx in range(len(region_points))]))

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
        lines = MultiLineString(lines)
        regions = self._create_regions(lines)

        if regions:
            # 1. Probability of colliding into an obstacle
            for obs, region in product(obs_list, regions):
                if region.contains(obs):
                    region_prob += (region.area / self.boundaries.area)
                    obs_prob += (self.obs_size / region.area)
                    nonnavigable.append(region.area)
            collision_prob = region_prob * obs_prob
            # 2. Variance on region area
            region_var = variance([region.area for region in regions])
            # 3. Amount of navigable space
            navigable_space = self.boundaries.area - sum(nonnavigable)
            # TODO 4. Variance on navigable space across problems
            navigable_space_var = 3
        
            cost = 0.3*collision_prob + 0.15*region_var + (-0.3*navigable_space) + 0.25*navigable_space_var
            
        return cost

    def _generate_lines(self):
        bounds = (-3, 3)
        optim_val, optim_coeffs = math.inf, None
        for num in range(2, self.num_languages+2):
            x0 = (bounds[1] - bounds[0])*np.random.rand(num, 4)+bounds[0]
            res = minimize(self._optimizer, x0, method='nelder-mead',
                           options={'xatol': 1e-8})

            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x

        optim_coeffs = np.reshape(optim_coeffs, (-1, 4))
        return optim_coeffs
    
    def get_langauge(self):
        return self._generate_lines()