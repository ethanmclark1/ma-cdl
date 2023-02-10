import math
import numpy as np

from shapely import Polygon
from itertools import product
from scipy.spatial import KDTree

class Language():
    def __init__(self, num_cols):
        self.num_layers = 3
        self.num_cols = num_cols
        self.reference_points, self.kd_trees = self._create_hierarchy()
        
    # Dissect specified region into smaller regions then find center point(s) of each region
    def _anatomize(self, num_cols, layer):
        map_length = 2
        origin_points = []
        delta = map_length / num_cols ** layer
        point = -1 + delta
        while round(point, 8) < 1:
            origin_points.append(round(point, 8))
            point += delta
        return origin_points
    
    # Ensure existing critical points do not carry over to new layer of critical points
    def _remove_duplicates(self, origin_points, prev_origin_points):
        # Convert to np.float32 to prevent inequalities due to precision errors
        temp_origin_points = np.array(origin_points, np.float32)
        indices = np.argwhere(np.isin(temp_origin_points, prev_origin_points))
        return self._sort(list(np.delete(origin_points, indices)))
        
    # Order center points in clockwise fashion
    def _sort(self, origin_points):
        ordered = []
        half = len(origin_points) / 2
        while len(origin_points) > half:
            ordered.append(origin_points.pop(origin_points.index(max(origin_points))))
        while origin_points: 
            ordered.append(origin_points.pop(origin_points.index(min(origin_points))))
        return ordered
    
    # Generate reference points for each layer to construct hierachy
    def _create_hierarchy(self):
        prev_origin_points = np.array(())
        reference_points, kd_trees = [], []
        
        for i in range(self.num_layers):
            origin_points = self._anatomize(self.num_cols, i+1)
            origin_points = self._remove_duplicates(origin_points, prev_origin_points)  
            prev_origin_points = np.concatenate([prev_origin_points, origin_points], axis=0, dtype=np.float32)
            # Cartesian product to create new reference points for each layer
            reference_points.append(list(product(origin_points, repeat=2)))
            kd_trees.append(KDTree(reference_points[i]))
            
        return reference_points, kd_trees
    
    # Reshape origin_points into 2d array with size of (num_cols, references_per_col)
    def _reshape(self, origin_points, references_per_col):
        i = []
        start, end = 0, references_per_col
        for _ in range(references_per_col):
            j = [origin_points[reference] for reference in range(start, end)]
            i.append(j)
            start += references_per_col
            end += references_per_col
        return i
    
    # Split search interval into half then compare with position to find region label of position in square
    def _get_region(self, pos, origin_points):
        references_per_col = self.num_cols - 1
        origin_points = self._reshape(origin_points, references_per_col)   
        dist = origin_points[0][0][1] - origin_points[0][1][1]

        for (col_num, col), idx in product(enumerate(origin_points), range(references_per_col)):
            # N-1 columns
            if (pos > col[idx]).all():
                label = self.num_cols*col_num + idx + 1
                bottom_left_corner = col[idx]
                bottom_right_corner = (col[idx][0] + dist, col[idx][1])
                top_left_corner = (col[idx][0], col[idx][1] + dist)
                top_right_corner = (col[idx][0] + dist, col[idx][1] + dist)
                break
            elif (pos > col[-1-idx])[0] and (pos < col[-1-idx])[1]:
                label = self.num_cols*(col_num+1) - idx
                bottom_left_corner = (col[-1-idx][0], col[-1-idx][1] - dist)
                bottom_right_corner = (col[-1-idx][0] + dist, col[-1-idx][1] - dist)
                top_left_corner = col[-1-idx]
                top_right_corner = (col[-1-idx][0] + dist, col[-1-idx][1])
                break
            
            # Nth column
            if col_num == len(origin_points) - 1:
                if (pos < col[idx])[0] and (pos > col[idx])[1]:
                    label = self.num_cols*(col_num+1) + idx + 1
                    bottom_left_corner = (col[idx][0] - dist, col[idx][1])
                    bottom_right_corner = col[idx]
                    top_left_corner = (col[idx][0] - dist, col[idx][1] + dist)
                    top_right_corner = (col[idx][0], col[idx][1] + dist)
                    break
                elif (pos < col[-1-idx]).all():
                    label = self.num_cols*(col_num+2) - idx
                    bottom_left_corner = (col[-1-idx][0] - dist, col[-1-idx][1] - dist)
                    bottom_right_corner = (col[-1-idx][0], col[-1-idx][1] - dist)
                    top_left_corner = (col[-1-idx][0] - dist, col[-1-idx][1])
                    top_right_corner = col[-1-idx]
                    break
        
        bounds = [bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner]
        return label, bounds
    
    # Retrieve center point(s) of region given by proximity to position
    def _get_origin(self, pos, origin_points):
        try:
            for i, layer in enumerate(self.reference_points, start=1):
                if origin_points[0] in layer:
                    _, center_point_idx = self.kd_trees[i].query(pos)
                    center_point = self.reference_points[i][center_point_idx]
                    _, center_points_idx = self.kd_trees[i].query(center_point, k=len(origin_points))
                    if isinstance(center_points_idx, np.int64):
                        center_points_idx = [center_points_idx]
                    origin_points = [self.reference_points[i][idx] for idx in center_points_idx]
                    return self._sort(origin_points), False
        except IndexError:
            return -1, True
    
    # Create symbol list to append to direction set by traversing through hierarchy
    def _localize(self, pos, obstacles, origin_points, symbols=[], bounds=[]):
        pos_label, label_bound = self._get_region(pos, origin_points)
        obs_label = [self._get_region(obs, origin_points) for obs in obstacles][0]
        
        if pos_label not in obs_label:
            symbols.append(pos_label)     
            bounds.append(label_bound)
        else:
            origin_points, limit_reached = self._get_origin(pos, origin_points)
            if not limit_reached:                    
                symbols, bounds = self._localize(pos, obstacles, origin_points, symbols, bounds)
                symbols.insert(0, pos_label)
                bounds.insert(0, label_bound)
            
        return symbols, bounds
    
    # Clean direction_set by removing redundant prefixes and 'END' tokens
    # -1 indicate layer traversal, -2 indicates layer reversal, and [0-16] indicates movement
    def _translate(self, symbol_list):
        directions = []
        prev_seq, cur_seq = [], []
        for symbol in symbol_list:
            if symbol == 'END':
                # Layer reversal
                if prev_seq: 
                    min_len = min(len(prev_seq), len(cur_seq))
                    temp_prev, temp_cur = prev_seq[:min_len], cur_seq[:min_len]
                    diff_idx = [i for i, (a, b) in enumerate(zip(temp_prev, temp_cur)) if a != b]
                    if diff_idx and diff_idx[0] < len(prev_seq) - 1:
                        directions.extend([-2] * ((len(prev_seq) - diff_idx[0]) // 2))
                        
                for i, cur in enumerate(cur_seq):
                    if i >= len(prev_seq) or cur != prev_seq[i]:
                        directions.append(cur)
                prev_seq = cur_seq
                cur_seq = []
            else:
                # Layer traversal
                cur_seq.extend((-1, symbol)) if cur_seq else cur_seq.append(symbol)

        return directions
    
    def _build_polygons(self, bounds_list):
        polygons = []
        for bounds in bounds_list:
            polygons.append(Polygon(bounds))
        return polygons
    
    # Gather directions from path to goal using alphabet symbols
    def direct(self, path, obstacles):
        bounds_list = []
        symbol_list = []
        origin_points = self.reference_points[0]
        for pos in path:
            symbols, bounds = self._localize(pos, obstacles, origin_points)
            symbols += ["END"]
            if not symbol_list:
                symbol_list.extend(symbols)
                bounds_list.extend(bounds)
            elif symbol_list[-len(symbols):] != symbols:
                symbol_list.extend(symbols)
                bounds_list.extend(bounds)
            symbols.clear()
            bounds.clear()

        directions = self._translate(symbol_list)
        polygons = self._build_polygons(bounds_list)
        return directions, polygons
    

def LanguageFactory(min_symbols, max_symbols):
    language_set = {}
    for i in range(min_symbols, max_symbols + 1):
        language_set[i] = Language(i)
    return language_set
