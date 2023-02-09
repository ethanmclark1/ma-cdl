import math
import numpy as np

from itertools import product
from scipy.spatial import KDTree

class Language():
    def __init__(self, num_cols):
        self.num_layers = 3
        self.num_cols = num_cols
        self.reference_points, self.kd_trees = self._create_hierarchy()
        
    # Dissect specified region into smaller regions then find center point(s) of each region
    def _anatomize(self, num_cols):
        map_length = 2
        origin_points = []
        delta = map_length / num_cols
        point = -1 + delta
        while round(point, 8) < 1:
            origin_points.append(round(point, 8))
            point += delta
        return origin_points
    
    # Ensure existing critical points does not carry over to new layer of critical points
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
            origin_points = self._anatomize(self.num_cols + i)
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
    
    # Split search interval into half then compare with position to find region
    def _get_region(self, pos, origin_points):
        references_per_col = self.num_cols - 1
        origin_points = self._reshape(origin_points, references_per_col)   
        for (col_num, col), idx in product(enumerate(origin_points), range(references_per_col)):
            # N-1 columns
            if (pos > col[idx]).all():
                region = self.num_cols*col_num + idx + 1
                break
            elif (pos > col[-1-idx])[0] and (pos < col[-1-idx])[1]:
                region = self.num_cols*(col_num+1) - idx
                break
            
            # Nth column
            if col_num == len(origin_points) - 1:
                if (pos < col[idx])[0] and (pos > col[idx])[1]:
                    region = self.num_cols*(col_num+1) + idx + 1
                    break
                elif (pos < col[-1-idx]).all():
                    region = self.num_cols*(col_num+2) - idx
                    break
                
        return region
    
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
    def _localize(self, pos, obstacles, origin_points, symbols=[]):
        pos_region = self._get_region(pos, origin_points)
        obs_regions = [self._get_region(obs, origin_points) for obs in obstacles]
        
        if pos_region not in obs_regions:
            symbols.append(pos_region)            
        else:
            origin_points, limit_reached = self._get_origin(pos, origin_points)
            if not limit_reached:                    
                symbols = self._localize(pos, obstacles, origin_points, symbols)
                symbols.insert(0, pos_region)
            
        return symbols
    
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
    
    # Gather directions from path to goal using alphabet symbols
    def direct(self, path, obstacles):
        symbol_list = []
        origin_points = self.reference_points[0]
        for pos in path:
            symbols = self._localize(pos, obstacles, origin_points)
            symbols += ["END"]
            if not symbol_list:
                symbol_list.extend(symbols)
            elif symbol_list[-len(symbols):] != symbols:
                symbol_list.extend(symbols)
            symbols.clear()

        directions = self._translate(symbol_list)
        return directions
    

def LanguageFactory(min_symbols, max_symbols):
    language_set = {}
    for i in range(min_symbols, max_symbols + 1):
        language_set[i] = Language(i)
    return language_set
