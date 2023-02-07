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
    
    def _get_region(self, pos, origin_points):
        references_per_col = self.num_cols - 1
        start = col_count = 0
        end = references_per_col
        while end <= len(origin_points):
            col_points = origin_points[start:end]
            for i in range(references_per_col):
                if (pos > col_points[0]).all():
                    region = 1 + col_count * self.num_cols
                    break
                elif (pos < origin_points[-1]).all():
                    region = 2 + col_count * self.num_cols
                    break
            col_count += 1
            start = end
            end += references_per_col

        if (pos > origin_points[0]).all():
            region = 1
        elif (pos > origin_points[0])[0] and (pos > origin_points[-1])[1]:
            reigon = 2
        elif (pos < origin_points[0])[0] and (pos > origin_points[-1])[1]:
            region = 3
        elif (pos < origin_points[-1]).all():
            region = 4
        
        # Nonant
        if (pos > origin_points[0]).all():
            region = 1
        elif (pos > origin_points[0])[0] and (pos > origin_points[1])[1]:
            region = 2
        elif (pos > origin_points[0])[0]:
            region = 3
        elif (pos > origin_points[2]).all():
            region = 4
        elif (pos > origin_points[2])[0] and (pos > origin_points[3])[1]:
            region = 5
        elif (pos > origin_points[2])[0]:
            region = 6
        elif (pos < origin_points[2])[0] and (pos > origin_points[2])[1]:
            region = 7
        elif (pos < origin_points[2])[0] and (pos > origin_points[3])[1]:
            region = 8
        elif (pos < origin_points[3]).all():
            region = 9

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
    # 0 indicate layer traversal, -1 indicates layer reversal, and [0-16] indicates movement
    def _translate(self, symbol_list):
        directions = []
        prev_sequence, cur_sequence = [], []
        for symbol in symbol_list:
            if symbol == 'END':
                # TODO: Layer reversal
                # Find where two lists differ and check if the index in which they differ is less than that of the previous list
                # If it is not, then add -1 to the end of direction list (len(prev_list) - diff) / 2 amount of times                    
                for i, cur in enumerate(cur_sequence):
                    if i >= len(prev_sequence) or cur != prev_sequence[i]:
                        directions.append(cur)
                prev_sequence = cur_sequence
                cur_sequence = []
            else:
                if cur_sequence:
                    cur_sequence.extend((0, symbol))
                else:
                    cur_sequence.append(symbol)

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
    

def LanguageFactory(max_symbols):
    language_set = {}
    for i in range(2, max_symbols + 1):
        language_set[i] = Language(i)
    return language_set
