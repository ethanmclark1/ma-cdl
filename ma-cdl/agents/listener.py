import itertools
import numpy as np
import networkx as nx

from languages.utils.cdl import CDL


class Listener:
    def __init__(self):
        self.sum = None
        self.has_ace = None
    
    def set_state(self, sum, has_ace):
        self.sum = sum
        self.has_ace = has_ace

    