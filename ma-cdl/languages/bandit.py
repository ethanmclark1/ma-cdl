from languages.utils.cdl import CDL

"""Infinitely Armed Bandit"""
# TODO: Entire thing
class Bandit(CDL):
    def __init__(self, agent_radius):
        super().__init__(agent_radius)