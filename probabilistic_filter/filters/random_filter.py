import numpy as np

class RandomFilter:
    """
    Args:
        rate (int): percentage of incoming queries that return positive
    """
    def __init__(self, rate, seed=2025):
        self.rate = rate
        self.rng = np.random.default_rng(seed=seed)
    
    def insert(self, v):
        pass

    def query(self, v):
        return self.rng.uniform(0,1) > self.rate