import numpy as np

class CosineSimilarityLSH():
    """
    Locally-sensitive hash function based on cosine similarity.
    Implemented with random hyperplane projections.
    """

    def __init__(self, dim, num_bits, seed=2025):
        """
        Args:
            dim (int): Dimension of input vectors.
            num_bits (int): Number of bits in the hash.
        """
        self.dim = dim
        self.num_bits = num_bits
        # Random hyperplanes
        rng = np.random.default_rng(seed=seed)
        self.hyperplanes = rng.standard_normal(size=(num_bits, dim))

    def hash(self, v):
        """
        Return an integer hash code produced by num_bits sign comparisons
        """
        projections = self.hyperplanes @ v
        bits = (projections >= 0).astype(np.uint8)
        # Convert bit array to integer
        return int("".join(map(str, bits)), 2)