import numpy as np
import hashlib

from filters.cosine_lsh import CosineSimilarityLSH

class RACESketch:
    def __init__(self, dim, num_hash, table_length, bits_per_lsh, threshold=0.1, seed=2025):
        """
        Args:
            dim (int): dimensions of input vectors
            num_hash (int): number of independent LSH functions
            table_length (int): length of each hash table
            bits_per_lsh (int): number of bits per LSH signature
            threshold (float): density threshold for membership query
        """
        self.dim = dim
        self.num_hash = num_hash
        self.table_length = table_length
        self.threshold = threshold

        # Hash tables
        self.sketch = np.zeros((num_hash, table_length))

        rng = np.random.default_rng(seed=seed)

        # Create num_hash independent LSH functions
        self.lsh_functions = [
            CosineSimilarityLSH(dim=dim, num_bits=bits_per_lsh, seed=rng.integers(low=0, high=100000))
            for _ in range(num_hash)
        ]
    
    def _lsh_hash_to_index(self, hashcode):
        """
        Map LSH hashcode value to index in bitarray
        """
        digest = hashlib.sha256(str(hashcode).encode()).digest()
        return int.from_bytes(digest[:4], "little") % self.table_length

    def insert(self, v):
        """
        Insert vector v into RACE Sketch
        """
        hashcodes = [lsh.hash(v) for lsh in self.lsh_functions]
        idxs = [self._lsh_hash_to_index(h) for h in hashcodes]
        self.sketch[np.arange(0, self.num_hash), idxs] += 1
    
    def query(self, v):
        """
        Query if a vector similar to v was inserted
            Returns True if estimated density is above threshold
            Returns False otherwise
        """
        hashcodes = [lsh.hash(v) for lsh in self.lsh_functions]
        idxs = [self._lsh_hash_to_index(h) for h in hashcodes]
        density_total = np.sum(self.sketch[np.arange(0, self.num_hash), idxs])
        density_est = density_total / np.sum(self.sketch)
        return density_est >= self.threshold
