import numpy as np
import hashlib

from filters.cosine_lsh import CosineSimilarityLSH

class LocallySensitiveBloomFilter:
    """
    A locally-sensitive bloom filter using cosine similarity LSH.
    """
    def __init__(self, dim, m=100000, k=8, bits_per_lsh=16, seed=2025):
        """
        Args:
            dim (int): dimension of input vectors
            m (int): number of bits in bloom filter
            k (int): number of independent LSH functions
            bits_per_lsh (int): number of bits per LSH signature
        """
        self.dim = dim
        self.m = m
        self.k = k
        self.bits_per_lsh = bits_per_lsh

        # Bloom filter bit array
        self.bitarray = np.zeros(m, dtype=bool)

        rng = np.random.default_rng(seed=seed)

        # Create k independent LSH functions
        self.lsh_functions = [
            CosineSimilarityLSH(dim=dim, num_bits=bits_per_lsh, seed=rng.integers(low=0, high=100000))
            for _ in range(k)
        ]
    
    def _lsh_hash_to_index(self, hashcode):
        """
        Map LSH hashcode value to index in bitarray
        """
        digest = hashlib.sha256(str(hashcode).encode()).digest()
        return int.from_bytes(digest[:4], "little") % self.m

    def insert(self, v):
        """
        Insert vector v into bloom filter
        """
        for lsh in self.lsh_functions:
            hashcode = lsh.hash(v)
            idx = self._lsh_hash_to_index(hashcode)
            self.bitarray[idx] = True
    
    def query(self, v):
        """
        Query if a vector similar to v was inserted
        """
        for lsh in self.lsh_functions:
            hashcode = lsh.hash(v)
            idx = self._lsh_hash_to_index(hashcode)
            if not self.bitarray[idx]:
                return False
        return True
