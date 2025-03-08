import numpy as np
import cupy as cp  # GPU acceleration
from numba import jit
import h5py  # For storing massive HDC vectors
from tqdm import tqdm
from scipy.stats import special_ortho_group
import logging
import os
from enum import Enum, auto
from functools import lru_cache
from collections import defaultdict
import torch
import argparse
import sys
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
import math
import psutil


class HDCVectorSpace:
    """Base class for hyperdimensional computing vector spaces."""

    def __init__(
        self,
        dimension: int = None,
        device: str = "auto",
        vector_type: str = "bipolar",
        seed: int = None,
        data_size: int = None,  # New parameter
    ):
        """Initialize HDC vector space.

        Args:
            dimension: Vector dimension (auto-determined if None and data_size is provided)
            device: 'gpu', 'cpu', or 'auto'
            vector_type: 'bipolar' or 'binary'
            seed: Random seed for reproducibility
            data_size: Estimated size of the data (for auto-determining dimension)
        """
        # Auto-select device
        self.device = device.lower()
        if self.device == "auto":
            self.device = "gpu" if cp.is_available() else "cpu"

        # Set compute module
        self.xp = cp if self.device == "gpu" and cp.is_available() else np
        if self.device == "gpu" and self.xp is np:
            logger.warning("GPU requested but not available. Falling back to CPU.")
            self.device = "cpu"

        # Set random seed
        if seed is not None:
            self.xp.random.seed(seed)
            np.random.seed(seed)

        # Set vector type
        self.vector_type = vector_type

        # Dimension auto-tuning
        if dimension is not None:
            self.dim = dimension
        elif data_size is not None:
            self.dim = get_optimal_dimension(data_size)  # Use utility function
        else:
            self.dim = 5000  # Keep a default, but it's now *last* resort.

        # Cache system
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        # Create base vectors
        self.is_initialized = False

    def initialize(self, elements: List[str] = None):
        """Initialize the vector space with base elements."""
        if elements is None:
            elements = ["A", "T", "G", "C"]  # DNA bases by default

        # Create orthogonal base vectors
        self.base_vectors = self._create_orthogonal_vectors(elements)

        # Create position vectors
        max_pos = 30  # Support up to 30-mers
        self.position_vectors = self._create_orthogonal_vectors(
            [f"pos_{i}" for i in range(max_pos)],
            orthogonal_to=list(self.base_vectors.values()),
        )

        self.is_initialized = True

    def _create_orthogonal_vectors(
        self, elements: List[str], orthogonal_to: List[Any] = None
    ) -> Dict[str, Any]:
        """Create set of orthogonal vectors."""
        total_vectors = len(elements)
        if orthogonal_to:
            total_vectors += len(orthogonal_to)

        # Check if we can create perfectly orthogonal vectors
        if self.dim >= total_vectors:
            # Use special orthogonal group for perfect orthogonality
            vectors = {}

            # Either extend existing orthogonal set or create new one
            if orthogonal_to:
                # Get size of existing set
                n_existing = len(orthogonal_to)

                # First create a full orthogonal matrix
                full_ortho = special_ortho_group.rvs(total_vectors)

                # Extract existing vectors
                existing_matrix = np.zeros((n_existing, total_vectors))
                for i, vec in enumerate(orthogonal_to):
                    # Extract the main components
                    vec_np = vec.get() if hasattr(vec, "get") else vec
                    existing_matrix[i, : min(len(vec_np), total_vectors)] = vec_np[
                        :total_vectors
                    ]

                # Perform Gram-Schmidt to get vectors orthogonal to existing ones
                remaining_vectors = []
                for i in range(total_vectors):
                    v = full_ortho[i]
                    # Make v orthogonal to existing vectors
                    for existing in existing_matrix:
                        v = v - np.dot(v, existing) * existing
                    # Normalize
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:  # Only keep if not close to zero
                        v /= norm
                        remaining_vectors.append(v)
                        if len(remaining_vectors) >= len(elements):
                            break

                # Create vectors from the orthogonal set
                for i, element in enumerate(elements):
                    if i < len(remaining_vectors):
                        # Pad to full dimension
                        v = np.zeros(self.dim)
                        v[:total_vectors] = remaining_vectors[i]
                        vectors[element] = self.xp.array(v)
            else:
                # Fresh orthogonal set
                ortho_matrix = special_ortho_group.rvs(total_vectors)

                for i, element in enumerate(elements):
                    # Pad to full dimension
                    v = np.zeros(self.dim)
                    v[:total_vectors] = ortho_matrix[i]
                    vectors[element] = self.xp.array(v)
        else:
            # Not enough dimensions, create quasi-orthogonal vectors
            vectors = self._create_quasiorthogonal_vectors(elements, orthogonal_to)

        # Normalize all vectors
        return {k: self._normalize(v) for k, v in vectors.items()}

    def _create_quasiorthogonal_vectors(
        self, elements: List[str], orthogonal_to: List[Any] = None
    ) -> Dict[str, Any]:
        """Create quasi-orthogonal vectors using random projections."""
        vectors = {}

        # Start with random vectors
        for element in elements:
            v = self._random_vector()

            # Make somewhat orthogonal to existing vectors
            if orthogonal_to:
                for existing in orthogonal_to:
                    # Gram-Schmidt process
                    dot_product = self.xp.dot(v, existing)
                    v = v - dot_product * existing

            vectors[element] = v

        return vectors

    def _random_vector(self) -> Any:
        """Generate a random vector based on vector type."""
        if self.vector_type == "binary":
            return self.xp.random.randint(0, 2, size=self.dim)
        else:  # bipolar
            return self.xp.random.choice([-1, 1], size=self.dim)

    def _normalize(self, vector: Any) -> Any:
        """Normalize vector to unit length."""
        norm = self.xp.linalg.norm(vector)
        if norm < 1e-10:
            return vector  # Avoid division by zero
        return vector / norm

    def bind(self, v1: Any, v2: Any) -> Any:
        """Bind operation (element-wise multiplication for bipolar, XOR for binary)."""
        if self.vector_type == "binary":
            return self.xp.bitwise_xor(v1.astype(bool), v2.astype(bool)).astype(int)
        else:  # bipolar
            return v1 * v2

    def bundle(self, v1: Any, v2: Any, alpha: float = 0.5) -> Any:
        """Bundle operation (weighted average)."""
        return alpha * v1 + (1 - alpha) * v2

    def permute(self, vector: Any, shift: int = 1) -> Any:
        """Permute vector by circular shift."""
        return self.xp.roll(vector, shift)

    def similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two vectors."""
        return float(self.xp.dot(v1, v2))

    @jit(nopython=True, parallel=True)
    def _batch_similarity_cpu(self, query: Any, references: Any) -> Any:
        """Compute similarities between query and references (CPU optimized)."""
        results = np.zeros(len(references))
        for i in range(len(references)):
            results[i] = np.dot(query, references[i])
        return results

    def batch_similarity(self, query: Any, references: Any) -> Any:
        """Compute similarities between query and multiple reference vectors."""
        # Convert to array if list
        if isinstance(references, list):
            references = self.xp.array(references)

        # Select optimal implementation based on device
        if self.device == "gpu":
            return self.xp.dot(query, references.T)
        else:
            # Use numba-accelerated version for CPU
            if hasattr(self.xp, "asnumpy"):
                query_np = self.xp.asnumpy(query)
                refs_np = self.xp.asnumpy(references)
            else:
                query_np = query
                refs_np = references

            return self._batch_similarity_cpu(query_np, refs_np)

    def clear_cache(self):
        """Clear the vector cache."""
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}


class DNASupercomputer(HDCVectorSpace):
    def __init__(
        self,
        dimension=10000,
        device="gpu",
        vector_type="bipolar",
        init_method="orthogonal",
        seed=None,  # Add seed for reproducibility
    ):
        """Initialize a DNA supercomputer using HDC.

        Args:
            dimension: Dimensionality of HDC vectors
            device: 'gpu' or 'cpu'
            vector_type: 'bipolar' {-1,1} or 'binary' {0,1}
            init_method: 'random', 'orthogonal', or 'quasi-orthogonal'
            seed: Random seed
        """
        # Call the HDCVectorSpace constructor to set up core HDC properties
        super().__init__(dimension, device, vector_type, seed)

        self.init_method = init_method  # Store init_method

        logger.info(
            f"Initializing DNASupercomputer with {dimension}D vectors on {self.device}"
        )

        # Initialize base and position vectors using HDCVectorSpace methods
        self.initialize(elements=["A", "T", "G", "C"])  # Initialize base vectors
        # Initialize position vectors (more than needed to support variable k-mer sizes)
        max_kmer_size = 30  # Support up to 30-mers
        # Initialize within HDCVectorSpace
        self.initialize_positions(max_kmer_size)

        # Cache for k-mer encodings (using LRU cache for efficiency)
        self.kmer_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Sequence statistics for adaptive encoding
        self.seq_stats = {base: 0 for base in "ATGC"}
        self.total_bases = 0

        logger.info("Initialization complete")

    def initialize_positions(self, count):
        """Initialize position vectors within HDCVectorSpace."""
        position_elements = [f"pos_{i}" for i in range(count)]
        self.position_vectors = self._create_orthogonal_vectors(
            position_elements, orthogonal_to=list(self.base_vectors.values())
        )

    def bind(self, v1, v2):  # Override bind, but call super
        return super().bind(v1, v2)

    def bundle(self, v1, v2, alpha=0.5):  # Override bundle, but call super
        return super().bundle(v1, v2, alpha)

    def permute(self, vector, shift=1):  # Override permute, but call super
        return super().permute(vector, shift)

    def encode_kmer(self, kmer):
        """Encode a k-mer into an HDC vector."""
        if kmer in self.kmer_cache:
            self.cache_hits += 1
            return self.kmer_cache[kmer]

        self.cache_misses += 1

        # Fast path for single bases
        if len(kmer) == 1 and kmer in self.base_vectors:
            return self.base_vectors[kmer]

        result = self.xp.zeros(self.dim)
        for i, base in enumerate(kmer):
            if base not in self.base_vectors:
                # Handle non-standard bases (N, R, Y, etc.) by approximation
                if base == "N":  # Unknown base
                    # Average of all bases
                    bound = (
                        sum(
                            self.bind(
                                v,
                                self.position_vectors[
                                    f"pos_{i % len(self.position_vectors)}"
                                ],
                            )
                            for v in self.base_vectors.values()
                        )
                        / 4
                    )
                elif base in "RY":  # R = purine (A/G), Y = pyrimidine (C/T)
                    purines = ["A", "G"] if base == "R" else ["C", "T"]
                    bound = (
                        sum(
                            self.bind(
                                self.base_vectors[b],
                                self.position_vectors[
                                    f"pos_{i % len(self.position_vectors)}"
                                ],
                            )
                            for b in purines
                        )
                        / 2
                    )
                else:
                    # Fallback - treat as noise
                    bound = self.xp.zeros(self.dim)
            else:
                # Update statistics for adaptive encoding
                self.seq_stats[base] += 1
                self.total_bases += 1

                # Bind base with its position
                bound = self.bind(
                    self.base_vectors[base],
                    self.position_vectors[f"pos_{i % len(self.position_vectors)}"],
                )

            # Bundle into result
            result = self.bundle(result, bound, alpha=0.9)  # Emphasize new information

        result = self._normalize(result)
        self.kmer_cache[kmer] = result

        # Prune cache if it gets too big
        if len(self.kmer_cache) > 1000000:
            self._prune_cache()

        return result

    def _prune_cache(self):
        """Prune the k-mer cache to prevent memory issues."""
        logger.info(f"Pruning k-mer cache (size: {len(self.kmer_cache)})")
        # Keep only the k-mers with shorter lengths (more common)
        sorted_kmers = sorted(self.kmer_cache.keys(), key=len)
        to_keep = sorted_kmers[:500000]  # Keep half

        new_cache = {k: self.kmer_cache[k] for k in to_keep}
        self.kmer_cache = new_cache
        logger.info(f"Pruned cache to {len(self.kmer_cache)} entries")

    def encode_sequence(self, sequence, k=7, stride=1, chunk_size=1000):
        """Encode a DNA sequence using sliding k-mers, with chunking for large sequences."""
        if len(sequence) < k:
            return self.encode_kmer(sequence)

        result = self.xp.zeros(self.dim)
        n_kmers = 0

        # Process sequence in chunks to save memory
        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]

            for i in range(0, len(chunk) - k + 1, stride):
                kmer = chunk[i : i + k]

                # Skip k-mers with too many non-standard bases
                if kmer.count("N") > k // 2:
                    continue

                # Encode and bundle
                kmer_vector = self.encode_kmer(kmer)
                result = self.bundle(result, kmer_vector, alpha=0.9)
                n_kmers += 1

        if n_kmers == 0:
            logger.warning(f"No valid k-mers found in sequence: {sequence[:20]}...")
            return self.xp.zeros(self.dim)

        return self._normalize(result)

    @jit(nopython=True, parallel=True)
    def _similarity_kernel(self, v1, v2_array):
        """Compute similarities between one vector and many (JIT accelerated)."""
        results = np.zeros(len(v2_array))
        for i in range(len(v2_array)):
            results[i] = np.dot(v1, v2_array[i])
        return results

    def batch_similarity(self, query_vector, reference_vectors):
        return super().batch_similarity(query_vector, reference_vectors)

    def find_similar_sequences(self, query_seq, reference_seqs, k=7, top_n=5):
        """Find most similar sequences to the query."""
        query_vector = self.encode_sequence(query_seq, k=k)

        # Encode all reference sequences
        ref_vectors = []
        for seq in tqdm(reference_seqs, desc="Encoding references"):
            ref_vectors.append(self.encode_sequence(seq, k=k))

        # Calculate similarities
        similarities = self.batch_similarity(query_vector, ref_vectors)

        # Get top matches
        if self.device == "gpu":
            similarities = self.xp.asnumpy(similarities)

        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return [(reference_seqs[i], similarities[i]) for i in top_indices]

    def save_vectors(self, vectors, filename):
        """Save HDC vectors to disk using HDF5."""
        with h5py.File(filename, "w") as f:
            # Save base vectors
            base_group = f.create_group("base_vectors")
            for base, vector in self.base_vectors.items():
                # Convert to NumPy if on GPU
                vector_np = self.xp.asnumpy(vector) if self.device == "gpu" else vector
                base_group.create_dataset(base, data=vector_np)

            # Save position vectors
            pos_group = f.create_group("position_vectors")
            for pos_name, vector in self.position_vectors.items():
                # Convert to NumPy if on GPU
                vector_np = self.xp.asnumpy(vector) if self.device == "gpu" else vector
                pos_group.create_dataset(pos_name, data=vector_np)

    def load_vectors(self, filename):
        """Load HDC vectors from disk."""
        base_vectors = {}
        position_vectors = {}

        with h5py.File(filename, "r") as f:
            # Load base vectors
            if "base_vectors" in f:
                base_group = f["base_vectors"]
                for base in base_group:
                    vector = base_group[base][:]
                    if self.device == "gpu":
                        vector = self.xp.array(vector)
                    base_vectors[base] = vector

            # Load position vectors
            if "position_vectors" in f:
                pos_group = f["position_vectors"]
                for pos_name in pos_group:
                    vector = pos_group[pos_name][:]
                    if self.device == "gpu":
                        vector = self.xp.array(vector)
                    position_vectors[pos_name] = vector

        self.base_vectors = base_vectors
        self.position_vectors = position_vectors
        return self

    def get_cache_stats(self):
        """Return cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return {"hits": 0, "misses": 0, "ratio": 0}

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "ratio": self.cache_hits / total,
            "size": len(self.kmer_cache),
        }

    def build_linkage_map(self, snp_positions, similarity_matrix, threshold=0.7):
        """Build a linkage disequilibrium map using HDC vectors.

        Args:
            snp_positions: List of SNP positions
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for linkage

        Returns:
            NetworkX graph representing the linkage map
        """
        import networkx as nx

        G = nx.Graph()

        for i, pos1 in enumerate(snp_positions):
            G.add_node(pos1)
            for j, pos2 in enumerate(snp_positions[i + 1 :], i + 1):
                sim = similarity_matrix[i][j]
                if sim > threshold:
                    G.add_edge(pos1, pos2, weight=sim)

        return G


class BiologicalHDC(DNASupercomputer):
    """HDC encoding system that integrates biological knowledge into vector representations."""

    def __init__(self, dimension=10000, device="gpu", *args, **kwargs):
        super().__init__(dimension, device, *args, **kwargs)

        # Feature providers and their weights
        self.feature_providers = {}
        self.feature_weights = {}

        # Cache for feature vectors to avoid recalculation
        self.feature_cache = {}

        # Register the built-in feature providers
        self._register_default_feature_providers()

    def _register_default_feature_providers(self):
        """Register the default biological feature providers."""
        self.register_feature_provider(
            "gc_content", self._compute_gc_content, weight=0.5
        )
        self.register_feature_provider(
            "sequence_complexity", self._compute_complexity, weight=0.3
        )
        # Other providers are registered when data is loaded

    def register_feature_provider(self, name, provider_fn, weight=1.0):
        """Register a new biological feature provider.

        Args:
            name: Unique identifier for this feature
            provider_fn: Function that computes the feature value for a k-mer
            weight: Relative importance of this feature (0.0-1.0)
        """
        self.feature_providers[name] = provider_fn
        self.feature_weights[name] = weight
        logger.info(f"Registered feature provider: {name} with weight {weight}")

    def load_conservation_data(self, conservation_file, weight=0.8):
        """Load conservation scores and register as a feature provider.

        Args:
            conservation_file: File containing conservation scores
            weight: Importance weight for conservation
        """
        # Load conservation data (format depends on source)
        conservation_scores = self._load_conservation_file(conservation_file)

        # Create a conservation provider function
        def conservation_provider(kmer, position=None):
            if position is None:
                return 0.5  # Default if position unknown

            # Get scores for this region
            scores = [
                conservation_scores.get(position + i, 0.0) for i in range(len(kmer))
            ]
            return sum(scores) / len(scores) if scores else 0.5

        # Register the provider
        self.register_feature_provider("conservation", conservation_provider, weight)
        return self

    def load_annotations(self, annotation_file, weight=0.7):
        """Load genomic annotations and register as a feature provider.

        Args:
            annotation_file: File containing genomic annotations (GFF/GTF/BED)
            weight: Importance weight for annotations
        """
        # Load annotations (format depends on source)
        annotations = self._load_annotation_file(annotation_file)

        # Create annotation vectors for each type
        annotation_vectors = {}
        for ann_type in set(ann["type"] for ann in annotations.values()):
            # Generate a stable random vector for this annotation type
            seed_val = hash(ann_type) % 10000
            np.random.seed(seed_val)
            ann_vector = self._normalize(
                self.xp.array(np.random.uniform(-1, 1, self.dim))
            )
            annotation_vectors[ann_type] = ann_vector

        # Create an annotation provider function
        def annotation_provider(kmer, position=None):
            if position is None:
                return None  # Can't compute without position

            # Get annotations for this region
            region_annotations = []
            for i in range(len(kmer)):
                if position + i in annotations:
                    region_annotations.append(annotations[position + i])

            if not region_annotations:
                return None

            # Combine annotation vectors
            result = self.xp.zeros(self.dim)
            for ann in region_annotations:
                ann_type = ann["type"]
                if ann_type in annotation_vectors:
                    result = self.bundle(
                        result, annotation_vectors[ann_type], alpha=0.7
                    )

            return self._normalize(result)

        # Register the provider
        self.register_feature_provider("annotations", annotation_provider, weight)
        return self

    def load_epigenetic_data(self, epigenetic_file, weight=0.6):
        """Load epigenetic data and register as a feature provider.

        Args:
            epigenetic_file: File containing epigenetic data
            weight: Importance weight for epigenetic data
        """
        # Load epigenetic data (format depends on source)
        epigenetic_data = self._load_epigenetic_file(epigenetic_file)

        # Create an epigenetic provider function
        def epigenetic_provider(kmer, position=None):
            if position is None:
                return None  # Can't compute without position

            # Get epigenetic data for this region
            region_data = {}
            for i in range(len(kmer)):
                if position + i in epigenetic_data:
                    for key, value in epigenetic_data[position + i].items():
                        region_data[key] = region_data.get(key, 0) + value

            if not region_data:
                return None

            # Normalize values
            for key in region_data:
                region_data[key] /= len(kmer)

            # Create a vector based on epigenetic data
            result = self.xp.zeros(self.dim)
            for key, value in region_data.items():
                # Use a seeded random vector for stability
                seed_val = hash(key) % 10000
                np.random.seed(seed_val)
                feature_vec = self._normalize(
                    self.xp.array(np.random.uniform(-1, 1, self.dim))
                )
                # Scale by value
                feature_vec *= value
                result = self.bundle(result, feature_vec, alpha=0.7)

            return self._normalize(result)

        # Register the provider
        self.register_feature_provider("epigenetics", epigenetic_provider, weight)
        return self

    def _compute_gc_content(self, kmer, position=None):
        """Compute GC content for a k-mer."""
        gc_count = kmer.count("G") + kmer.count("C")
        return gc_count / len(kmer)

    def _compute_complexity(self, kmer, position=None):
        """Compute sequence complexity (Shannon entropy) for a k-mer."""
        from math import log2

        base_counts = {base: kmer.count(base) / len(kmer) for base in set(kmer)}
        entropy = -sum(p * log2(p) for p in base_counts.values() if p > 0)
        max_entropy = log2(min(4, len(kmer)))  # Maximum possible entropy
        return entropy / max_entropy if max_entropy > 0 else 0

    def encode_kmer(self, kmer, position=None):
        """Encode a k-mer with biological feature integration.

        Args:
            kmer: The k-mer to encode
            position: Genomic position of the k-mer (if known)

        Returns:
            HDC vector representing the k-mer
        """
        # Check cache first
        cache_key = (kmer, position)
        if cache_key in self.kmer_cache:
            return self.kmer_cache[cache_key]

        # Basic encoding
        result = self.xp.zeros(self.dim)

        # 1. First encode the sequence itself
        for i, base in enumerate(kmer):
            if base not in self.base_vectors:
                # Handle non-standard bases
                if base == "N":  # Unknown base
                    # Average of all bases
                    bound = (
                        sum(
                            self.bind(
                                v, self.position_vectors[i % len(self.position_vectors)]
                            )
                            for v in self.base_vectors.values()
                        )
                        / 4
                    )
                else:
                    # Skip unusual bases
                    continue
            else:
                # Bind base with position
                bound = self.bind(
                    self.base_vectors[base],
                    self.position_vectors[i % len(self.position_vectors)],
                )

            # Bundle
            result = self.bundle(result, bound, alpha=0.8)

        # 2. Integrate biological features
        feature_vector = self._compute_feature_vector(kmer, position)

        if feature_vector is not None:
            # Combine with base encoding
            result = self.bundle(result, feature_vector, alpha=0.6)

        # Normalize and cache
        result = self._normalize(result)
        self.kmer_cache[cache_key] = result
        return result

    def _compute_feature_vector(self, kmer, position=None):
        """Compute integrated feature vector for a k-mer."""
        # Check feature cache
        cache_key = (kmer, position)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = {}
        # Compute each feature
        for name, provider in self.feature_providers.items():
            feature_val = provider(kmer, position)
            if feature_val is not None:
                features[name] = feature_val

        if not features:
            return None

        # Combine vector-type features
        vector_features = {}
        scalar_features = {}

        for name, value in features.items():
            if isinstance(value, (self.xp.ndarray, np.ndarray)):
                vector_features[name] = value
            else:
                scalar_features[name] = value

        # Create a combined feature vector
        feature_vector = self.xp.zeros(self.dim)

        # 1. Integrate scalar features by binding with feature-specific vectors
        for name, value in scalar_features.items():
            # Get a stable random vector for this feature
            seed_val = hash(name) % 10000
            np.random.seed(seed_val)
            feature_vec = self._normalize(
                self.xp.array(np.random.uniform(-1, 1, self.dim))
            )

            # Scale by value and weight
            feature_vec *= value * self.feature_weights.get(name, 1.0)
            feature_vector = self.bundle(feature_vector, feature_vec, alpha=0.7)

        # 2. Integrate vector features
        for name, vec in vector_features.items():
            # Scale by weight
            weighted_vec = vec * self.feature_weights.get(name, 1.0)
            feature_vector = self.bundle(feature_vector, weighted_vec, alpha=0.7)

        # Cache and return
        normalized = self._normalize(feature_vector)
        self.feature_cache[cache_key] = normalized
        return normalized

    def encode_sequence(self, sequence, k=7, stride=1, chunk_size=None):
        """Encode a DNA sequence with biological feature integration."""
        # This actually calls our parent class implementation but passes position information
        return self._encode_sequence_with_position(sequence, k, stride, chunk_size)

    def _encode_sequence_with_position(self, sequence, k=7, stride=1, chunk_size=None):
        """Position-aware sequence encoding that passes genomic coordinates to feature providers."""
        import psutil

        # Quick path for short sequences
        if len(sequence) < k:
            return self.encode_kmer(sequence)

        # Dynamic chunk size based on available memory
        if chunk_size is None:
            available_memory = psutil.virtual_memory().available
            bytes_per_vector = self.dim * 8  # 64-bit floats
            chunk_size = min(
                len(sequence), int(0.5 * available_memory / bytes_per_vector)
            )

        # Process in chunks
        result = self.xp.zeros(self.dim)
        n_kmers = 0

        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]

            # Process k-mers in this chunk
            for i in range(0, len(chunk) - k + 1, stride):
                kmer = chunk[i : i + k]

                # Skip k-mers with too many non-standard bases
                if kmer.count("N") > k // 3:
                    continue

                # Encode with position information
                position = chunk_start + i  # Global position in sequence
                kmer_vector = self.encode_kmer(kmer, position)

                # Add to result
                result = self.bundle(result, kmer_vector, alpha=0.9)
                n_kmers += 1

        if n_kmers == 0:
            logger.warning(f"No valid k-mers found in sequence: {sequence[:20]}...")
            return self.xp.zeros(self.dim)

        return self._normalize(result)

    def _predict_motifs(self, kmer, position=None):
        """Predict motifs within a kmer using position weight matrices."""
        # Quickly check for common motifs without loading external data
        # This is a lightweight built-in feature that doesn't require file loading
        common_motifs = {
            "TATA": 0.8,  # TATA box
            "CAAT": 0.7,  # CAAT box
            "GATA": 0.6,  # GATA binding factor
            "CACGTG": 0.8,  # E-box
            "GCCNNNGGC": 0.7,  # GC box
            "TTGACA": 0.6,  # -35 region
            "TATAAT": 0.6,  # -10 region (pribnow box)
            "AATAAA": 0.7,  # Poly-A signal
            "CCAAT": 0.6,  # CCAAT box
            "GAGAG": 0.5,  # GAGA factor binding site
        }

        # Score the kmer against motifs
        scores = {}
        for motif, importance in common_motifs.items():
            if len(motif) > len(kmer):
                continue

            # Check for exact matches and fuzzy matches
            if motif in kmer:
                scores[motif] = importance
            else:
                # Simple fuzzy matching with one mismatch allowed
                for i in range(len(kmer) - len(motif) + 1):
                    subseq = kmer[i : i + len(motif)]
                    mismatches = sum(a != b for a, b in zip(subseq, motif) if b != "N")
                    if mismatches <= 1:
                        scores[motif] = importance * (1 - mismatches / len(motif))

        if not scores:
            return 0.0

        # Return the highest score
        return max(scores.values())

    def load_motif_data(self, motif_file, weight=0.7):
        """Load protein-binding motifs and register as a feature provider.

        Args:
            motif_file: File containing motif positions (e.g., BED format) or PWMs
            weight: Importance weight for motifs
        """
        # Load motif data
        motifs = self._load_motif_file(motif_file)

        # Create a motif provider function
        def motif_provider(kmer, position=None):
            if position is None:
                # If position unknown, use pattern matching
                return self._predict_motifs(kmer)

            # Check if this region overlaps with any motifs
            overlapping_motifs = []
            for motif in motifs:
                # Check for overlap
                motif_end = motif["start"] + len(motif.get("pattern", ""))
                if motif["start"] <= position + len(kmer) and motif_end >= position:
                    overlapping_motifs.append(motif)

            if not overlapping_motifs:
                return None

            # Create a vector for the motifs
            result = self.xp.zeros(self.dim)
            for motif in overlapping_motifs:
                # Use a seeded random vector for stability
                seed_val = hash(motif["name"]) % 10000
                np.random.seed(seed_val)
                motif_vec = self._normalize(
                    self.xp.array(np.random.uniform(-1, 1, self.dim))
                )
                # Scale by motif score (if available)
                motif_vec *= motif.get("score", 1.0)
                result = self.bundle(result, motif_vec, alpha=0.7)

            return self._normalize(result)

        # Register the provider
        self.register_feature_provider("motifs", motif_provider, weight)
        return self

    def _load_motif_file(self, filename):
        """Load protein-binding motifs from file."""
        try:
            import pandas as pd

            # For JASPAR/MEME format (PWM)
            if filename.endswith((".jaspar", ".pfm", ".meme")):
                # This is a simplification - real implementation would parse PWM formats
                pwms = {}
                with open(filename, "r") as f:
                    lines = f.readlines()
                    current_motif = None
                    for line in lines:
                        if line.startswith(">"):
                            current_motif = line[1:].strip()
                            pwms[current_motif] = {"pwm": [], "name": current_motif}
                        elif current_motif and line.strip():
                            # Parse PWM rows
                            values = [float(x) for x in line.strip().split()]
                            if values:
                                pwms[current_motif]["pwm"].append(values)

                # Convert PWMs to motifs with consensus patterns
                motifs = []
                for name, data in pwms.items():
                    if "pwm" in data and data["pwm"]:
                        # Generate consensus sequence
                        consensus = ""
                        for pos in zip(*data["pwm"]):
                            max_idx = pos.index(max(pos))
                            consensus += "ACGT"[max_idx]

                        motifs.append(
                            {
                                "name": name,
                                "pattern": consensus,
                                "start": 0,  # Default position
                                "end": len(consensus),
                                "score": 1.0,
                                "pwm": data["pwm"],
                            }
                        )
                return motifs

            # For BED format
            elif filename.endswith(".bed"):
                df = pd.read_csv(filename, sep="\t", header=None)
                motifs = []
                for _, row in df.iterrows():
                    motifs.append(
                        {
                            "name": row[3] if len(row) > 3 else "unknown",
                            "start": int(row[1]),
                            "end": int(row[2]),
                            "score": float(row[4]) if len(row) > 4 else 1.0,
                            "pattern": row[5] if len(row) > 5 else None,
                        }
                    )
                return motifs

            # For custom format
            else:
                df = pd.read_csv(filename, sep="\t")
                return [
                    {
                        "name": row["name"],
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "score": float(row.get("score", 1.0)),
                        "pattern": row.get("pattern", None),
                    }
                    for _, row in df.iterrows()
                ]

        except Exception as e:
            logger.error(f"Error loading motif data: {e}")
            return []  # Return empty list on error

    # File loading methods - implementations depend on file formats
    def _load_conservation_file(self, filename):
        """Load conservation scores from file."""
        # Mock implementation - replace with actual loading code
        try:
            import pandas as pd

            # For WIG/BigWig format
            if filename.endswith((".wig", ".bw", ".bigwig")):
                import pyBigWig

                bw = pyBigWig.open(filename)
                # This is simplified - real implementation would need to handle chromosomes
                scores = {
                    i: bw.values("chr1", i, i + 1)[0] for i in range(bw.chroms("chr1"))
                }
                bw.close()
                return scores

            # For BED format
            elif filename.endswith(".bed"):
                df = pd.read_csv(filename, sep="\t", header=None)
                scores = {}
                for _, row in df.iterrows():
                    for pos in range(row[1], row[2]):
                        scores[pos] = float(row[4])  # Assuming score is in 5th column
                return scores

            # For custom tab-delimited format
            else:
                df = pd.read_csv(filename, sep="\t")
                return {
                    int(row["position"]): float(row["score"])
                    for _, row in df.iterrows()
                }

        except Exception as e:
            logger.error(f"Error loading conservation data: {e}")
            return {}  # Return empty dict on error

    def _load_annotation_file(self, filename):
        """Load genomic annotations from file."""
        # Mock implementation - replace with actual loading code
        try:
            import pandas as pd

            # For GFF/GTF format
            if filename.endswith((".gff", ".gtf")):
                df = pd.read_csv(filename, sep="\t", comment="#", header=None)
                annotations = {}
                for _, row in df.iterrows():
                    ann_type = row[2]
                    start, end = int(row[3]), int(row[4])
                    for pos in range(start, end + 1):
                        annotations[pos] = {"type": ann_type}
                return annotations

            # For BED format
            elif filename.endswith(".bed"):
                df = pd.read_csv(filename, sep="\t", header=None)
                annotations = {}
                for _, row in df.iterrows():
                    start, end = int(row[1]), int(row[2])
                    ann_type = row[3] if len(row) > 3 else "region"
                    for pos in range(start, end + 1):
                        annotations[pos] = {"type": ann_type}
                return annotations

            else:
                # Custom format
                df = pd.read_csv(filename, sep="\t")
                return {
                    int(row["position"]): {"type": row["annotation_type"]}
                    for _, row in df.iterrows()
                }

        except Exception as e:
            logger.error(f"Error loading annotation data: {e}")
            return {}  # Return empty dict on error

    def _load_epigenetic_file(self, filename):
        """Load epigenetic data from file."""
        # Mock implementation - replace with actual loading code
        try:
            import pandas as pd

            # For BED format
            if filename.endswith(".bed"):
                df = pd.read_csv(filename, sep="\t", header=None)
                data = {}
                for _, row in df.iterrows():
                    start, end = int(row[1]), int(row[2])
                    value = float(row[4]) if len(row) > 4 else 1.0
                    mark_type = row[3] if len(row) > 3 else "unknown"
                    for pos in range(start, end + 1):
                        if pos not in data:
                            data[pos] = {}
                        data[pos][mark_type] = value
                return data

            # For WIG/BigWig format
            elif filename.endswith((".wig", ".bw", ".bigwig")):
                import pyBigWig

                bw = pyBigWig.open(filename)
                # Extract mark type from filename
                mark_type = os.path.basename(filename).split(".")[0]
                # This is simplified - real implementation would need to handle chromosomes
                data = {}
                chrom = "chr1"  # Example
                for i in range(bw.chroms(chrom)):
                    value = bw.values(chrom, i, i + 1)[0]
                    if value is not None and not np.isnan(value):
                        data[i] = {mark_type: value}
                bw.close()
                return data

            else:
                # Custom format
                df = pd.read_csv(filename, sep="\t")
                data = {}
                for _, row in df.iterrows():
                    pos = int(row["position"])
                    if pos not in data:
                        data[pos] = {}
                    data[pos][row["mark_type"]] = float(row["value"])
                return data

        except Exception as e:
            logger.error(f"Error loading epigenetic data: {e}")
            return {}  # Return empty dict on error


class DNAEncoder(HDCVectorSpace):
    """HDC-based DNA sequence encoder with auto-tuning capabilities."""

    def __init__(
        self,
        dimension: int = None,
        device: str = "auto",
        vector_type: str = "bipolar",
        seed: int = None,
        data_size: int = None,  # Add data_size
    ):
        super().__init__(dimension, device, vector_type, seed, data_size)

        # DNA-specific caches
        self.kmer_cache = {}
        self.sequence_cache = {}

        # DNA base statistics for adaptive encoding
        self.stats = {
            "base_counts": {"A": 0, "T": 0, "G": 0, "C": 0, "N": 0},
            "total_bases": 0,
            "gc_content": 0.5,  # Default
            "seq_length_distrib": [],
        }

        # Initialize with standard DNA bases
        self.initialize(["A", "T", "G", "C"])

        # Track most commonly used k-mer size
        self.kmer_usage = defaultdict(int)
        self.optimal_k = 7  # Default

    def encode_kmer(self, kmer: str, position: int = None) -> Any:
        """Encode a k-mer into HDC vector.

        Args:
            kmer: DNA k-mer string
            position: Position in genome (for biological features)

        Returns:
            HDC vector representing the k-mer
        """
        # Update k-mer usage statistics
        self.kmer_usage[len(kmer)] += 1

        # Check cache first
        cache_key = (kmer, position)
        if cache_key in self.kmer_cache:
            self.cache_stats["hits"] += 1
            return self.kmer_cache[cache_key]

        self.cache_stats["misses"] += 1

        # Fast path for single bases
        if len(kmer) == 1 and kmer in self.base_vectors:
            return self.base_vectors[kmer]

        # Standard encoding with position binding
        result = self.xp.zeros(self.dim)
        for i, base in enumerate(kmer):
            # Handle standard and non-standard bases
            if base in self.base_vectors:
                # Update base statistics
                if base in self.stats["base_counts"]:
                    self.stats["base_counts"][base] += 1
                    self.stats["total_bases"] += 1

                # Bind base with position
                bound = self.bind(
                    self.base_vectors[base],
                    self.position_vectors[i % len(self.position_vectors)],
                )
            else:
                # Handle ambiguity codes
                bound = self._handle_ambiguity_code(base, i)

            # Bundle with existing result
            result = self.bundle(result, bound, alpha=0.9)

        # Normalize and cache
        result = self._normalize(result)
        self.kmer_cache[cache_key] = result

        # Prune cache if too large
        self._prune_cache_if_needed()

        return result

    def _handle_ambiguity_code(self, base: str, position: int) -> Any:
        """Handle ambiguity codes in DNA/RNA."""
        # IUPAC nucleotide codes
        ambiguity_map = {
            "N": ["A", "C", "G", "T"],  # Any base
            "R": ["A", "G"],  # Purine
            "Y": ["C", "T"],  # Pyrimidine
            "M": ["A", "C"],  # Amino
            "K": ["G", "T"],  # Keto
            "S": ["C", "G"],  # Strong
            "W": ["A", "T"],  # Weak
            "B": ["C", "G", "T"],  # Not A
            "D": ["A", "G", "T"],  # Not C
            "H": ["A", "C", "T"],  # Not G
            "V": ["A", "C", "G"],  # Not T
        }

        # Get possible bases
        bases = ambiguity_map.get(base.upper(), [])

        # If recognized ambiguity code, compute average of possible bases
        if bases:
            pos_vector = self.position_vectors[position % len(self.position_vectors)]
            vectors = [
                self.bind(self.base_vectors[b], pos_vector)
                for b in bases
                if b in self.base_vectors
            ]
            if vectors:
                return sum(vectors) / len(vectors)

        # Unknown base, return zero vector
        return self.xp.zeros(self.dim)

    def _prune_cache_if_needed(self):
        """Prune the k-mer cache if it grows too large."""
        # Get available system memory
        mem_info = psutil.virtual_memory()
        mem_used_pct = mem_info.percent

        # Adjust cache threshold based on available memory
        threshold = 10000000 if mem_used_pct < 70 else 1000000

        if len(self.kmer_cache) > threshold:
            logger.info(f"Pruning k-mer cache ({len(self.kmer_cache)} entries)")

            # First, sort by k-mer length (keep shorter ones)
            sorted_by_len = sorted(
                self.kmer_cache.keys(),
                key=lambda k: len(k[0]) if isinstance(k, tuple) else len(k),
            )

            # Keep half the cache, prioritizing shorter k-mers
            to_keep = sorted_by_len[: len(sorted_by_len) // 2]
            self.kmer_cache = {k: self.kmer_cache[k] for k in to_keep}

            logger.info(f"Cache pruned to {len(self.kmer_cache)} entries")

    def encode_sequence(
        self, sequence: str, k: int = None, stride: int = None, chunk_size: int = None
    ) -> Any:
        """Encode a DNA sequence using sliding k-mers with adaptive parameters.

        Args:
            sequence: DNA sequence string
            k: k-mer size (auto-determined if None)
            stride: Step size for sliding window (auto-determined if None)
            chunk_size: Chunk size for processing (auto-determined if None)

        Returns:
            HDC vector representing the sequence
        """
        # Auto-determine optimal k-mer size if not specified
        if k is None:
            k = self._get_optimal_k(sequence)

        # Auto-determine optimal stride based on sequence length
        if stride is None:
            stride = max(
                1, len(sequence) // 10000
            )  # 1 for short sequences, larger for longer ones

        # Auto-determine chunk size based on available memory
        if chunk_size is None:
            chunk_size = self._get_optimal_chunk_size(len(sequence))

        # Quick path for sequences shorter than k
        if len(sequence) < k:
            return self.encode_kmer(sequence)

        # Check cache for identical sequence and parameters
        cache_key = (sequence, k, stride)
        if cache_key in self.sequence_cache:
            return self.sequence_cache[cache_key]

        # Initialize the result vector
        result = self.xp.zeros(self.dim)
        n_kmers = 0

        # Process sequence in chunks to save memory
        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]

            # Process k-mers in this chunk
            for i in range(0, len(chunk) - k + 1, stride):
                kmer = chunk[i : i + k]

                # Skip k-mers with too many non-standard bases
                if kmer.count("N") > k // 3:
                    continue

                # Get position-aware encoding if biological features might be used
                position = (
                    chunk_start + i
                    if hasattr(self, "use_biological_features")
                    else None
                )

                # Encode and bundle
                kmer_vector = self.encode_kmer(kmer, position)
                result = self.bundle(result, kmer_vector, alpha=0.9)
                n_kmers += 1

        # Handle the case with no valid k-mers
        if n_kmers == 0:
            logger.warning(f"No valid k-mers found in sequence: {sequence[:20]}...")
            return self.xp.zeros(self.dim)

        # Normalize and cache (only for reasonably-sized sequences)
        result = self._normalize(result)
        if len(sequence) < 10000:  # Don't cache large sequences
            self.sequence_cache[cache_key] = result

        # Update sequence statistics
        self._update_sequence_stats(sequence)

        return result

    def _get_optimal_k(self, sequence: str) -> int:
        """Determine optimal k-mer size based on sequence and statistics."""
        # If we have usage statistics, use the most common k-mer size
        if sum(self.kmer_usage.values()) > 1000:
            self.optimal_k = max(self.kmer_usage.items(), key=lambda x: x[1])[0]
            return self.optimal_k

        # Otherwise, calculate based on sequence properties
        if len(sequence) < 20:
            return min(5, len(sequence))  # Small k for short sequences

        # Base on sequence complexity
        return detect_optimal_kmer_size(sequence)

    def _get_optimal_chunk_size(self, sequence_length: int) -> int:
        """Determine optimal chunk size based on system memory."""
        # Get available system memory
        mem_info = psutil.virtual_memory()
        available_bytes = mem_info.available

        # Estimate memory needed per base (conservatively)
        bytes_per_base = (
            self.dim * 8 * 2
        )  # Vector size * bytes per float * overhead factor

        # Use at most 25% of available memory
        max_chunk_size = int(0.25 * available_bytes / bytes_per_base)

        # Limit to reasonable range
        chunk_size = min(max_chunk_size, sequence_length)
        chunk_size = max(chunk_size, 1000)  # At least 1000 bases per chunk

        return chunk_size

    def _update_sequence_stats(self, sequence: str):
        """Update sequence statistics for adaptive encoding."""
        # Update length distribution
        self.stats["seq_length_distrib"].append(len(sequence))
        if len(self.stats["seq_length_distrib"]) > 100:
            self.stats["seq_length_distrib"] = self.stats["seq_length_distrib"][-100:]

        # Sample bases if sequence is long
        sample = (
            sequence
            if len(sequence) < 1000
            else "".join(
                sequence[i : i + 1]
                for i in range(0, len(sequence), len(sequence) // 1000)
            )
        )

        # Update base counts
        for base in sample:
            if base in self.stats["base_counts"]:
                self.stats["base_counts"][base] += 1
                self.stats["total_bases"] += 1

        # Update GC content
        if self.stats["total_bases"] > 0:
            gc_count = self.stats["base_counts"]["G"] + self.stats["base_counts"]["C"]
            self.stats["gc_content"] = gc_count / self.stats["total_bases"]

    def find_similar_sequences(
        self, query: str, references: List[str], k: int = None, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar sequences to the query.

        Args:
            query: Query sequence
            references: List of reference sequences
            k: k-mer size (auto-determined if None)
            top_n: Number of top matches to return

        Returns:
            List of (sequence, similarity) tuples
        """
        # Auto-determine k if not specified
        if k is None:
            k = self._get_optimal_k(query)

        # Encode query
        query_vector = self.encode_sequence(query, k=k)

        # Encode all reference sequences
        ref_vectors = []
        for seq in tqdm(references, desc="Encoding references"):
            ref_vectors.append(self.encode_sequence(seq, k=k))

        # Calculate similarities
        similarities = self.batch_similarity(query_vector, ref_vectors)

        # Get top matches
        if self.device == "gpu":
            similarities = self.xp.asnumpy(similarities)

        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return [(references[i], similarities[i]) for i in top_indices]

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics and memory usage."""
        # Calculate cache memory usage
        kmer_cache_size = len(self.kmer_cache)
        seq_cache_size = len(self.sequence_cache)

        # Get base statistics
        base_stats = {
            k: v
            for k, v in self.stats["base_counts"].items()
            if self.stats["total_bases"] > 0
        }
        if self.stats["total_bases"] > 0:
            base_stats = {
                k: v / self.stats["total_bases"] for k, v in base_stats.items()
            }

        # Get average sequence length
        avg_len = sum(self.stats["seq_length_distrib"]) / max(
            1, len(self.stats["seq_length_distrib"])
        )

        return {
            "kmer_cache_size": kmer_cache_size,
            "sequence_cache_size": seq_cache_size,
            "cache_hit_ratio": self.cache_stats["hits"]
            / max(1, self.cache_stats["hits"] + self.cache_stats["misses"]),
            "base_distribution": base_stats,
            "gc_content": self.stats["gc_content"],
            "avg_sequence_length": avg_len,
            "optimal_k": self._get_optimal_k(""),
            "dimension": self.dim,
            "device": self.device,
        }

    def save_vectors(self, vectors: List[Any], filename: str):
        """Save HDC vectors to disk using HDF5."""
        # Convert to NumPy if on GPU
        if self.device == "gpu":
            vectors = [self.xp.asnumpy(v) for v in vectors]

        with h5py.File(filename, "w") as f:
            f.create_dataset("vectors", data=np.array(vectors))

            # Save metadata
            f.attrs["dimension"] = self.dim
            f.attrs["device"] = self.device
            f.attrs["vector_type"] = self.vector_type

    def load_vectors(self, filename: str) -> List[Any]:
        """Load HDC vectors from disk."""
        with h5py.File(filename, "r") as f:
            vectors = f["vectors"][:]

            # Check metadata
            if "dimension" in f.attrs and f.attrs["dimension"] != self.dim:
                logger.warning(
                    f"Loaded vectors have dimension {f.attrs['dimension']}, but encoder has {self.dim}"
                )

        # Convert to device array if on GPU
        if self.device == "gpu":
            return [self.xp.array(v) for v in vectors]
        return list(vectors)


class MetaHDConservation:
    """Meta-learning conservation scorer using HDC vectors."""

    def __init__(self, hdc_computer, dimension=1000, memory_depth=5, alpha=0.1):
        self.hdc = hdc_computer
        self.dim = dimension
        self.depth = memory_depth
        self.alpha = alpha  # Make alpha a parameter

        # Meta-memory: stores HDC vector patterns
        self.meta_patterns = {
            "conserved": self.hdc._random_vector(),  # Initialize with random vectors
            "non_conserved": self.hdc._random_vector(),
            "context": self.hdc._random_vector(),  # Single context vector
        }
        self.context_weights = [
            (0.9**i) for i in range(self.depth)
        ]  # Exponential decay for context
        self.context_vectors = []

    def _similarity(self, v1, v2):
        """Compute cosine similarity (more standard than resonance)."""
        return float(self.hdc.similarity(v1, v2))

    def _meta_encode(self, sequence):
        """Generate meta-features from HDC vector patterns."""
        base_vector = self.hdc.encode_sequence(sequence)

        features = {
            "conserved_similarity": self._similarity(
                base_vector, self.meta_patterns["conserved"]
            ),
            "non_conserved_similarity": self._similarity(
                base_vector, self.meta_patterns["non_conserved"]
            ),
            "context_similarity": self._similarity(
                base_vector, self.meta_patterns["context"]
            ),
        }
        return base_vector, features

    def update_meta_patterns(self, vector, conservation_score):
        """Update meta-patterns based on observed conservation."""
        # Adaptive learning rate
        alpha = self.alpha * (1.0 - conservation_score)  # Higher score -> lower alpha

        if conservation_score > 0.7:  # Highly conserved
            self.meta_patterns["conserved"] = self.hdc.bundle(
                self.meta_patterns["conserved"], vector, alpha=1 - alpha
            )
        elif conservation_score < 0.3:  # Not conserved
            self.meta_patterns["non_conserved"] = self.hdc.bundle(
                self.meta_patterns["non_conserved"], vector, alpha=1 - alpha
            )

        # Update context (weighted average of recent vectors)
        self.context_vectors.append(vector)
        if len(self.context_vectors) > self.depth:
            self.context_vectors = self.context_vectors[-self.depth :]

        weighted_context = self.hdc.xp.zeros(self.dim)
        total_weight = 0
        for i, v in enumerate(reversed(self.context_vectors)):
            weight = self.context_weights[i]
            weighted_context += weight * v
            total_weight += weight

        if total_weight > 0:
            self.meta_patterns["context"] = self.hdc._normalize(weighted_context)

    def score(self, sequence):
        """Score conservation using meta-HDC patterns."""
        vector, features = self._meta_encode(sequence)

        # Weighted combination of features
        score = (
            0.6 * features["conserved_similarity"]
            - 0.3 * features["non_conserved_similarity"]
            + 0.1 * features["context_similarity"]
        )

        # Clip to [0, 1]
        score = np.clip(score, 0, 1)
        self.update_meta_patterns(vector, score)  # update after score
        return score

    def get_meta_patterns(self):
        """Returns a copy of the meta_patterns. Useful for saving/loading"""
        return {k: v.copy() for k, v in self.meta_patterns.items()}

    def set_meta_patterns(self, patterns):
        """Set the meta patterns.  Used for loading."""
        for k, v in patterns.items():
            if k in self.meta_patterns:
                self.meta_patterns[k] = v.copy()


class BiologicalEncoder(DNAEncoder):
    """Extended DNA encoder that integrates biological features."""

    def __init__(
        self,
        dimension: int = None,
        device: str = "auto",
        vector_type: str = "bipolar",
        seed: int = None,
        data_size: int = None,  # Add data_size
    ):
        super().__init__(dimension, device, vector_type, seed, data_size)

        # Flag for enabling biological features
        self.use_biological_features = True

        # Feature providers and weights
        self.feature_providers = {}
        self.feature_weights = {}

        # Feature cache
        self.feature_cache = {}

        # Register default feature providers
        self._register_default_features()

    def _register_default_features(self):
        """Register default biological feature providers."""
        # Basic features that don't require external data
        self.register_feature("gc_content", self._compute_gc_content, weight=0.5)
        self.register_feature("complexity", self._compute_complexity, weight=0.3)
        self.register_feature("motifs", self._detect_motifs, weight=0.4)

    def register_feature(self, name: str, provider_fn: Callable, weight: float = 1.0):
        """Register a biological feature provider.

        Args:
            name: Unique name for the feature
            provider_fn: Function computing the feature
            weight: Importance weight for this feature
        """
        self.feature_providers[name] = provider_fn
        self.feature_weights[name] = weight
        logger.info(f"Registered feature '{name}' with weight {weight}")

    def encode_kmer(self, kmer: str, position: int = None) -> Any:
        """Encode a k-mer with biological feature integration.

        Args:
            kmer: DNA k-mer
            position: Genomic position

        Returns:
            HDC vector with biological features
        """
        # Check cache first
        cache_key = (kmer, position)
        if cache_key in self.kmer_cache:
            self.cache_stats["hits"] += 1
            return self.kmer_cache[cache_key]

        self.cache_stats["misses"] += 1

        # Base encoding from parent class
        base_vector = super().encode_kmer(kmer, position)

        # Integrate biological features if enabled
        if self.use_biological_features:
            # Compute feature vector
            feature_vector = self._compute_feature_vector(kmer, position)

            if feature_vector is not None:
                # Combine base encoding with features
                integrated = self.bundle(base_vector, feature_vector, alpha=0.7)
                integrated = self._normalize(integrated)
            else:
                integrated = base_vector
        else:
            integrated = base_vector

        # Cache and return
        self.kmer_cache[cache_key] = integrated
        self._prune_cache_if_needed()

        return integrated

    def _compute_feature_vector(self, kmer: str, position: int = None) -> Optional[Any]:
        """Compute combined feature vector from all providers.

        Args:
            kmer: DNA k-mer
            position: Genomic position

        Returns:
            Combined feature vector or None
        """
        # Check feature cache first
        cache_key = (kmer, position)
        """Compute combined feature vector from all providers.
        
        Args:
            kmer: DNA k-mer
            position: Genomic position
            
        Returns:
            Combined feature vector or None
        """
        # Check feature cache first
        cache_key = (kmer, position)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Collect features from all providers
        features = {}
        for name, provider in self.feature_providers.items():
            try:
                value = provider(kmer, position)
                if value is not None:
                    features[name] = value
            except Exception as e:
                logger.debug(f"Feature provider '{name}' failed: {e}")

        if not features:
            return None

        # Separate vector features from scalar features
        vector_features = {}
        scalar_features = {}

        for name, value in features.items():
            if isinstance(value, (self.xp.ndarray, np.ndarray)):
                vector_features[name] = value
            else:
                scalar_features[name] = value

        # Initialize combined feature vector
        feature_vector = self.xp.zeros(self.dim)

        # Integrate scalar features by binding with feature-specific vectors
        for name, value in scalar_features.items():
            # Get a stable vector for this feature (seeded by name)
            seed_val = hash(name) % 10000
            np.random.seed(seed_val)
            feature_vec = self._normalize(
                self.xp.array(np.random.uniform(-1, 1, self.dim))
            )

            # Scale by value and importance weight
            weight = self.feature_weights.get(name, 1.0)
            feature_vec *= value * weight

            # Bundle into combined vector
            feature_vector = self.bundle(feature_vector, feature_vec, alpha=0.7)

        # Integrate vector features
        for name, vec in vector_features.items():
            weight = self.feature_weights.get(name, 1.0)
            weighted_vec = vec * weight
            feature_vector = self.bundle(feature_vector, weighted_vec, alpha=0.7)

        # Normalize the result
        normalized = self._normalize(feature_vector)

        # Cache and return
        self.feature_cache[cache_key] = normalized
        return normalized

    # Feature provider functions

    def _compute_gc_content(self, kmer: str, position: int = None) -> float:
        """Compute GC content feature."""
        gc_count = kmer.count("G") + kmer.count("C")
        return gc_count / len(kmer) if len(kmer) > 0 else 0.5

    def _compute_complexity(self, kmer: str, position: int = None) -> float:
        """Compute sequence complexity using Shannon entropy."""
        # Count each base
        base_counts = {}
        for base in kmer:
            if base in "ACGT":
                base_counts[base] = base_counts.get(base, 0) + 1

        # Calculate entropy
        total = sum(base_counts.values())
        if total <= 1:  # Not enough data for meaningful entropy
            return 0.5

        entropy = 0
        for count in base_counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(min(4, len(base_counts)))
        return entropy / max_entropy if max_entropy > 0 else 0.5

    @lru_cache(maxsize=128)
    def _detect_motifs(self, kmer: str, position: int = None) -> float:
        """Detect common genetic motifs."""
        # Common regulatory and functional motifs
        motif_scores = {
            # Core promoter elements
            "TATAAA": 0.9,  # TATA box
            "TTGACA": 0.8,  # -35 box (prokaryotes)
            "TATAAT": 0.8,  # -10 box (prokaryotes)
            # Eukaryotic regulatory elements
            "CCAAT": 0.7,  # CCAAT box
            "CACGTG": 0.8,  # E-box
            "GCCNNNGGC": 0.7,  # GC box
            # Transcription termination
            "AATAAA": 0.8,  # Poly(A) signal
            # DNA binding motifs
            "GATA": 0.6,  # GATA binding factor
            "CAAT": 0.7,  # CAAT box
            "GAGAG": 0.6,  # GAGA factor
            # Specialized motifs
            "CG": 0.5,  # CpG sites
            "ATG": 0.7,  # Start codon
            "TAA": 0.6,  # Stop codon
            "TAG": 0.6,  # Stop codon
            "TGA": 0.6,  # Stop codon
        }

        # Find highest scoring motif match
        best_score = 0
        for motif, importance in motif_scores.items():
            if len(motif) > len(kmer):
                continue

            # Check for exact match
            if motif in kmer:
                best_score = max(best_score, importance)
                continue

            # Check for fuzzy match with one mismatch allowed
            for i in range(len(kmer) - len(motif) + 1):
                subseq = kmer[i : i + len(motif)]
                mismatches = sum(a != b for a, b in zip(subseq, motif) if b != "N")
                if mismatches <= 1:  # Allow one mismatch
                    match_score = importance * (1 - mismatches / len(motif))
                    best_score = max(best_score, match_score)

        return best_score

    # Data loading methods

    def load_annotations(
        self, filename: str, weight: float = 0.7
    ) -> "BiologicalEncoder":
        """Load genomic annotations (genes, regulatory elements, etc.)."""
        # Load the annotations
        annotations = self._load_annotations_file(filename)

        # Create vectors for annotation types
        annotation_vectors = {}
        for ann_type in set(ann["type"] for ann in annotations.values()):
            # Create stable vector for this annotation type
            seed_val = hash(ann_type) % 10000
            np.random.seed(seed_val)
            annotation_vectors[ann_type] = self._normalize(
                self.xp.array(np.random.uniform(-1, 1, self.dim))
            )

        # Create annotation provider function
        def annotation_provider(kmer: str, position: int = None) -> Optional[Any]:
            if position is None:
                return None  # Need position for annotations

            # Get annotations overlapping this region
            region_annotations = []
            for i in range(len(kmer)):
                pos = position + i
                if pos in annotations:
                    region_annotations.append(annotations[pos])

            if not region_annotations:
                return None

            # Combine annotation vectors
            result = self.xp.zeros(self.dim)
            for ann in region_annotations:
                ann_type = ann["type"]
                if ann_type in annotation_vectors:
                    result = self.bundle(
                        result, annotation_vectors[ann_type], alpha=0.7
                    )

            return self._normalize(result)

        # Register the provider
        self.register_feature("annotations", annotation_provider, weight)
        return self

    def load_conservation(
        self, filename: str, weight: float = 0.8
    ) -> "BiologicalEncoder":
        """Load evolutionary conservation scores."""
        # Load conservation data
        conservation = self._load_conservation_file(filename)

        # Create conservation provider function
        def conservation_provider(kmer: str, position: int = None) -> Optional[float]:
            if position is None:
                return None  # Need position for conservation

            # Get scores for this region
            scores = []
            for i in range(len(kmer)):
                pos = position + i
                if pos in conservation:
                    scores.append(conservation[pos])

            if not scores:
                return None

            # Return average conservation score
            return sum(scores) / len(scores)

        # Register the provider
        self.register_feature("conservation", conservation_provider, weight)
        return self

    def load_epigenetic_data(
        self, filename: str, weight: float = 0.6
    ) -> "BiologicalEncoder":
        """Load epigenetic data (methylation, histone marks, etc.)."""
        # Load epigenetic data
        epigenetic_data = self._load_epigenetic_file(filename)

        # Create epigenetic provider function
        def epigenetic_provider(kmer: str, position: int = None) -> Optional[Any]:
            if position is None:
                return None  # Need position for epigenetics

            # Get epigenetic marks for this region
            region_data = {}
            for i in range(len(kmer)):
                pos = position + i
                if pos in epigenetic_data:
                    for mark, value in epigenetic_data[pos].items():
                        region_data[mark] = region_data.get(mark, 0) + value

            if not region_data:
                return None

            # Normalize by region length
            region_data = {k: v / len(kmer) for k, v in region_data.items()}

            # Create vector representation
            result = self.xp.zeros(self.dim)
            for mark, value in region_data.items():
                # Create stable vector for this mark
                seed_val = hash(mark) % 10000
                np.random.seed(seed_val)
                mark_vec = self._normalize(
                    self.xp.array(np.random.uniform(-1, 1, self.dim))
                )
                mark_vec *= value
                result = self.bundle(result, mark_vec, alpha=0.7)

            return self._normalize(result)

        # Register the provider
        self.register_feature("epigenetics", epigenetic_provider, weight)
        return self

    def load_motif_data(
        self, filename: str, weight: float = 0.7
    ) -> "BiologicalEncoder":
        """Load transcription factor binding motifs."""
        # Load motif data
        motifs = self._load_motif_file(filename)

        # Create motif provider function
        def motif_provider(kmer: str, position: int = None) -> Optional[Any]:
            # If position unknown, use built-in motif detection
            if position is None:
                return self._detect_motifs(kmer)

            # Find overlapping motifs
            overlapping_motifs = []
            for motif in motifs:
                motif_end = motif["start"] + len(motif.get("pattern", ""))
                if position <= motif_end and position + len(kmer) >= motif["start"]:
                    overlapping_motifs.append(motif)

            if not overlapping_motifs:
                return None

            # Create vector representation
            result = self.xp.zeros(self.dim)
            for motif in overlapping_motifs:
                # Create stable vector for this motif
                seed_val = hash(motif["name"]) % 10000
                np.random.seed(seed_val)
                motif_vec = self._normalize(
                    self.xp.array(np.random.uniform(-1, 1, self.dim))
                )
                motif_vec *= motif.get("score", 1.0)
                result = self.bundle(result, motif_vec, alpha=0.7)

            return self._normalize(result)

        # Register the provider
        self.register_feature("known_motifs", motif_provider, weight)
        return self

    # File parsing helpers

    def _load_annotations_file(self, filename: str) -> Dict[int, Dict[str, Any]]:
        """Parse genomic annotation file."""
        ext = filename.lower().split(".")[-1]
        annotations = {}

        try:
            import pandas as pd

            if ext in ["gff", "gff3", "gtf"]:
                # Parse GFF/GTF format
                df = pd.read_csv(filename, sep="\t", comment="#", header=None)
                colnames = [
                    "seqid",
                    "source",
                    "type",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "phase",
                    "attributes",
                ]
                df.columns = colnames[: len(df.columns)]

                for _, row in df.iterrows():
                    ann_type = row["type"]
                    start, end = int(row["start"]), int(row["end"])

                    # Add annotation at each position in range
                    for pos in range(start, end + 1):
                        annotations[pos] = {"type": ann_type}

            elif ext == "bed":
                # Parse BED format
                df = pd.read_csv(filename, sep="\t", header=None)

                for _, row in df.iterrows():
                    start, end = int(row[1]), int(row[2])
                    ann_type = row[3] if len(row) > 3 else "region"

                    for pos in range(start, end + 1):
                        annotations[pos] = {"type": ann_type}

            else:
                logger.warning(f"Unknown annotation file format: {ext}")

        except Exception as e:
            logger.error(f"Error parsing annotation file: {e}")

        return annotations

    def _load_conservation_file(self, filename: str) -> Dict[int, float]:
        """Parse conservation score file."""
        ext = filename.lower().split(".")[-1]
        conservation = {}

        try:
            if ext in ["wig", "bigwig", "bw"]:
                # Parse WIG/BigWig format
                import pyBigWig

                bw = pyBigWig.open(filename)

                # Get chromosomes and their lengths
                for chrom in bw.chroms():
                    # Sample scores to avoid massive dict
                    chrom_len = bw.chroms(chrom)
                    for pos in range(0, chrom_len, 100):  # Sample every 100bp
                        end = min(pos + 100, chrom_len)
                        try:
                            scores = bw.values(chrom, pos, end)
                            for i, score in enumerate(scores):
                                if score is not None and not np.isnan(score):
                                    conservation[pos + i] = float(score)
                        except Exception as e:
                            logger.debug(f"Error reading conservation data: {e}")
                            pass

                bw.close()

            elif ext == "bed":
                # Parse BED format with conservation scores
                import pandas as pd

                df = pd.read_csv(filename, sep="\t", header=None)

                for _, row in df.iterrows():
                    start, end = int(row[1]), int(row[2])
                    score = float(row[4]) if len(row) > 4 else 0.0

                    for pos in range(start, end + 1):
                        conservation[pos] = score

            else:
                logger.warning(f"Unknown conservation file format: {ext}")

        except Exception as e:
            logger.error(f"Error parsing conservation file: {e}")

        return conservation

    def _load_epigenetic_file(self, filename: str) -> Dict[int, Dict[str, float]]:
        """Parse epigenetic data file."""
        ext = filename.lower().split(".")[-1]
        epigenetic_data = {}

        try:
            if ext in ["wig", "bigwig", "bw"]:
                # Parse WIG/BigWig format
                import pyBigWig

                bw = pyBigWig.open(filename)

                # Extract mark type from filename
                mark_type = os.path.basename(filename).split(".")[0]

                # Get chromosomes and their lengths
                for chrom in bw.chroms():
                    # Sample scores to avoid massive dict
                    chrom_len = bw.chroms(chrom)
                    for pos in range(0, chrom_len, 100):  # Sample every 100bp
                        end = min(pos + 100, chrom_len)
                        try:
                            values = bw.values(chrom, pos, end)
                            for i, value in enumerate(values):
                                if value is not None and not np.isnan(value):
                                    if (pos + i) not in epigenetic_data:
                                        epigenetic_data[pos + i] = {}
                                    epigenetic_data[pos + i][mark_type] = float(value)
                        except Exception as e:
                            logger.debug(f"Error reading epigenetic data: {e}")
                            pass

                bw.close()

            elif ext == "bed":
                # Parse BED format
                import pandas as pd

                df = pd.read_csv(filename, sep="\t", header=None)

                for _, row in df.iterrows():
                    start, end = int(row[1]), int(row[2])
                    mark_type = row[3] if len(row) > 3 else "unknown"
                    value = float(row[4]) if len(row) > 4 else 1.0

                    for pos in range(start, end + 1):
                        if pos not in epigenetic_data:
                            epigenetic_data[pos] = {}
                        epigenetic_data[pos][mark_type] = value

            else:
                logger.warning(f"Unknown epigenetic file format: {ext}")

        except Exception as e:
            logger.error(f"Error parsing epigenetic file: {e}")

        return epigenetic_data

    def _load_motif_file(self, filename: str) -> List[Dict[str, Any]]:
        """Parse motif data file."""
        ext = filename.lower().split(".")[-1]
        motifs = []

        try:
            if ext in ["jaspar", "pfm", "meme"]:
                # Parse position weight matrix formats
                with open(filename, "r") as f:
                    lines = f.readlines()

                    current_motif = None
                    pwm = []

                    for line in lines:
                        if line.startswith(">") or line.startswith("MOTIF"):
                            # New motif
                            if current_motif and pwm:
                                # Save previous motif
                                consensus = self._pwm_to_consensus(pwm)
                                motifs.append(
                                    {
                                        "name": current_motif,
                                        "start": 0,
                                        "end": len(consensus),
                                        "pattern": consensus,
                                        "pwm": pwm,
                                    }
                                )

                            # Start new motif
                            current_motif = (
                                line.split()[1]
                                if line.startswith("MOTIF")
                                else line[1:].strip()
                            )
                            pwm = []

                        elif (
                            current_motif and line.strip() and not line.startswith("#")
                        ):
                            # Parse PWM row
                            values = [
                                float(x) for x in line.strip().split() if x.strip()
                            ]
                            if values:
                                pwm.append(values)

                    # Save last motif
                    if current_motif and pwm:
                        consensus = self._pwm_to_consensus(pwm)
                        motifs.append(
                            {
                                "name": current_motif,
                                "start": 0,
                                "end": len(consensus),
                                "pattern": consensus,
                                "pwm": pwm,
                            }
                        )

            elif ext == "bed":
                # Parse BED format
                import pandas as pd

                df = pd.read_csv(filename, sep="\t", header=None)

                for _, row in df.iterrows():
                    motif = {
                        "name": row[3] if len(row) > 3 else "unknown",
                        "start": int(row[1]),
                        "end": int(row[2]),
                        "score": float(row[4]) if len(row) > 4 else 1.0,
                    }

                    # Add pattern if available
                    if len(row) > 5:
                        motif["pattern"] = row[5]

                    motifs.append(motif)

            else:
                logger.warning(f"Unknown motif file format: {ext}")

        except Exception as e:
            logger.error(f"Error parsing motif file: {e}")

        return motifs

    def _pwm_to_consensus(self, pwm: List[List[float]]) -> str:
        """Convert position weight matrix to consensus sequence."""
        consensus = ""
        bases = "ACGT"

        for pos in zip(*pwm):
            max_idx = pos.index(max(pos))
            consensus += bases[max_idx]

        return consensus


class GeneticAnalyzer:
    """High-level API for genetic sequence analysis using HDC."""

    def __init__(
        self,
        dimension: int = None,
        device: str = "auto",
        use_biological: bool = False,
        data_size: int = None,
    ):
        """Initialize genetic analyzer"""
        # Initialize appropriate encoder
        self.encoder_class = BiologicalEncoder if use_biological else DNAEncoder
        self.encoder = self.encoder_class(
            dimension=dimension, device=device, data_size=data_size
        )  # Pass data_size

        # Store configuration
        self.config = {
            "dimension": self.encoder.dim,
            "device": self.encoder.device,
            "use_biological": use_biological,
        }

        logger.info(
            f"Initialized GeneticAnalyzer (dim={self.encoder.dim}, "
            f"device={self.encoder.device}, bio={use_biological})"
        )

    def encode_sequences(self, sequences: List[str], k: int = None) -> List[Any]:
        """Encode multiple sequences into HDC vectors.

        Args:
            sequences: List of DNA sequences
            k: k-mer size (auto-determined if None)

        Returns:
            List of HDC vectors
        """
        # Auto-determine optimal k if needed
        if k is None and sequences:
            k = self.encoder._get_optimal_k(sequences[0])

        # Encode all sequences
        vectors = []
        for seq in tqdm(sequences, desc="Encoding sequences"):
            vectors.append(self.encoder.encode_sequence(seq, k=k))

        logger.info(f"Encoded {len(sequences)} sequences with k={k}")
        return vectors

    def compute_similarity_matrix(
        self, sequences: List[str], k: int = None
    ) -> np.ndarray:
        """Compute pairwise similarity matrix for sequences.

        Args:
            sequences: List of DNA sequences
            k: k-mer size (auto-determined if None)

        Returns:
            Similarity matrix (NxN array)
        """
        n = len(sequences)
        similarity_matrix = np.zeros((n, n))

        # Encode all sequences
        vectors = self.encode_sequences(sequences, k=k)

        # Compute all pairwise similarities
        for i in tqdm(range(n), desc="Computing similarities"):
            # Diagonal (self-similarity) is always 1.0
            similarity_matrix[i, i] = 1.0

            # Compare with all other sequences
            for j in range(i + 1, n):
                sim = float(self.encoder.similarity(vectors[i], vectors[j]))
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Matrix is symmetric

        return similarity_matrix

    def find_similar(
        self, query: str, references: List[str], k: int = None, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar sequences to a query.

        Args:
            query: Query sequence
            references: Reference sequences to search
            k: k-mer size (auto-determined if None)
            top_n: Number of top matches to return

        Returns:
            List of (sequence, similarity) tuples
        """
        return self.encoder.find_similar_sequences(query, references, k, top_n)

    def analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a sequence.

        Args:
            sequence: DNA sequence

        Returns:
            Dictionary of analysis results
        """
        # Basic sequence stats
        stats = {
            "length": len(sequence),
            "gc_content": (sequence.count("G") + sequence.count("C")) / len(sequence)
            if sequence
            else 0,
            "base_counts": {base: sequence.count(base) for base in "ACGTN"},
        }

        # Detect optimal k-mer size
        stats["optimal_k"] = detect_optimal_kmer_size(sequence)

        # Entropy (complexity)
        base_counts = {
            base: sequence.count(base) for base in set(sequence) if base in "ACGT"
        }
        if sum(base_counts.values()) > 0:
            probs = [
                count / sum(base_counts.values()) for count in base_counts.values()
            ]
            stats["entropy"] = -sum(p * math.log2(p) for p in probs if p > 0)
            stats["normalized_entropy"] = stats["entropy"] / math.log2(len(base_counts))

        # If biological encoder is available, add more features
        if isinstance(self.encoder, BiologicalEncoder):
            # Check for motifs
            motif_positions = []
            window_size = 10

            for i in range(0, len(sequence) - window_size + 1, window_size // 2):
                window = sequence[i : i + window_size]
                score = self.encoder._detect_motifs(window)
                if score > 0.6:  # Threshold for interesting motif
                    motif_positions.append((i, score))

            stats["motif_positions"] = motif_positions

        return stats

    def _save_hdf5_group(self, h5file: h5py.File, group_name: str, data: dict):
        """Helper function to save a dictionary of data to an HDF5 group."""
        group = h5file.create_group(group_name)
        for key, value in data.items():
            if isinstance(value, (np.ndarray, cp.ndarray)):
                # Convert Cupy arrays to Numpy for saving
                value = value.get() if isinstance(value, cp.ndarray) else value
                group.create_dataset(str(key), data=value)
            elif isinstance(value, dict):
                # Recursively save nested dictionaries
                self._save_hdf5_group(group, str(key), value)
            else:
                # Save other data types as attributes
                group.attrs[str(key)] = value

    def _load_hdf5_group(self, h5file: h5py.File, group_name: str) -> dict:
        """Helper function to load data from an HDF5 group into a dictionary."""
        group = h5file[group_name]
        data = {}
        for key in group:
            if isinstance(group[key], h5py.Dataset):
                data[key] = group[key][:]
                if self.encoder.device == "gpu":
                    data[key] = self.encoder.xp.array(data[key])  # To GPU if needed
            else:  # Assume it is a group
                data[key] = self._load_hdf5_group(group, key)  # recursion

        for key in group.attrs:  # load the attributes
            data[key] = group.attrs[key]
        return data

    def save(self, filename: str):
        """Save analyzer state to disk (unified)."""
        with h5py.File(filename, "w") as f:
            # Save configuration
            f.attrs.update(self.config)

            # Save encoder state
            encoder_data = {
                "base_vectors": self.encoder.base_vectors,
                "position_vectors": self.encoder.position_vectors,
            }
            self._save_hdf5_group(f, "encoder", encoder_data)

            # Save MetaHDConservation state (if it exists)
            if hasattr(self, "meta_conservation"):
                meta_data = self.meta_conservation.get_meta_patterns()
                self._save_hdf5_group(f, "meta_conservation", meta_data)

        logger.info(f"Saved analyzer state to {filename}")

    def load(self, filename: str):
        """Load analyzer state from disk (unified)."""
        with h5py.File(filename, "r") as f:
            # Load configuration
            self.config = dict(f.attrs)

            # Reinitialize encoder with loaded config
            use_biological = self.config.get("use_biological", False)
            self.encoder_class = BiologicalEncoder if use_biological else DNAEncoder
            self.encoder = self.encoder_class(
                dimension=self.config.get("dimension"),
                device=self.config.get("device", "auto"),
                data_size=self.config.get("data_size"),  # Load data_size!
            )

            # Load encoder state
            if "encoder" in f:
                encoder_data = self._load_hdf5_group(f, "encoder")
                self.encoder.base_vectors = encoder_data["base_vectors"]
                self.encoder.position_vectors = encoder_data["position_vectors"]

            # Load MetaHDConservation state (if it exists)
            if "meta_conservation" in f:
                if not hasattr(self, "meta_conservation"):
                    self.meta_conservation = MetaHDConservation(self.encoder)
                meta_data = self._load_hdf5_group(f, "meta_conservation")
                self.meta_conservation.set_meta_patterns(meta_data)

        logger.info(f"Loaded analyzer state from {filename}")
        return self

    def load_meta_conservation(self, weight=0.9):
        """Loads and integrates the MetaHDConservation scorer."""
        self.meta_conservation = MetaHDConservation(self)

        def meta_conservation_provider(
            kmer: str, position: int = None
        ) -> Optional[float]:
            return self.meta_conservation.score(kmer)

        self.register_feature("meta_conservation", meta_conservation_provider, weight)
        return self


class AgentType(Enum):
    """Different specialized agent types for genetic analysis."""

    MOTIF_DETECTOR = auto()  # Detects regulatory elements and motifs
    STRUCTURE_ANALYZER = auto()  # Analyzes DNA/RNA secondary structure properties
    CONSERVATION_TRACKER = auto()  # Focuses on evolutionary conservation patterns
    CODON_SPECIALIST = auto()  # Specializes in coding regions and codon bias
    BOUNDARY_FINDER = auto()  # Specialized in detecting sequence boundaries/junctions
    GENERALIST = auto()  # Balanced approach across all features
    IMPUTER = auto()  # Imputation specific agent


class GeneticAgent:
    """Base agent class with core genetic analysis and RL capabilities."""

    def __init__(
        self,
        supercomputer,
        agent_id=None,
        agent_type=None,
        learning_rate=0.01,
        discount_factor=0.95,
        mode="general",  # New: 'general' or 'imputation'
    ):
        self.sc = supercomputer  # HDC supercomputer backbone
        self.id = agent_id if agent_id is not None else id(self)
        self.type = agent_type if agent_type is not None else AgentType.GENERALIST
        self.xp = supercomputer.xp  # Use same compute backend as supercomputer

        # RL parameters
        self.lr = learning_rate
        self.gamma = discount_factor

        # Knowledge representation
        self.q_vectors = {}  # State-action value vectors
        self.experience_buffer = []
        self.action_space = ["A", "T", "G", "C"]

        # Sensing and confidence
        self.confidence = 0.5
        self.history = []

        # Agent specialization
        self.specialization = self._initialize_specialization()

        # Performance tracking
        self.rewards = []
        self.imputation_accuracy = []  # Keep this to track imputation performance.
        self.training_rewards = []

        # NEW: Operating mode
        self.mode = mode
        if self.mode == "imputation":
            self.type = AgentType.IMPUTER  # Override type if in imputation mode

    def _initialize_specialization(self):
        """Define agent specialization parameters based on type."""
        # Base parameters for all agent types
        params = {
            "k_size": 5,  # Default k-mer size
            "feature_weights": {
                "motifs": 1.0,
                "gc_content": 1.0,
                "complexity": 1.0,
                "conservation": 1.0,
                "structure": 1.0,
                "coding": 1.0,
            },
        }

        # Specialized parameters by agent type
        if self.type == AgentType.MOTIF_DETECTOR:
            params["k_size"] = 6
            params["feature_weights"]["motifs"] = 2.5
            params["feature_weights"]["conservation"] = 1.5
            params["feature_weights"]["gc_content"] = 0.5

        elif self.type == AgentType.STRUCTURE_ANALYZER:
            params["k_size"] = 4
            params["feature_weights"]["structure"] = 2.5
            params["feature_weights"]["gc_content"] = 1.8
            params["feature_weights"]["complexity"] = 1.5

        elif self.type == AgentType.CONSERVATION_TRACKER:
            params["k_size"] = 7
            params["feature_weights"]["conservation"] = 2.5
            params["feature_weights"]["motifs"] = 1.2
            params["feature_weights"]["complexity"] = 0.8

        elif self.type == AgentType.CODON_SPECIALIST:
            params["k_size"] = 3
            params["feature_weights"]["coding"] = 2.5
            params["feature_weights"]["gc_content"] = 1.3
            params["feature_weights"]["conservation"] = 1.2

        elif self.type == AgentType.BOUNDARY_FINDER:
            params["k_size"] = 8
            params["feature_weights"]["motifs"] = 2.0
            params["feature_weights"]["complexity"] = 2.0
            params["feature_weights"]["conservation"] = 1.5

        elif self.type == AgentType.IMPUTER:  # Imputation agent
            params["k_size"] = 7  # Good default for imputation
            # No specific feature weighting for the general-purpose imputer.

        return params

    def analyze(self, sequence, position=None, features=None):
        """Analyze a DNA sequence or region using the agent's specialization."""
        k = self.specialization["k_size"]

        # Handle full sequence vs. specific position
        if position is None:
            # Analyze full sequence with agent's k-mer size preference
            vector = self.sc.encode_sequence(sequence, k=k)
        else:
            # Analyze specific region
            if position + k <= len(sequence):
                kmer = sequence[position : position + k]
                vector = self.sc.encode_kmer(kmer, position)
            else:
                return None

        # Apply agent's feature weighting if biological features are available
        if features and hasattr(self.sc, "_compute_feature_vector"):
            weights = self.specialization["feature_weights"]
            # Create weighted combination based on specialization
            for feature, value in features.items():
                if feature in weights and isinstance(value, (int, float)):
                    scalar = weights.get(feature, 1.0) * value
                    # Scale the vector by the weighted feature
                    vector_norm = float(self.xp.linalg.norm(vector))
                    if vector_norm > 0:
                        influence = 0.2 * scalar / sum(weights.values())
                        vector *= 1.0 + influence
                        # Re-normalize
                        vector = vector / self.xp.linalg.norm(vector)

        # Update agent confidence based on reading consistency
        if len(self.history) > 0:
            vector_norm = float(self.xp.linalg.norm(vector))
            history_norms = [float(self.xp.linalg.norm(v)) for v in self.history]
            consistency = 1.0 - min(
                1.0, abs(vector_norm - sum(history_norms) / len(history_norms))
            )
            self.confidence = 0.8 * self.confidence + 0.2 * consistency

        # Update history with latest reading
        self.history.append(vector)
        if len(self.history) > 10:
            self.history = self.history[-10:]  # Keep only most recent readings

        return {
            "agent_id": self.id,
            "agent_type": self.type.name,
            "position": position,
            "reading": vector,
            "confidence": self.confidence,
        }

    def predict_base(self, state_vector, epsilon=0.1):
        """Predict next base using epsilon-greedy policy with Q-vectors."""
        # Exploration
        if self.xp.random.random() < epsilon:
            return self.xp.random.choice(self.action_space)

        # Exploitation - choose best action using Q-values
        q_values = {}
        state_hash = hash(state_vector.tobytes())

        for action in self.action_space:
            state_action = f"{state_hash}_{action}"
            if state_action not in self.q_vectors:
                # Initialize with base vector if not seen before
                self.q_vectors[state_action] = self.sc.base_vectors[action].copy()

            # Q-value as similarity
            q_values[action] = float(
                self.xp.dot(state_vector, self.q_vectors[state_action])
            )

        return max(q_values, key=lambda a: q_values[a])

    def update_q_vector(self, state, action, reward, next_state):
        """Update Q-vector using TD learning in HDC space."""
        state_hash = hash(state.tobytes())
        state_action = f"{state_hash}_{action}"

        if state_action not in self.q_vectors:
            self.q_vectors[state_action] = self.sc.base_vectors[action].copy()

        # Get maximum Q-value for next state
        next_q_values = []
        next_hash = hash(next_state.tobytes())

        for next_action in self.action_space:
            next_state_action = f"{next_hash}_{next_action}"
            if next_state_action not in self.q_vectors:
                self.q_vectors[next_state_action] = self.sc.base_vectors[
                    next_action
                ].copy()

            next_q_values.append(
                float(self.xp.dot(next_state, self.q_vectors[next_state_action]))
            )

        max_next_q = max(next_q_values) if next_q_values else 0

        # TD update
        current_q = float(self.xp.dot(state, self.q_vectors[state_action]))
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        # HDC-compatible update using bundling
        delta = self.lr * td_error
        self.q_vectors[state_action] = self.sc.bundle(
            self.q_vectors[state_action],
            state,
            alpha=1.0 - min(0.9, abs(delta)),  # Limit alpha to prevent extreme values
        )

    def add_experience(self, state, action, reward, next_state):
        """Add experience tuple to buffer for replay training."""
        self.experience_buffer.append((state, action, reward, next_state))

        # Keep buffer at manageable size
        if len(self.experience_buffer) > 10000:
            # Keep 50% of buffer, prioritizing more recent experiences
            self.experience_buffer = self.experience_buffer[-5000:]

    def train_from_buffer(self, batch_size=32):
        """Train on random batch from experience buffer."""
        if len(self.experience_buffer) < batch_size:
            return 0

        # Sample random batch without replacement
        indices = self.xp.random.choice(
            len(self.experience_buffer),
            min(batch_size, len(self.experience_buffer)),
            replace=False,
        )

        # Train on each experience
        updates = 0
        for idx in indices:
            state, action, reward, next_state = self.experience_buffer[idx]
            self.update_q_vector(state, action, reward, next_state)
            updates += 1

        return updates

    def get_state_vector(self, context):  # Added from GeneticRLAgent
        """Convert context into state vector."""
        if isinstance(context, str):
            return self.sc.encode_sequence(context)
        elif isinstance(context, list):
            # Combine multiple context segments
            vectors = [self.sc.encode_sequence(seg) for seg in context]
            combined = sum(vectors) / len(vectors)
            return self.sc.xp.array(combined)
        else:
            return context  # Assume already a vector

    def impute_segment(
        self, known_segments, missing_length, epsilon=0.05
    ):  # Added from GeneticRLAgent
        """Impute a missing DNA segment using RL in HDC space."""
        # Encode context (known segments)
        context_vector = self.get_state_vector(known_segments)

        # Generate candidate bases iteratively
        imputed_segment = ""
        state_vector = context_vector.copy()

        for _ in range(missing_length):
            # Choose action (next base)
            next_base = self.predict_base(state_vector, epsilon)

            # Update state
            imputed_segment += next_base
            next_vector = self.sc.encode_kmer(
                imputed_segment[-min(len(imputed_segment), 7) :]
            )
            next_state = self.sc.bundle(state_vector, next_vector, alpha=0.7)
            state_vector = next_state

        return imputed_segment

    def calculate_reward(self, imputed, actual):  # Added from GeneticRLAgent
        """Calculate reward for imputation quality."""
        # Base reward for matching bases
        match_reward = sum(i == a for i, a in zip(imputed, actual)) / max(
            len(actual), 1
        )

        # Bonus for preserving GC content
        imputed_gc = (imputed.count("G") + imputed.count("C")) / max(len(imputed), 1)
        actual_gc = (actual.count("G") + actual.count("C")) / max(len(actual), 1)
        gc_reward = 1 - abs(imputed_gc - actual_gc)

        # Bonus for k-mer similarity (using HDC)
        kmer_sim = 0
        if len(imputed) >= 3 and len(actual) >= 3:
            imputed_vec = self.sc.encode_sequence(imputed, k=3)
            actual_vec = self.sc.encode_sequence(actual, k=3)
            kmer_sim = self.xp.dot(imputed_vec, actual_vec)

        # Combined reward
        return 0.6 * match_reward + 0.2 * gc_reward + 0.2 * kmer_sim

    def train(  # Added and adapted from GeneticRLAgent
        self, training_sequences, epochs=10, epsilon_start=0.3, epsilon_decay=0.9
    ):
        """Train the RL agent on known sequences."""
        logger.info(
            f"Starting training for {epochs} epochs on {len(training_sequences)} sequences"
        )

        epsilon = epsilon_start

        for epoch in range(epochs):
            total_reward = 0
            total_segments = 0

            for sequence in tqdm(
                training_sequences, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                # Skip sequences that are too short
                if len(sequence) < 15:
                    continue

                if self.mode == "imputation":
                    # Create artificial missing segments
                    seq_len = len(sequence)
                    gap_start = self.xp.random.randint(0, seq_len // 2)
                    gap_length = self.xp.random.randint(3, min(10, seq_len // 4))

                    known_prefix = sequence[:gap_start]
                    known_suffix = sequence[gap_start + gap_length :]
                    actual_missing = sequence[gap_start : gap_start + gap_length]

                    # Skip if segments are too short
                    if len(known_prefix) < 3 or len(known_suffix) < 3:
                        continue

                    # Impute the missing segment
                    imputed = self.impute_segment(
                        [known_prefix, known_suffix], gap_length, epsilon
                    )

                    # Calculate reward
                    reward = self.calculate_reward(imputed, actual_missing)
                    total_reward += reward
                    total_segments += 1

                    # Update Q-vectors through experience replay
                    self._update_from_experience(
                        known_prefix,
                        known_suffix,
                        imputed,
                        actual_missing,
                        reward,  # This was already in the previous definition
                    )
                else:  # mode == "general"
                    # For general training sample subsequences.

                    seq_len = len(sequence)
                    subseq_start = self.xp.random.randint(
                        0, seq_len - 10
                    )  # at least 10 bases

                    context = sequence[
                        subseq_start : subseq_start + 10
                    ]  # 10 bases context
                    targets = sequence[
                        subseq_start + 10 : subseq_start + 15
                    ]  # 5 bases targets

                    if len(context) < 5 or len(targets) < 1:
                        continue

                    state_vector = self.get_state_vector(context)

                    for target_base in targets:
                        predicted_base = self.predict_base(state_vector, epsilon)
                        reward = 1.0 if predicted_base == target_base else -0.2
                        total_reward += reward
                        total_segments += 1

                        next_kmer = (
                            context[-min(len(context), 7) :] + predicted_base
                        )  # Append *predicted* base.
                        next_vector = self.sc.encode_kmer(next_kmer)
                        next_state = self.sc.bundle(
                            state_vector, next_vector, alpha=0.7
                        )

                        self.add_experience(
                            state_vector, predicted_base, reward, next_state
                        )
                        state_vector = next_state  # Prepare for next iteration

                    self.train_from_buffer()  # Train on batch

            # Decay exploration rate
            epsilon *= epsilon_decay

            # Track performance
            avg_reward = total_reward / max(total_segments, 1)
            self.training_rewards.append(avg_reward)
            if self.mode == "imputation":
                logger.info(
                    f"Epoch {epoch + 1} - Average reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1} - Average reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}"
                )

            # Evaluate on a sample
            if epoch % 2 == 0 and self.mode == "imputation":
                self.evaluate(training_sequences[: min(100, len(training_sequences))])

        return self.training_rewards

    def _update_from_experience(
        self, prefix, suffix, imputed, actual, final_reward
    ):  # Added from GeneticAgent
        """Update Q-vectors based on experience."""
        # Create state sequence
        state_vector = self.get_state_vector([prefix, suffix])

        for i, (imp_base, act_base) in enumerate(zip(imputed, actual)):
            # Current state
            current_state = state_vector.copy()

            # Get reward
            match_reward = 1.0 if imp_base == act_base else -0.2

            # Update context for next state
            next_kmer = imp_base
            if i > 0:
                next_kmer = imputed[max(0, i - 6) : i + 1]
            next_vector = self.sc.encode_kmer(next_kmer)
            next_state = self.sc.bundle(state_vector, next_vector, alpha=0.7)

            # Store experience
            self.experience_buffer.append(
                (current_state, imp_base, match_reward, next_state)
            )

            # Update state
            state_vector = next_state

        # Train on random batch from experience buffer  (Moved to train, no need to repeat)
        # if len(self.experience_buffer) > 100:
        #    batch_size = min(64, len(self.experience_buffer))
        #    batch_indices = self.xp.random.choice(
        #        len(self.experience_buffer), batch_size, replace=False
        #    )
        #
        #    for idx in batch_indices:
        #        state, action, reward, next_state = self.experience_buffer[idx]
        #        self.update_q_vector(state, action, reward, next_state)
        #
        #    # Limit buffer size
        #    if len(self.experience_buffer) > 10000:
        #        self.experience_buffer = self.experience_buffer[-5000:]

    def evaluate(self, test_sequences, gap_length=5):  # Added from GeneticRLAgent
        """Evaluate imputation performance on test sequences."""
        correct = 0
        total = 0

        for seq in tqdm(test_sequences[:100], desc="Evaluating"):
            if len(seq) < 15:
                continue

            # Create test gap
            seq_len = len(seq)
            gap_start = seq_len // 3

            prefix = seq[:gap_start]
            actual = seq[gap_start : gap_start + gap_length]
            suffix = seq[gap_start + gap_length :]

            # Impute
            imputed = self.impute_segment([prefix, suffix], gap_length, epsilon=0)

            # Count matches
            matches = sum(i == a for i, a in zip(imputed, actual))
            correct += matches
            total += gap_length

        accuracy = correct / max(total, 1)
        self.imputation_accuracy.append(accuracy)
        logger.info(f"Evaluation - Accuracy: {accuracy:.4f} ({correct}/{total} bases)")

        return accuracy

    def save(self, h5file, prefix=""):  # Added from GeneticAgent
        """Save agent state to an open H5 file."""
        prefix = f"{prefix}/" if prefix else ""
        group = h5file.create_group(f"{prefix}agent_{self.id}")

        # Save metadata
        group.attrs["id"] = self.id
        group.attrs["type"] = self.type.name
        group.attrs["confidence"] = self.confidence
        group.attrs["mode"] = self.mode  # Save the mode.

        # Save Q-vectors
        qv_group = group.create_group("q_vectors")
        for key, vector in self.q_vectors.items():
            # Convert to NumPy if needed
            vector_np = (
                self.xp.asnumpy(vector) if hasattr(self.xp, "asnumpy") else vector
            )
            qv_group.create_dataset(str(key), data=vector_np)

        # Save rewards
        if self.rewards:
            group.create_dataset("rewards", data=self.rewards)

        # Save imputation accuracies
        if self.imputation_accuracy:
            group.create_dataset("imputation_accuracy", data=self.imputation_accuracy)

        if self.training_rewards:
            group.create_dataset("training_rewards", data=self.training_rewards)

    def load(self, h5file, prefix=""):  # Added from GeneticAgent
        """Load agent state from an open H5 file."""
        prefix = f"{prefix}/" if prefix else ""
        group = h5file[f"{prefix}agent_{self.id}"]

        # Load metadata
        self.confidence = group.attrs["confidence"]
        self.type = AgentType[group.attrs["type"]]  # Load agent type as enum
        self.mode = group.attrs.get("mode", "general")  # Load the mode.

        # Load Q-vectors
        qv_group = group["q_vectors"]
        for key in qv_group:
            vector = qv_group[key][:]
            if self.sc.device == "gpu":
                vector = self.xp.array(vector)
            self.q_vectors[key] = vector

        # Load rewards
        if "rewards" in group:
            self.rewards = group["rewards"][:]

        # Load imputation accuracies
        if "imputation_accuracy" in group:
            self.imputation_accuracy = group["imputation_accuracy"][:]

        if "training_rewards" in group:
            self.training_rewards = group["training_rewards"][:]


class GeneticSwarm:
    """Advanced genetic sequence prediction system using adaptive swarm intelligence.

    This swarm-based system dynamically configures specialized agents based on
    sequence characteristics, enabling more effective sequence analysis, prediction,
    and imputation for genomic data.
    """

    def __init__(
        self, supercomputer, swarm_size=10, agent_types=None, sample_sequences=None
    ):
        """Initialize the genetic swarm with data-driven agent configuration.

        Args:
            supercomputer: HDC computing engine
            swarm_size: Total number of agents in the swarm
            agent_types: Dict mapping AgentType to count, or None for data-driven defaults
            sample_sequences: Sample DNA sequences for data-driven initialization
        """
        self.sc = supercomputer
        self.xp = supercomputer.xp

        # Agent configuration - use data-driven approach if no config provided
        if agent_types is None and sample_sequences:
            agent_types = self._determine_agent_distribution(
                sample_sequences, swarm_size
            )
        elif agent_types is None:
            # Default distribution if no samples available
            agent_count = max(6, swarm_size)
            agent_types = {
                AgentType.MOTIF_DETECTOR: max(1, agent_count // 5),
                AgentType.STRUCTURE_ANALYZER: max(1, agent_count // 5),
                AgentType.CONSERVATION_TRACKER: max(1, agent_count // 5),
                AgentType.CODON_SPECIALIST: max(1, agent_count // 5),
                AgentType.BOUNDARY_FINDER: max(1, agent_count // 10),
                AgentType.GENERALIST: max(1, agent_count // 10),
            }

        # Initialize agent swarm
        self.agents = []
        agent_id = 0
        for agent_type, count in agent_types.items():
            for _ in range(count):
                agent = GeneticAgent(supercomputer, agent_id, agent_type)
                self.agents.append(agent)
                agent_id += 1

        logger.info(f"Initialized swarm with {len(self.agents)} agents")

        # Dynamic agent weighting
        self.agent_weights = {agent.id: 0.5 for agent in self.agents}

        # Swarm convergence
        self.convergence = 0.0

        # Performance tracking
        self.swarm_rewards = []
        self.training_history = []
        self.validation_history = []
        self.current_epoch = 0

        # Cache for feature extraction
        self.feature_cache = {}

        # Call adaptive initialization if sample sequences are provided
        if sample_sequences:
            self._adaptive_initialization(sample_sequences)

    def _determine_agent_distribution(self, sequences, swarm_size):
        """Determine optimal agent distribution based on sequence characteristics.

        Args:
            sequences: List of DNA/RNA sequences for analyzing
            swarm_size: Total number of agents to distribute

        Returns:
            Dict mapping AgentType to agent count
        """
        # Initialize all agent type counts to 0
        distribution = {agent_type: 0 for agent_type in AgentType}

        # Ensure we always have at least one generalist
        distribution[AgentType.GENERALIST] = 1
        remaining_slots = swarm_size - 1

        # If not enough sequences for analysis, use fixed distribution
        if not sequences or len(sequences) < 3:
            return self._get_default_distribution(swarm_size)

        # Analyze sequence characteristics on a sample
        sample_size = min(10, len(sequences))
        sample_seqs = (
            sequences[:sample_size] if len(sequences) > sample_size else sequences
        )

        # Calculate feature scores to determine agent allocation
        feature_scores = {
            "motifs": 0,
            "structure": 0,
            "conservation": 0,
            "coding": 0,
            "boundary": 0,
        }

        # Analyze each sample sequence
        for seq in sample_seqs:
            # Skip sequences that are too short
            if len(seq) < 30:
                continue

            # Calculate feature scores
            scores = self._analyze_sequence_features(seq)

            # Accumulate scores
            for feature, score in scores.items():
                if feature in feature_scores:
                    feature_scores[feature] += score

        # If no valid sequences were analyzed, use default distribution
        if all(score == 0 for score in feature_scores.values()):
            return self._get_default_distribution(swarm_size)

        # Normalize scores
        total_score = sum(feature_scores.values())
        if total_score > 0:
            normalized_scores = {k: v / total_score for k, v in feature_scores.items()}
        else:
            normalized_scores = {k: 1.0 / len(feature_scores) for k in feature_scores}

        # Map feature scores to agent types
        agent_scores = {
            AgentType.MOTIF_DETECTOR: normalized_scores["motifs"],
            AgentType.STRUCTURE_ANALYZER: normalized_scores["structure"],
            AgentType.CONSERVATION_TRACKER: normalized_scores["conservation"],
            AgentType.CODON_SPECIALIST: normalized_scores["coding"],
            AgentType.BOUNDARY_FINDER: normalized_scores["boundary"],
        }

        # Allocate agents proportionally, ensuring each type gets at least 1 agent if score > 0.1
        for agent_type, score in agent_scores.items():
            if score > 0.1 and distribution[agent_type] == 0:
                distribution[agent_type] = 1
                remaining_slots -= 1

        # Distribute remaining slots proportionally
        if remaining_slots > 0:
            # Get non-zero agent types
            nonzero_types = [
                at
                for at, count in distribution.items()
                if count > 0 or agent_scores.get(at, 0) > 0
            ]
            nonzero_types = [at for at in nonzero_types if at != AgentType.GENERALIST]

            # Calculate allocation for remaining slots
            if nonzero_types:
                # Normalize scores for non-zero types
                nonzero_scores = {at: agent_scores.get(at, 0) for at in nonzero_types}
                total_nonzero = sum(nonzero_scores.values())

                if total_nonzero > 0:
                    # Distribute proportionally
                    for at in nonzero_types:
                        # Calculate proportional allocation
                        alloc = int(
                            remaining_slots * (nonzero_scores[at] / total_nonzero)
                        )
                        distribution[at] += alloc
                        remaining_slots -= alloc

            # Assign any leftover slots to generalist
            distribution[AgentType.GENERALIST] += remaining_slots

        logger.info(f"Data-driven agent distribution: {distribution}")
        return distribution

    def _get_default_distribution(self, swarm_size):
        """Get default agent distribution."""
        agent_count = max(6, swarm_size)
        return {
            AgentType.MOTIF_DETECTOR: max(1, agent_count // 5),
            AgentType.STRUCTURE_ANALYZER: max(1, agent_count // 5),
            AgentType.CONSERVATION_TRACKER: max(1, agent_count // 5),
            AgentType.CODON_SPECIALIST: max(1, agent_count // 5),
            AgentType.BOUNDARY_FINDER: max(1, agent_count // 10),
            AgentType.GENERALIST: max(1, agent_count // 5),
        }

    def _analyze_sequence_features(self, sequence):
        """Analyze sequence to calculate feature scores for agent distribution.

        Args:
            sequence: DNA/RNA sequence

        Returns:
            Dict of feature scores
        """
        # Initialize feature scores
        scores = {
            "motifs": 0,
            "structure": 0,
            "conservation": 0,
            "coding": 0,
            "boundary": 0,
        }

        # Skip if sequence is too short
        if len(sequence) < 30:
            return scores

        # Sample positions throughout sequence
        window_size = 10
        stride = max(1, len(sequence) // 20)  # Sample ~20 windows

        for pos in range(0, len(sequence) - window_size, stride):
            # Extract features
            features = self._extract_features(sequence, pos, window_size)

            # Update feature scores
            scores["motifs"] += features.get("motifs", 0)
            scores["structure"] += features.get("structure", 0)
            scores["conservation"] += features.get("conservation", 0)
            scores["coding"] += features.get("coding", 0)

            # Check for potential boundaries (sharp changes in features)
            if pos > 0:
                prev_features = self._extract_features(
                    sequence, pos - stride, window_size
                )
                feature_change = sum(
                    abs(features.get(f, 0) - prev_features.get(f, 0))
                    for f in ["gc_content", "complexity", "motifs"]
                )
                scores["boundary"] += feature_change

        # Normalize by number of windows
        num_windows = max(1, (len(sequence) - window_size) // stride)
        for feature in scores:
            scores[feature] /= num_windows

        return scores

    def _adaptive_initialization(self, sample_sequences):
        """Perform adaptive initialization of agents based on sequence characteristics.

        Args:
            sample_sequences: Sample DNA sequences for initialization
        """
        # Skip if no sequences provided
        if not sample_sequences:
            return

        logger.info("Performing adaptive agent initialization")

        # Initialize specialization parameters for each agent
        for agent in self.agents:
            # Adapt k-mer size based on sequence complexity
            if sample_sequences:
                # Sample a sequence
                seq_idx = self.xp.random.randint(0, len(sample_sequences))
                seq = sample_sequences[seq_idx]

                # Detect optimal k-mer size if sequence is long enough
                if len(seq) > 30:
                    optimal_k = detect_optimal_kmer_size(seq)

                    # Adjust k-mer size based on agent type
                    if agent.type == AgentType.MOTIF_DETECTOR:
                        agent.specialization["k_size"] = min(8, optimal_k + 1)
                    elif agent.type == AgentType.STRUCTURE_ANALYZER:
                        agent.specialization["k_size"] = max(3, min(6, optimal_k - 1))
                    elif agent.type == AgentType.CONSERVATION_TRACKER:
                        agent.specialization["k_size"] = optimal_k
                    elif agent.type == AgentType.CODON_SPECIALIST:
                        agent.specialization["k_size"] = 3  # Codon size
                    elif agent.type == AgentType.BOUNDARY_FINDER:
                        agent.specialization["k_size"] = max(6, optimal_k + 2)
                    else:  # GENERALIST
                        agent.specialization["k_size"] = optimal_k

        # Calculate average GC content for feature weight adjustments
        if sample_sequences:
            gc_contents = []
            for seq in sample_sequences:
                if len(seq) > 0:
                    gc = (seq.count("G") + seq.count("C")) / len(seq)
                    gc_contents.append(gc)

            if gc_contents:
                avg_gc = sum(gc_contents) / len(gc_contents)

                # Adjust feature weights based on GC content
                for agent in self.agents:
                    if agent.type == AgentType.MOTIF_DETECTOR:
                        # Higher motif focus in GC-rich regions
                        if avg_gc > 0.55:
                            agent.specialization["feature_weights"]["motifs"] *= 1.5
                    elif agent.type == AgentType.STRUCTURE_ANALYZER:
                        # GC content affects structure stability
                        if avg_gc > 0.60:
                            agent.specialization["feature_weights"]["structure"] *= 1.3
                            agent.specialization["feature_weights"]["gc_content"] *= 1.2

        logger.info("Adaptive initialization complete")

    @lru_cache(maxsize=128)
    def _extract_features(self, sequence, position, window_size):
        """Extract biological features at a position (cached for efficiency)."""
        # Check if result is in cache
        cache_key = (sequence[position : position + window_size], position)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Extract window around position
        start = max(0, position - window_size // 2)
        end = min(len(sequence), position + window_size // 2)
        window = sequence[start:end]

        if not window:
            return {}

        # Calculate basic features
        features = {}

        # GC content
        gc_count = window.count("G") + window.count("C")
        features["gc_content"] = gc_count / len(window)

        # Motif detection (common regulatory elements)
        motifs = {
            "TATA": 0.9,
            "CAAT": 0.8,
            "GATA": 0.7,
            "AATAAA": 0.85,
            "CCAAT": 0.75,
            "GCCNNNGGC": 0.8,
            "CACGTG": 0.85,
        }

        motif_score = 0.0
        for motif, importance in motifs.items():
            if len(motif) <= len(window):
                for i in range(len(window) - len(motif) + 1):
                    subseq = window[i : i + len(motif)]
                    # Check exact match
                    if subseq == motif:
                        motif_score = max(motif_score, importance)
                    # Check fuzzy match (allowing Ns and one mismatch)
                    elif sum(a != b and b != "N" for a, b in zip(subseq, motif)) <= 1:
                        motif_score = max(motif_score, importance * 0.7)

        features["motifs"] = motif_score

        # Sequence complexity (unique k-mers ratio)
        k = 3  # Use 3-mers for complexity
        unique_kmers = set()
        for i in range(len(window) - k + 1):
            kmer = window[i : i + k]
            if "N" not in kmer:
                unique_kmers.add(kmer)

        features["complexity"] = (
            len(unique_kmers) / (len(window) - k + 1) if len(window) >= k else 0.5
        )

        # Coding potential (simple codon bias measure)
        if len(window) >= 3:
            codon_counts = defaultdict(int)
            for i in range(0, len(window) - 2, 3):
                codon = window[i : i + 3]
                if "N" not in codon:
                    codon_counts[codon] += 1

            # Common codons in coding regions
            common_codons = {
                "ATG",
                "CAG",
                "GAG",
                "AAG",
                "CTG",
                "GCC",
                "TTC",
                "AAC",
                "GGC",
                "TAC",
            }
            coding_score = sum(codon_counts[codon] for codon in common_codons) / max(
                1, sum(codon_counts.values())
            )
            features["coding"] = coding_score
        else:
            features["coding"] = 0.5

        # Simple structure propensity (stems/hairpins tend to have complementary bases)
        if len(window) >= 6:
            complementary_pairs = 0
            total_pairs = 0

            complement = {"A": "T", "T": "A", "G": "C", "C": "G"}

            for i in range(len(window) // 2):
                left = window[i]
                right = window[len(window) - i - 1]

                if left in complement and right in complement:
                    total_pairs += 1
                    if complement[left] == right:
                        complementary_pairs += 1

            features["structure"] = (
                complementary_pairs / total_pairs if total_pairs > 0 else 0.5
            )
        else:
            features["structure"] = 0.5

        # Mock conservation score (would use real data in production)
        features["conservation"] = 0.5

        # Cache result
        self.feature_cache[cache_key] = features

        # Limit cache size
        if len(self.feature_cache) > 1000:
            keys_to_remove = list(self.feature_cache.keys())[:-500]  # Keep newest 500
            for key in keys_to_remove:
                self.feature_cache.pop(key, None)

        return features

    def predict_sequence(self, context, length, epsilon=0.05):
        """Predict a DNA sequence using the agent swarm."""
        predicted = ""

        # Create initial state from context
        state_vector = self.sc.encode_sequence(context)

        # Generate sequence one base at a time
        for _ in range(length):
            # Get predictions from all agents
            predictions = defaultdict(list)
            for agent in self.agents:
                base = agent.predict_base(state_vector, epsilon)
                # Store agent ID and its weighted vote
                weight = self.agent_weights[agent.id] * agent.confidence
                predictions[base].append((agent.id, weight))

            # Weighted voting - sum up weights for each predicted base
            base_scores = {
                base: sum(weight for _, weight in preds)
                for base, preds in predictions.items()
            }

            # Select winning base
            next_base = max(base_scores, key=base_scores.get)
            predicted += next_base

            # Update state with new base
            next_kmer = predicted[-min(len(predicted), 7) :]
            next_vector = self.sc.encode_kmer(next_kmer)
            state_vector = self.sc.bundle(state_vector, next_vector, alpha=0.7)

        return predicted

    def impute_segment(self, prefix, suffix, gap_length, epsilon=0.01):
        """Impute a missing DNA segment between prefix and suffix."""
        # Handle empty inputs
        if not prefix and not suffix:
            return "N" * gap_length

        # Create context from prefix
        context = prefix[-10:] if len(prefix) > 10 else prefix

        # Initial prediction using just the prefix
        candidates = []

        # Generate multiple candidates with different random influences
        # for beam-search like exploration
        n_candidates = 5
        for i in range(n_candidates):
            # Vary epsilon to get diverse candidates
            cand_epsilon = epsilon + (i * 0.05)
            cand = self.predict_sequence(context, gap_length, epsilon=cand_epsilon)
            candidates.append(cand)

        # If we have a suffix, score candidates by how well they connect to it
        if suffix:
            suffix_context = suffix[:10] if len(suffix) > 10 else suffix
            suffix_vector = self.sc.encode_sequence(suffix_context)

            # Score each candidate by similarity to suffix
            best_score = -float("inf")
            best_candidate = candidates[0]

            for cand in candidates:
                # Get vector for candidate
                cand_vector = self.sc.encode_sequence(cand)

                # Score based on similarity to suffix
                similarity = float(self.xp.dot(cand_vector, suffix_vector))

                if similarity > best_score:
                    best_score = similarity
                    best_candidate = cand

            return best_candidate
        else:
            # Without suffix, return the first candidate (standard prediction)
            return candidates[0]

    def analyze_region(self, sequence, position, window_size=10):
        """Analyze a region of DNA using all agents in the swarm."""
        # Extract features for this region
        features = self._extract_features(sequence, position, window_size)

        # Collect readings from all agents
        agent_readings = []
        for agent in self.agents:
            reading = agent.analyze(sequence, position, features)
            if reading:
                agent_readings.append(reading)

        # Analyze consensus and divergence
        readings = [r["reading"] for r in agent_readings]
        consensus = None

        if readings:
            # Calculate swarm consensus (average reading)
            consensus = sum(readings) / len(readings)

            # Measure disagreement as average distance from consensus
            divergence = sum(
                float(self.xp.linalg.norm(r - consensus)) for r in readings
            ) / len(readings)

            # Update convergence score (higher when agents agree)
            self.convergence = 1.0 / (1.0 + divergence)

        return {
            "position": position,
            "agent_readings": agent_readings,
            "features": features,
            "consensus": consensus,
            "convergence": self.convergence,
        }

    def train_on_sequence(self, sequence, epochs=1, batch_size=32):
        """Train the swarm on a DNA sequence."""
        if len(sequence) < 20:
            return 0

        total_reward = 0
        total_samples = 0

        for epoch in range(epochs):
            # Sample positions throughout sequence
            n_samples = min(100, max(1, len(sequence) // 10))
            positions = self.xp.random.choice(
                len(sequence) - 15, min(n_samples, len(sequence) - 15), replace=False
            )

            for pos in positions:
                # Define context and target
                context_size = 10
                target_size = 5

                context = sequence[pos : pos + context_size]
                target = sequence[pos + context_size : pos + context_size + target_size]

                if len(context) < 5 or len(target) < 3:
                    continue

                # Create initial state vector
                state_vector = self.sc.encode_sequence(context)

                # Train each agent on this example
                for agent in self.agents:
                    current_state = state_vector.copy()
                    agent_reward = 0

                    for i, base in enumerate(target):
                        # Agent predicts next base
                        predicted = agent.predict_base(current_state, epsilon=0.2)

                        # Calculate reward
                        match_reward = 1.0 if predicted == base else -0.2
                        agent_reward += match_reward

                        # Calculate next state
                        next_kmer = context[-min(len(context), 7) :] + target[: i + 1]
                        next_vector = self.sc.encode_sequence(next_kmer)

                        # Store experience
                        agent.add_experience(
                            current_state, predicted, match_reward, next_vector
                        )

                        # Update current state for next prediction
                        current_state = next_vector.copy()

                    # Train on batch of experiences
                    agent.train_from_buffer(batch_size)

                    # Track agent's performance
                    avg_reward = agent_reward / len(target)
                    agent.rewards.append(avg_reward)

                    total_reward += agent_reward
                    total_samples += len(target)

            # Update agent weights based on recent performance
            self._update_agent_weights()

        # Calculate overall reward for this training sequence
        overall_reward = total_reward / max(1, total_samples)
        self.swarm_rewards.append(overall_reward)

        return overall_reward

    def _update_agent_weights(self):
        """Update weights of agents based on performance."""
        # Calculate recent performance for each agent
        agent_performance = {}

        for agent in self.agents:
            # Use recent rewards (last 10) if available
            if agent.rewards:
                recent_rewards = (
                    agent.rewards[-10:] if len(agent.rewards) > 10 else agent.rewards
                )
                agent_performance[agent.id] = sum(recent_rewards) / len(recent_rewards)
            else:
                agent_performance[agent.id] = 0.5  # Default performance

        # Calculate performance by agent type
        type_performance = defaultdict(list)
        for agent in self.agents:
            type_performance[agent.type].append(agent_performance[agent.id])

        # Calculate average performance by type
        avg_by_type = {
            agent_type: sum(perfs) / len(perfs)
            for agent_type, perfs in type_performance.items()
        }

        # Update weights using a blend of individual and type performance
        for agent in self.agents:
            individual_perf = agent_performance[agent.id]
            type_perf = avg_by_type[agent.type]

            # Weighted blend (70% individual, 30% type average)
            new_weight = (0.7 * individual_perf + 0.3 * type_perf) * agent.confidence

            # Update smoothly to avoid dramatic changes
            self.agent_weights[agent.id] = (
                0.8 * self.agent_weights[agent.id] + 0.2 * new_weight
            )

    def train(
        self, sequences, epochs=5, batch_size=32, validation_set=None, validate_every=1
    ):
        """Train the model on a set of sequences.

        Args:
            sequences: List of DNA/RNA sequences for training
            epochs: Number of training epochs
            batch_size: Batch size for experience replay
            validation_set: Separate sequences for validation (uses training set if None)
            validate_every: Run validation every N epochs

        Returns:
            Dict containing training metrics
        """
        start_epoch = self.current_epoch
        total_sequences = len(sequences)
        logger.info(f"Training on {total_sequences} sequences for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch += 1
            epoch_reward = 0
            trained_sequences = 0

            # Process each training sequence in random order
            sequence_order = np.random.permutation(total_sequences)

            for idx in tqdm(sequence_order, desc=f"Epoch {self.current_epoch}"):
                sequence = sequences[idx]
                if len(sequence) < 30:
                    continue

                # Train swarm on this sequence
                reward = self.train_on_sequence(
                    sequence, epochs=1, batch_size=batch_size
                )
                epoch_reward += reward
                trained_sequences += 1

            # Calculate average reward for this epoch
            avg_reward = epoch_reward / max(1, trained_sequences)
            self.training_history.append(avg_reward)

            # Run validation if scheduled
            if (
                validation_set is not None
                and (self.current_epoch - start_epoch) % validate_every == 0
            ):
                val_accuracy = self.evaluate(validation_set)
                self.validation_history.append(val_accuracy)
                logger.info(
                    f"Epoch {self.current_epoch} - Train: {avg_reward:.4f}, Val: {val_accuracy:.4f}"
                )
            else:
                logger.info(f"Epoch {self.current_epoch} - Train: {avg_reward:.4f}")

            # Report agent type performance
            self._report_agent_performance()

        # Final metrics
        metrics = {
            "final_reward": self.training_history[-1] if self.training_history else 0,
            "training_history": self.training_history,
            "validation_history": self.validation_history,
        }

        return metrics

    def _report_agent_performance(self):
        """Report performance metrics for each agent type."""
        # Group agents by type
        type_agents = defaultdict(list)
        for agent in self.agents:
            type_agents[agent.type].append(agent.id)

        # Calculate average weight by type
        for agent_type, agent_ids in type_agents.items():
            avg_weight = sum(self.agent_weights[aid] for aid in agent_ids) / len(
                agent_ids
            )
            logger.info(
                f"  {agent_type.name}: avg_weight={avg_weight:.4f}, count={len(agent_ids)}"
            )

    def evaluate(self, test_sequences, gap_length=10, num_samples=None):
        """Evaluate the swarm on test sequences."""
        if not test_sequences:
            return 0.0

        total_correct = 0
        total_bases = 0

        # Determine number of samples
        if num_samples is None:
            num_samples = min(100, sum(max(0, len(seq) - 30) for seq in test_sequences))

        sample_count = 0

        # Evaluate on random segments from test sequences
        for sequence in test_sequences:
            if len(sequence) < 30:
                continue

            # Determine samples to take from this sequence
            seq_samples = max(
                1,
                int(
                    num_samples * (len(sequence) / sum(len(s) for s in test_sequences))
                ),
            )

            for _ in range(seq_samples):
                if sample_count >= num_samples:
                    break

                # Generate random gap
                gap_start = self.xp.random.randint(10, len(sequence) - gap_length - 10)

                prefix = sequence[:gap_start]
                suffix = sequence[gap_start + gap_length :]
                actual = sequence[gap_start : gap_start + gap_length]

                # Impute the gap
                predicted = self.impute_segment(prefix, suffix, gap_length)

                # Calculate accuracy
                matches = sum(p == a for p, a in zip(predicted, actual))
                total_correct += matches
                total_bases += len(actual)

                sample_count += 1

        accuracy = total_correct / max(1, total_bases)
        return accuracy

    def predict(self, context, length, epsilon=0.01):
        """Predict DNA sequence given a context."""
        return self.predict_sequence(context, length, epsilon)

    def suggest_exploration(self, sequence, explored_positions=None, max_suggestions=5):
        """Suggest regions for further exploration based on swarm interest/disagreement."""
        if explored_positions is None:
            explored_positions = set()

        # Score unexplored positions
        position_scores = {}
        for pos in range(0, len(sequence) - 10, 5):  # Use stride for efficiency
            if pos in explored_positions:
                continue

            # Analyze region with all agents
            result = self.analyze_region(sequence, pos, window_size=10)
            agent_readings = result.get("agent_readings", [])

            if not agent_readings:
                continue

            # Calculate interest score:
            # 1. Based on disagreement (higher variance = more interesting)
            readings = [r["reading"] for r in agent_readings]
            reading_norms = [float(self.xp.linalg.norm(r)) for r in readings]

            # Variance indicates disagreement
            if len(reading_norms) > 1:
                disagreement = float(self.xp.var(reading_norms))
            else:
                disagreement = 0.0

            # 2. Based on feature richness
            features = result.get("features", {})
            feature_score = sum(v for k, v in features.items() if v > 0.5)

            # Combined score
            position_scores[pos] = disagreement * (1.0 + feature_score)

        # Return top scoring positions
        top_positions = sorted(
            position_scores.keys(), key=lambda p: position_scores[p], reverse=True
        )
        return top_positions[:max_suggestions]

    def analyze_sequence(self, sequence, window_size=10, stride=5):
        """Perform comprehensive analysis of an entire sequence."""
        results = []

        # Analyze sequence in sliding windows with stride
        for pos in range(0, len(sequence) - window_size + 1, stride):
            result = self.analyze_region(sequence, pos, window_size)
            results.append(result)

        # Find interesting regions
        interesting_positions = self.suggest_exploration(sequence, max_suggestions=10)

        # Create analysis summary
        summary = {
            "length": len(sequence),
            "results": results,
            "interesting_positions": interesting_positions,
            "gc_content": (sequence.count("G") + sequence.count("C")) / len(sequence),
            "convergence": self.convergence,
        }

        return summary

    def save(self, filename):
        """Save the swarm model to disk."""
        with h5py.File(filename, "w") as f:
            # Save each agent
            agents_group = f.create_group("agents")
            for agent in self.agents:
                agent.save(agents_group)

            # Save collective data
            f.create_dataset("swarm_rewards", data=self.swarm_rewards)

            # Save agent weights
            weights_group = f.create_group("agent_weights")
            for agent_id, weight in self.agent_weights.items():
                weights_group.attrs[str(agent_id)] = weight

            # Save training history
            if self.training_history:
                f.create_dataset("training_history", data=self.training_history)
            if self.validation_history:
                f.create_dataset("validation_history", data=self.validation_history)

            # Save current epoch
            f.attrs["current_epoch"] = self.current_epoch

    def load(self, filename):
        """Load the swarm model from disk."""
        with h5py.File(filename, "r") as f:
            # Load agents
            if "agents" in f:
                agents_group = f["agents"]

                # Clear existing agents
                self.agents = []

                # Recreate agents
                for agent_name in agents_group:
                    if not agent_name.startswith("agent_"):
                        continue

                    agent_group = agents_group[agent_name]
                    agent_id = agent_group.attrs["id"]
                    agent_type = AgentType[agent_group.attrs["type"]]

                    # Create new agent
                    agent = GeneticAgent(self.sc, agent_id, agent_type)

                    # Load agent data
                    agent.load(agents_group)
                    self.agents.append(agent)

            # Load swarm rewards
            if "swarm_rewards" in f:
                self.swarm_rewards = f["swarm_rewards"][:]

            # Load agent weights
            if "agent_weights" in f:
                weights_group = f["agent_weights"]
                self.agent_weights = {}
                for key in weights_group.attrs:
                    agent_id = int(key)
                    self.agent_weights[agent_id] = weights_group.attrs[key]

            # Load training history
            if "training_history" in f:
                self.training_history = f["training_history"][:]
            if "validation_history" in f:
                self.validation_history = f["validation_history"][:]

            # Load current epoch
            if "current_epoch" in f.attrs:
                self.current_epoch = f.attrs["current_epoch"]


"""
Unified Genetic Analysis CLI - HDC-powered DNA sequence analysis framework
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"genetic_analysis_{time.strftime('%Y%m%d-%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("genetic-analysis")


# -------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------


def get_optimal_dimension(data_size: int, target_accuracy: float = 0.95) -> int:
    """Calculate optimal HDC dimension based on data characteristics.

    Uses Johnson-Lindenstrauss lemma to determine dimension that preserves
    distances with desired accuracy.
    """
    # Min dimension to preserve distances with target_accuracy probability
    min_dim = math.ceil(8 * math.log(data_size) / (target_accuracy**2))

    # Round to nearest power of 2 for computational efficiency
    power = math.ceil(math.log2(min_dim))
    optimal_dim = 2**power

    # Cap at reasonable limits
    return max(1024, min(optimal_dim, 16384))


def detect_optimal_kmer_size(sequence: str) -> int:
    """Detect optimal k-mer size based on sequence complexity."""
    # Sample the sequence if it's very long
    if len(sequence) > 10000:
        samples = [
            sequence[i : i + 1000] for i in range(0, len(sequence), len(sequence) // 10)
        ]
        sequence = "".join(samples)

    # Calculate sequence complexity (unique k-mers ratio) for different k
    complexity = {}
    for k in range(3, 12):
        if len(sequence) < k:
            continue

        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i : i + k]
            if "N" not in kmer:
                kmers.add(kmer)

        # Unique k-mers ratio
        complexity[k] = len(kmers) / (len(sequence) - k + 1)

    if not complexity:
        return 7  # Default

    # Find k where complexity starts to level off
    prev_k, prev_c = list(complexity.items())[0]
    for k, c in list(complexity.items())[1:]:
        if c - prev_c < 0.05:  # Diminishing returns threshold
            return prev_k
        prev_k, prev_c = k, c

    return min(complexity.keys(), key=lambda k: abs(complexity[k] - 0.7))


# -------------------------------------------------------------------------------
# Argument Parsing - Unified CLI
# -------------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments for all genetic analysis modes."""
    parser = argparse.ArgumentParser(
        description="Genetic Analysis Suite - HDC-powered DNA sequence analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common arguments
    parser.add_argument("--input", type=str, help="Input FASTA/FASTQ file")
    parser.add_argument("--output", type=str, help="Output file/directory")
    parser.add_argument("--dim", type=int, default=5000, help="HDC vector dimension")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Computation device",
    )
    parser.add_argument("--model", type=str, help="Path to saved model file")

    # Create subparsers for different operation modes
    subparsers = parser.add_subparsers(
        dest="mode", help="Operation mode", required=True
    )

    # Mode: encode
    encode_parser = subparsers.add_parser(
        "encode", help="Encode sequences as HDC vectors"
    )
    encode_parser.add_argument(
        "--kmer", type=int, default=7, help="k-mer size for encoding"
    )

    # Mode: train
    train_parser = subparsers.add_parser("train", help="Train genetic models")
    train_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--agent-type",
        type=str,
        default="swarm",
        choices=["swarm", "simple"],
        help="Agent architecture",
    )
    train_parser.add_argument(
        "--swarm-size", type=int, default=10, help="Number of agents in swarm"
    )

    # Mode: predict
    predict_parser = subparsers.add_parser(
        "predict", help="Predict sequence continuations"
    )
    predict_parser.add_argument(
        "--context", type=int, default=20, help="Context length for prediction"
    )
    predict_parser.add_argument(
        "--length", type=int, default=20, help="Length to predict"
    )

    # Mode: impute
    impute_parser = subparsers.add_parser("impute", help="Impute missing segments")
    impute_parser.add_argument(
        "--gap-length", type=int, default=10, help="Length of gap to impute"
    )
    impute_parser.add_argument(
        "--prefix", type=str, help="Sequence before gap (overrides --input)"
    )
    impute_parser.add_argument(
        "--suffix", type=str, help="Sequence after gap (overrides --input)"
    )

    # Mode: analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze sequence properties"
    )
    analyze_parser.add_argument(
        "--window", type=int, default=10, help="Analysis window size"
    )
    analyze_parser.add_argument("--stride", type=int, default=5, help="Window stride")
    analyze_parser.add_argument(
        "--similarity", action="store_true", help="Compute sequence similarity matrix"
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if not args.output:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.output = f"genetic_results_{args.mode}_{timestamp}"

    return args


# -------------------------------------------------------------------------------
# I/O Utilities
# -------------------------------------------------------------------------------


def load_sequences(input_path: str) -> Tuple[List[str], List[str]]:
    """Load sequences from a FASTA/FASTQ file.

    Returns:
        Tuple of (sequences, sequence_ids)
    """
    try:
        from Bio import SeqIO

        format_map = {
            ".fasta": "fasta",
            ".fa": "fasta",
            ".fna": "fasta",
            ".fastq": "fastq",
            ".fq": "fastq",
        }

        # Determine format from file extension
        ext = os.path.splitext(input_path.lower())[1]
        file_format = format_map.get(ext, "fasta")  # Default to FASTA

        sequences = []
        sequence_ids = []

        for record in SeqIO.parse(input_path, file_format):
            sequences.append(str(record.seq).upper())
            sequence_ids.append(record.id)

        logger.info(f"Loaded {len(sequences)} sequences from {input_path}")
        return sequences, sequence_ids

    except Exception as e:
        logger.error(f"Error loading sequences: {e}")
        sys.exit(1)


def save_results(results, output_path: str, format: str = "auto"):
    """Save analysis results to file.

    Args:
        results: Data to save (vectors, matrix, or JSON-serializable objects)
        output_path: Path to save results
        format: Output format, or 'auto' to detect from extension
    """
    # Create output directory if it's a path with directories
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine format from extension if set to auto
    if format == "auto":
        ext = os.path.splitext(output_path.lower())[1]
        if ext in [".h5", ".hdf5"]:
            format = "hdf5"
        elif ext in [".npy", ".npz"]:
            format = "numpy"
        elif ext in [".json"]:
            format = "json"
        elif ext in [".txt", ".csv"]:
            format = "text"
        else:
            format = "pickle"  # Default fallback

    try:
        # Save based on format
        if format == "hdf5":
            import h5py

            with h5py.File(output_path, "w") as f:
                if isinstance(results, list) and all(
                    isinstance(item, np.ndarray) for item in results
                ):
                    # List of vectors
                    f.create_dataset("vectors", data=np.array(results))
                elif isinstance(results, np.ndarray):
                    # Single array/matrix
                    f.create_dataset("data", data=results)
                else:
                    # Try to save as attributes or groups
                    for k, v in results.items():
                        if isinstance(v, np.ndarray):
                            f.create_dataset(k, data=v)
                        else:
                            f.attrs[k] = v

        elif format == "numpy":
            if isinstance(results, list) and all(
                isinstance(item, np.ndarray) for item in results
            ):
                np.save(output_path, np.array(results))
            elif isinstance(results, np.ndarray):
                np.save(output_path, results)
            else:
                np.savez(output_path, **results)

        elif format == "json":
            import json

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    return json.JSONEncoder.default(self, obj)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)

        elif format == "text":
            with open(output_path, "w") as f:
                if isinstance(results, str):
                    f.write(results)
                elif isinstance(results, list):
                    for item in results:
                        f.write(f"{item}\n")
                elif isinstance(results, dict):
                    for k, v in results.items():
                        f.write(f"{k}: {v}\n")

        else:  # pickle fallback
            import pickle

            with open(output_path, "wb") as f:
                pickle.dump(results, f)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")


# -------------------------------------------------------------------------------
# Unified Run Function
# -------------------------------------------------------------------------------


def run_genetic_analysis(args):
    """Main entry point for all genetic analysis operations."""
    # Determine compute device
    if args.device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Import the right modules based on operation mode
    from genetic_modules import (
        DNASupercomputer,
        BiologicalHDC,
        GeneticAgent,
        GeneticSwarm,
    )

    # Initialize appropriate supercomputer
    use_biological = args.mode in ["analyze", "impute"]
    supercomputer_class = BiologicalHDC if use_biological else DNASupercomputer
    supercomputer = supercomputer_class(dimension=args.dim, device=device)

    # Load data if input file provided
    sequences = []
    sequence_ids = []
    if args.input:
        sequences, sequence_ids = load_sequences(args.input)

    # Execute selected operation mode
    if args.mode == "encode":
        # Encode sequences as HDC vectors
        if not sequences:
            logger.error("No input sequences provided for encoding")
            return

        vectors = []
        for seq in tqdm(sequences, desc="Encoding sequences"):
            vectors.append(supercomputer.encode_sequence(seq, k=args.kmer))

        save_results(vectors, args.output)

        # Report cache stats
        cache_stats = supercomputer.get_cache_stats()
        logger.info(
            f"Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses "
            f"({cache_stats['ratio']:.2%} hit ratio)"
        )

        return vectors

    elif args.mode == "train":
        # Train genetic model
        if not sequences:
            logger.error("No input sequences provided for training")
            return

        # Initialize appropriate agent/swarm
        if args.agent_type == "swarm":
            model = GeneticSwarm(supercomputer, swarm_size=args.swarm_size)
        else:
            model = GeneticAgent(supercomputer)

        # Load existing model if provided
        if args.model and os.path.exists(args.model):
            logger.info(f"Loading existing model from {args.model}")
            model.load(args.model)

        # Split into train/validation sets
        train_size = int(0.8 * len(sequences))
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:] if train_size < len(sequences) else None

        # Train the model
        logger.info(
            f"Training {args.agent_type} model on {len(train_sequences)} sequences"
        )
        train_results = model.train(
            train_sequences,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_set=val_sequences,
        )

        # Save the trained model
        model_path = f"{args.output}_model.h5"
        model.save(model_path)
        logger.info(f"Trained model saved to {model_path}")

        # Save training metrics
        metrics_path = f"{args.output}_metrics.json"
        save_results(train_results, metrics_path, format="json")

        return model

    elif args.mode == "predict":
        # Predict sequence continuations
        if not args.model:
            logger.error("No model file provided for prediction")
            return

        # Load the model
        model = GeneticSwarm(supercomputer)
        model.load(args.model)

        # Generate predictions
        predictions = []

        if sequences:
            # Predict from input sequences
            for i, seq in enumerate(sequences):
                if len(seq) < args.context:
                    logger.warning(f"Sequence {i} too short for prediction, skipping")
                    continue

                # Use specified context length from beginning
                context = seq[: args.context]

                # Predict specified length
                predict_length = min(args.length, len(seq) - args.context)
                actual = (
                    seq[args.context : args.context + predict_length]
                    if predict_length > 0
                    else ""
                )

                # Generate prediction
                predicted = model.predict(context, predict_length)

                # Calculate accuracy if actual sequence available
                accuracy = 0
                if actual:
                    matches = sum(p == a for p, a in zip(predicted, actual))
                    accuracy = matches / len(actual)

                predictions.append(
                    {
                        "id": sequence_ids[i],
                        "context": context,
                        "predicted": predicted,
                        "actual": actual,
                        "accuracy": accuracy,
                    }
                )
        else:
            # Interactive mode - ask for context
            context = input("Enter DNA context sequence: ").strip().upper()
            if not context:
                logger.error("No context provided")
                return

            # Predict specified length
            predicted = model.predict(context, args.length)

            predictions.append(
                {
                    "id": "interactive",
                    "context": context,
                    "predicted": predicted,
                    "actual": "",
                    "accuracy": 0,
                }
            )

        # Save predictions
        save_results(predictions, f"{args.output}_predictions.json", format="json")

        # Print predictions
        for pred in predictions:
            logger.info(f"Sequence: {pred['id']}")
            logger.info(f"  Context: {pred['context']}")
            logger.info(f"  Predicted: {pred['predicted']}")
            if pred["actual"]:
                logger.info(f"  Actual: {pred['actual']}")
                logger.info(f"  Accuracy: {pred['accuracy']:.4f}")
            logger.info("---")

        return predictions

    elif args.mode == "impute":
        # Impute missing segments
        if not args.model:
            logger.error("No model file provided for imputation")
            return

        # Load the model
        model = GeneticSwarm(supercomputer)
        model.load(args.model)

        # Generate imputations
        imputations = []

        if args.prefix is not None and args.suffix is not None:
            # Direct imputation mode
            prefix = args.prefix.strip().upper()
            suffix = args.suffix.strip().upper()

            # Impute the gap
            imputed = model.impute_segment(prefix, suffix, args.gap_length)

            imputations.append(
                {"id": "direct", "prefix": prefix, "suffix": suffix, "imputed": imputed}
            )

        elif sequences:
            # Impute from input sequences
            for i, seq in enumerate(sequences):
                if (
                    len(seq) < args.gap_length + 20
                ):  # Need enough sequence for meaningful gaps
                    logger.warning(f"Sequence {i} too short for imputation, skipping")
                    continue

                # Create gap in middle third of sequence
                seq_len = len(seq)
                gap_start = seq_len // 3

                prefix = seq[:gap_start]
                actual = seq[gap_start : gap_start + args.gap_length]
                suffix = seq[gap_start + args.gap_length :]

                # Impute the gap
                imputed = model.impute_segment(prefix, suffix, args.gap_length)

                # Calculate accuracy
                matches = sum(i == a for i, a in zip(imputed, actual))
                accuracy = matches / args.gap_length

                imputations.append(
                    {
                        "id": sequence_ids[i],
                        "gap_position": gap_start,
                        "prefix": prefix[:20] + "..."
                        if len(prefix) > 20
                        else prefix,  # Truncate for display
                        "suffix": "..." + suffix[-20:]
                        if len(suffix) > 20
                        else suffix,  # Truncate for display
                        "imputed": imputed,
                        "actual": actual,
                        "accuracy": accuracy,
                    }
                )

        else:
            # Interactive mode
            prefix = input("Enter sequence before gap: ").strip().upper()
            suffix = input("Enter sequence after gap: ").strip().upper()

            if not prefix and not suffix:
                logger.error("Need at least prefix or suffix for imputation")
                return

            # Impute the gap
            imputed = model.impute_segment(prefix, suffix, args.gap_length)

            imputations.append(
                {
                    "id": "interactive",
                    "prefix": prefix,
                    "suffix": suffix,
                    "imputed": imputed,
                }
            )

        # Save imputations
        save_results(imputations, f"{args.output}_imputations.json", format="json")

        # Print imputations
        for imp in imputations:
            logger.info(f"Sequence: {imp['id']}")
            logger.info(f"  Prefix: {imp['prefix']}")
            logger.info(f"  Imputed: {imp['imputed']}")
            logger.info(f"  Suffix: {imp['suffix']}")
            if "actual" in imp:
                logger.info(f"  Actual: {imp['actual']}")
                logger.info(f"  Accuracy: {imp['accuracy']:.4f}")
            logger.info("---")

        return imputations

    elif args.mode == "analyze":
        # Analyze sequences
        if not sequences:
            logger.error("No input sequences provided for analysis")
            return

        # Initialize agent if model provided, otherwise use basic supercomputer
        if args.model:
            logger.info(f"Loading model from {args.model}")
            model = GeneticSwarm(supercomputer)
            model.load(args.model)
            analyzer = model
        else:
            analyzer = supercomputer

        # Generate analyses
        analyses = []

        for i, seq in enumerate(sequences):
            if len(seq) < args.window:
                logger.warning(
                    f"Sequence {i} too short for analysis window size {args.window}, skipping"
                )
                continue

            # Basic sequence stats
            seq_stats = {
                "id": sequence_ids[i],
                "length": len(seq),
                "gc_content": (seq.count("G") + seq.count("C")) / len(seq),
                "base_counts": {base: seq.count(base) for base in "ACGTN"},
            }

            # Advanced analysis if using a model
            if args.model:
                full_analysis = analyzer.analyze_sequence(
                    seq, window_size=args.window, stride=args.stride
                )
                seq_stats.update(
                    {
                        "interesting_positions": full_analysis["interesting_positions"],
                        "convergence": full_analysis["convergence"],
                    }
                )

            analyses.append(seq_stats)

        # Calculate sequence similarity matrix if requested
        if args.similarity:
            similarity_matrix = np.zeros((len(sequences), len(sequences)))

            for i, seq1 in enumerate(tqdm(sequences, desc="Computing similarities")):
                # Encode first sequence
                vec1 = supercomputer.encode_sequence(seq1, k=5)

                # Compare with all other sequences (including itself)
                for j, seq2 in enumerate(sequences):
                    if i == j:
                        similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
                        continue

                    # Encode second sequence
                    vec2 = supercomputer.encode_sequence(seq2, k=5)

                    # Compute similarity
                    sim = float(supercomputer.xp.dot(vec1, vec2))
                    similarity_matrix[i, j] = sim

            # Save similarity matrix
            save_results(
                similarity_matrix, f"{args.output}_similarity.npy", format="numpy"
            )

            # Also save as CSV for easier viewing
            sim_csv = "Sequence," + ",".join(sequence_ids) + "\n"
            for i, seq_id in enumerate(sequence_ids):
                sim_csv += (
                    seq_id
                    + ","
                    + ",".join(f"{sim:.4f}" for sim in similarity_matrix[i])
                    + "\n"
                )

            with open(f"{args.output}_similarity.csv", "w") as f:
                f.write(sim_csv)

        # Save analyses
        save_results(analyses, f"{args.output}_analysis.json", format="json")

        return analyses

    else:
        logger.error(f"Unknown mode: {args.mode}")
        return


# -------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_genetic_analysis(args)
