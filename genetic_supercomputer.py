import numpy as np
import cupy as cp  # GPU acceleration
import h5py  # For storing massive HDC vectors
from tqdm import tqdm
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
from dataclasses import dataclass, field
import json
import pickle
from Bio import SeqIO
import gradio as gr
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

try:  # Try importing CuPy with graceful fallback
    import cupy as cp

    HAS_GPU = cp.is_available()
except ImportError:
    cp = None
    HAS_GPU = False

# Constants for IUPAC codes and other repeated data
IUPAC_AMBIGUITY_MAP = {
    "N": ["A", "C", "G", "T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "M": ["A", "C"],
    "K": ["G", "T"],
    "S": ["C", "G"],
    "W": ["A", "T"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
}

COMMON_MOTIFS = {  # Moved to module level for reuse
    "TATA": 0.8,  # TATA box
    "CAAT": 0.7,  # CAAT box
    "GATA": 0.6,
    "CACGTG": 0.8,
    "GCCNNNGGC": 0.7,
    "TTGACA": 0.6,
    "TATAAT": 0.6,
    "AATAAA": 0.7,
    "CCAAT": 0.6,
    "GAGAG": 0.5,
}

DEFAULT_ALPHABET = ["A", "T", "G", "C"]
DEFAULT_BUNDLE_DECAY = 0.9
DEFAULT_MAX_KMER_SIZE = 30


@dataclass
class GSCConfig:
    """Unified configuration for Genetic Supercomputer.
    Parameters are automatically tuned based on data characteristics and hardware.
    """

    dimension: Optional[int] = None
    device: str = "auto"  # 'gpu', 'cpu', or 'auto'
    vector_type: str = "bipolar"
    seed: Optional[int] = None

    alphabet: List[str] = field(default_factory=lambda: DEFAULT_ALPHABET)
    data_size: Optional[int] = None
    avg_sequence_length: Optional[int] = None

    max_kmer_size: int = DEFAULT_MAX_KMER_SIZE
    bundle_decay: float = DEFAULT_BUNDLE_DECAY
    cache_size: Optional[int] = None
    chunk_size: Optional[int] = None
    position_vectors: Optional[int] = None  # Number of position vectors
    # Added accuracy_target for dimension calculation
    accuracy_target: float = 0.95

    def __post_init__(self):
        """Initialize derived parameters."""
        self.sys_memory = psutil.virtual_memory().total
        self.gpu_memory = self._detect_gpu_memory()
        self.cpu_cores = os.cpu_count() or 4

        if self.device == "auto":
            self.device = "gpu" if HAS_GPU else "cpu"

        if self.dimension is None:
            self.dimension = self._derive_optimal_dimension()

        if self.cache_size is None:
            self._set_cache_size()  # Use a separate method for cache sizing

        if self.chunk_size is None:
            self._set_chunk_size()

        if self.position_vectors is None:
            # Default: Enough for largest kmer, or 1/10th of dimension, whichever is smaller
            self.position_vectors = min(self.max_kmer_size, self.dimension // 10)

    def _detect_gpu_memory(self) -> Optional[int]:
        if not HAS_GPU:
            return None
        try:
            return cp.cuda.Device().mem_info[0]
        except Exception:
            return None

    def _derive_optimal_dimension(self) -> int:
        """Calculate optimal dimension using Johnson-Lindenstrauss lemma or defaults."""
        if self.data_size and self.data_size > 1:  # Added check for valid data_size
            # Johnson-Lindenstrauss lemma
            jl_dim = int(8 * math.log(self.data_size) / (1 - self.accuracy_target) ** 2)
            # Adjust for alphabet (empirical factor). This can be tuned
            alphabet_factor = max(1.0, math.log2(len(self.alphabet)) / 2.0)
            dim = 2 ** math.ceil(math.log2(jl_dim * alphabet_factor))

            # Memory constraint
            mem_limit = max(self.sys_memory, self.gpu_memory or 0) * 0.1
            bytes_per_dim = 4 if self.vector_type == "bipolar" else 1
            max_dim_by_memory = int(
                mem_limit / (bytes_per_dim * (self.data_size or 1000))
            )

            return max(1024, min(dim, max_dim_by_memory, 32768))

        # Hardware-based fallback
        for mem, dim in [(32, 10000), (16, 8192), (8, 4096)]:
            if self.sys_memory > mem * 1024**3:
                return dim
        return 2048

    def _set_cache_size(self):
        """Set cache size based on available memory."""
        mem_gb = self.sys_memory / (1024**3)
        # Scale cache size with memory, but also consider dimension
        base_cache_size = int(mem_gb * 50000)
        dimension_factor = max(
            1, self.dimension // 1000
        )  # Larger dimension = smaller cache
        self.cache_size = max(100000, base_cache_size // dimension_factor)

    def _set_chunk_size(self):
        """Set chunk size based on memory and vector dimension."""
        mem_available = min(
            self.sys_memory * 0.2,
            self.gpu_memory * 0.5 if self.gpu_memory else float("inf"),
        )
        bytes_per_vector = self.dimension * (4 if self.vector_type == "bipolar" else 1)
        # Ensure at least 100, max 10000, but scale with available memory
        self.chunk_size = min(10000, max(100, int(mem_available / bytes_per_vector)))

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__  # Much cleaner to just return the dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GSCConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def __str__(self) -> str:
        return (
            f"GSCConfig(dim={self.dimension}, device={self.device}, "
            f"cache={self.cache_size // 1000}K, chunk={self.chunk_size})"
        )


class Utils:
    """Utility functions for genetic sequence analysis and processing."""

    @staticmethod
    def get_optimal_dimension(data_size: int, target_accuracy: float = 0.95) -> int:
        """Calculate optimal HDC dimension using Johnson-Lindenstrauss lemma."""
        min_dim = math.ceil(8 * math.log(data_size) / (target_accuracy**2))
        power = math.ceil(math.log2(min_dim))
        return max(1024, min(2**power, 16384))

    @staticmethod
    def detect_optimal_kmer_size(sequence: str) -> int:
        """Detect optimal k-mer size based on sequence complexity."""
        if len(sequence) > 10000:
            sequence = "".join(
                [
                    sequence[i : i + 1000]
                    for i in range(0, len(sequence), len(sequence) // 10)
                ]
            )

        complexity = {
            k: len(
                set(
                    sequence[i : i + k]
                    for i in range(len(sequence) - k + 1)
                    if "N" not in sequence[i : i + k]
                )
            )
            / (len(sequence) - k + 1)
            for k in range(3, 12)
            if len(sequence) >= k
        }

        if not complexity:
            return 7  # Default

        for k, c in list(complexity.items())[1:]:
            if c - list(complexity.values())[0] < 0.05:
                return list(complexity.keys())[0]
        return min(complexity.keys(), key=lambda k: abs(complexity[k] - 0.7))

    @staticmethod
    def generate_kmers(sequence: str, k: int, stride: int = 1) -> List[str]:
        """Generate k-mers from a sequence with a given stride."""
        return [sequence[i : i + k] for i in range(0, len(sequence) - k + 1, stride)]

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content of a sequence."""
        gc_count = sequence.upper().count("G") + sequence.upper().count("C")
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0

    @staticmethod
    def calculate_sequence_entropy(sequence: str) -> float:
        """Calculate Shannon entropy of a sequence."""
        base_counts = {base: sequence.count(base) for base in set(sequence)}
        total = sum(base_counts.values())
        if total == 0:
            return 0.0
        probs = [count / total for count in base_counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def detect_motifs(sequence: str, motif_list: List[str]) -> Dict[str, List[int]]:
        """Detect positions of motifs in a sequence."""
        motif_positions = defaultdict(list)
        for motif in motif_list:
            motif_len = len(motif)
            for i in range(len(sequence) - motif_len + 1):
                if sequence[i : i + motif_len] == motif:
                    motif_positions[motif].append(i)
        return motif_positions

    @staticmethod
    def load_sequences(file_path: str) -> Tuple[List[str], List[str]]:
        """Load sequences from a FASTA/FASTQ file."""
        try:
            format_map = {
                ".fasta": "fasta",
                ".fa": "fasta",
                ".fna": "fasta",
                ".fastq": "fastq",
                ".fq": "fastq",
            }
            ext = file_path[file_path.rfind(".") :].lower()
            file_format = format_map.get(ext, "fasta")

            sequences = [
                str(record.seq).upper()
                for record in SeqIO.parse(file_path, file_format)
            ]
            sequence_ids = [record.id for record in SeqIO.parse(file_path, file_format)]
            return sequences, sequence_ids
        except Exception as e:
            logger.error(f"Error loading sequences: {e}")
            raise

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Generate the reverse complement of a DNA sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join(complement[base] for base in reversed(sequence))

    @staticmethod
    def calculate_kmer_frequencies(sequences: List[str], k: int) -> Dict[str, float]:
        """Calculate normalized k-mer frequencies across a list of sequences."""
        kmer_counts = defaultdict(int)
        total_kmers = 0

        for seq in sequences:
            kmers = Utils.generate_kmers(seq, k)
            for kmer in kmers:
                kmer_counts[kmer] += 1
                total_kmers += 1

        return {kmer: count / total_kmers for kmer, count in kmer_counts.items()}

    @staticmethod
    def find_consensus_sequence(sequences: List[str]) -> str:
        """Find the consensus sequence from a list of aligned sequences."""
        if not sequences:
            return ""
        consensus = []
        for i in range(len(sequences[0])):
            base_counts = defaultdict(int)
            for seq in sequences:
                base_counts[seq[i]] += 1
            consensus.append(max(base_counts.items(), key=lambda x: x[1])[0])
        return "".join(consensus)

    @staticmethod
    def calculate_pairwise_similarity(seq1: str, seq2: str, k: int = 5) -> float:
        """Calculate pairwise similarity between two sequences using k-mer overlap."""
        kmers1 = set(Utils.generate_kmers(seq1, k))
        kmers2 = set(Utils.generate_kmers(seq2, k))
        intersection = kmers1.intersection(kmers2)
        union = kmers1.union(kmers2)
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def calculate_conservation_scores(sequences: List[str]) -> List[float]:
        """Calculate conservation scores for each position in aligned sequences."""
        if not sequences:
            return []
        conservation_scores = []
        for i in range(len(sequences[0])):
            base_counts = defaultdict(int)
            for seq in sequences:
                base_counts[seq[i]] += 1
            max_freq = max(base_counts.values())
            conservation_scores.append(max_freq / len(sequences))
        return conservation_scores


class GenomicDataLoader:
    """Class for loading and saving genomic data files."""

    def __init__(self):
        self.logger = logging.getLogger("GenomicDataLoader")

    def load_sequences(self, input_path: str) -> Tuple[List[str], List[str]]:
        """Load sequences from a FASTA/FASTQ file."""
        try:
            format_map = {
                ".fasta": "fasta",
                ".fa": "fasta",
                ".fna": "fasta",
                ".fastq": "fastq",
                ".fq": "fastq",
            }

            ext = os.path.splitext(input_path.lower())[1]
            file_format = format_map.get(ext, "fasta")

            sequences = [
                str(record.seq).upper()
                for record in SeqIO.parse(input_path, file_format)
            ]
            sequence_ids = [
                record.id for record in SeqIO.parse(input_path, file_format)
            ]

            self.logger.info(f"Loaded {len(sequences)} sequences from {input_path}")
            return sequences, sequence_ids

        except Exception as e:
            self.logger.error(f"Error loading sequences: {e}")
            sys.exit(1)

    def save_results(self, results, output_path: str, format: str = "auto"):
        """Save analysis results to file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if format == "auto":
            ext = os.path.splitext(output_path.lower())[1]
            format = {
                ".h5": "hdf5",
                ".npy": "numpy",
                ".npz": "numpy",
                ".json": "json",
                ".txt": "text",
                ".csv": "text",
            }.get(ext, "pickle")

        try:
            if format == "hdf5":
                with h5py.File(output_path, "w") as f:
                    if isinstance(results, (list, np.ndarray)):
                        f.create_dataset("data", data=np.array(results))
                    elif isinstance(results, dict):
                        for k, v in results.items():
                            if isinstance(v, (np.ndarray, list)):
                                f.create_dataset(k, data=np.array(v))
                            else:
                                f.attrs[k] = v

            elif format == "numpy":
                np.save(output_path, np.array(results))

            elif format == "json":

                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return json.JSONEncoder.default(self, obj)

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, cls=NumpyEncoder)

            elif format == "text":
                with open(output_path, "w") as f:
                    if isinstance(results, str):
                        f.write(results)
                    elif isinstance(results, (list, dict)):
                        for item in results:
                            f.write(f"{item}\n")

            else:  # pickle fallback
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

            self.logger.info(f"Results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


class HDCVectorSpace:
    """Hyperdimensional computing vector space optimized for genetic sequences."""

    def __init__(self, config: "GSCConfig"):
        self.dim = config.dimension
        self.device = "gpu" if (config.device == "gpu" and HAS_GPU) else "cpu"
        self.xp = cp if self.device == "gpu" else np
        self.vector_type = config.vector_type  # 'binary' or 'bipolar'
        self.dtype = np.int8 if self.vector_type == "binary" else np.float32
        self.bundle_decay = config.bundle_decay
        if config.seed is not None:
            self.xp.random.seed(config.seed)
        self.base_vectors = {}
        self.position_vectors = {}
        self._vector_cache = lru_cache(maxsize=config.cache_size)(self._encode_kmer)

    def initialize(self, alphabet: List[str] = None) -> "HDCVectorSpace":
        """Initialize vector space with orthogonal base vectors."""
        alphabet = alphabet or ["A", "T", "G", "C"]
        n_vectors = len(alphabet) + self.config.position_vectors

        if self.vector_type == "binary":
            basis = self.xp.random.randint(
                2, size=(n_vectors, self.dim), dtype=self.dtype
            )
        else:  # Bipolar vectors
            basis = self.xp.random.choice(
                [-1, 1], size=(n_vectors, self.dim), dtype=self.dtype
            )

        basis, _ = self.xp.linalg.qr(basis)
        for i, base in enumerate(alphabet):  # Assign base vectors
            self.base_vectors[base] = self._normalize(basis[i])
        for i in range(len(alphabet), n_vectors):  # Assign position vectors
            self.position_vectors[f"pos_{i - len(alphabet)}"] = self._normalize(
                basis[i]
            )

        return self

    def encode_sequence(self, sequence: str, k: int = 7) -> Any:
        """Encode complete sequence using sliding k-mers."""
        if len(sequence) < k:
            return self._encode_kmer(sequence)

        vectors = [
            self._encode_kmer(sequence[i : i + k])
            for i in range(0, len(sequence) - k + 1)
        ]

        # Combine vectors with exponential decay
        result = vectors[0]
        for vec in vectors[1:]:
            result = self.bundle(result, vec, self.bundle_decay)
        return self._normalize(result)

    def _encode_kmer(self, kmer: str) -> Any:
        """Core k-mer encoding with optimized vectorization."""
        if len(kmer) == 1 and kmer in self.base_vectors:
            return self.base_vectors[kmer]

        # Vectorized operations for standard bases
        if all(base in self.base_vectors for base in kmer):
            xp = self.xp
            base_vecs = xp.stack([self.base_vectors[b] for b in kmer])
            pos_vecs = xp.stack(
                [
                    self.position_vectors[f"pos_{i % len(self.position_vectors)}"]
                    for i in range(len(kmer))
                ]
            )

            # Vectorized binding and weighted bundling
            weights = xp.array([self.bundle_decay**i for i in range(len(kmer))])
            return self._normalize(
                xp.sum(
                    xp.multiply(base_vecs, pos_vecs)
                    * (weights / weights.sum()).reshape(-1, 1),
                    axis=0,
                )
            )

        # Handle ambiguous bases
        return self._encode_ambiguous(kmer)

    def _encode_ambiguous(self, kmer: str) -> Any:
        """Handle k-mers with ambiguous bases."""
        result = self.xp.zeros(self.dim, dtype=self.dtype)

        for i, base in enumerate(kmer):
            if base in self.base_vectors:
                vec = self.base_vectors[base]
            elif base in IUPAC_AMBIGUITY_MAP:
                # Average vector of possible bases
                bases = [b for b in IUPAC_AMBIGUITY_MAP[base] if b in self.base_vectors]
                if not bases:
                    continue
                vec = sum(self.base_vectors[b] for b in bases) / len(bases)
            else:
                continue

            pos_vec = self.position_vectors[f"pos_{i % len(self.position_vectors)}"]
            bound = self.xp.multiply(vec, pos_vec)
            result = self.bundle(result, bound, self.bundle_decay)

        return self._normalize(result)

    def bind(self, v1: Any, v2: Any) -> Any:
        """Binding operation (element-wise multiplication)."""
        return self.xp.multiply(v1, v2)

    def bundle(self, v1: Any, v2: Any, alpha: float = None) -> Any:
        """Bundling operation (weighted sum)."""
        alpha = alpha if alpha is not None else self.bundle_decay
        if self.vector_type == "binary":
            # For binary vectors, use thresholding after bundling
            result = alpha * v1 + (1 - alpha) * v2
            return self.xp.where(result > 0.5, 1, 0)
        else:
            # For bipolar vectors, use continuous values
            return alpha * v1 + (1 - alpha) * v2

    def similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two vectors."""
        if self.vector_type == "binary":
            return float(self.xp.sum(v1 == v2) / self.dim)
        else: # bipolar 
            return float(self.xp.dot(v1, v2))

    def _normalize(self, v: Any) -> Any:
        """Normalize a vector."""
        norm = self.xp.linalg.norm(v)
        if norm < 1e-10:
            return v
        if self.vector_type == "binary":
            return self.xp.where(v / norm > 0.5, 1, 0)
        else: # bipolar
            return v / norm


class DNASupercomputer:
    """DNA sequence analysis using HDC."""

    def __init__(self, config: Optional[GSCConfig] = None, **kwargs):
        """Initialize DNA supercomputer."""
        self.config = config or GSCConfig(**kwargs)
        self.hdc = HDCVectorSpace(self.config)
        self.base_vectors = {}
        self.position_vectors = {}
        self.kmer_cache = lru_cache(maxsize=self.config.cache_size)(
            self._encode_kmer_uncached
        )
        self.seq_stats = {
            "base_counts": {base: 0 for base in self.config.alphabet},
            "total_bases": 0,
            "gc_content": 0.5,
        }
        self.data_loader = GenomicDataLoader()
        self._initialize_vector_space()
        logger.info(
            f"DNASupercomputer initialized with {self.hdc.dim}D vectors on {self.hdc.device}"
        )

    def _initialize_vector_space(self):
        self.hdc.initialize(self.config.alphabet)
        self.base_vectors = self.hdc.base_vectors
        self.position_vectors = self.hdc.position_vectors

    def encode_kmer(self, kmer):
        """Encode a k-mer with caching."""
        return self.kmer_cache(kmer)  # Use the LRU cache

    def _encode_kmer_uncached(self, kmer):
        """Core k-mer encoding logic (called via cache)."""
        if len(kmer) == 1 and kmer in self.base_vectors:
            return self.base_vectors[kmer]

        xp = self.hdc.xp
        if self.hdc.device == "gpu" and all(base in self.base_vectors for base in kmer):
            base_vecs = xp.array([self.base_vectors[base] for base in kmer])
            pos_vecs = xp.array(
                [
                    self.position_vectors[f"pos_{i % len(self.position_vectors)}"]
                    for i in range(len(kmer))
                ]
            )
            bindings = xp.multiply(base_vecs, pos_vecs)  # Element-wise multiplication
            weights = xp.array([self.config.bundle_decay**i for i in range(len(kmer))])
            weights /= weights.sum()  # Normalize
            result = xp.sum(bindings * weights.reshape(-1, 1), axis=0)  # Weighted sum
            return self.hdc._normalize(result)

        result = xp.zeros(self.hdc.dim, dtype=xp.float32)
        for i, base in enumerate(kmer):
            self._update_base_stats(base)  # Simplified stats update

            if base in self.base_vectors:
                bound = self.hdc.bind(
                    self.base_vectors[base],
                    self.position_vectors[f"pos_{i % len(self.position_vectors)}"],
                )
            elif base in IUPAC_AMBIGUITY_MAP:  # Use constant
                possible_bases = [
                    b for b in IUPAC_AMBIGUITY_MAP[base] if b in self.base_vectors
                ]
                if possible_bases:
                    pos_vec = self.position_vectors[
                        f"pos_{i % len(self.position_vectors)}"
                    ]

                    bound = self.hdc.bind(
                        sum(self.base_vectors[b] for b in possible_bases)
                        / len(possible_bases),
                        pos_vec,
                    )
                else:
                    continue  # Skip if no known possible bases
            else:
                continue

            result = self.hdc.bundle(result, bound, self.config.bundle_decay)
        return self.hdc._normalize(result)

    def _update_base_stats(self, base):
        """Update base statistics using Utils."""
        if base in self.config.alphabet:
            self.seq_stats["base_counts"][base] += 1
            self.seq_stats["total_bases"] += 1

            if base in "GC":
                total = self.seq_stats["total_bases"]
                gc_count = (
                    self.seq_stats["base_counts"]["G"]
                    + self.seq_stats["base_counts"]["C"]
                )
                self.seq_stats["gc_content"] = gc_count / total if total > 0 else 0.5

    def encode_sequence(self, sequence, k=None, stride=None, chunk_size=None):
        """Encode sequence with adaptive parameters using Utils."""
        k = k or Utils.detect_optimal_kmer_size(sequence)
        stride = stride or max(1, k // 3)
        chunk_size = chunk_size or self.config.chunk_size
        xp = self.hdc.xp

        if len(sequence) < k:
            return self.encode_kmer(sequence)

        result = xp.zeros(self.hdc.dim, dtype=xp.float32)
        n_kmers = 0

        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]
            kmers = Utils.generate_kmers(chunk, k, stride)
            chunk_vectors = []

            for kmer in kmers:
                if kmer.count("N") <= k // 3:
                    kmer_vector = self.encode_kmer(kmer)
                    chunk_vectors.append(kmer_vector)

            if chunk_vectors:
                stacked = xp.array(chunk_vectors)
                weights = xp.array(
                    [self.config.bundle_decay**i for i in range(len(chunk_vectors))]
                )
                weights /= weights.sum()  # Normalize
                chunk_result = xp.sum(stacked * weights.reshape(-1, 1), axis=0)
                result = self.hdc.bundle(result, chunk_result, self.config.bundle_decay)
                n_kmers += len(chunk_vectors)

        self.seq_stats["gc_content"] = Utils.calculate_gc_content(sequence)

        return (
            self.hdc._normalize(result)
            if n_kmers > 0
            else xp.zeros(self.hdc.dim, dtype=xp.float32)
        )

    def load_sequences(self, file_path):
        """Load sequences from file using the GenomicDataLoader."""
        try:
            sequences, sequence_ids = self.data_loader.load_sequences(file_path)
            logger.info(f"Loaded {len(sequences)} sequences from {file_path}")
            return sequences, sequence_ids
        except Exception as e:
            logger.error(f"Failed to load sequences: {str(e)}")
            raise

    def find_similar_sequences(self, query_seq, reference_seqs, k=None, top_n=5):
        """Find similar sequences using Utils for optimal k-mer size."""
        k = k or Utils.detect_optimal_kmer_size(query_seq)
        query_vector = self.encode_sequence(query_seq, k=k)
        batch_size = min(100, len(reference_seqs))
        all_similarities = []

        for batch_start in range(0, len(reference_seqs), batch_size):
            batch_end = min(batch_start + batch_size, len(reference_seqs))
            batch = reference_seqs[batch_start:batch_end]
            batch_vectors = [self.encode_sequence(seq, k=k) for seq in batch]
            batch_similarities = self.hdc.batch_similarity(query_vector, batch_vectors)
            all_similarities.extend(
                batch_similarities.get()
                if self.hdc.device == "gpu"
                else batch_similarities
            )

        top_indices = np.argsort(all_similarities)[-top_n:][::-1]
        return [(reference_seqs[i], all_similarities[i]) for i in top_indices]

    def analyze_sequence(self, sequence):
        """Analyze sequence properties using Utils."""
        stats = {}
        stats["length"] = len(sequence)
        stats["gc_content"] = Utils.calculate_gc_content(sequence)
        stats["entropy"] = Utils.calculate_sequence_entropy(sequence)
        stats["normalized_entropy"] = (
            stats["entropy"] / math.log2(4) if stats["entropy"] > 0 else 0
        )
        stats["optimal_k"] = Utils.detect_optimal_kmer_size(sequence)
        common_motifs = ["TATA", "GATA", "CAAT", "AATAAA", "GCCNNNGGC"]
        motif_positions = Utils.detect_motifs(sequence, common_motifs)
        stats["motif_positions"] = [
            (pos, 0.7) for motif in motif_positions for pos in motif_positions[motif]
        ]

        return stats

    def calculate_pairwise_similarities(self, sequences, k=None):
        """Calculate pairwise similarities between all sequences."""
        n = len(sequences)
        if n == 0:
            return np.zeros((0, 0))

        sample_seq = sequences[0]
        k = k or Utils.detect_optimal_kmer_size(sample_seq)
        vectors = [self.encode_sequence(seq, k=k) for seq in sequences]

        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # Self-similarity is 1.0
            for j in range(i + 1, n):
                sim = float(self.hdc.similarity(vectors[i], vectors[j]))
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix

        return similarity_matrix

    def save_vectors(self, filename):
        """Save HDC vectors to HDF5 using GenomicDataLoader."""
        try:
            # Create a dictionary with all the data to save
            data_to_save = {
                "config": self.config.to_dict(),
                "base_vectors": {},
                "position_vectors": {},
            }

            # Add vectors (converting GPU vectors to NumPy if necessary)
            for key, vector in self.base_vectors.items():
                data_to_save["base_vectors"][key] = (
                    vector.get() if self.hdc.device == "gpu" else vector
                )

            for key, vector in self.position_vectors.items():
                data_to_save["position_vectors"][key] = (
                    vector.get() if self.hdc.device == "gpu" else vector
                )

            # Use the data loader to save the results
            self.data_loader.save_results(data_to_save, filename, format="hdf5")
            logger.info(f"Saved model vectors to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving vectors: {str(e)}")
            return False

    def load_vectors(self, filename):
        """Load HDC vectors from HDF5."""
        try:
            with h5py.File(filename, "r") as f:
                if "config" in f:
                    config_dict = {k: v for k, v in f["config"].attrs.items()}
                    self.config = GSCConfig.from_dict(config_dict)  # Use from_dict

                for group_name, vector_dict in [
                    ("base_vectors", self.base_vectors),
                    ("position_vectors", self.position_vectors),
                ]:
                    if group_name in f:
                        group = f[group_name]
                        for name in group:
                            vector = group[name][:]
                            vector_dict[name] = (
                                self.hdc.xp.array(vector)
                                if self.hdc.device == "gpu"
                                else vector
                            )
            logger.info(f"Loaded model vectors from {filename}")
            return self
        except Exception as e:
            logger.error(f"Error loading vectors: {str(e)}")
            raise

    def save_analysis_results(self, results, output_path, format="auto"):
        """Save analysis results using GenomicDataLoader."""
        try:
            self.data_loader.save_results(results, output_path, format)
            logger.info(f"Saved analysis results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    def get_cache_stats(self):
        """Return cache stats."""
        cache_info = self.kmer_cache.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "ratio": cache_info.hits / max(1, cache_info.hits + cache_info.misses),
            "size": cache_info.currsize,
            "max_size": cache_info.maxsize,
        }

    def clear_cache(self):
        """Clear k-mer cache."""
        self.kmer_cache.cache_clear()
        return self


@dataclass
class FeatureProvider:
    """Represents a biological feature provider."""

    name: str
    provider_fn: Callable[[str, Optional[int]], Any]
    weight: float = 1.0


class BiologicalEncoder:
    """HDC encoder integrating biological features, now without inheritance."""

    def __init__(self, config: GSCConfig):
        self.config = config
        self.hdc = HDCVectorSpace(config)  # Composition instead of inheritance
        self.feature_providers: Dict[str, FeatureProvider] = {}
        self.feature_cache: Dict[Tuple[str, Optional[int]], Any] = {}  # Feature cache
        self.kmer_cache = lru_cache(maxsize=self.config.cache_size)(
            self._encode_kmer_uncached
        )
        self.data_loader = GenomicDataLoader()  # Initialize the GenomicDataLoader
        self.utils = Utils()  # Initialize the Utils class
        self._register_default_features()
        self.hdc.initialize()  # Initialize after setting up base vectors

    def _register_default_features(self):
        """Register default feature providers."""
        self.register_feature_provider(
            FeatureProvider("gc_content", self._compute_gc_content, 0.5)
        )
        self.register_feature_provider(
            FeatureProvider("complexity", self._compute_complexity, 0.3)
        )
        self.register_feature_provider(
            FeatureProvider("motifs", self._detect_motifs, 0.4)  # Local motifs
        )

    def register_feature_provider(self, provider: FeatureProvider):
        """Register a feature provider."""
        self.feature_providers[provider.name] = provider
        logger.info(
            f"Registered feature provider: {provider.name} with weight {provider.weight}"
        )

    def _compute_gc_content(self, kmer: str, position: Optional[int] = None) -> float:
        """Compute GC content using Utils."""
        return self.utils.calculate_gc_content(kmer)

    def _compute_complexity(self, kmer: str, position: Optional[int] = None) -> float:
        """Compute Shannon entropy using Utils."""
        return self.utils.calculate_sequence_entropy(kmer)

    @lru_cache(maxsize=128)  # Cache motif detection
    def _detect_motifs(self, kmer: str, position: Optional[int] = None) -> float:
        """Detect common motifs using Utils."""
        motif_list = list(COMMON_MOTIFS.keys())
        motif_positions = self.utils.detect_motifs(kmer, motif_list)
        if not motif_positions:
            return 0.0
        # Return the highest motif score
        return max(COMMON_MOTIFS[motif] for motif in motif_positions.keys())

    def _compute_feature_vector(
        self, kmer: str, position: Optional[int] = None
    ) -> Optional[Any]:
        """Compute combined feature vector."""
        cache_key = (kmer, position)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = {
            name: provider.provider_fn(kmer, position)
            for name, provider in self.feature_providers.items()
            if provider.provider_fn(kmer, position) is not None
        }

        if not features:
            return None

        # Separate vector and scalar features
        vector_features = {
            name: value
            for name, value in features.items()
            if isinstance(value, (self.hdc.xp.ndarray, np.ndarray))
        }
        scalar_features = {
            name: value
            for name, value in features.items()
            if not isinstance(value, (self.hdc.xp.ndarray, np.ndarray))
        }
        feature_vector = self.hdc.xp.zeros(self.hdc.dim)

        # Integrate scalar features
        for name, value in scalar_features.items():
            seed_val = hash(name) % 10000  # Consistent seeding
            np.random.seed(seed_val)
            feature_vec = self.hdc._normalize(
                self.hdc.xp.array(np.random.uniform(-1, 1, self.hdc.dim))
            )
            weight = self.feature_providers[name].weight  # Get weight from provider
            feature_vec *= value * weight
            feature_vector = self.hdc.bundle(feature_vector, feature_vec, alpha=0.7)

        # Integrate vector features
        for name, vec in vector_features.items():
            weight = self.feature_providers[name].weight
            weighted_vec = vec * weight
            feature_vector = self.hdc.bundle(feature_vector, weighted_vec, alpha=0.7)

        normalized = self.hdc._normalize(feature_vector)
        self.feature_cache[cache_key] = normalized
        return normalized

    def _encode_kmer_uncached(self, kmer: str, position: Optional[int] = None) -> Any:
        """Core k-mer encoding, including biological features (uncached)."""
        if len(kmer) == 1 and kmer in self.hdc.base_vectors:
            return self.hdc.base_vectors[kmer]

        # 1. Base k-mer encoding (using HDCVectorSpace's methods)
        base_vector = self.hdc.encode_kmer(kmer)

        # 2. Integrate biological features
        feature_vector = self._compute_feature_vector(kmer, position)
        if feature_vector is not None:
            # Combine base and feature vectors
            combined_vector = self.hdc.bundle(
                base_vector, feature_vector, alpha=0.6
            )  # Combine the two
            return self.hdc._normalize(combined_vector)  # Normalize combined
        return base_vector  # Return the base vector if no features.

    def encode_kmer(self, kmer: str, position: Optional[int] = None) -> Any:
        """Encode k-mer (using cache)."""
        return self.kmer_cache(kmer, position)

    def encode_sequence(
        self,
        sequence: str,
        k: Optional[int] = None,
        stride: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """Encode sequence with biological features."""
        k = k or self.utils.detect_optimal_kmer_size(sequence)
        stride = stride or max(1, k // 3)
        chunk_size = chunk_size or self.config.chunk_size
        xp = self.hdc.xp
        if len(sequence) < k:
            return self.encode_kmer(sequence)

        result = xp.zeros(self.hdc.dim, dtype=xp.float32)
        n_kmers = 0
        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]
            kmers = self.utils.generate_kmers(chunk, k, stride)
            chunk_vectors = []

            for i, kmer in enumerate(kmers):
                if kmer.count("N") <= k // 3:
                    position = chunk_start + (i * stride)
                    kmer_vector = self.encode_kmer(kmer, position)
                    chunk_vectors.append(kmer_vector)

            if chunk_vectors:  # Combine chunk vectors with weighted average
                stacked = xp.array(chunk_vectors)
                weights = xp.array(
                    [self.config.bundle_decay**i for i in range(len(chunk_vectors))]
                )
                weights /= weights.sum()  # Normalize
                chunk_result = xp.sum(stacked * weights.reshape(-1, 1), axis=0)
                result = self.hdc.bundle(result, chunk_result, self.config.bundle_decay)
                n_kmers += len(chunk_vectors)

        return (
            self.hdc._normalize(result)
            if n_kmers > 0
            else xp.zeros(self.hdc.dim, dtype=xp.float32)
        )

    def _optimal_kmer_size(self, sequence: str) -> int:
        """Determine optimal k-mer size using Utils."""
        return self.utils.detect_optimal_kmer_size(sequence)

    def load_conservation_data(
        self, conservation_file: str, weight: float = 0.8
    ) -> "BiologicalEncoder":
        """Load conservation scores using the GenomicDataLoader."""
        conservation_scores = self.data_loader.load_conservation_file(conservation_file)

        def conservation_provider(
            kmer: str, position: Optional[int] = None
        ) -> Optional[float]:
            if position is None:
                return 0.5
            scores = [
                conservation_scores.get(position + i, 0.0) for i in range(len(kmer))
            ]
            return sum(scores) / len(scores) if scores else 0.5

        self.register_feature_provider(
            FeatureProvider("conservation", conservation_provider, weight)
        )
        return self

    def load_annotations(
        self, annotation_file: str, weight: float = 0.7
    ) -> "BiologicalEncoder":
        """Load genomic annotations using the GenomicDataLoader."""
        annotations = self.data_loader.load_annotation_file(annotation_file)
        annotation_vectors = {}

        for ann_type in set(ann["type"] for ann in annotations.values()):
            seed_val = hash(ann_type) % 10000
            np.random.seed(seed_val)  # Consistent seeding
            ann_vector = self.hdc._normalize(
                self.hdc.xp.array(np.random.uniform(-1, 1, self.hdc.dim))
            )
            annotation_vectors[ann_type] = ann_vector

        def annotation_provider(
            kmer: str, position: Optional[int] = None
        ) -> Optional[Any]:
            if position is None:
                return None
            region_annotations = [
                annotations.get(position + i) for i in range(len(kmer))
            ]
            region_annotations = [
                ann for ann in region_annotations if ann is not None
            ]  # Filter None

            if not region_annotations:
                return None

            result = self.hdc.xp.zeros(self.hdc.dim)
            for ann in region_annotations:
                if ann["type"] in annotation_vectors:
                    result = self.hdc.bundle(
                        result, annotation_vectors[ann["type"]], alpha=0.7
                    )
            return self.hdc._normalize(result)

        self.register_feature_provider(
            FeatureProvider("annotations", annotation_provider, weight)
        )
        return self

    def load_epigenetic_data(
        self, epigenetic_file: str, weight: float = 0.6
    ) -> "BiologicalEncoder":
        """Load epigenetic data using the GenomicDataLoader."""
        epigenetic_data = self.data_loader.load_epigenetic_file(epigenetic_file)

        def epigenetic_provider(
            kmer: str, position: Optional[int] = None
        ) -> Optional[Any]:
            if position is None:
                return None

            region_data = {}
            for i in range(len(kmer)):
                if position + i in epigenetic_data:
                    for key, value in epigenetic_data[position + i].items():
                        region_data[key] = region_data.get(key, 0) + value

            if not region_data:
                return None

            # Normalize and create vector.
            for key in region_data:
                region_data[key] /= len(kmer)
            result = self.hdc.xp.zeros(self.hdc.dim)

            for key, value in region_data.items():
                seed_val = hash(key) % 10000
                np.random.seed(seed_val)
                feature_vec = self.hdc._normalize(
                    self.hdc.xp.array(np.random.uniform(-1, 1, self.hdc.dim))
                )
                feature_vec *= value
                result = self.hdc.bundle(result, feature_vec, alpha=0.7)
            return self.hdc._normalize(result)

        self.register_feature_provider(
            FeatureProvider("epigenetics", epigenetic_provider, weight)
        )
        return self

    def load_motif_data(
        self, motif_file: str, weight: float = 0.7
    ) -> "BiologicalEncoder":
        """Load motif data using the GenomicDataLoader."""
        motifs = self.data_loader.load_motif_file(motif_file)

        def motif_provider(kmer: str, position: Optional[int] = None) -> Optional[Any]:
            if position is None:
                return self._detect_motifs(kmer)  # Fallback to local motif detection

            overlapping_motifs = [
                motif
                for motif in motifs
                if motif["start"] <= position + len(kmer)
                and motif["start"] + len(motif.get("pattern", "")) >= position
            ]

            if not overlapping_motifs:
                return None

            result = self.hdc.xp.zeros(self.hdc.dim)
            for motif in overlapping_motifs:
                seed_val = hash(motif["name"]) % 10000  # Consistent seed
                np.random.seed(seed_val)
                motif_vec = self.hdc._normalize(
                    self.hdc.xp.array(np.random.uniform(-1, 1, self.hdc.dim))
                )
                motif_vec *= motif.get("score", 1.0)
                result = self.hdc.bundle(result, motif_vec, alpha=0.7)
            return self.hdc._normalize(result)

        self.register_feature_provider(
            FeatureProvider("known_motifs", motif_provider, weight)
        )
        return self


@dataclass
class DNAEncoderStats:
    """Data class to store DNA encoder statistics."""

    base_counts: Dict[str, int] = field(
        default_factory=lambda: {base: 0 for base in "ACGTN"}
    )
    total_bases: int = 0
    gc_content: float = 0.5
    seq_length_distrib: List[int] = field(default_factory=list)
    kmer_usage: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    optimal_k: int = 7

    def update_base_counts(self, sequence: str):
        """Update base counts and GC content from a sequence."""
        for base in sequence:
            if base in self.base_counts:
                self.base_counts[base] += 1
                self.total_bases += 1
        if self.total_bases > 0:
            gc_count = self.base_counts["G"] + self.base_counts["C"]
            self.gc_content = gc_count / self.total_bases

    def update_sequence_length(self, sequence_length: int):
        self.seq_length_distrib.append(sequence_length)
        if len(self.seq_length_distrib) > 100:
            self.seq_length_distrib = self.seq_length_distrib[-100:]

    def update_kmer_usage(self, kmer_length: int):
        self.kmer_usage[kmer_length] += 1

    def get_average_length(self) -> float:
        return sum(self.seq_length_distrib) / max(1, len(self.seq_length_distrib))

    def get_base_distribution(self) -> Dict[str, float]:
        return {
            k: v / self.total_bases if self.total_bases > 0 else 0.0
            for k, v in self.base_counts.items()
        }


class DNAEncoder:
    """HDC-based DNA sequence encoder."""

    def __init__(self, config: GSCConfig):
        """Initialize the DNA encoder."""
        self.config = config
        self.hdc = HDCVectorSpace(config)
        self.stats = DNAEncoderStats()
        self.kmer_cache = lru_cache(maxsize=config.cache_size)(
            self._encode_kmer_uncached
        )
        self.sequence_cache = {}
        self.utils = Utils()  # Initialize the Utils class

    def _encode_kmer_uncached(self, kmer: str, position: Optional[int] = None) -> Any:
        """Core k-mer encoding logic (uncached)."""
        if len(kmer) == 1 and kmer in self.hdc.base_vectors:
            return self.hdc.base_vectors[kmer]
        return (
            self.hdc._encode_clean_kmer(kmer)
            if all(base in self.hdc.base_vectors for base in kmer)
            else self.hdc._encode_ambiguous_kmer(kmer)
        )

    def encode_kmer(self, kmer: str, position: Optional[int] = None) -> Any:
        """Encode a k-mer (using cache)."""
        self.stats.update_kmer_usage(len(kmer))
        return self.kmer_cache(kmer, position)

    def encode_sequence(
        self,
        sequence: str,
        k: Optional[int] = None,
        stride: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """Encode a DNA sequence with adaptive parameters using Utils."""
        k = k or self._get_optimal_k(sequence)
        stride = stride or max(1, k // 3)
        chunk_size = chunk_size or self.config.chunk_size
        xp = self.hdc.xp

        if len(sequence) < k:
            return self.encode_kmer(sequence)

        cache_key = (sequence, k, stride)
        if cache_key in self.sequence_cache:
            return self.sequence_cache[cache_key]

        result = xp.zeros(self.hdc.dim, dtype=xp.float32)
        n_kmers = 0

        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size + k - 1, len(sequence))
            chunk = sequence[chunk_start:chunk_end]
            kmers = self.utils.generate_kmers(chunk, k, stride)
            chunk_vectors = [
                self.encode_kmer(kmer) for kmer in kmers if kmer.count("N") <= k // 3
            ]

            if chunk_vectors:
                stacked = xp.array(chunk_vectors)
                weights = xp.array(
                    [self.config.bundle_decay**i for i in range(len(chunk_vectors))]
                )
                weights /= weights.sum()
                chunk_result = xp.sum(stacked * weights.reshape(-1, 1), axis=0)
                result = self.hdc.bundle(result, chunk_result, self.config.bundle_decay)
                n_kmers += len(chunk_vectors)

        if n_kmers == 0:
            logger.warning(f"No valid k-mers in sequence: {sequence[:20]}...")
            return xp.zeros(self.hdc.dim, dtype=xp.float32)

        result = self.hdc._normalize(result)
        if len(sequence) < 10000:  # Cache result for short sequences
            self.sequence_cache[cache_key] = result
        self.stats.update_sequence_length(len(sequence))
        self.stats.update_base_counts(sequence)
        return result

    def _get_optimal_k(self, sequence: str) -> int:
        """Determine optimal k-mer size using Utils."""
        if sum(self.stats.kmer_usage.values()) > 1000:
            self.stats.optimal_k = max(
                self.stats.kmer_usage, key=self.stats.kmer_usage.get
            )
            return self.stats.optimal_k
        if len(sequence) < 20:
            return min(5, len(sequence))
        return self.utils.detect_optimal_kmer_size(sequence)

    def find_similar_sequences(
        self, query: str, references: List[str], k: Optional[int] = None, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar sequences using k-mer similarity."""
        k = k or self._get_optimal_k(query)
        query_vector = self.encode_sequence(query, k=k)
        ref_vectors = [self.encode_sequence(seq, k=k) for seq in references]
        similarities = self.hdc.batch_similarity(query_vector, ref_vectors)
        similarities_np = (
            similarities.get() if isinstance(similarities, cp.ndarray) else similarities
        )
        top_indices = np.argsort(similarities_np)[-top_n:][::-1]
        return [(references[i], similarities_np[i]) for i in top_indices]

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        kmer_cache_info = self.kmer_cache.cache_info()
        return {
            "kmer_cache_size": kmer_cache_info.currsize,
            "sequence_cache_size": len(self.sequence_cache),
            "cache_hit_ratio": kmer_cache_info.hits
            / max(1, kmer_cache_info.hits + kmer_cache_info.misses),
            "base_distribution": self.stats.get_base_distribution(),
            "gc_content": self.stats.gc_content,
            "avg_sequence_length": self.stats.get_average_length(),
            "optimal_k": self.stats.optimal_k,
            "dimension": self.hdc.dim,
            "device": self.hdc.device,
        }

    def save_vectors(self, vectors: List[Any], filename: str):
        """Save vectors to HDF5."""
        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "vectors",
                data=np.array(
                    [v.get() if isinstance(v, cp.ndarray) else v for v in vectors]
                ),
            )
            f.attrs["dimension"] = self.hdc.dim
            f.attrs["device"] = self.hdc.device
            f.attrs["vector_type"] = self.hdc.vector_type

    def load_vectors(self, filename: str) -> List[Any]:
        """Load vectors from HDF5."""
        with h5py.File(filename, "r") as f:
            vectors = f["vectors"][:]
            if "dimension" in f.attrs and f.attrs["dimension"] != self.hdc.dim:
                logger.warning("Loaded vectors have different dimension.")
            return [
                self.hdc.xp.array(v) if self.hdc.device == "gpu" else v for v in vectors
            ]


@dataclass
class MetaHDConfig:
    """Configuration for MetaHDConservation."""

    memory_depth: int = 5
    alpha: float = 0.1  # Learning rate
    conserved_threshold: float = 0.7
    non_conserved_threshold: float = 0.3
    context_decay: float = 0.9
    conserved_weight: float = 0.6
    non_conserved_weight: float = -0.3
    context_weight: float = 0.1


class MetaHDConservation:
    """Meta-learning conservation scorer using HDC vectors."""

    def __init__(
        self,
        hdc_computer: "HDCVectorSpace",  # Type hint for clarity
        config: Optional[MetaHDConfig] = None,
        **kwargs,
    ):
        """Initialize MetaHDConservation.

        Args:
            hdc_computer: The HDCVectorSpace instance to use.
            config: Optional MetaHDConfig instance.  If None, a default
                config is created, and any provided kwargs are used to
                override the defaults.
            **kwargs:  Used to override individual config parameters.
                Takes precedence over values in the `config` argument.
        """

        self.hdc = hdc_computer
        self.config = config or MetaHDConfig(**kwargs)
        # Apply kwargs overrides (cleaner way to handle overrides)
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.xp = self.hdc.xp  # Use the same compute backend

        # Initialize meta-patterns as HDC vectors
        self.meta_patterns: Dict[str, Any] = {
            "conserved": self.hdc._random_vector(),
            "non_conserved": self.hdc._random_vector(),
            "context": self.hdc._random_vector(),
        }

        # Pre-compute context weights for efficiency
        self.context_weights = self.xp.array(
            [self.config.context_decay**i for i in range(self.config.memory_depth)]
        )
        self.context_weights /= self.xp.sum(self.context_weights)  # Normalize weights

        self.context_vectors: List[Any] = []

    def _meta_encode(self, sequence: str) -> Tuple[Any, Any]:
        """Generate meta-features from HDC vector patterns (using batch similarity)."""
        base_vector = self.hdc.encode_sequence(sequence)
        # Use batch similarity for efficiency
        similarities = self.hdc.batch_similarity(
            base_vector, list(self.meta_patterns.values())
        )

        # Create a dictionary with meaningful keys.  Order matters here.
        feature_names = [
            "conserved_similarity",
            "non_conserved_similarity",
            "context_similarity",
        ]
        features = dict(zip(feature_names, similarities))
        return base_vector, features

    def update_meta_patterns(self, vector: Any, conservation_score: float):
        """Update meta-patterns using HDC bundling (vectorized)."""
        # Adaptive learning rate
        alpha = self.config.alpha * (1.0 - conservation_score)

        # Update conserved and non-conserved patterns using bundling
        if conservation_score > self.config.conserved_threshold:
            self.meta_patterns["conserved"] = self.hdc.bundle(
                self.meta_patterns["conserved"], vector, alpha=1 - alpha
            )
        elif conservation_score < self.config.non_conserved_threshold:
            self.meta_patterns["non_conserved"] = self.hdc.bundle(
                self.meta_patterns["non_conserved"], vector, alpha=1 - alpha
            )

        # Update context vector (weighted average of recent vectors)
        self.context_vectors.append(vector)
        if len(self.context_vectors) > self.config.memory_depth:
            self.context_vectors = self.context_vectors[-self.config.memory_depth :]

        # Efficient weighted average using precomputed weights.  No loop needed!
        if self.context_vectors:  # Ensure not empty before stacking
            stacked_vectors = self.xp.stack(
                self.context_vectors[-self.config.memory_depth :]
            )
            weighted_context = self.xp.dot(
                self.context_weights[: stacked_vectors.shape[0]], stacked_vectors
            )  # Use dot product
            self.meta_patterns["context"] = self.hdc._normalize(weighted_context)

    def score(self, sequence: str) -> float:
        """Score conservation using meta-HDC patterns (vectorized)."""
        vector, features = self._meta_encode(sequence)

        # Weighted combination of features (now more readable)
        score = (
            self.config.conserved_weight * features["conserved_similarity"]
            + self.config.non_conserved_weight * features["non_conserved_similarity"]
            + self.config.context_weight * features["context_similarity"]
        )

        # Clip score to [0, 1] and handle potential NaN values
        score = np.clip(score, 0, 1) if not np.isnan(score) else 0.5
        self.update_meta_patterns(vector, score)  # Update after score calculation
        return float(score)  # Ensure return type is float

    def get_meta_patterns(self) -> Dict[str, Any]:
        """Returns a copy of the meta_patterns."""
        return {k: v.copy() for k, v in self.meta_patterns.items()}

    def set_meta_patterns(self, patterns: Dict[str, Any]):
        """Set the meta patterns."""
        for k, v in patterns.items():
            if k in self.meta_patterns:
                # Ensure patterns are on the correct device.
                self.meta_patterns[k] = (
                    self.xp.array(v) if isinstance(v, np.ndarray) else v.copy()
                )


@dataclass
class SequenceStats:
    """Data class to store sequence statistics."""

    length: int = 0
    gc_content: float = 0.0
    base_counts: Dict[str, int] = field(
        default_factory=lambda: {base: 0 for base in "ACGTN"}
    )
    optimal_k: int = 7
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    motif_positions: List[Tuple[int, float]] = field(default_factory=list)

    def update_stats(self, sequence: str):
        """Update sequence statistics."""
        self.length = len(sequence)
        self.gc_content = (
            (sequence.count("G") + sequence.count("C")) / self.length
            if self.length > 0
            else 0
        )
        self.base_counts = {base: sequence.count(base) for base in "ACGTN"}
        self.optimal_k = Utils.detect_optimal_kmer_size(sequence)
        self._calculate_entropy(sequence)

    def _calculate_entropy(self, sequence: str):
        """Calculate sequence entropy."""
        base_counts = {
            base: sequence.count(base) for base in set(sequence) if base in "ACGT"
        }
        total = sum(base_counts.values())
        if total > 0:
            probs = [count / total for count in base_counts.values()]
            self.entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            self.normalized_entropy = self.entropy / math.log2(len(base_counts))


class GeneticAnalyzer:
    """High-level API for genetic sequence analysis using HDC."""

    def __init__(
        self,
        dimension: int = None,
        device: str = "auto",
        use_biological: bool = False,
        data_size: int = None,
    ):
        """Initialize genetic analyzer."""
        self.encoder_class = BiologicalEncoder if use_biological else DNAEncoder
        self.encoder = self.encoder_class(
            dimension=dimension, device=device, data_size=data_size
        )
        self.config = {
            "dimension": self.encoder.dim,
            "device": self.encoder.device,
            "use_biological": use_biological,
        }
        logger.info(
            f"Initialized GeneticAnalyzer (dim={self.encoder.dim}, device={self.encoder.device}, bio={use_biological})"
        )

    def encode_sequences(
        self, sequences: List[str], k: Optional[int] = None
    ) -> List[Any]:
        """Encode multiple sequences into HDC vectors."""
        k = k or self.encoder._get_optimal_k(sequences[0]) if sequences else 7
        vectors = [
            self.encoder.encode_sequence(seq, k=k)
            for seq in tqdm(sequences, desc="Encoding sequences")
        ]
        logger.info(f"Encoded {len(sequences)} sequences with k={k}")
        return vectors

    def compute_similarity_matrix(
        self, sequences: List[str], k: Optional[int] = None
    ) -> np.ndarray:
        """Compute pairwise similarity matrix for sequences."""
        n = len(sequences)
        similarity_matrix = np.zeros((n, n))
        vectors = self.encode_sequences(sequences, k=k)

        for i in tqdm(range(n), desc="Computing similarities"):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = float(self.encoder.similarity(vectors[i], vectors[j]))
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def find_similar(
        self, query: str, references: List[str], k: Optional[int] = None, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar sequences to a query."""
        return self.encoder.find_similar_sequences(query, references, k, top_n)

    def analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a sequence."""
        stats = SequenceStats()
        stats.update_stats(sequence)

        if isinstance(self.encoder, BiologicalEncoder):
            stats.motif_positions = [
                (i, score)
                for i in range(0, len(sequence) - 10 + 1, 5)
                if (score := self.encoder._detect_motifs(sequence[i : i + 10])) > 0.6
            ]

        return stats.__dict__

    def _save_hdf5_group(self, h5file: h5py.File, group_name: str, data: dict):
        """Helper function to save a dictionary of data to an HDF5 group."""
        group = h5file.create_group(group_name)
        for key, value in data.items():
            if isinstance(value, (np.ndarray, cp.ndarray)):
                value = value.get() if isinstance(value, cp.ndarray) else value
                group.create_dataset(str(key), data=value)
            elif isinstance(value, dict):
                self._save_hdf5_group(group, str(key), value)
            else:
                group.attrs[str(key)] = value

    def _load_hdf5_group(self, h5file: h5py.File, group_name: str) -> dict:
        """Helper function to load data from an HDF5 group into a dictionary."""
        group = h5file[group_name]
        data = {}
        for key in group:
            if isinstance(group[key], h5py.Dataset):
                data[key] = group[key][:]
                if self.encoder.device == "gpu":
                    data[key] = self.encoder.xp.array(data[key])
            else:
                data[key] = self._load_hdf5_group(group, key)
        for key in group.attrs:
            data[key] = group.attrs[key]
        return data

    def save(self, filename: str):
        """Save analyzer state to disk."""
        with h5py.File(filename, "w") as f:
            f.attrs.update(self.config)
            encoder_data = {
                "base_vectors": self.encoder.base_vectors,
                "position_vectors": self.encoder.position_vectors,
            }
            self._save_hdf5_group(f, "encoder", encoder_data)

            if hasattr(self, "meta_conservation"):
                meta_data = self.meta_conservation.get_meta_patterns()
                self._save_hdf5_group(f, "meta_conservation", meta_data)

        logger.info(f"Saved analyzer state to {filename}")

    def load(self, filename: str):
        """Load analyzer state from disk."""
        with h5py.File(filename, "r") as f:
            self.config = dict(f.attrs)
            use_biological = self.config.get("use_biological", False)
            self.encoder_class = BiologicalEncoder if use_biological else DNAEncoder
            self.encoder = self.encoder_class(
                dimension=self.config.get("dimension"),
                device=self.config.get("device", "auto"),
                data_size=self.config.get("data_size"),
            )

            if "encoder" in f:
                encoder_data = self._load_hdf5_group(f, "encoder")
                self.encoder.base_vectors = encoder_data["base_vectors"]
                self.encoder.position_vectors = encoder_data["position_vectors"]

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


@dataclass
class AgentConfig:
    """Configuration for genetic agent behavior and learning."""

    class Type(Enum):
        """Agent specialization types with their core functionality."""

        MOTIF_DETECTOR = auto()  # Regulatory elements and motifs
        STRUCTURE_ANALYZER = auto()  # DNA/RNA structure properties
        CONSERVATION_TRACKER = auto()  # Evolutionary patterns
        CODON_SPECIALIST = auto()  # Coding regions and bias
        BOUNDARY_FINDER = auto()  # Sequence boundaries
        GENERALIST = auto()  # Balanced analysis
        IMPUTER = auto()  # Gap filling

        @property
        def k_size(self) -> int:
            """Get optimal k-mer size for this agent type."""
            return {
                self.MOTIF_DETECTOR: 6,  # Common motif length
                self.STRUCTURE_ANALYZER: 4,  # Structure element size
                self.CONSERVATION_TRACKER: 7,  # Conservation window
                self.CODON_SPECIALIST: 3,  # Codon length
                self.BOUNDARY_FINDER: 8,  # Junction context
                self.IMPUTER: 7,  # Gap context
                self.GENERALIST: 5,  # Balanced size
            }[self]

        @property
        def feature_weights(self) -> Dict[str, float]:
            """Get specialized feature weights for this agent type."""
            base_weights = {
                "motifs": 1.0,
                "gc_content": 1.0,
                "complexity": 1.0,
                "conservation": 1.0,
                "structure": 1.0,
                "coding": 1.0,
            }

            specializations = {
                self.MOTIF_DETECTOR: {"motifs": 2.5, "conservation": 1.5},
                self.STRUCTURE_ANALYZER: {"structure": 2.5, "gc_content": 1.8},
                self.CONSERVATION_TRACKER: {"conservation": 2.5, "motifs": 1.2},
                self.CODON_SPECIALIST: {"coding": 2.5, "gc_content": 1.3},
                self.BOUNDARY_FINDER: {"motifs": 2.0, "complexity": 2.0},
                # IMPUTER and GENERALIST use base weights
            }

            weights = base_weights.copy()
            weights.update(specializations.get(self, {}))
            return weights

    # Core hyperparameters
    agent_type: Type = Type.GENERALIST
    mode: str = "general"  # 'general' or 'imputation'
    learning_rate: float = 0.01
    gamma: float = 0.95
    epsilon_start: float = 0.3
    epsilon_decay: float = 0.9
    buffer_size: int = 10000
    batch_size: int = 32
    history_size: int = 10

    # Dynamic properties based on agent type
    k_size: int = field(init=False)
    feature_weights: Dict[str, float] = field(init=False)

    def __post_init__(self):
        """Initialize type-specific parameters."""
        self.k_size = self.agent_type.k_size
        self.feature_weights = self.agent_type.feature_weights

        # Override settings for imputation mode
        if self.mode == "imputation":
            self.agent_type = self.Type.IMPUTER
            self.k_size = self.Type.IMPUTER.k_size


@dataclass
class AgentState:
    """Tracks agent's internal state and memory."""

    q_vectors: Dict[str, Any] = field(default_factory=dict)
    experience_buffer: List[tuple] = field(default_factory=list)
    history: List[Any] = field(default_factory=list)
    confidence: float = 0.5
    rewards: List[float] = field(default_factory=list)

    def update_confidence(self, vector: Any, xp: Any) -> None:
        """Update confidence based on reading consistency."""
        if self.history:
            vector_norm = float(xp.linalg.norm(vector))
            history_norms = [float(xp.linalg.norm(v)) for v in self.history]
            self.confidence = 0.8 * self.confidence + 0.2 * (
                1.0 - min(1.0, abs(vector_norm - np.mean(history_norms)))
            )
        self.history.append(vector)
        self.history = self.history[-10:]  # Keep last 10 readings


class GeneticAgent:
    """HDC-powered reinforcement learning agent for genetic sequence analysis."""

    def __init__(
        self, supercomputer: "HDCVectorSpace", config: Optional[AgentConfig] = None
    ):
        self.sc = supercomputer
        self.xp = supercomputer.xp
        self.config = config or AgentConfig(agent_type=AgentConfig.Type.GENERALIST)
        self.state = AgentState()
        self.action_space = ["A", "T", "G", "C"]

    def predict_base(self, state_vector: Any, epsilon: float = 0.1) -> str:
        """Predict next base using vectorized Q-value computation."""
        # Exploration
        if self.xp.random.random() < epsilon:
            return self.xp.random.choice(self.action_space)

        # Exploitation - get Q-values for all actions in parallel
        state_hash = hash(state_vector.tobytes())
        q_values = {
            action: self._get_q_value(state_hash, action, state_vector)
            for action in self.action_space
        }
        return max(q_values.items(), key=lambda x: x[1])[0]

    def _get_q_value(self, state_hash: int, action: str, state_vector: Any) -> float:
        """Get Q-value, initializing if needed using HDC base vectors."""
        key = f"{state_hash}_{action}"
        if key not in self.state.q_vectors:
            self.state.q_vectors[key] = self.sc.base_vectors[action].copy()
        return float(self.xp.dot(state_vector, self.state.q_vectors[key]))

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        """TD learning update using HDC operations."""
        state_hash = hash(state.tobytes())
        key = f"{state_hash}_{action}"

        # Get max future Q-value using vectorized operations
        next_hash = hash(next_state.tobytes())
        next_q = max(
            self._get_q_value(next_hash, a, next_state) for a in self.action_space
        )

        # TD update using HDC bundling
        current_q = self._get_q_value(state_hash, action, state)
        td_error = reward + self.config.gamma * next_q - current_q
        delta = self.config.learning_rate * td_error

        self.state.q_vectors[key] = self.sc.bundle(
            self.state.q_vectors[key], state, alpha=1.0 - min(0.9, abs(delta))
        )

    def train_on_segment(self, context: Any, target: str) -> float:
        """Train on sequence segment using vectorized operations."""
        state = context.copy()
        total_reward = 0

        for target_base in target:
            # Make prediction and get reward
            pred_base = self.predict_base(state, self.config.epsilon_start)
            reward = 1.0 if pred_base == target_base else -0.2
            total_reward += reward

            # Get next state using HDC operations
            next_vector = self.sc.encode_kmer(pred_base)
            next_state = self.sc.bundle(state, next_vector, alpha=0.7)

            # Update Q-values and store experience
            self.update(state, pred_base, reward, next_state)
            self.state.experience_buffer.append((state, pred_base, reward, next_state))

            # Trim buffer if needed
            if len(self.state.experience_buffer) > self.config.buffer_size:
                self.state.experience_buffer = self.state.experience_buffer[
                    -self.config.buffer_size :
                ]

            state = next_state

        # Update agent state tracking
        self.state.rewards.append(total_reward / len(target))
        self.state.update_confidence(state, self.xp)

        return total_reward

    def train_batch(self, batch_size: Optional[int] = None) -> int:
        """Train on random batch from experience buffer."""
        batch_size = batch_size or self.config.batch_size
        if len(self.state.experience_buffer) < batch_size:
            return 0

        # Sample and train on batch
        indices = self.xp.random.choice(
            len(self.state.experience_buffer), batch_size, replace=False
        )
        for idx in indices:
            state, action, reward, next_state = self.state.experience_buffer[idx]
            self.update(state, action, reward, next_state)

        return len(indices)


@dataclass
class SwarmConfig:
    """Configuration for swarm behavior and dynamics."""

    size: int = 10
    min_agent_count: int = 6
    beam_width: int = 5  # For beam search in imputation
    window_size: int = 10
    stride: int = 5

    # Swarm dynamics
    individual_weight: float = 0.7
    type_weight: float = 0.3
    momentum: float = 0.8

    # Feature importance thresholds
    feature_threshold: float = 0.1
    interest_threshold: float = 0.5

    # Performance tracking
    history_size: int = 10
    cache_size: int = 1000


@dataclass
class FeatureMetrics:
    """Metrics for sequence feature analysis."""

    motifs: float = 0.0
    structure: float = 0.0
    conservation: float = 0.0
    coding: float = 0.0
    boundary: float = 0.0
    gc_content: float = 0.0
    complexity: float = 0.0

    def normalize(self) -> "FeatureMetrics":
        total = sum(getattr(self, f) for f in self.__dataclass_fields__)
        if total > 0:
            for field in self.__dataclass_fields__:
                setattr(self, field, getattr(self, field) / total)
        return self


class GeneticSwarm:
    """HDC-powered swarm intelligence for genetic sequence analysis."""

    def __init__(
        self, supercomputer: "HDCVectorSpace", config: Optional[SwarmConfig] = None
    ):
        self.sc = supercomputer
        self.xp = supercomputer.xp
        self.config = config or SwarmConfig()

        # Initialize swarm components
        self.agents: List[GeneticAgent] = []
        self.weights: Dict[int, float] = {}
        self.convergence = 0.0
        self.feature_cache = {}

        # Performance tracking
        self.rewards = []
        self.history = {"train": [], "val": []}
        self.epoch = 0

    def initialize_swarm(self, sequences: Optional[List[str]] = None) -> None:
        """Initialize swarm with data-driven agent distribution."""
        distribution = (
            self._analyze_distribution(sequences)
            if sequences
            else self._default_distribution()
        )

        # Create agents based on distribution
        agent_id = 0
        for agent_type, count in distribution.items():
            for _ in range(count):
                agent = GeneticAgent(
                    self.sc,
                    agent_type=agent_type,
                    config=AgentConfig(agent_type=agent_type),
                )
                self.agents.append(agent)
                self.weights[agent_id] = 0.5
                agent_id += 1

        if sequences:
            self._adapt_to_sequences(sequences)

    def _analyze_distribution(
        self, sequences: List[str]
    ) -> Dict[AgentConfig.Type, int]:
        """Determine optimal agent distribution from sequence characteristics."""
        if len(sequences) < 3:
            return self._default_distribution()

        # Analyze sequence features
        metrics = self._compute_sequence_metrics(sequences[:10])

        # Map features to agent types
        type_scores = {
            AgentConfig.Type.MOTIF_DETECTOR: metrics.motifs,
            AgentConfig.Type.STRUCTURE_ANALYZER: metrics.structure,
            AgentConfig.Type.CONSERVATION_TRACKER: metrics.conservation,
            AgentConfig.Type.CODON_SPECIALIST: metrics.coding,
            AgentConfig.Type.BOUNDARY_FINDER: metrics.boundary,
            AgentConfig.Type.GENERALIST: 0.1,  # Always keep some generalists
        }

        return self._allocate_agents(type_scores)

    def _compute_sequence_metrics(self, sequences: List[str]) -> FeatureMetrics:
        """Compute feature metrics for sequences using vectorized operations."""
        metrics = FeatureMetrics()

        for seq in sequences:
            if len(seq) < 30:
                continue

            # Vectorized feature extraction
            windows = [
                seq[i : i + self.config.window_size]
                for i in range(
                    0, len(seq) - self.config.window_size, self.config.stride
                )
            ]

            if not windows:
                continue

            # Compute features in parallel using HDC operations
            features = self._extract_features_batch(windows)
            for feature_type in metrics.__dataclass_fields__:
                current = getattr(metrics, feature_type)
                setattr(
                    metrics,
                    feature_type,
                    current + np.mean([f.get(feature_type, 0) for f in features]),
                )

        return metrics.normalize()

    def predict_sequence(self, context: str, length: int, epsilon: float = 0.05) -> str:
        """Generate sequence prediction using swarm consensus."""
        state = self.sc.encode_sequence(context)
        result = []

        for _ in range(length):
            # Get weighted predictions from all agents
            predictions = defaultdict(float)
            for agent in self.agents:
                base = agent.predict_base(state, epsilon)
                weight = self.weights[agent.id] * agent.confidence
                predictions[base] += weight

            # Select best base and update state
            next_base = max(predictions.items(), key=lambda x: x[1])[0]
            result.append(next_base)

            # Update state vector using latest context
            kmer = "".join(result[-7:])
            next_vec = self.sc.encode_kmer(kmer)
            state = self.sc.bundle(state, next_vec, alpha=0.7)

        return "".join(result)

    def impute_segment(self, prefix: str, suffix: str, length: int) -> str:
        """Impute missing segment using beam search and swarm consensus."""
        if not prefix and not suffix:
            return "N" * length

        # Initialize beam with prefix context
        context = prefix[-10:] if len(prefix) > 10 else prefix
        beam = [
            (self.predict_sequence(context, length, epsilon=0.01 + i * 0.05), 0.0)
            for i in range(self.config.beam_width)
        ]

        if not suffix:
            return beam[0][0]

        # Score candidates using suffix similarity
        suffix_vec = self.sc.encode_sequence(suffix[:10])
        scored_candidates = [
            (cand, float(self.xp.dot(self.sc.encode_sequence(cand), suffix_vec)))
            for cand, _ in beam
        ]

        return max(scored_candidates, key=lambda x: x[1])[0]

    def train(
        self, sequences: List[str], epochs: int = 5, val_set: Optional[List[str]] = None
    ) -> Dict:
        """Train swarm using parallelized sequence processing."""
        for epoch in range(epochs):
            self.epoch += 1

            # Process sequences in parallel batches
            rewards = []
            for seq in sequences:
                if len(seq) < 30:
                    continue

                reward = self._train_sequence(seq)
                rewards.append(reward)

            # Update agent weights and track performance
            self._update_weights()
            avg_reward = np.mean(rewards) if rewards else 0
            self.history["train"].append(avg_reward)

            if val_set:
                val_acc = self.evaluate(val_set)
                self.history["val"].append(val_acc)

        return {"final_reward": self.history["train"][-1], "history": self.history}

    def _train_sequence(self, sequence: str) -> float:
        """Train all agents on a single sequence using HDC operations."""
        total_reward = 0
        state = self.sc.encode_sequence(sequence[:10])

        for i in range(10, len(sequence) - 5, 5):
            target = sequence[i : i + 5]

            # Train each agent
            for agent in self.agents:
                reward = agent.train_on_segment(state, target)
                total_reward += reward

            # Update state
            state = self.sc.encode_sequence(sequence[i : i + 10])

        return total_reward / max(1, len(sequence) - 15)


class HDCGenomicUI:
    def __init__(self):
        # Initialize with moderate defaults that work on most hardware
        self.config = GSCConfig(
            dimension=4096,  # Good balance between accuracy and speed
            device="auto",  # Let it detect GPU if available
            max_kmer_size=12,
            bundle_decay=0.85,
        )

        # Create the DNA supercomputer instance
        self.sc = DNASupercomputer(self.config)
        self.bio_encoder = BiologicalEncoder(self.config)

    def process_sequence_input(self, input_text, file_obj):
        """Handle input from either text area or file upload"""
        if file_obj is not None:
            # Process uploaded file (FASTA/FASTQ)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                tmp.write(file_obj)
                tmp_name = tmp.name

            sequences, seq_ids = self.sc.data_loader.load_sequences(tmp_name)
            os.unlink(tmp_name)  # Clean up temp file
            return sequences, seq_ids

        elif input_text.strip():
            # Process direct text input
            sequences = [seq.strip() for seq in input_text.split(">") if seq.strip()]
            if len(sequences) == 1 and "\n" not in sequences[0]:
                # Simple sequence without FASTA formatting
                return [sequences[0].strip()], ["seq1"]
            else:
                # Try to parse as FASTA
                parsed_seqs = []
                seq_ids = []
                for seq_block in sequences:
                    if not seq_block.strip():
                        continue
                    lines = seq_block.split("\n")
                    seq_ids.append(lines[0].strip())
                    parsed_seqs.append("".join(lines[1:]).strip())
                return parsed_seqs, seq_ids

        return [], []

    def encode_and_visualize(self, input_text, file_obj, kmer_size):
        """Encode sequences and visualize the vector space"""
        sequences, seq_ids = self.process_sequence_input(input_text, file_obj)

        if not sequences:
            return "No sequences provided", None, None

        # Encode sequences
        k = (
            int(kmer_size)
            if kmer_size
            else Utils.detect_optimal_kmer_size(sequences[0])
        )
        vectors = [self.sc.encode_sequence(seq, k=k) for seq in sequences]

        # Create visualization of vector space using PCA
        if len(vectors) > 1:
            # Convert to numpy arrays if on GPU
            if self.sc.hdc.device == "gpu":
                vectors_np = [v.get() for v in vectors]
            else:
                vectors_np = vectors

            # Compute pairwise similarities - USING vectors_np now!
            n_seqs = len(sequences)
            sim_matrix = np.zeros((n_seqs, n_seqs))
            for i in range(n_seqs):
                for j in range(i, n_seqs):
                    # Use vectors_np instead of vectors
                    if self.sc.hdc.device == "gpu":
                        sim = float(np.dot(vectors_np[i], vectors_np[j]))
                    else:
                        sim = float(
                            self.sc.hdc.similarity(vectors_np[i], vectors_np[j])
                        )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            # Create heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(sim_matrix, cmap="viridis")
            plt.colorbar(label="Similarity")
            plt.title(f"Sequence Similarity Matrix (k={k})")

            # Add sequence IDs as labels if not too many
            if n_seqs <= 20:
                plt.xticks(range(n_seqs), seq_ids, rotation=90)
                plt.yticks(range(n_seqs), seq_ids)

            heatmap_path = tempfile.mktemp(suffix=".png")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

            # Compute some stats
            stats = {
                seq_id: self.sc.analyze_sequence(seq)
                for seq_id, seq in zip(seq_ids, sequences)
            }

            # Format stats as text
            stats_text = ""
            for seq_id, seq_stats in stats.items():
                stats_text += f"Sequence: {seq_id}\n"
                stats_text += f"Length: {seq_stats.get('length', 0)}\n"
                stats_text += f"GC Content: {seq_stats.get('gc_content', 0):.2f}\n"
                stats_text += f"Entropy: {seq_stats.get('normalized_entropy', 0):.2f}\n"
                stats_text += f"Optimal k-mer: {seq_stats.get('optimal_k', k)}\n\n"

            return stats_text, Image.open(heatmap_path), f"Used k-mer size: {k}"
        else:
            # Single sequence analysis
            stats = self.sc.analyze_sequence(sequences[0])
            stats_text = f"Sequence length: {stats.get('length', 0)}\n"
            stats_text += f"GC Content: {stats.get('gc_content', 0):.2f}\n"
            stats_text += f"Entropy: {stats.get('normalized_entropy', 0):.2f}\n"
            stats_text += f"Optimal k-mer: {stats.get('optimal_k', k)}\n"

            # Generate motif plot if available
            motif_positions = stats.get("motif_positions", [])
            if motif_positions:
                plt.figure(figsize=(12, 3))
                positions, scores = zip(*motif_positions)
                plt.scatter(positions, scores, alpha=0.7)
                plt.xlabel("Position")
                plt.ylabel("Motif Score")
                plt.title("Potential Motif Locations")
                plt.ylim(0, 1)

                motif_path = tempfile.mktemp(suffix=".png")
                plt.tight_layout()
                plt.savefig(motif_path)
                plt.close()

                return stats_text, Image.open(motif_path), f"Used k-mer size: {k}"

            return stats_text, None, f"Used k-mer size: {k}"

    def impute_sequence(self, prefix, suffix, gap_length):
        """Fill in missing sequence between prefix and suffix"""
        if not prefix and not suffix:
            return "Need at least prefix or suffix to impute sequence"

        # Create vectors for context
        prefix_vec = self.sc.encode_sequence(prefix) if prefix else None
        suffix_vec = self.sc.encode_sequence(suffix) if suffix else None

        # Simple imputation strategy (could be enhanced with the GeneticSwarm)
        if prefix and suffix:
            # Use both contexts
            gap_length = int(gap_length) if gap_length else 10

            # Generate 5 candidates using varied bundle weights
            candidates = []
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                # Start with prefix context
                result = []
                state = prefix_vec

                for _ in range(gap_length):
                    # Predict next base using HDC similarity
                    scores = {}
                    for base in "ACGT":
                        base_vec = self.sc.hdc.base_vectors[base]
                        bound = self.sc.hdc.bind(state, base_vec)

                        # Score by similarity to suffix context
                        if suffix_vec is not None:
                            scores[base] = float(
                                self.sc.hdc.similarity(bound, suffix_vec)
                            )
                        else:
                            # If no suffix, use similarity to past context
                            scores[base] = float(self.sc.hdc.similarity(bound, state))

                    # Choose best base
                    next_base = max(scores.items(), key=lambda x: x[1])[0]
                    result.append(next_base)

                    # Update state with prefix context + new base
                    next_vec = self.sc.hdc.encode_kmer(next_base)
                    state = self.sc.hdc.bundle(state, next_vec, alpha=alpha)

                candidates.append(("".join(result), alpha))

            # Score candidates by compatibility with suffix
            candidate_scores = []
            for seq, alpha in candidates:
                cand_vec = self.sc.encode_sequence(seq)
                score = float(self.sc.hdc.similarity(cand_vec, suffix_vec))
                candidate_scores.append((seq, score, alpha))

            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # Format results
            result_text = f"Top imputation candidates for {gap_length}bp gap:\n\n"
            for i, (seq, score, alpha) in enumerate(candidate_scores):
                result_text += f"{i + 1}. {seq} (score: {score:.3f}, α: {alpha})\n"

            return result_text

        elif prefix:
            # Just predict continuation
            gap_length = int(gap_length) if gap_length else 10
            state = prefix_vec
            result = []

            for _ in range(gap_length):
                scores = {}
                for base in "ACGT":
                    base_vec = self.sc.hdc.base_vectors[base]
                    bound = self.sc.hdc.bind(state, base_vec)
                    scores[base] = float(self.sc.hdc.similarity(bound, state))

                next_base = max(scores.items(), key=lambda x: x[1])[0]
                result.append(next_base)

                next_vec = self.sc.hdc.encode_kmer(next_base)
                state = self.sc.hdc.bundle(state, next_vec, alpha=0.7)

            return f"Predicted continuation: {prefix}→{''.join(result)}"

        else:  # suffix only
            # Predict what might come before
            gap_length = int(gap_length) if gap_length else 10
            result = []

            # This is trickier - we'll work backwards
            state = suffix_vec
            for _ in range(gap_length):
                scores = {}
                for base in "ACGT":
                    base_vec = self.sc.hdc.base_vectors[base]
                    # Position-binding to simulate reverse direction
                    pos_vec = self.sc.hdc.position_vectors.get(
                        "pos_0", self.sc.hdc.base_vectors[base]
                    )
                    bound = self.sc.hdc.bind(base_vec, pos_vec)
                    bound = self.sc.hdc.bind(bound, state)
                    scores[base] = float(self.sc.hdc.similarity(bound, state))

                prev_base = max(scores.items(), key=lambda x: x[1])[0]
                result.insert(0, prev_base)

                prev_vec = self.sc.hdc.encode_kmer(prev_base)
                state = self.sc.hdc.bundle(prev_vec, state, alpha=0.3)

            return f"Predicted prefix: {''.join(result)}→{suffix}"

    def build_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="HDC Genomic Supercomputer") as app:
            gr.Markdown("# 🧬 HDC Genomic Supercomputer")
            gr.Markdown("Analyze DNA sequences using Hyperdimensional Computing")

            with gr.Tab("Encode & Analyze"):
                with gr.Row():
                    with gr.Column():
                        sequence_input = gr.Textbox(
                            label="Enter DNA sequence(s)",
                            placeholder="ATGCAAGTGCAATATTACGA...",
                            lines=10,
                        )
                        fasta_upload = gr.File(label="Or upload FASTA/FASTQ file")
                        kmer_size = gr.Slider(
                            label="k-mer size (0 for auto)",
                            minimum=0,
                            maximum=20,
                            step=1,
                            value=0,
                        )
                        analyze_btn = gr.Button("Analyze Sequence(s)")

                    with gr.Column():
                        stats_output = gr.Textbox(label="Analysis Results", lines=10)
                        viz_output = gr.Image(label="Visualization")
                        info_output = gr.Textbox(label="Info")

                analyze_btn.click(
                    fn=self.encode_and_visualize,
                    inputs=[sequence_input, fasta_upload, kmer_size],
                    outputs=[stats_output, viz_output, info_output],
                )

            with gr.Tab("Impute Sequence"):
                with gr.Row():
                    with gr.Column():
                        prefix_input = gr.Textbox(
                            label="Prefix (sequence before gap)",
                            placeholder="ATGCAAGTGCAATATTACGA...",
                            lines=5,
                        )
                        suffix_input = gr.Textbox(
                            label="Suffix (sequence after gap)",
                            placeholder="TTACGAGCATTGCAATGA...",
                            lines=5,
                        )
                        gap_length = gr.Slider(
                            label="Gap length to impute",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=10,
                        )
                        impute_btn = gr.Button("Impute Missing Sequence")

                    with gr.Column():
                        impute_output = gr.Textbox(label="Imputation Results", lines=15)

                impute_btn.click(
                    fn=self.impute_sequence,
                    inputs=[prefix_input, suffix_input, gap_length],
                    outputs=[impute_output],
                )

            with gr.Tab("About"):
                gr.Markdown(
                    """
                # HDC Genomic Supercomputer
                
                This interface provides access to a Hyperdimensional Computing (HDC) based DNA analysis system. 
                
                ## What is HDC?
                Hyperdimensional Computing is a computational framework inspired by patterns of neural activity in the brain.
                It represents information using high-dimensional vectors (thousands of dimensions), enabling robust pattern
                recognition and efficient computation on complex data like DNA sequences.
                
                ## Features
                - Sequence encoding in high-dimensional vector spaces
                - Sequence similarity analysis
                - Motif detection
                - Sequence imputation (filling in missing segments)
                
                ## Current Configuration
                - Vector dimension: {0}
                - Computing device: {1}
                - Max k-mer size: {2}
                """.format(
                        self.config.dimension,
                        self.config.device,
                        self.config.max_kmer_size,
                    )
                )

        return app


# -------------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------------


def configure_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"GSC_{time.strftime('%Y%m%d-%H%M%S')}.log"),
        ],
    )
    return logging.getLogger("GSC")


logger = configure_logging()


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

    # Subparsers for different operation modes
    subparsers = parser.add_subparsers(dest="mode", required=True)

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
        args.output = f"genetic_results_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}"

    return args


# -------------------------------------------------------------------------------
# Unified Run Function
# -------------------------------------------------------------------------------


def run_genetic_analysis(args):
    """Main entry point for all genetic analysis operations."""
    device = (
        "gpu" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    logger.info(f"Using device: {device}")

    use_biological = args.mode in ["analyze", "impute"]
    supercomputer = (BiologicalEncoder if use_biological else DNASupercomputer)(
        dimension=args.dim, device=device
    )

    data_loader = GenomicDataLoader()
    sequences, sequence_ids = [], []
    if args.input:
        sequences, sequence_ids = data_loader.load_sequences(args.input)

    if args.mode == "encode":
        if not sequences:
            logger.error("No input sequences provided for encoding")
            return

        vectors = [
            supercomputer.encode_sequence(seq, k=args.kmer)
            for seq in tqdm(sequences, desc="Encoding sequences")
        ]
        data_loader.save_results(vectors, args.output)
        logger.info(f"Cache performance: {supercomputer.get_cache_stats()}")

    elif args.mode == "train":
        if not sequences:
            logger.error("No input sequences provided for training")
            return

        model = (
            GeneticSwarm(supercomputer, swarm_size=args.swarm_size)
            if args.agent_type == "swarm"
            else GeneticAgent(supercomputer)
        )
        if args.model and os.path.exists(args.model):
            logger.info(f"Loading existing model from {args.model}")
            model.load(args.model)

        train_sequences = sequences[: int(0.8 * len(sequences))]
        val_sequences = (
            sequences[int(0.8 * len(sequences)) :] if len(sequences) > 1 else None
        )

        train_results = model.train(
            train_sequences,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_set=val_sequences,
        )
        data_loader.save_results(
            train_results, f"{args.output}_metrics.json", format="json"
        )
        model.save(f"{args.output}_model.h5")

    elif args.mode == "predict":
        if not args.model:
            logger.error("No model file provided for prediction")
            return

        model = GeneticSwarm(supercomputer)
        model.load(args.model)

        if sequences:
            predictions = [
                {
                    "id": sequence_ids[i],
                    "context": seq[: args.context],
                    "predicted": model.predict(
                        seq[: args.context], min(args.length, len(seq) - args.context)
                    ),
                    "actual": seq[args.context : args.context + args.length]
                    if len(seq) > args.context
                    else "",
                    "accuracy": sum(
                        p == a
                        for p, a in zip(
                            model.predict(
                                seq[: args.context],
                                min(args.length, len(seq) - args.context),
                            ),
                            seq[args.context : args.context + args.length],
                        )
                    )
                    / min(args.length, len(seq) - args.context)
                    if len(seq) > args.context
                    else 0,
                }
                for i, seq in enumerate(sequences)
                if len(seq) >= args.context
            ]
        else:
            context = input("Enter DNA context sequence: ").strip().upper()
            predictions = [
                {
                    "id": "interactive",
                    "context": context,
                    "predicted": model.predict(context, args.length),
                    "actual": "",
                    "accuracy": 0,
                }
            ]

        data_loader.save_results(
            predictions, f"{args.output}_predictions.json", format="json"
        )

    elif args.mode == "impute":
        if not args.model:
            logger.error("No model file provided for imputation")
            return

        model = GeneticSwarm(supercomputer)
        model.load(args.model)

        if args.prefix and args.suffix:
            imputations = [
                {
                    "id": "direct",
                    "prefix": args.prefix,
                    "suffix": args.suffix,
                    "imputed": model.impute_segment(
                        args.prefix, args.suffix, args.gap_length
                    ),
                }
            ]
        elif sequences:
            imputations = [
                {
                    "id": sequence_ids[i],
                    "gap_position": len(seq) // 3,
                    "prefix": seq[: len(seq) // 3],
                    "suffix": seq[len(seq) // 3 + args.gap_length :],
                    "imputed": model.impute_segment(
                        seq[: len(seq) // 3],
                        seq[len(seq) // 3 + args.gap_length :],
                        args.gap_length,
                    ),
                    "actual": seq[len(seq) // 3 : len(seq) // 3 + args.gap_length],
                    "accuracy": sum(
                        i == a
                        for i, a in zip(
                            model.impute_segment(
                                seq[: len(seq) // 3],
                                seq[len(seq) // 3 + args.gap_length :],
                                args.gap_length,
                            ),
                            seq[len(seq) // 3 : len(seq) // 3 + args.gap_length],
                        )
                    )
                    / args.gap_length,
                }
                for i, seq in enumerate(sequences)
                if len(seq) >= args.gap_length + 20
            ]
        else:
            prefix = input("Enter sequence before gap: ").strip().upper()
            suffix = input("Enter sequence after gap: ").strip().upper()
            imputations = [
                {
                    "id": "interactive",
                    "prefix": prefix,
                    "suffix": suffix,
                    "imputed": model.impute_segment(prefix, suffix, args.gap_length),
                }
            ]

        data_loader.save_results(
            imputations, f"{args.output}_imputations.json", format="json"
        )

    elif args.mode == "analyze":
        if not sequences:
            logger.error("No input sequences provided for analysis")
            return

        analyzer = GeneticSwarm(supercomputer) if args.model else supercomputer
        if args.model:
            analyzer.load(args.model)

        analyses = [
            SequenceStats().update_stats(seq).__dict__
            for seq in sequences
            if len(seq) >= args.window
        ]

        if args.similarity:
            similarity_matrix = np.array(
                [
                    [
                        float(
                            supercomputer.similarity(
                                supercomputer.encode_sequence(seq1, k=5),
                                supercomputer.encode_sequence(seq2, k=5),
                            )
                        )
                        for seq1 in sequences
                        for seq2 in sequences
                    ]
                ]
            ).reshape(len(sequences), len(sequences))
            data_loader.save_results(
                similarity_matrix, f"{args.output}_similarity.npy", format="numpy"
            )
        data_loader.save_results(
            analyses, f"{args.output}_analysis.json", format="json"
        )

    else:
        logger.error(f"Unknown mode: {args.mode}")


# -------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    # Check if we're launching in UI mode
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        ui = HDCGenomicUI()
        demo = ui.build_interface()
        demo.launch(share="--share" in sys.argv)
    else: # Use CLI mode
        args = parse_args()
        run_genetic_analysis(args)
