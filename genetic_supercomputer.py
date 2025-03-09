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
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
import math
import psutil
from dataclasses import dataclass, field
import json
from Bio import SeqIO
import gradio as gr
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.base import BaseEstimator
import requests
from pybedtools import BedTool
import pysam
import pandas as pd
import random
import functools
import sqlite3
from io import BytesIO
"""
The Genetic Supercomputer is a novel system for analyzing DNA and RNA sequences using Hyperdimensional Computing (HDC). Instead of traditional methods like alignment, it transforms sequences into high-dimensional vectors, where patterns and relationships are represented as geometric relationships. This allows for very fast comparisons and analysis, even with very large datasets.

The core idea is to represent k-mers (short DNA subsequences) as random, high-dimensional vectors. Longer sequences are then encoded by combining the vectors of their constituent k-mers. This creates a "vector space" where similar sequences have similar vectors. The system can also integrate biological information (like known motifs, conservation scores from multiple sequence alignments, and epigenetic data) directly into these vector representations, making them more biologically meaningful.

The system uses a "swarm intelligence" approach for advanced analysis. A group of "agents" (each a simple reinforcement learning program) explores this high-dimensional vector space. Each agent has a "position" in the space, represented by an HDC vector. Agents can:
- Specialize: Different agents can focus on different aspects of sequence analysis. Some might be good at finding motifs, others at identifying conserved regions, and others at predicting the impact of mutations.
- Move: Agents move through the HDC space, guided by their own experience and by the "collective intelligence" of the swarm. They are attracted to regions of the space that represent interesting or important sequence features.
- Learn: Agents use reinforcement learning to improve their ability to predict the next base in a sequence, identify motifs, or perform other tasks.
- Collaborate: Agents indirectly communicate by sharing information through their movements and by influencing a "global attractor" that represents the overall consensus of the swarm.

This swarm approach makes the system robust and adaptable. It can discover complex patterns in genomic data that might be missed by traditional algorithms.

Potential Uses:
- Finding new regulatory elements (motifs) in DNA.
- Predicting the functional impact of mutations.
- Comparing genomes from different species to find conserved regions.
- Filling in gaps in incomplete sequence data (imputation).
- Identifying genes and other functional elements in newly sequenced genomes.
- Discovering potential drug targets.
- Designing synthetic DNA sequences with desired properties.
- Analyzing complex mixtures of DNA from different organisms (metagenomics).
- Analyzing Epigenetic Data
- Comparing Sequences

How to Use the UI:

The system provides a user-friendly web interface (built with Gradio) that lets you perform these analyses without needing to write code. Here's a summary of the main tabs:

Encode & Analyze:
- Paste in a DNA sequence (or upload a FASTA/FASTQ file).
- The system encodes the sequence into an HDC vector.
- It displays basic statistics (length, GC content, etc.).
- If you provide multiple sequences, it shows a similarity matrix (heatmap) visualizing how similar they are.
- For single, it may also display motif information.

Impute Sequence:
- Provide a sequence with a missing segment (represented by a "gap"). You give the sequence before the gap ("prefix") and the sequence after the gap ("suffix").
- Specify the length of the gap.
- The system uses the swarm to predict the most likely sequence to fill the gap.

Motif Discovery:
- Paste in a DNA sequence.
- Specify the length of motifs to search for.
- The system uses the HDC-based motif discovery to find potential motifs and shows their locations.

Variant Analysis:
- Paste in an original DNA sequence and a modified sequence with a mutation (variant).
- The system calculates an "impact score" showing how different the variant is from the original.

Comparative Genomics:
- Input multiple sequences (one per line)
- Compares the HDC vectors of sequences
- Displays a similarity matrix and heatmap
- Epigenetic Analysis:
- Paste in a DNA sequence.
- Upload a file containing epigenetic data (e.g., in BED or bigWig format).
- The system analyzes the relationship between the sequence and the epigenetic marks.

Configuration (Advanced):
- Allows expert users to adjust various parameters of the system (e.g., vector dimension, learning rates, feature weights).

About:
- Provides the details about HDC and the project.

To launch the UI, run the script with the --ui flag: python genetic_supercomputer.py --ui

In essence, this project combines the power of HDC for efficient representation, swarm intelligence for robust search, and biological knowledge integration for meaningful analysis, all within a user-friendly framework. It represents a novel approach to tackling complex problems in genomics.

Key Features:

- HDC Encoding: Transforms DNA/RNA sequences into high-dimensional vectors
  using PyTorch, capturing k-mer patterns.  Includes both a basic k-mer
  encoder (`DNAEncoder`) and a `BiologicalEncoder` that incorporates
  features like GC content, motifs, conservation scores, and epigenetic data.
- Swarm Intelligence: Employs a swarm of reinforcement learning agents
  (`GeneticSwarm`) that collaboratively explore the HDC space for tasks like
  sequence prediction and imputation. Agents can specialize in different
  sequence analysis aspects.
- Scalability: Designed for large datasets and high-dimensional vectors,
  leveraging PyTorch for efficient CPU and GPU computation.
- Modular Design: Organized into well-defined classes (e.g.,
  `HDCVectorSpace`, `BiologicalEncoder`, `GeneticSwarm`, `HDCMotifDiscovery`,
  `GenomicDataLoader`, `DatabaseSchema`).
- Database Integration: Uses an SQLite database (`DatabaseSchema`) to store
  sequences and HDC vectors, facilitating large-scale data management.
- Command-Line Interface (CLI): A unified CLI for encoding, training,
  prediction, imputation, and analysis.
- Interactive Web Interface (UI): A Gradio-based UI (`HDCGenomicUI`) provides
  interactive sequence analysis, visualization, and configuration.
- Data-Driven Adaptation: The system automatically adjusts parameters like
  k-mer size and agent distribution based on sequence characteristics.
- Extensibility: Designed for easy extension with new biological features,
  agent types, and analysis methods.
- File Format Support: Handles FASTA, FASTQ, BED, GFF, VCF, bigWig, and
  other common genomic file formats via `GenomicDataLoader`.

How to Use (CLI - Examples):

1. Encode sequences:
   python genetic_supercomputer.py encode --input sequences.fasta --output vectors.h5

2. Train a swarm model:
   python genetic_supercomputer.py train --input training.fasta --output model.pt --epochs 10 --agent-type swarm

3. Predict sequence continuation:
   python genetic_supercomputer.py predict --model model.pt --context "ATGC" --length 20

4. Impute a missing sequence segment:
   python genetic_supercomputer.py impute --model model.pt --prefix "ATGC" --suffix "TGCA" --gap-length 10

5. Analyze sequence properties:
   python genetic_supercomputer.py analyze --input sequences.fasta

6. Launch the interactive UI:
    python genetic_supercomputer.py --ui

How to Use (API - Example):

from genetic_supercomputer import DNASupercomputer, GeneticSwarm, GSCConfig

# Create a configuration:
config = GSCConfig(dimension=4096, device="cpu")  # or "gpu"

# Create a supercomputer instance:
sc = DNASupercomputer(config)

# Encode a sequence:
sequence = "ATGCGTAGCTAGCTAGCTAGCTA"
vector = sc.encode_sequence(sequence)  # Returns a PyTorch tensor

# Create and initialize a swarm (optional):
swarm = GeneticSwarm(sc)
sequences = ["ATGCGTAG", "CTAGCTAG"] # Example sequences
swarm.initialize_swarm(sequences)

# Train the swarm (optional):
swarm.train(sequences, epochs=5)

# ... perform analysis, prediction, etc., using the swarm or supercomputer ...

Project Structure:

- genetic_supercomputer.py: Contains the core classes and functions, including:
    - HDCVectorSpace:  Manages HDC vectors (creation, operations, normalization).
    - DNASupercomputer:  Provides DNA-specific functionality (encoding).
    - BiologicalEncoder:  Encodes sequences with integrated biological features.
    - GeneticAgent:  Reinforcement learning agent for prediction.
    - GeneticSwarm:  Swarm intelligence system for collaborative analysis.
    - MetaHDConservation:  Learns and predicts conservation patterns.
    - HDCMotifDiscovery:  System for discovering and analyzing sequence motifs.
    - GenomicDataLoader: Handles loading and saving data in various formats.
    - DatabaseSchema:  Manages the SQLite database for storing data.
    - GSCConfig:  Unified configuration for the system.
    - AgentType, AgentConfig, AgentState:  Classes for agent management.
    - Utils: Utility functions for sequence analysis.
    - and supporting classes and functions.

- ui.py: (If you have a separate ui.py) Contains the Gradio UI code.
- requirements.txt : lists the required pip packages

Installation:

1. Clone the Repository:
    git clone https://github.com/jeffmeloy/GSC
    cd GSC

2. Create a Virtual Environment (Recommended):
   python3 -m venv venv
   source venv/bin/activate

3. Install Dependencies:
    pip install -r requirements.txt

    Note for GPU usage: If you have a CUDA-enabled GPU and want to use it
    with PyTorch, install the appropriate PyTorch version with CUDA support.
    Refer to the PyTorch documentation for installation instructions.
    (Typically, this involves `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu11x` , replacing `cu11x` as necessary).

TODO:
1. Update code to only use biploar encoding for the HDC vectors 
2. Data Sources and Feature Providers 
    
    Prioritize Plant-Specific Databases:
    - Ensembl Plants: This should be your primary data source. It provides comprehensive genome assemblies, gene annotations, variation data, and comparative genomics information for a wide range of plant species. Use the GenomicDataLoader to integrate data from Ensembl Plants.
    - NCBI RefSeq (Plants): A good source for curated reference genomes.
    - TAIR (The Arabidopsis Information Resource): Essential if you're working with Arabidopsis thaliana, the model plant organism.
    - Phytozome (JGI): A large collection of plant genomes, with a focus on comparative genomics.
    - Gramene: A curated resource for comparative genomics in grasses and related species.
    - PlantTFDB: A database of plant transcription factors. This is crucial for your FeatureProvider for TFBS (transcription factor binding sites).
    - Support Specific Crop Databases (e.g., rice, maize, wheat, soybean): there are often specialized databases for those species (e.g., RiceXPro, MaizeGDB, SoyBase).

    Feature Provider Focus:
    - Transcription Factor Binding Sites (TFBS): This becomes even more critical in plants, as gene regulation plays a huge role in development and response to the environment. Use PlantTFDB, JASPAR Plants, and potentially motif data derived from ChIP-seq experiments (from ENCODE or publications).
    - Promoter Regions: Focus on identifying and characterizing promoter elements (TATA box, CAAT box, etc.).
    - Introns/Exons: Accurate gene structure annotation is very important. Use data from Ensembl Plants and other sources.
    - Repeated Elements: Plant genomes often have a high proportion of repetitive DNA. Include features related to repeat density and type (e.g., transposable elements).
    - Conservation: Use comparative genomics data (e.g., from Ensembl Plants or Phytozome) to calculate conservation scores. Focus on plant-specific alignments.
    - Epigenetic Marks: If you're integrating epigenetic data, focus on marks relevant to plants (e.g., DNA methylation, histone modifications). Look for data from publications and specialized databases.
    - Regulatory RNAs: Consider features related to non-coding RNAs (ncRNAs), which play important roles in plant gene regulation.
    - SNPs/Variants: Plant breeding and crop improvement rely heavily on identifying and utilizing genetic variation. Integrate SNP data from relevant databases.

2. HDCMotifDiscovery:
    COMMON_MOTIFS: Update COMMON_MOTIFS dictionary to include more plant-specific motifs. Research common promoter elements and transcription factor binding sites in plants.
    Motif Databases: Make sure your GenomicDataLoader can load and parse motif data from plant-specific sources (JASPAR Plants, PlantTFDB, etc.).

3. GeneticAgent and GeneticSwarm:
    Agent Specializations: You can tailor your agent specializations to plant-specific tasks:
    - MOTIF_DETECTOR: Focus on plant transcription factor binding sites.
    - STRUCTURE_ANALYZER: Focus on promoter regions, introns/exons, and repeat elements.
    - CONSERVATION_TRACKER: Train on plant-specific alignments.
    - VARIANT_ANALYZER: (New specialization) Focus on identifying the impact of SNPs and other variants on gene function.
    - REPEAT_DETECTOR (New)

4. BiologicalEncoder:
    Ensure load_motif_data, load_epigenetic_data, load_annotations, and load_conservation_data methods are working correctly with plant-specific data sources.

5. GSCConfig:
* Keep the ability to customize the alphabet. While "ACGT" is standard, researchers might want to include other bases for specific analyses.

6. Example Data (for Testing):
    Use well-annotated plant genome: Arabidopsis thaliana is the classic choice. 
    Download data in standard formats:
    - FASTA for the genome sequence.
    - GFF3 for gene annotations.
    - BED/bigWig for conservation scores (if available).
    - BED/narrowPeak/broadPeak for epigenetic marks (if available).
    - Motif data from JASPAR Plants or PlantTFDB.

7. Testing (Plant-Specific):
    - Motif Discovery: Test whether your HDCMotifDiscovery system can rediscover known plant motifs.
    - Gene Structure Prediction: Test whether your system can accurately identify gene features (promoters, exons, introns).
    - Variant Analysis: Test whether your system can predict the impact of known variants (e.g., SNPs associated with particular traits).
    - Conservation: Test whether your MetaHDConservation class can learn to identify conserved regions in plant genomes.
"""

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


# Constants
DEFAULT_ALPHABET = ["A", "T", "G", "C"]
DEFAULT_BUNDLE_DECAY = 0.9
DEFAULT_MAX_KMER_SIZE = 30
HAS_GPU = cp.is_available() if cp else False



class DatabaseSchema:
    """
    Encapsulates the SQLite database schema for the Genetic Supercomputer.
    Provides methods for creating and managing the database tables.
    """

    def __init__(self, db_path: str = "genetic_data.db"):
        """
        Initializes the DatabaseSchema.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._register_adapters()  # Register at initialization

    def _register_adapters(self):
        """Registers adapters and converters for custom data types."""
        sqlite3.register_adapter(np.ndarray, self._adapt_array)
        sqlite3.register_converter("array", self._convert_array)


    def connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(
                self.db_path, detect_types=sqlite3.PARSE_DECLTYPES
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("Database connection closed.")

    def _adapt_array(self, arr: np.ndarray) -> sqlite3.Binary:
        """Converts a NumPy array to a SQLite BLOB."""
        out = BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def _convert_array(self, text: bytes) -> np.ndarray:
        """Converts a SQLite BLOB back to a NumPy array."""
        out = BytesIO(text)
        out.seek(0)
        return np.load(out)


    def create_tables(self):
        """Creates the database tables if they don't exist."""
        try:
            # Sequences Table
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sequences (
                    sequence_id TEXT PRIMARY KEY,
                    sequence TEXT NOT NULL,
                    length INTEGER NOT NULL,
                    gc_content REAL,
                    metadata TEXT
                )
                """
            )

            # HDC Vectors Table
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hdc_vectors (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id TEXT,
                    encoding_method TEXT,
                    kmer_size INTEGER,
                    dimension INTEGER,
                    vector array NOT NULL,
                    FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
                )
                """
            )
            self.conn.commit()
            logger.info("Tables created successfully.")

        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()  # Rollback changes if error occurs
            raise

    def drop_tables(self):
        """Drops all tables (USE WITH CAUTION)."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS hdc_vectors")
            self.cursor.execute("DROP TABLE IF EXISTS sequences")
            self.conn.commit()
            logger.info("Tables dropped successfully.")

        except sqlite3.Error as e:
            logger.error(f"Error dropping tables: {e}")
            self.conn.rollback()
            raise

    def insert_sequence(self, sequence_id: str, sequence: str, gc_content: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Inserts a new sequence into the 'sequences' table.

        Args:
            sequence_id: Unique ID for the sequence.
            sequence: The DNA sequence string.
            gc_content: Pre-calculated GC content (optional).
            metadata:  Dictionary of metadata (optional)
        """

        if gc_content is None:
                gc_content = Utils.calculate_gc_content(sequence)
        if metadata is not None:
            metadata_json = json.dumps(metadata) # Convert dict to JSON string
        else:
            metadata_json = None

        try:
            self.cursor.execute("""
                INSERT INTO sequences (sequence_id, sequence, length, gc_content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """, (sequence_id, sequence, len(sequence), gc_content, metadata_json))
            self.conn.commit()
            # logger.debug(f"Sequence {sequence_id} inserted successfully.") #Use debug
        except sqlite3.IntegrityError as e:
            logger.warning(f"Sequence {sequence_id} already exists: {e}")
            self.conn.rollback()
        except sqlite3.Error as e:
            logger.error(f"Error inserting sequence {sequence_id}: {e}")
            self.conn.rollback()
            raise


    def insert_hdc_vector(self, sequence_id: str, encoding_method: str,
                          kmer_size: int, dimension: int, hdc_vector: np.ndarray) -> None:

        """
        Inserts hdc vector
        """
        try:
            self.cursor.execute("""
            INSERT INTO hdc_vectors (sequence_id, encoding_method, kmer_size, dimension, vector)
            VALUES (?, ?, ?, ?, ?)
            """, (sequence_id, encoding_method, kmer_size, dimension, hdc_vector))
            self.conn.commit()
            # logger.debug(f"HDC vector inserted successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error inserting HDC Vector: {e}")
            self.conn.rollback()
            raise


    def get_sequence(self, sequence_id: str) -> Optional[str]:
        """Retrieves a sequence by its ID."""
        self.cursor.execute("SELECT sequence FROM sequences WHERE sequence_id = ?", (sequence_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        return None


    def get_hdc_vector(self, sequence_id: str, encoding_method: str) -> Optional[np.ndarray]:
        """
        Retrieves an HDC vector by sequence ID and encoding method.

        Args:
            sequence_id: ID of the sequence.
            encoding_method: Method used to encode the sequence

        Returns:
            The HDC vector as a NumPy array, or None if not found.
        """
        self.cursor.execute("""
            SELECT vector
            FROM hdc_vectors
            WHERE sequence_id = ? AND encoding_method = ?
        """, (sequence_id, encoding_method))
        result = self.cursor.fetchone()
        if result:
            return result[0]  # The vector is the first (and only) element in the tuple
        return None

    def get_all_sequence_ids(self) -> List[str]:
        """Retrieves all sequence IDs from the database."""
        self.cursor.execute("SELECT sequence_id FROM sequences")
        return [row[0] for row in self.cursor.fetchall()]

    def get_all_sequences(self) -> List[Tuple[str, str]]:
        "Retrieve all sequences and their ids"
        self.cursor.execute("SELECT sequence_id, sequence FROM sequences")
        return self.cursor.fetchall()

    def clear_database(self):
        """Remove all entries from the database, but keep the schema"""
        try:
            self.cursor.execute("DELETE FROM hdc_vectors")
            self.cursor.execute("DELETE FROM sequences")
            self.conn.commit()
            logger.info("All data cleared from the database.")
        except sqlite3.Error as e:
            logger.error(f"Error clearing database: {e}")
            self.conn.rollback()
            raise

    def sequence_exists(self, sequence_id: str) -> bool:
        """
        Checks if a sequence with the given ID already exists in the database.

        Args:
            sequence_id: The sequence ID to check.

        Returns:
            True if the sequence exists, False otherwise.
        """
        self.cursor.execute("SELECT 1 FROM sequences WHERE sequence_id = ?", (sequence_id,))
        return self.cursor.fetchone() is not None

    def vector_exists(self, sequence_id: str, encoding_method:str) -> bool:
        """Checks if an HDC vector exists for the sequence and encoding."""
        self.cursor.execute("""
            SELECT 1
            FROM hdc_vectors
            WHERE sequence_id = ? AND encoding_method = ?
            """, (sequence_id, encoding_method))
        return self.cursor.fetchone() is not None


    def get_metadata(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a sequence.

        Args:
            sequence_id: The ID of the sequence.

        Returns:
            The metadata as a dictionary, or None if no metadata is found
            or if the sequence doesn't exist.
        """

        self.cursor.execute("SELECT metadata FROM sequences WHERE sequence_id = ?", (sequence_id,))
        result = self.cursor.fetchone()
        if result:
            metadata_json = result[0]
            if metadata_json:  # Check if metadata is not NULL
                return json.loads(metadata_json) #Convert JSON string back to dict
            else:
                return None #No metadata
        return None


@dataclass
class GSCConfig:
    """Unified configuration for Genetic Supercomputer with automated tuning.
    Parameters are automatically tuned based on data characteristics and hardware.
    """

    # Core parameters (maintaining compatibility with existing code)
    dimension: Optional[int] = None
    device: str = "auto"  # 'gpu', 'cpu', or 'auto'
    vector_type: str = "bipolar"
    seed: Optional[int] = None

    # Data-specific parameters
    alphabet: List[str] = field(default_factory=lambda: DEFAULT_ALPHABET)
    data_size: Optional[int] = None
    avg_sequence_length: Optional[int] = None

    # Hyperparameters
    max_kmer_size: int = DEFAULT_MAX_KMER_SIZE
    bundle_decay: float = DEFAULT_BUNDLE_DECAY
    cache_size: Optional[int] = None
    chunk_size: Optional[int] = None
    position_vectors: Optional[int] = None
    accuracy_target: float = 0.95

    # Task-Specific Parameters (from HDCGenomicUIConfig)
    motif_detection_enabled: bool = True  # Enable motif discovery
    variant_analysis_enabled: bool = True  # Enable variant impact analysis
    sequence_imputation_enabled: bool = True  # Enable sequence imputation
    epigenetic_integration_enabled: bool = True  # Enable epigenetic data integration

    motif_detection_threshold: float = 0.8  # Motif discovery threshold
    motif_window_size: int = 10  # Window size for motif discovery
    variant_analysis_threshold: float = (
        0.75  # Threshold for variant analysis (higher means stricter)
    )
    epigenetic_integration_mode: str = (
        "combined"  # Epigenetic data integration mode: "combined", "separate", etc.
    )

    def __post_init__(self):
        """Initialize derived parameters and perform automated tuning."""
        # Detect hardware capabilities
        self.sys_memory = psutil.virtual_memory().total
        self.gpu_memory = self._detect_gpu_memory()
        self.cpu_cores = os.cpu_count() or 4

        # Set device
        if self.device == "auto":
            self.device = "gpu" if HAS_GPU else "cpu"

        # Initialize parameters using rule-based heuristics
        if self.dimension is None:
            self.dimension = self._derive_optimal_dimension()
        if self.cache_size is None:
            self._set_cache_size()
        if self.chunk_size is None:
            self._set_chunk_size()
        if self.position_vectors is None:
            self.position_vectors = min(self.max_kmer_size, self.dimension // 10)

        # Refine parameters using automated tuning
        self._auto_tune()

        # Synchronize task-specific parameters from UI config
        self._sync_with_ui_config()

    def _sync_with_ui_config(self):
        """
        Synchronize task-specific parameters from the UI configuration.
        This ensures that task-specific flags properly affect computational behavior.
        """
        # Sync feature enablement with computational parameters
        if not hasattr(self, "motif_detection_enabled"):
            # Default values for backward compatibility
            self.motif_detection_enabled = True
            self.variant_analysis_enabled = True
            self.sequence_imputation_enabled = True
            self.epigenetic_integration_enabled = True

            self.motif_detection_threshold = 0.8
            self.motif_window_size = 10
            self.variant_analysis_threshold = 0.75
            self.epigenetic_integration_mode = "combined"

        # Apply computational adjustments based on feature flags
        if not self.motif_detection_enabled:
            self.motif_detection_threshold = (
                0.0  # Effectively disable by setting threshold to zero
            )

        if not self.variant_analysis_enabled:
            self.variant_analysis_threshold = 0.0  # Disable variant analysis

        if not self.epigenetic_integration_enabled:
            self.epigenetic_integration_mode = "none"  # Disable epigenetic integration

        # Adjust computational resources based on enabled features
        self._optimize_resources_for_enabled_features()

    def _optimize_resources_for_enabled_features(self):
        """
        Optimize computational resource allocation based on which features are enabled.
        This reduces memory and CPU usage when certain features are disabled.
        """
        # Count enabled features
        enabled_features = sum(
            [
                self.motif_detection_enabled,
                self.variant_analysis_enabled,
                self.sequence_imputation_enabled,
                self.epigenetic_integration_enabled,
            ]
        )

        # Adjust cache size based on enabled features
        if self.cache_size is not None:
            if enabled_features < 4:
                # Reduce cache size proportionally if some features are disabled
                self.cache_size = max(
                    10000, int(self.cache_size * (0.7 + 0.1 * enabled_features))
                )

        # Adjust chunk size for more efficient processing based on enabled features
        if self.chunk_size is not None and enabled_features < 3:
            # Increase chunk size when fewer features are enabled
            self.chunk_size = min(
                20000, int(self.chunk_size * (1.5 - 0.1 * enabled_features))
            )

    def update_from_ui_config(self, ui_config):
        """
        Update the GSC configuration from an HDCGenomicUIConfig instance.
        This allows for runtime reconfiguration without reinitialization.

        Args:
            ui_config: An HDCGenomicUIConfig instance with updated parameters

        Returns:
            True if core parameters changed requiring reinitialization, False otherwise
        """
        # Track if core parameters change
        core_params_changed = False

        # Core parameters that require reinitialization if changed
        if self.dimension != ui_config.dimension:
            self.dimension = ui_config.dimension
            core_params_changed = True

        if self.device != ui_config.device:
            self.device = ui_config.device
            core_params_changed = True

        if self.vector_type != ui_config.vector_type:
            self.vector_type = ui_config.vector_type
            core_params_changed = True

        # Update other parameters that don't require reinitialization
        self.max_kmer_size = ui_config.max_kmer_size
        self.bundle_decay = ui_config.bundle_decay

        # Update task-specific parameters
        self.motif_detection_enabled = ui_config.motif_detection_enabled
        self.variant_analysis_enabled = ui_config.variant_analysis_enabled
        self.sequence_imputation_enabled = ui_config.sequence_imputation_enabled
        self.epigenetic_integration_enabled = ui_config.epigenetic_integration_enabled
        self.motif_detection_threshold = ui_config.motif_detection_threshold
        self.motif_window_size = ui_config.motif_window_size
        self.variant_analysis_threshold = ui_config.variant_analysis_threshold
        self.epigenetic_integration_mode = ui_config.epigenetic_integration_mode

        # Resynchronize and optimize
        self._sync_with_ui_config()

        return core_params_changed

    def _detect_gpu_memory(self) -> Optional[int]:
        """Detect available GPU memory."""
        if not HAS_GPU:
            return None
        try:
            return cp.cuda.Device().mem_info[0]
        except Exception:
            return None

    def _derive_optimal_dimension(self) -> int:
        """Calculate optimal dimension using Johnson-Lindenstrauss lemma or defaults."""
        if self.data_size and self.data_size > 1:
            # Johnson-Lindenstrauss lemma
            jl_dim = int(8 * math.log(self.data_size) / (1 - self.accuracy_target) ** 2)
            # Adjust for alphabet (empirical factor)
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
        base_cache_size = int(mem_gb * 50000)
        dimension_factor = max(1, self.dimension // 1000)
        self.cache_size = max(100000, base_cache_size // dimension_factor)

    def _set_chunk_size(self):
        """Set chunk size based on memory and vector dimension."""
        mem_available = min(
            self.sys_memory * 0.2,
            self.gpu_memory * 0.5 if self.gpu_memory else float("inf"),
        )
        bytes_per_vector = self.dimension * (4 if self.vector_type == "bipolar" else 1)
        self.chunk_size = min(10000, max(100, int(mem_available / bytes_per_vector)))

    def _auto_tune(self):
        """Refine hyperparameters using Bayesian optimization."""
        # Define search space
        search_space = {
            "dimension": Integer(1024, 16384, name="dimension"),
            "bundle_decay": Real(0.5, 0.95, name="bundle_decay"),
            "cache_size": Integer(10000, 1000000, name="cache_size"),
            "chunk_size": Integer(100, 10000, name="chunk_size"),
        }

        # Define objective function
        def objective(params):
            self.dimension = params["dimension"]
            self.bundle_decay = params["bundle_decay"]
            self.cache_size = params["cache_size"]
            self.chunk_size = params["chunk_size"]
            return self._evaluate_performance()

        # Perform Bayesian optimization
        opt = BayesSearchCV(
            estimator=BaseEstimator(),
            search_spaces=search_space,
            n_iter=10,  # Number of iterations
            cv=3,  # Cross-validation folds
            random_state=self.seed,
        )
        opt.fit(None, None)  # Dummy fit
        best_params = opt.best_params_

        # Update configuration with best parameters
        self.dimension = best_params["dimension"]
        self.bundle_decay = best_params["bundle_decay"]
        self.cache_size = best_params["cache_size"]
        self.chunk_size = best_params["chunk_size"]

    def _evaluate_performance(self):
        """Evaluate HDC performance (e.g., accuracy, memory usage, runtime)."""
        # Placeholder for actual evaluation logic
        # For now, return a dummy score based on dimension and bundle_decay
        return -abs(self.dimension - 4096) / 4096 - abs(self.bundle_decay - 0.85) / 0.85

    def _sync_with_ui_config(self):
        """Synchronize task-specific parameters from the UI configuration."""
        # Sync with task-specific parameters
        if self.motif_detection_enabled:
            self.motif_detection_threshold = 0.8  # Set default value if enabled
        if not self.variant_analysis_enabled:
            self.variant_analysis_threshold = 0.0  # Disable variant analysis
        if not self.epigenetic_integration_enabled:
            self.epigenetic_integration_mode = "none"  # Disable epigenetic integration

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GSCConfig":
        """Create a GSCConfig instance from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"GSCConfig(dim={self.dimension}, device={self.device}, "
            f"cache={self.cache_size // 1000}K, chunk={self.chunk_size})"
        )


class GenomicDataLoader:
    """Enhanced class for loading and managing genomic data from files and databases.

    Usage Example:
        # Initialize the data loader
        data_loader = GenomicDataLoader()

        # Load sequences from a FASTA file
        sequences, sequence_ids = data_loader.load_sequences("example.fasta")

        # Load regions from a BED file
        regions, region_ids = data_loader.load_sequences("example.bed", file_type="bed")

        # Fetch data from ENCODE
        encode_data = data_loader.fetch_encode_data("ENCFF123ABC")

        # Load metadata
        metadata = data_loader.load_metadata("metadata.json")
    """

    def __init__(self):
        """Initialize the GenomicDataLoader with logging."""
        self.logger = logging.getLogger("GenomicDataLoader")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def load_sequences(
        self, input_path: str, file_type: str = "auto"
    ) -> Tuple[List[str], List[str]]:
        """
        Load sequences or genomic regions from a file or database.

        Args:
            input_path: Path to the input file or database accession number.
            file_type: Type of file or database (e.g., "fasta", "bed", "encode").

        Returns:
            A tuple containing:
                - List of sequences or genomic regions.
                - List of corresponding IDs or descriptions.
        """
        if file_type == "auto":
            file_type = self._detect_file_type(input_path)

        if file_type in ["fasta", "fa", "fna", "fastq", "fq"]:
            return self._load_fasta_fastq(input_path, file_type)
        elif file_type == "bed":
            return self._load_bed(input_path)
        elif file_type == "gff":
            return self._load_gff(input_path)
        elif file_type == "vcf":
            return self._load_vcf(input_path)
        elif file_type == "encode":
            return self._load_encode_data(input_path)
        elif file_type == "ncbi":
            return self._load_ncbi_data(input_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _detect_file_type(self, input_path: str) -> str:
        """
        Detect the file type based on the file extension.

        Args:
            input_path: Path to the input file.

        Returns:
            Detected file type (e.g., "fasta", "bed").
        """
        ext = input_path[input_path.rfind(".") :].lower()
        format_map = {
            ".fasta": "fasta",
            ".fa": "fasta",
            ".fna": "fasta",
            ".fastq": "fastq",
            ".fq": "fastq",
            ".bed": "bed",
            ".gff": "gff",
            ".vcf": "vcf",
        }
        return format_map.get(ext, "fasta")  # Default to FASTA

    def _load_fasta_fastq(
        self, input_path: str, file_type: str
    ) -> Tuple[List[str], List[str]]:
        """
        Load sequences from a FASTA/FASTQ file.

        Args:
            input_path: Path to the FASTA/FASTQ file.
            file_type: Type of file ("fasta" or "fastq").

        Returns:
            A tuple containing:
                - List of sequences.
                - List of sequence IDs.
        """
        try:
            sequences = [
                str(record.seq).upper() for record in SeqIO.parse(input_path, file_type)
            ]
            sequence_ids = [record.id for record in SeqIO.parse(input_path, file_type)]
            self.logger.info(f"Loaded {len(sequences)} sequences from {input_path}")
            return sequences, sequence_ids
        except Exception as e:
            self.logger.error(f"Error loading FASTA/FASTQ file: {e}")
            raise

    def _load_bed(self, input_path: str) -> Tuple[List[str], List[str]]:
        """
        Load regions from a BED file.

        Args:
            input_path: Path to the BED file.

        Returns:
            A tuple containing:
                - List of regions (chromosome, start, end).
                - List of region IDs.
        """
        try:
            bed = BedTool(input_path)
            regions = [
                (interval.chrom, interval.start, interval.end) for interval in bed
            ]
            region_ids = [f"{chrom}:{start}-{end}" for chrom, start, end in regions]
            self.logger.info(f"Loaded {len(regions)} regions from {input_path}")
            return regions, region_ids
        except Exception as e:
            self.logger.error(f"Error loading BED file: {e}")
            raise

    def _load_gff(self, input_path: str) -> Tuple[List[str], List[str]]:
        """
        Load annotations from a GFF file.

        Args:
            input_path: Path to the GFF file.

        Returns:
            A tuple containing:
                - List of annotations (chromosome, start, end, name).
                - List of annotation IDs.
        """
        try:
            gff = BedTool(input_path)
            annotations = [
                (interval.chrom, interval.start, interval.end, interval.name)
                for interval in gff
            ]
            annotation_ids = [
                f"{chrom}:{start}-{end}" for chrom, start, end, _ in annotations
            ]
            self.logger.info(f"Loaded {len(annotations)} annotations from {input_path}")
            return annotations, annotation_ids
        except Exception as e:
            self.logger.error(f"Error loading GFF file: {e}")
            raise

    def _load_vcf(self, input_path: str) -> Tuple[List[str], List[str]]:
        """
        Load variants from a VCF file.

        Args:
            input_path: Path to the VCF file.

        Returns:
            A tuple containing:
                - List of variants.
                - List of variant IDs.
        """
        try:
            vcf = pysam.VariantFile(input_path)
            variants = [str(record) for record in vcf]
            variant_ids = [record.id for record in vcf]
            self.logger.info(f"Loaded {len(variants)} variants from {input_path}")
            return variants, variant_ids
        except Exception as e:
            self.logger.error(f"Error loading VCF file: {e}")
            raise

    def _load_encode_data(self, accession: str) -> Tuple[List[str], List[str]]:
        """
        Fetch and load data from the ENCODE database.

        Args:
            accession: ENCODE accession number.

        Returns:
            A tuple containing:
                - List of sequences or regions.
                - List of corresponding IDs.
        """
        try:
            data = self.fetch_encode_data(accession)
            sequences = data.get("sequences", [])
            sequence_ids = data.get("ids", [])
            self.logger.info(
                f"Loaded {len(sequences)} sequences from ENCODE accession {accession}"
            )
            return sequences, sequence_ids
        except Exception as e:
            self.logger.error(f"Error loading ENCODE data: {e}")
            raise

    def _load_ncbi_data(self, accession: str) -> Tuple[List[str], List[str]]:
        """
        Fetch and load data from the NCBI database.

        Args:
            accession: NCBI accession number.

        Returns:
            A tuple containing:
                - List of sequences.
                - List of sequence IDs.
        """
        try:
            data = self.fetch_ncbi_data(accession)
            sequences = data.get("sequences", [])
            sequence_ids = data.get("ids", [])
            self.logger.info(
                f"Loaded {len(sequences)} sequences from NCBI accession {accession}"
            )
            return sequences, sequence_ids
        except Exception as e:
            self.logger.error(f"Error loading NCBI data: {e}")
            raise

    def fetch_encode_data(
        self, accession: str, file_type: str = "bed"
    ) -> Dict[str, Any]:
        """
        Fetch data from the ENCODE database.

        Args:
            accession: ENCODE accession number.
            file_type: Type of file to fetch (e.g., "bed", "fasta").

        Returns:
            A dictionary containing the fetched data.
        """
        base_url = "https://www.encodeproject.org"
        url = f"{base_url}/search/?type=File&accession={accession}&file_format={file_type}&format=json"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data for accession {accession}")

        data = response.json()
        return data["@graph"][0]  # Return the first result

    def fetch_ncbi_data(
        self, accession: str, file_type: str = "fasta"
    ) -> Dict[str, Any]:
        """
        Fetch data from the NCBI database.

        Args:
            accession: NCBI accession number.
            file_type: Type of file to fetch (e.g., "fasta", "gff").

        Returns:
            A dictionary containing the fetched data.
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "nucleotide",
            "id": accession,
            "rettype": file_type,
            "retmode": "text",
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data for accession {accession}")

        return response.text

    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from a JSON or CSV file.

        Args:
            metadata_path: Path to the metadata file.

        Returns:
            A dictionary containing the metadata.
        """
        try:
            if metadata_path.endswith(".json"):
                with open(metadata_path, "r") as f:
                    return json.load(f)
            elif metadata_path.endswith(".csv"):
                return pd.read_csv(metadata_path).to_dict(orient="records")
            else:
                raise ValueError(f"Unsupported metadata format: {metadata_path}")
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            raise

    def save_results(self, results: Any, output_path: str, format: str = "auto"):
        """
        Save analysis results to a file.

        Args:
            results: The data to save (can be a dictionary, list, NumPy array, etc.).
            output_path: Path to the output file.
            format: Output file format ('auto', 'json', 'csv', 'hdf5', 'npy', 'txt').
        """
        try:
            if format == "auto":
                format = self._detect_output_format(output_path)

            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            elif format == "csv":
                if isinstance(results, list) and all(
                    isinstance(item, dict) for item in results
                ):
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
                elif isinstance(results, np.ndarray):
                    np.savetxt(output_path, results, delimiter=",")
                else:
                    raise ValueError(
                        "CSV format only supports lists of dictionaries or NumPy arrays."
                    )
            elif format == "hdf5":
                with h5py.File(output_path, "w") as f:
                    self._save_hdf5_recursively(f, results)
            elif format == "npy":
                np.save(output_path, results)
            elif format == "txt":  # Simple text format for lists or strings
                with open(output_path, "w") as f:
                    if isinstance(results, list):
                        for item in results:
                            f.write(str(item) + "\n")
                    else:
                        f.write(str(results))
            else:
                raise ValueError(f"Unsupported output format: {format}")

            self.logger.info(f"Saved results to {output_path} in {format} format")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def _detect_output_format(self, output_path: str) -> str:
        """Detect output file format based on extension."""
        ext = output_path[output_path.rfind(".") :].lower()
        format_map = {
            ".json": "json",
            ".csv": "csv",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".npy": "npy",
            ".txt": "txt",
        }
        return format_map.get(ext, "txt")  # Default to text

    def _save_hdf5_recursively(
        self, h5file: h5py.File, data: Any, group_name: str = ""
    ):
        """Recursively save data to HDF5, handling nested structures."""
        if isinstance(data, dict):
            group = h5file.create_group(group_name) if group_name else h5file
            for key, value in data.items():
                self._save_hdf5_recursively(group, value, str(key))
        elif isinstance(data, (list, tuple)):
            # Convert lists/tuples to NumPy arrays for efficient storage
            try:
                data_array = np.array(data)
                h5file.create_dataset(group_name, data=data_array)
            except (
                Exception
            ) as e:  # Fallback for mixed types (e.g. list of strings and lists)
                self.logger.info(f"{group_name} is a list of mixed types: {e}")
                for idx, item in enumerate(data):
                    self._save_hdf5_recursively(h5file, item, f"{group_name}_{idx}")

        elif isinstance(data, (np.ndarray, cp.ndarray)):
            data_np = data.get() if isinstance(data, cp.ndarray) else data
            h5file.create_dataset(group_name, data=data_np)
        elif isinstance(data, (int, float, str, bool)):
            h5file.attrs[group_name] = data  # Store basic types as attributes
        else:
            self.logger.warning(
                f"Unsupported data type for HDF5: {type(data)}. Skipping {group_name}."
            )

    def load_conservation_file(self, conservation_file: str) -> Dict[int, float]:
        """
        Load conservation scores from a file (e.g., phastCons, phyloP).

        Args:
            conservation_file: Path to the conservation file (wigFix, bigWig, or bedGraph).

        Returns:
            A dictionary mapping genomic position to conservation score.
        """
        conservation_scores = {}
        try:
            if conservation_file.endswith((".wig", ".wigFix")):
                # Handle wigFix format (variableStep or fixedStep)
                with open(conservation_file, "r") as f:
                    chrom = None
                    step = 1
                    span = 1
                    start = 1
                    for line in f:
                        line = line.strip()
                        if line.startswith("variableStep"):
                            parts = line.split()
                            chrom = [p.split("=")[1] for p in parts if "chrom=" in p][0]
                            span_list = [p.split("=")[1] for p in parts if "span=" in p]
                            span = int(span_list[0]) if span_list else 1

                        elif line.startswith("fixedStep"):
                            parts = line.split()
                            chrom = [p.split("=")[1] for p in parts if "chrom=" in p][0]
                            start = int(
                                [p.split("=")[1] for p in parts if "start=" in p][0]
                            )
                            step_list = [p.split("=")[1] for p in parts if "step=" in p]
                            step = int(step_list[0]) if step_list else 1
                            span_list = [p.split("=")[1] for p in parts if "span=" in p]
                            span = int(span_list[0]) if span_list else 1

                        elif chrom:  # Only proceed if we are within a track
                            if line.startswith("variableStep") or line.startswith(
                                "fixedStep"
                            ):
                                continue  # In case the file has multiple tracks
                            try:
                                if "variableStep" in locals():  # Check if variableStep
                                    pos, score = line.split()
                                    pos, score = int(pos), float(score)
                                    for i in range(span):
                                        conservation_scores[pos + i] = score
                                else:  # fixedStep
                                    score = float(line)
                                    for i in range(span):
                                        conservation_scores[start + i] = score
                                    start += step
                            except ValueError:  # Handle potential parsing errors
                                self.logger.warning(
                                    f"Skipping line in wigFix file: {line}"
                                )
            elif conservation_file.endswith((".bw", ".bigWig")):
                # Handle bigWig format (requires pyBigWig)
                import pyBigWig  # Import only if needed

                bw = pyBigWig.open(conservation_file)
                # This assumes you have a single chromosome; adapt if needed
                for chrom in bw.chroms():
                    # Use intervals() to get all (start, end, value) tuples
                    for start, end, value in bw.intervals(chrom):
                        for i in range(start, end):
                            conservation_scores[i] = value
                bw.close()

            elif conservation_file.endswith(".bedGraph"):
                # Handle bedGraph format
                with open(conservation_file, "r") as f:
                    for line in f:
                        chrom, start, end, score = line.strip().split()
                        start, end, score = int(start), int(end), float(score)
                        for i in range(start, end):
                            conservation_scores[i] = score
            else:
                raise ValueError(
                    "Unsupported conservation file format.  Use wigFix, bigWig, or bedGraph."
                )

            self.logger.info(f"Loaded conservation scores from {conservation_file}")
            return conservation_scores

        except FileNotFoundError:
            self.logger.error(f"Conservation file not found: {conservation_file}")
            raise
        except ImportError as e:
            if "pyBigWig" in str(e):
                self.logger.error(
                    "pyBigWig is required to read bigWig files. Install it with: pip install pybigwig"
                )
            else:
                self.logger.error(f"Import error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading conservation file: {e}")
            raise

    def load_annotation_file(self, annotation_file: str) -> Dict[int, Dict[str, Any]]:
        """
        Load genomic annotations from a file (e.g., GFF, BED).

        Args:
            annotation_file: Path to the annotation file.

        Returns:
            A dictionary mapping genomic position to annotation information.
        """
        annotations = {}
        try:
            if annotation_file.endswith((".gff", ".gff3")):
                # Handle GFF/GFF3 format
                for record in SeqIO.parse(annotation_file, "gff3"):
                    for feature in record.features:
                        for i in range(feature.location.start, feature.location.end):
                            annotations[i] = {
                                "type": feature.type,
                                "source": feature.qualifiers.get("source", [""])[0],
                                "score": feature.qualifiers.get("score", [None])[0],
                                "strand": feature.strand,
                                "attributes": feature.qualifiers,
                            }

            elif annotation_file.endswith(".bed"):
                # Handle BED format
                bed = BedTool(annotation_file)
                for interval in bed:
                    for i in range(interval.start, interval.end):
                        annotations[i] = {
                            "type": interval.name or "region",  # Use name if available
                            "score": interval.score,
                            "strand": interval.strand,
                            "chrom": interval.chrom,
                        }
            else:
                raise ValueError("Unsupported annotation file format. Use GFF or BED.")

            self.logger.info(f"Loaded annotations from {annotation_file}")
            return annotations

        except FileNotFoundError:
            self.logger.error(f"Annotation file not found: {annotation_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading annotation file: {e}")
            raise

    def load_epigenetic_file(self, epigenetic_file: str) -> Dict[int, Dict[str, float]]:
        """
        Load epigenetic data from a file (e.g., bedGraph, bigWig, narrowPeak, broadPeak)

        Args:
            epigenetic_file: Path to the epigenetic data file.

        Returns:
          A dictionary where keys are genomic positions (integers) and values
          are dictionaries.  The inner dictionaries map epigenetic mark names
          (e.g., "H3K27ac", "DNase") to their corresponding values (floats) at
          that position.

        """
        epigenetic_data = {}
        try:
            if epigenetic_file.endswith(".bedGraph"):
                with open(epigenetic_file, "r") as f:
                    for line in f:
                        chrom, start, end, value = line.strip().split()
                        start, end, value = int(start), int(end), float(value)
                        # Assuming the file name contains the mark name
                        mark = epigenetic_file.split(".")[0].split("/")[-1]
                        for i in range(start, end):
                            epigenetic_data.setdefault(i, {})[mark] = value

            elif epigenetic_file.endswith((".bw", ".bigWig")):
                import pyBigWig

                bw = pyBigWig.open(epigenetic_file)
                mark = epigenetic_file.split(".")[0].split("/")[
                    -1
                ]  # Extract mark from filename
                for chrom in bw.chroms():
                    for start, end, value in bw.intervals(chrom):
                        for i in range(start, end):
                            epigenetic_data.setdefault(i, {})[mark] = value
                bw.close()

            elif epigenetic_file.endswith((".narrowPeak", ".broadPeak")):
                with open(epigenetic_file, "r") as f:
                    for line in f:
                        fields = line.strip().split()
                        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                        # narrowPeak/broadPeak files may not always have a defined value for all fields
                        mark = (
                            fields[3] if len(fields) > 3 else "peak"
                        )  # Use peak name (or "peak")
                        value = (
                            float(fields[4]) if len(fields) > 4 else 1.0
                        )  # Use signalValue, or default

                        for i in range(start, end):
                            epigenetic_data.setdefault(i, {})[mark] = value
            else:
                raise ValueError(
                    "Unsupported epigenetic file format. Use bedGraph, bigWig, narrowPeak, or broadPeak."
                )

            self.logger.info(f"Loaded epigenetic data from {epigenetic_file}")
            return epigenetic_data

        except FileNotFoundError:
            self.logger.error(f"Epigenetic file not found: {epigenetic_file}")
            raise
        except ImportError as e:
            if "pyBigWig" in str(e):
                self.logger.error(
                    "pyBigWig is required to read bigWig files. Install it with: pip install pybigwig"
                )
            else:
                self.logger.error(f"Import error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading epigenetic data: {e}")
            raise

    def load_motif_file(self, motif_file: str) -> List[Dict[str, Any]]:
        """
        Load motif data from a file (e.g., JASPAR, MEME, HOMER).

        Args:
            motif_file: Path to the motif file.

        Returns:
            A list of dictionaries, where each dictionary represents a motif and
            contains information like 'name', 'start', 'end', 'score', and 'pattern'.
        """
        motifs = []
        try:
            if motif_file.endswith(".jaspar"):
                # Handle JASPAR format
                with open(motif_file, "r") as f:
                    for record in SeqIO.parse(f, "jaspar"):
                        # Assuming simple JASPAR format (adjust as needed)
                        motifs.append(
                            {
                                "name": record.name,
                                "pattern": str(
                                    record.seq
                                ),  # Simplified; use matrix if available
                                "score": 1.0,  # No scores in simple jaspar format
                                "start": 0,
                                "end": len(record.seq),
                            }
                        )

            elif motif_file.endswith(".meme"):
                # Handle MEME format (requires Biopython)
                from Bio import motifs  # Import only if needed

                with open(motif_file) as handle:
                    for motif in motifs.parse(handle, "meme"):
                        # Extract information, adapt based on your needs
                        for instance in motif.instances:
                            motifs.append(
                                {
                                    "name": motif.name,
                                    "pattern": str(instance),
                                    "score": instance.score
                                    if hasattr(instance, "score")
                                    else 1.0,  # Fallback in case there is no score
                                    "start": 0,  # Not directly available in MEME instances
                                    "end": len(instance),
                                }
                            )

            elif motif_file.endswith(".homer"):
                with open(motif_file, "r") as f:
                    for line in f:
                        if line.startswith(">"):
                            parts = line.strip().split("\t")
                            motif_info = parts[0][1:].split(",")  # Remove '>' and split
                            motif_name = motif_info[0]
                            # HOMER format usually contains position weight matrices, not instances.
                            # Here we treat the consensus as if it was an instance.
                            consensus = parts[1]
                            motifs.append(
                                {
                                    "name": motif_name,
                                    "pattern": consensus,
                                    "score": 1.0,
                                    "start": 0,
                                    "end": len(consensus),
                                }
                            )

            else:  # Fallback: Try to parse as simple position/sequence pairs
                with open(motif_file, "r") as f:
                    for line in f:
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                # Assume format:  position  sequence [score]
                                start = int(parts[0])
                                pattern = parts[1]
                                score = float(parts[2]) if len(parts) > 2 else 1.0
                                motifs.append(
                                    {
                                        "name": f"motif_{start}",
                                        "start": start,
                                        "end": start + len(pattern),
                                        "pattern": pattern,
                                        "score": score,
                                    }
                                )
                        except (ValueError, IndexError):
                            self.logger.warning(f"Skipping line in motif file: {line}")

            self.logger.info(f"Loaded motif data from {motif_file}")
            return motifs

        except FileNotFoundError:
            self.logger.error(f"Motif file not found: {motif_file}")
            raise
        except ImportError as e:
            if "Bio" in str(e):
                self.logger.error(
                    "Biopython is required to read MEME/JASPAR files.  Install it with: pip install biopython"
                )
            else:
                self.logger.error(f"Import error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading motif file: {e}")
            raise


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
        else:  # bipolar
            return float(self.xp.dot(v1, v2))

    def _normalize(self, v: Any) -> Any:
        """Normalize a vector."""
        norm = self.xp.linalg.norm(v)
        if norm < 1e-10:
            return v
        if self.vector_type == "binary":
            return self.xp.where(v / norm > 0.5, 1, 0)
        else:  # bipolar
            return v / norm

    def matched_filter(self, sequence_vectors, filter_vector):
        """
        Applies a matched filter (represented by filter_vector) to a
        sequence of HDC vectors.  This is essentially the same as the
        convolve method, but we're using the "matched filter" terminology.
        """
        return self.convolve(sequence_vectors, filter_vector)

    def circular_shift(self, vector, shift):
        """
        Cyclic shift (circular convolution) of an HDC vector.

        Args:
            vector: The HDC vector to shift (NumPy or CuPy array).
            shift: The number of positions to shift.  Positive values shift
                   to the right, negative values shift to the left.

        Returns:
            The shifted HDC vector (NumPy or CuPy array).
        """
        xp = self.xp  # Use the appropriate array library (NumPy or CuPy)
        return xp.roll(vector, shift)


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
    def reverse_complement(sequence: str) -> str:
        """Generate the reverse complement of a DNA sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join(complement[base] for base in reversed(sequence))

    @staticmethod
    def detect_motifs_by_length(
        sequence: str, motif_length: int
    ) -> Dict[str, List[int]]:
        """
        Detect motifs in a DNA sequence based on a specified length.

        Args:
            sequence: The DNA sequence.
            motif_length: The length of motifs to detect.

        Returns:
            A dictionary mapping motifs to their positions in the sequence.
        """
        motifs = defaultdict(list)
        for i in range(len(sequence) - motif_length + 1):
            motif = sequence[i : i + motif_length]
            motifs[motif].append(i)
        return motifs

    @staticmethod
    def calculate_gc_content_with_threshold(
        sequence: str, threshold: float = 0.5
    ) -> bool:
        """
        Calculate GC content and check if it exceeds a threshold.

        Args:
            sequence: The DNA sequence.
            threshold: The GC content threshold (default: 0.5).

        Returns:
            True if GC content exceeds the threshold, False otherwise.
        """
        gc_content = Utils.calculate_gc_content(sequence)
        return gc_content > threshold

    @staticmethod
    def calculate_normalized_entropy(sequence: str) -> float:
        """
        Calculate normalized Shannon entropy of a sequence.

        Args:
            sequence: The DNA sequence.

        Returns:
            The normalized Shannon entropy (scaled to [0, 1]).
        """
        entropy = Utils.calculate_sequence_entropy(sequence)
        max_entropy = math.log2(4)  # Maximum entropy for DNA (4 bases)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def reverse_complement_with_ambiguity(sequence: str) -> str:
        """
        Generate the reverse complement of a DNA sequence, handling ambiguous bases.

        Args:
            sequence: The DNA sequence (may contain ambiguous bases).

        Returns:
            The reverse complement sequence.
        """
        complement = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
            "R": "Y",
            "Y": "R",
            "M": "K",
            "K": "M",
            "S": "S",
            "W": "W",
            "B": "V",
            "V": "B",
            "D": "H",
            "H": "D",
        }
        return "".join(complement[base] for base in reversed(sequence))

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

    @staticmethod  # Added to Utils
    def handle_ambiguous_base(base: str, hdc: HDCVectorSpace) -> Optional[Any]:
        """Handles an ambiguous base by averaging the HDC vectors of its
        possible bases. Returns None if no valid bases are found."""
        if base in hdc.base_vectors:
            return hdc.base_vectors[base]
        elif base in IUPAC_AMBIGUITY_MAP:
            valid_bases = [
                b for b in IUPAC_AMBIGUITY_MAP[base] if b in hdc.base_vectors
            ]
            if valid_bases:
                return sum(hdc.base_vectors[b] for b in valid_bases) / len(valid_bases)
        return None

    @staticmethod  # Added to Utils.
    def encode_kmer_basic(kmer: str, hdc: HDCVectorSpace) -> Any:
        """
        Basic k-mer encoding (without biological features).  This is suitable
        for use in DNAEncoder.
        """
        xp = hdc.xp

        if len(kmer) == 1 and kmer in hdc.base_vectors:
            return hdc.base_vectors[kmer]

        result = xp.zeros(hdc.dim, dtype=xp.float32)
        for i, base in enumerate(kmer):
            base_vector = Utils.handle_ambiguous_base(base, hdc)  # Use utility function
            if base_vector is not None:
                pos_vector = hdc.position_vectors[
                    f"pos_{i % len(hdc.position_vectors)}"
                ]
                bound = hdc.bind(base_vector, pos_vector)
                result = hdc.bundle(result, bound, hdc.bundle_decay)

        return hdc._normalize(result)

    @staticmethod
    def calculate_distances(
        vector1: Any,
        vector2: Any,
        methods: List[str] = ["euclidean", "mahalanobis", "hyperbolic"],
    ) -> Dict[str, float]:
        """Calculate multiple distance metrics between vectors."""
        distances = {}
        if "euclidean" in methods:
            distances["euclidean"] = float(torch.norm(vector1 - vector2))

        if "mahalanobis" in methods:
            diff = vector1 - vector2
            stacked = torch.stack([vector1, vector2], dim=1)
            cov_matrix = torch.cov(stacked.T)
            inv_cov = torch.linalg.inv(cov_matrix)
            distances["mahalanobis"] = float(
                torch.sqrt(torch.matmul(torch.matmul(diff.T, inv_cov), diff))
            )

        if "hyperbolic" in methods:
            x_norm, y_norm = torch.norm(vector1), torch.norm(vector2)
            x_p = vector1 / (1 + torch.sqrt(1 + x_norm**2))
            y_p = vector2 / (1 + torch.sqrt(1 + y_norm**2))
            distances["hyperbolic"] = float(torch.acosh(1 - 2 * torch.sum(x_p * y_p)))

        return distances

    @staticmethod
    def calculate_swarm_repulsion(
        positions: List[Any], min_distance: float = 0.1
    ) -> List[Any]:
        """Calculate repulsion forces to maintain swarm spacing."""
        forces = [torch.zeros_like(pos) for pos in positions]

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    diff = pos_i - pos_j
                    dist = torch.norm(diff)
                    if dist < min_distance:
                        # Repulsive force inversely proportional to distance
                        force = diff * (1 / dist**2 - 1 / min_distance**2)
                        forces[i] += force
                        forces[j] -= force
        return forces

    def convolve(self, sequence_vectors, filter_vector):
        """Convolve (efficient matrix-based implementation)."""
        xp = self.xp
        sequence_matrix = xp.stack(sequence_vectors)
        similarities = xp.dot(sequence_matrix, filter_vector)
        return similarities.tolist()

    def max_pool(self, feature_map, window_size=4):
        """Max pooling (efficient reshaping)."""
        xp = self.xp
        if not feature_map:
            return []
        num_pools = len(feature_map) // window_size
        if num_pools == 0:
            return [xp.max(feature_map)]
        reshaped = xp.array(feature_map[: num_pools * window_size]).reshape(
            num_pools, window_size
        )
        pooled = xp.max(reshaped, axis=1)
        return pooled.tolist()

    def circular_shift(self, vector, shift):
        """Cyclic shift."""
        xp = self.xp
        return xp.roll(vector, shift)

    @staticmethod
    def encode_sequence(sequence: str, k: int) -> np.ndarray:
        """
        Encode a DNA sequence using k-mer encoding.

        Args:
            sequence: The DNA sequence to encode.
            k: The k-mer size.

        Returns:
            A high-dimensional vector representing the sequence.
        """
        kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        vector = np.zeros(len(kmers) * 4)  # 4 bases (A, T, G, C)
        for i, kmer in enumerate(kmers):
            vector[i * 4 : (i + 1) * 4] = Utils._encode_kmer(kmer)
        return vector

    @staticmethod
    def _encode_kmer(kmer: str) -> np.ndarray:
        """
        Encode a single k-mer into a one-hot vector.

        Args:
            kmer: The k-mer to encode.

        Returns:
            A one-hot encoded vector.
        """
        encoding = {
            "A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "C": [0, 0, 0, 1],
        }
        return np.array([encoding[base] for base in kmer]).flatten()

    @staticmethod
    def compute_similarity(
        vector1: np.ndarray, vector2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two vectors using a specified metric.

        Args:
            vector1: The first vector.
            vector2: The second vector.
            metric: The similarity metric ("cosine", "euclidean", "jaccard").

        Returns:
            The similarity score.
        """
        if metric == "cosine":
            return np.dot(vector1, vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
        elif metric == "euclidean":
            return 1 / (1 + np.linalg.norm(vector1 - vector2))
        elif metric == "jaccard":
            intersection = np.sum(np.minimum(vector1, vector2))
            union = np.sum(np.maximum(vector1, vector2))
            return intersection / union
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    @staticmethod
    def impute_sequence(prefix: str, suffix: str, gap_length: int) -> str:
        """
        Impute missing segments in a DNA sequence.

        Args:
            prefix: The sequence before the gap.
            suffix: The sequence after the gap.
            gap_length: The length of the gap to impute.

        Returns:
            The imputed sequence.
        """
        # Simple imputation: Use the most frequent k-mer in the prefix and suffix
        k = min(len(prefix), len(suffix), gap_length)
        prefix_kmers = [prefix[i : i + k] for i in range(len(prefix) - k + 1)]
        suffix_kmers = [suffix[i : i + k] for i in range(len(suffix) - k + 1)]
        all_kmers = prefix_kmers + suffix_kmers
        most_frequent = max(set(all_kmers), key=all_kmers.count)
        return prefix + most_frequent * gap_length + suffix

    @staticmethod
    def analyze_variant(sequence: str, variant: str) -> float:
        """
        Analyze the impact of a genetic variant on a sequence.

        Args:
            sequence: The original sequence.
            variant: The variant sequence.

        Returns:
            The impact score (0 = no impact, 1 = high impact).
        """
        original_vector = Utils.encode_sequence(sequence, k=len(sequence))
        variant_vector = Utils.encode_sequence(variant, k=len(variant))
        return 1 - Utils.compute_similarity(original_vector, variant_vector)

    @staticmethod
    def compare_sequences(sequences: List[str]) -> np.ndarray:
        """
        Compare multiple sequences and compute pairwise similarity.

        Args:
            sequences: A list of DNA sequences.

        Returns:
            A pairwise similarity matrix.
        """
        n = len(sequences)
        similarity_matrix = np.zeros((n, n))
        vectors = [Utils.encode_sequence(seq, k=len(seq)) for seq in sequences]
        for i in range(n):
            for j in range(i, n):
                similarity_matrix[i, j] = Utils.compute_similarity(
                    vectors[i], vectors[j]
                )
                similarity_matrix[j, i] = similarity_matrix[i, j]
        return similarity_matrix

    @staticmethod
    def integrate_epigenetic_data(
        sequence: str, epigenetic_data: Dict[int, float]
    ) -> np.ndarray:
        """
        Integrate epigenetic data with a DNA sequence.

        Args:
            sequence: The DNA sequence.
            epigenetic_data: A dictionary mapping positions to epigenetic scores.

        Returns:
            A vector combining sequence and epigenetic data.
        """
        sequence_vector = Utils.encode_sequence(sequence, k=len(sequence))
        epigenetic_vector = np.zeros(len(sequence_vector))
        for pos, score in epigenetic_data.items():
            if pos < len(epigenetic_vector):
                epigenetic_vector[pos] = score
        return np.concatenate((sequence_vector, epigenetic_vector))


class HDCMotifDiscovery:
    """
    Hyperdimensional computing-based motif discovery system that leverages
    both known motifs and discovers new ones from sequence data.

    Features:
    - Bootstraps from known motifs in COMMON_MOTIFS
    - Automatically discovers and classifies novel motifs
    - Groups similar motifs into families
    - Provides consensus representations
    - Scores sequences for motif enrichment
    """

    def __init__(self, hdc_computer, projection_dim=128, similarity_threshold=0.82):
        """
        Initialize the HDC motif discovery system.

        Args:
            hdc_computer: An HDCVectorSpace instance
            projection_dim: Dimension for projection space (for clustering)
            similarity_threshold: Threshold for considering motifs similar
        """
        self.hdc = hdc_computer
        self.similarity_threshold = similarity_threshold
        self.projection_dim = projection_dim

        # Initialize storage for motifs
        self.motif_anchors = {}  # Known/seed motifs
        self.motif_clusters = {}  # Expanded motif families
        self.novel_motifs = {}  # Discovered motifs not close to known ones
        self.motif_families = {}  # Grouping of motifs by family

        # For embedding projection (dimensionality reduction)
        self.projector = self._build_projector(self.hdc.hdc.dim, projection_dim)

        # Bootstrap with COMMON_MOTIFS
        self._bootstrap_from_common_motifs()

    def _build_projector(self, input_dim, output_dim):
        """Build a simple projection matrix for dimensionality reduction."""
        # Use stable random projections for vector embedding
        xp = self.hdc.hdc.xp
        seed = 42  # Fixed seed for reproducibility
        rng = np.random.RandomState(seed)
        projection = rng.normal(0, 1.0 / np.sqrt(output_dim), (input_dim, output_dim))
        return xp.array(projection)

    def _bootstrap_from_common_motifs(self):
        """Initialize motif space using COMMON_MOTIFS as anchors."""
        for motif, weight in COMMON_MOTIFS.items():
            # Skip motifs with ambiguous bases for now
            if "N" in motif and len(motif) > 8:
                continue

            # Encode the motif using HDC
            motif_vec = self.hdc.encode_kmer(motif)

            # Guess the motif family based on common patterns
            family = self._guess_motif_family(motif)

            # Store as an anchor
            self.motif_anchors[motif] = {
                "vector": motif_vec,
                "weight": weight,
                "family": family,
                "consensus": motif,
                "variants": [motif],
                "scores": {},  # Will store scores against sequences
                "positions": {},  # Will track positions where found
            }

            # Initialize the cluster for this motif
            self.motif_clusters[motif] = []

            # Add to family grouping
            if family not in self.motif_families:
                self.motif_families[family] = []
            self.motif_families[family].append(motif)

    def _guess_motif_family(self, motif):
        """Guess the motif family based on sequence patterns."""
        if motif in ["TATA", "TATAA", "TATAAA"]:
            return "TATA_box"
        elif motif in ["CAAT", "CCAAT"]:
            return "CAAT_box"
        elif "GATA" in motif:
            return "GATA_factor"
        elif "CACGTG" in motif:
            return "E_box"
        elif "GCC" in motif and "GGC" in motif:
            return "GC_rich"
        elif motif in ["TTGACA", "TATAAT"]:
            return "Bacterial_promoter"
        elif motif == "AATAAA":
            return "PolyA_signal"
        elif "GAGAG" in motif:
            return "GA_repeat"
        else:
            return "Other"

    def discover_motifs(self, sequences, k_range=(5, 12), min_support=3, stride=1):
        """
        Discover motifs from a set of sequences.

        Args:
            sequences: List of DNA sequences
            k_range: Range of k-mer sizes to consider (min, max)
            min_support: Minimum number of occurrences required
            stride: Step size for sliding window

        Returns:
            Dictionary of discovered motifs
        """
        # Extract all k-mers from sequences
        all_kmers = []
        kmer_positions = {}

        for k in range(k_range[0], k_range[1] + 1):
            for seq_idx, seq in enumerate(sequences):
                for i in range(0, len(seq) - k + 1, stride):
                    kmer = seq[i : i + k]
                    # Skip k-mers with non-standard bases
                    if any(base not in "ACGT" for base in kmer):
                        continue
                    all_kmers.append(kmer)
                    if kmer not in kmer_positions:
                        kmer_positions[kmer] = []
                    kmer_positions[kmer].append((seq_idx, i))

        # Count occurrences and filter by min_support
        kmer_counts = {}
        for kmer in all_kmers:
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

        frequent_kmers = [
            kmer for kmer, count in kmer_counts.items() if count >= min_support
        ]

        # Encode k-mers and cluster them
        vectors = {}
        for kmer in frequent_kmers:
            vectors[kmer] = self.hdc.encode_kmer(kmer)

        # Find motifs similar to known anchors
        self._assign_to_existing_clusters(vectors, kmer_positions)

        # Discover novel motifs from remaining k-mers
        self._discover_novel_clusters(
            {
                k: v
                for k, v in vectors.items()
                if not any(k in cluster for cluster in self.motif_clusters.values())
            }
        )

        # Update consensus motifs for each cluster
        self._update_consensus_motifs()

        return self.get_all_motifs()

    def _assign_to_existing_clusters(self, kmer_vectors, kmer_positions):
        """Assign k-mers to existing motif clusters."""
        # For each k-mer, check if it's similar to any known motif
        for kmer, vector in kmer_vectors.items():
            best_anchor = None
            best_sim = -1

            # Compare to each anchor motif
            for anchor, data in self.motif_anchors.items():
                sim = float(self.hdc.hdc.similarity(vector, data["vector"]))
                if sim > self.similarity_threshold and sim > best_sim:
                    best_anchor = anchor
                    best_sim = sim

            # If similar to a known motif, add to its cluster
            if best_anchor:
                self.motif_clusters[best_anchor].append(kmer)
                self.motif_anchors[best_anchor]["variants"].append(kmer)

                # Store positions where this k-mer was found
                for seq_idx, pos in kmer_positions.get(kmer, []):
                    if seq_idx not in self.motif_anchors[best_anchor]["positions"]:
                        self.motif_anchors[best_anchor]["positions"][seq_idx] = []
                    self.motif_anchors[best_anchor]["positions"][seq_idx].append(pos)

    def _discover_novel_clusters(self, remaining_vectors, min_cluster_size=3):
        """Discover novel motif clusters from remaining k-mers."""
        if not remaining_vectors:
            return

        # Project vectors to lower dimension for clustering
        kmers = list(remaining_vectors.keys())
        vectors = list(remaining_vectors.values())

        # Skip if too few vectors
        if len(vectors) < min_cluster_size:
            return

        # Convert to numpy for clustering
        if self.hdc.hdc.device == "gpu":
            vectors_np = [v.get() for v in vectors]
        else:
            vectors_np = vectors

        # Compute pairwise similarity matrix
        n = len(vectors_np)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = float(np.dot(vectors_np[i], vectors_np[j]))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Try to use HDBSCAN for clustering if available
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric="precomputed",
                core_dist_n_jobs=-1,
            )
            # Convert similarity to distance
            distance_matrix = 1 - sim_matrix
            labels = clusterer.fit_predict(distance_matrix)

        except ImportError:
            # Fallback to simple threshold-based clustering
            from sklearn.cluster import AgglomerativeClustering

            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                metric="precomputed",
                linkage="average",
            )
            # Convert similarity to distance
            distance_matrix = 1 - sim_matrix
            labels = clusterer.fit_predict(distance_matrix)

        # Organize into clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Skip noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(kmers[i])

        # Add novel clusters that are large enough
        for cluster_id, members in clusters.items():
            if len(members) >= min_cluster_size:
                # Calculate consensus for this cluster
                consensus = self._derive_consensus(members)
                family = self._guess_motif_family(consensus)

                # Create a new anchor for this novel motif
                self.novel_motifs[consensus] = {
                    "vector": self.hdc.encode_kmer(consensus),
                    "weight": 0.5,  # Start with lower confidence
                    "family": family,
                    "consensus": consensus,
                    "variants": members,
                    "scores": {},
                    "positions": {},
                }

                # Add to family grouping
                if family not in self.motif_families:
                    self.motif_families[family] = []
                self.motif_families[family].append(consensus)

    def _derive_consensus(self, sequences):
        """Derive a consensus sequence from a set of related sequences."""
        if not sequences:
            return ""

        # Handle variable lengths by focusing on the mode length
        lengths = [len(seq) for seq in sequences]
        mode_length = max(set(lengths), key=lengths.count)
        filtered_seqs = [seq for seq in sequences if len(seq) == mode_length]

        # Build position-specific base counts
        counts = []
        for i in range(mode_length):
            pos_counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            for seq in filtered_seqs:
                if i < len(seq):
                    pos_counts[seq[i]] = pos_counts.get(seq[i], 0) + 1
            counts.append(pos_counts)

        # Generate consensus by taking the most frequent base at each position
        consensus = ""
        for pos_counts in counts:
            max_base = max(pos_counts.items(), key=lambda x: x[1])[0]
            consensus += max_base

        return consensus

    def _update_consensus_motifs(self):
        """Update consensus motifs for all clusters."""
        # Update consensus for known motifs with variants
        for motif, data in self.motif_anchors.items():
            if len(data["variants"]) > 1:
                consensus = self._derive_consensus(data["variants"])
                # Only update if consensus is different and not degenerate
                if consensus != motif and consensus != "":
                    data["consensus"] = consensus

    def score_sequence(self, sequence, window_size=20, stride=5):
        """
        Score a sequence for motif enrichment.

        Args:
            sequence: DNA sequence to score
            window_size: Size of sliding window
            stride: Step size for window

        Returns:
            Dictionary with motif scores and positions
        """
        result = {"motif_scores": {}, "motif_positions": {}, "family_scores": {}}

        # Initialize scores for all motifs
        all_motifs = list(self.motif_anchors.keys()) + list(self.novel_motifs.keys())
        for motif in all_motifs:
            result["motif_scores"][motif] = 0
            result["motif_positions"][motif] = []

        # Score sequence windows
        for i in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[i : i + window_size]
            window_vec = self.hdc.encode_sequence(window)

            # Score against known motifs
            for motif, data in self.motif_anchors.items():
                sim = float(self.hdc.hdc.similarity(window_vec, data["vector"]))
                if sim > self.similarity_threshold:
                    result["motif_scores"][motif] += sim * data["weight"]
                    result["motif_positions"][motif].append((i, sim))

            # Score against novel motifs
            for motif, data in self.novel_motifs.items():
                sim = float(self.hdc.hdc.similarity(window_vec, data["vector"]))
                if sim > self.similarity_threshold:
                    result["motif_scores"][motif] += sim * data["weight"]
                    result["motif_positions"][motif].append((i, sim))

        # Calculate family-level scores
        for family, motifs in self.motif_families.items():
            family_score = sum(result["motif_scores"].get(motif, 0) for motif in motifs)
            result["family_scores"][family] = family_score

        return result

    def get_all_motifs(self):
        """Get all motifs (known and novel)."""
        all_motifs = {}
        # Include known motifs
        for motif, data in self.motif_anchors.items():
            all_motifs[motif] = {
                "consensus": data["consensus"],
                "family": data["family"],
                "weight": data["weight"],
                "variants": data["variants"],
                "is_novel": False,
            }

        # Include novel motifs
        for motif, data in self.novel_motifs.items():
            all_motifs[motif] = {
                "consensus": data["consensus"],
                "family": data["family"],
                "weight": data["weight"],
                "variants": data["variants"],
                "is_novel": True,
            }

        return all_motifs

    def get_motifs_by_family(self):
        """Get motifs organized by family."""
        return self.motif_families

    def visualize_motif_network(self, output_file=None):
        """
        Visualize the motif similarity network.
        Requires networkx and matplotlib.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.Graph()

            # Add all motifs as nodes
            for motif, data in self.motif_anchors.items():
                G.add_node(
                    motif,
                    family=data["family"],
                    type="known",
                    weight=data["weight"],
                    size=len(data["variants"]),
                )

            for motif, data in self.novel_motifs.items():
                G.add_node(
                    motif,
                    family=data["family"],
                    type="novel",
                    weight=data["weight"],
                    size=len(data["variants"]),
                )

            # Add edges based on similarity
            all_motifs = {**self.motif_anchors, **self.novel_motifs}
            motif_list = list(all_motifs.keys())

            for i in range(len(motif_list)):
                for j in range(i + 1, len(motif_list)):
                    motif1 = motif_list[i]
                    motif2 = motif_list[j]
                    vec1 = all_motifs[motif1]["vector"]
                    vec2 = all_motifs[motif2]["vector"]
                    sim = float(self.hdc.hdc.similarity(vec1, vec2))
                    if sim > self.similarity_threshold:
                        G.add_edge(motif1, motif2, weight=sim)

            # Plot the network
            plt.figure(figsize=(12, 10))

            # Set node colors by family
            families = list(set(nx.get_node_attributes(G, "family").values()))
            color_map = plt.cm.get_cmap("tab20", len(families))
            family_colors = {family: color_map(i) for i, family in enumerate(families)}

            node_colors = [family_colors[G.nodes[n]["family"]] for n in G.nodes()]
            node_sizes = [G.nodes[n]["size"] * 100 for n in G.nodes()]

            # Create layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)

            # Draw network
            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
            )
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title("Motif Similarity Network")
            plt.axis("off")

            # Add legend for families
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=family,
                )
                for family, color in family_colors.items()
            ]
            plt.legend(
                handles=handles, title="Motif Families", loc="lower right", ncol=2
            )

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.show()

        except ImportError as e:
            print(f"Visualization requires networkx and matplotlib: {e}")
            return None

    def cluster_motifs(self, motifs):
        """Cluster motifs using distance metrics (Mahalanobis or Hyperbolic)."""
        clusters = []
        for i, motif1 in enumerate(motifs):
            cluster = []
            for j, motif2 in enumerate(motifs):
                if i != j:
                    # Calculate Mahalanobis or Hyperbolic distance between motifs
                    mahalanobis_dist = self._calculate_mahalanobis(motif1, motif2)
                    hyperbolic_dist = self._hyperbolic_distance(motif1, motif2)

                    # Apply a distance threshold for clustering
                    if mahalanobis_dist < 1.0 or hyperbolic_dist < 1.0:
                        cluster.append(motif2)
            clusters.append(cluster)

        # Organize motifs into families
        motif_families = self._group_motifs_by_family(clusters)
        return motif_families

    def _group_motifs_by_family(self, clusters):
        """Group motifs into families based on distance-based clustering."""
        motif_families = {}
        for cluster in clusters:
            # Calculate consensus for this cluster
            consensus = self._derive_consensus(cluster)
            family = self._guess_motif_family(consensus)

            # Create or update motif family
            if family not in motif_families:
                motif_families[family] = []
            motif_families[family].append(consensus)

        return motif_families

    def _calculate_mahalanobis(self, motif1, motif2):
        """Calculate the Mahalanobis distance between two motifs."""
        diff = motif1 - motif2
        cov_matrix = torch.cov(torch.stack([motif1, motif2], dim=1).T)
        inv_cov_matrix = torch.linalg.inv(cov_matrix)
        mahalanobis_dist = torch.sqrt(
            torch.matmul(torch.matmul(diff.T, inv_cov_matrix), diff)
        )
        return float(mahalanobis_dist)

    def _hyperbolic_distance(self, motif1, motif2):
        """Calculate hyperbolic distance between two motifs."""
        x_norm = torch.norm(motif1)
        y_norm = torch.norm(motif2)
        x_p = motif1 / (1 + torch.sqrt(1 + x_norm**2))
        y_p = motif2 / (1 + torch.sqrt(1 + y_norm**2))
        inner_term = -2 * torch.sum(x_p * y_p)
        return torch.acosh(1 + inner_term)


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
        """Analyze sequence properties using advanced HDC motif discovery."""
        # Initialize basic stats
        stats = {}
        stats["length"] = len(sequence)
        stats["gc_content"] = Utils.calculate_gc_content(sequence)
        stats["entropy"] = Utils.calculate_sequence_entropy(sequence)
        stats["normalized_entropy"] = (
            stats["entropy"] / math.log2(4) if stats["entropy"] > 0 else 0
        )
        stats["optimal_k"] = Utils.detect_optimal_kmer_size(sequence)

        # Lazy-load our motif discovery system
        if not hasattr(self, "motif_discovery"):
            self.motif_discovery = HDCMotifDiscovery(self)

        # Get comprehensive motif analysis
        motif_results = self.motif_discovery.score_sequence(sequence)

        # Extract motif positions with scores
        stats["motif_positions"] = []
        for motif, positions in motif_results["motif_positions"].items():
            for pos, score in positions:
                stats["motif_positions"].append((pos, score, motif))

        # Add motif enrichment scores
        stats["motif_scores"] = motif_results["motif_scores"]
        stats["motif_families"] = motif_results["family_scores"]

        # Add top motifs for quick reference
        top_motifs = sorted(
            motif_results["motif_scores"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        stats["top_motifs"] = [m for m, s in top_motifs if s > 0]

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
    """Represents a biological feature provider.

    Args:
        name: Name of the feature provider (e.g., "tfbs", "gc_content").
        provider_fn: Function to compute the feature vector for a given k-mer or position.
        weight: Weight of this feature in the final vector.
        data: Optional data required for the feature provider (e.g., TFBS positions).
    """

    name: str
    provider_fn: Optional[Callable[[str, Optional[int]], Any]] = None
    weight: float = 1.0
    data: Optional[Dict[str, Any]] = field(default_factory=dict)

    def compute_feature(
        self, kmer: str, position: Optional[int] = None
    ) -> Optional[Any]:
        """Compute the feature vector for a given k-mer and position."""
        if self.provider_fn is not None:
            return self.provider_fn(kmer, position)
        elif self.name == "tfbs":
            return self._compute_tfbs(kmer, position)
        else:
            raise NotImplementedError(
                f"No provider function defined for feature: {self.name}"
            )

    def _compute_tfbs(self, kmer: str, position: Optional[int] = None) -> Optional[Any]:
        """Compute TFBS feature vector for a k-mer."""
        if position is None:
            return None

        # Check if the k-mer overlaps with any TFBS
        tfbs_scores = {}
        for tf, sites in self.data.get("tfbs_data", {}).items():
            for start, end in sites:
                if start <= position < end:
                    tfbs_scores[tf] = (
                        1.0  # Binary score (1 if overlapping, 0 otherwise)
                    )

        if not tfbs_scores:
            return None

        # Create a feature vector based on TFBS scores
        feature_vector = np.zeros(self.data.get("dim", 1000))  # Default dimension
        for tf, score in tfbs_scores.items():
            seed_val = hash(tf) % 10000  # Consistent seeding
            np.random.seed(seed_val)
            tf_vec = np.random.uniform(-1, 1, self.data.get("dim", 1000))
            feature_vector += tf_vec * score

        return feature_vector / np.linalg.norm(feature_vector)  # Normalize


class BiologicalEncoder:
    def __init__(self, config: GSCConfig):
        self.config = config
        self.hdc = HDCVectorSpace(config)
        self.feature_providers: Dict[str, FeatureProvider] = {}
        self.feature_cache: Dict[
            Tuple[str, Optional[int]], Any
        ] = {}  # Feature cache, str for kmer
        self.kmer_cache = lru_cache(maxsize=self.config.cache_size)(
            self._encode_kmer_uncached
        )
        self.data_loader = GenomicDataLoader()
        self.utils = Utils()  # Access the utility functions
        self._register_default_features()
        self.hdc.initialize()  # Initialize HDC vectors
        self.filters = {}  # Initialize filters, used for convolutional method of encoding sequences

    def initialize_filters(self, filter_motifs: Optional[List[str]] = None):
        """
        Initializes the convolutional filters.  If filter_motifs is provided,
        the filters are initialized with the HDC vectors of those motifs.
        Otherwise, filters are initialized randomly.

        Args:
            filter_motifs: A list of motifs (strings) to use for initializing
                           the filters.  If None, random filters are created.
        """
        if filter_motifs:
            for motif in filter_motifs:
                self.filters[motif] = self.hdc.encode_sequence(
                    motif
                )  # Use encode_sequence, not encode_kmer
        else:
            # Create random filters.  You might want to adjust the number and
            # length of random filters. A good starting point is to have
            # filters that are similar in length to common motifs.
            num_random_filters = 10
            for i in range(num_random_filters):
                filter_len = self.utils.detect_optimal_kmer_size(
                    "".join(random.choices("ACGT", k=10))
                )  # Random length between 5 and 15
                random_motif = "".join(
                    self.hdc.xp.random.choice(
                        list(self.hdc.base_vectors.keys()), size=filter_len
                    )
                )
                self.filters[f"random_{i}"] = self.hdc.encode_sequence(random_motif)

    def encode_sequence_convolutional(
        self, sequence: str, pooling_window: int = 4
    ) -> Any:
        """
        Encodes a DNA sequence using a convolutional HDC approach.

        Args:
            sequence: The DNA sequence to encode (string).
            pooling_window: The size of the window for max pooling.

        Returns:
            The HDC vector representing the sequence (NumPy or CuPy array).
        """
        xp = self.hdc.xp

        if not sequence:
            return xp.zeros(self.hdc.dim, dtype=xp.float32)  # Handle empty sequence

        # 1.  Base Encoding (with Circular Shift for Position)
        base_vectors = []
        for i, base in enumerate(sequence):
            base_vector = self.utils.handle_ambiguous_base(
                base, self.hdc
            )  # Use Utils for ambiguous bases
            if base_vector is not None:
                encoded_base = self.hdc.circular_shift(base_vector, i)
                base_vectors.append(encoded_base)

        if not base_vectors:
            return xp.zeros(
                self.hdc.dim, dtype=xp.float32
            )  # Handle all-ambiguous-bases

        # 2. Convolution with Filters
        feature_maps = {}
        for filter_name, filter_vector in self.filters.items():
            feature_map = self.hdc.convolve(
                base_vectors, filter_vector
            )  # use the convolve function
            feature_maps[filter_name] = feature_map

        # 3. Max Pooling
        pooled_maps = {}
        for filter_name, feature_map in feature_maps.items():
            pooled_map = self.hdc.max_pool(feature_map, window_size=pooling_window)
            if not pooled_map:  # Handle the case where pooling results in an empty list (sequence shorter than window)
                pooled_maps[filter_name] = [0.0]  # Use 0 as default
            else:
                pooled_maps[filter_name] = pooled_map

        # 4. Bundling (Weighted Sum of Pooled Feature Maps)
        # Convert pooled maps to HDC vectors, handling potential empty maps.
        pooled_vectors = []
        for filter_name, pooled_map in pooled_maps.items():
            if pooled_map:  # Should always be true now, but good to keep checking.
                pooled_vector = xp.array(pooled_map, dtype=xp.float32)
                if pooled_vector.ndim == 1:  # If it is 1D, make 2D
                    pooled_vector = xp.expand_dims(pooled_vector, axis=0)
                if pooled_vector.shape[1] < self.hdc.dim:  # Pad if needed
                    padding_size = self.hdc.dim - pooled_vector.shape[1]
                    padding = xp.zeros((1, padding_size), dtype=xp.float32)
                    pooled_vector = xp.concatenate((pooled_vector, padding), axis=1)
                elif pooled_vector.shape[1] > self.hdc.dim:  # Truncate
                    pooled_vector = pooled_vector[:, : self.hdc.dim]
                pooled_vectors.append(
                    self.hdc._normalize(pooled_vector[0])
                )  # Back to 1D

        if (
            not pooled_vectors
        ):  # Handle the case where all filters resulted in empty pooled maps
            return xp.zeros(self.hdc.dim, dtype=xp.float32)

        # Stack, then bundle
        stacked_vectors = xp.stack(pooled_vectors)
        result = xp.sum(stacked_vectors, axis=0)
        return self.hdc._normalize(result)

    def _register_default_features(self):
        """Register default feature providers."""
        self.register_feature_provider(
            FeatureProvider("gc_content", self._compute_gc_content, 0.5)
        )
        self.register_feature_provider(
            FeatureProvider("complexity", self._compute_complexity, 0.3)
        )
        self.register_feature_provider(
            FeatureProvider("motifs", self._detect_motifs, 0.4)
        )  # Local motifs

    def register_feature_provider(self, provider: FeatureProvider):
        """Register a feature provider."""
        self.feature_providers[provider.name] = provider
        logger.info(
            f"Registered feature provider: {provider.name} with weight {provider.weight}"
        )

    def _compute_gc_content(
        self, kmer: str, position: Optional[int] = None
    ) -> float:  # Added position
        """Compute GC content using Utils."""
        return self.utils.calculate_gc_content(kmer)

    def _compute_complexity(
        self, kmer: str, position: Optional[int] = None
    ) -> float:  # Added position
        """Compute Shannon entropy using Utils."""
        return self.utils.calculate_sequence_entropy(kmer)

    @lru_cache(maxsize=128)
    def _detect_motifs(self, kmer: str, position: Optional[int] = None) -> float:
        """Detect motifs using HDC-powered motif discovery."""
        # Lazy-load our motif discovery system
        if not hasattr(self, "motif_discovery"):
            self.motif_discovery = HDCMotifDiscovery(self)

        # For short k-mers, do quick scoring against known motifs
        if len(kmer) < 15:
            kmer_vec = self.encode_kmer(kmer)

            # Score against all motifs (known + novel)
            best_score = 0.0

            # Check all motifs from both known and novel collections
            all_motifs = {
                **self.motif_discovery.motif_anchors,
                **self.motif_discovery.novel_motifs,
            }
            for _, data in all_motifs.items():
                sim = float(self.hdc.similarity(kmer_vec, data["vector"]))
                weighted_score = sim * data["weight"]
                best_score = max(best_score, weighted_score)

            # Return the score only if above threshold
            return best_score if best_score > 0.4 else 0.0

        # For longer sequences, use the fuller scoring approach
        score_results = self.motif_discovery.score_sequence(kmer)
        if not score_results["motif_scores"]:
            return 0.0

        # Get the highest motif score
        return max(score_results["motif_scores"].values())

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
        base_vector = self.utils.encode_kmer_basic(
            kmer, self.hdc
        )  # Use Utils for basic encoding

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
        return self.utils.encode_kmer_basic(kmer, self.hdc)  # Use Utils

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
            kmers = self.utils.generate_kmers(chunk, k, stride)  # Use Utils
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


class AgentType(Enum):
    """Agent specialization types."""

    MOTIF_DETECTOR = auto()
    VARIANT_ANALYZER = auto()
    CONSERVATION_TRACKER = auto()
    GENERALIST = auto()


@dataclass
class AgentConfig:
    """Configuration for genetic agent behavior and learning."""

    agent_type: AgentType = AgentType.GENERALIST
    learning_rate: float = 0.01
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: {"motifs": 1.0, "gc_content": 1.0}
    )
    epsilon_start: float = 0.3
    epsilon_decay: float = 0.9

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate with validation."""
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1.")
        self.learning_rate = learning_rate

    def set_feature_weights(self, feature: str, weight: float) -> None:
        """Set the weight for a specific feature with validation."""
        if not 0 <= weight <= 1:
            raise ValueError("Feature weight must be between 0 and 1.")
        self.feature_weights[feature] = weight

    def validate_config(self) -> bool:
        """Validate all configuration parameters."""
        return 0 < self.learning_rate <= 1 and all(
            0 <= weight <= 1 for weight in self.feature_weights.values()
        )

    @classmethod
    def create_motif_detector_config(cls) -> "AgentConfig":
        """Factory method for creating a motif detector configuration."""
        return cls(
            agent_type=AgentType.MOTIF_DETECTOR,
            feature_weights={"motifs": 1.0, "gc_content": 0.5},
        )

    @classmethod
    def create_variant_analyzer_config(cls) -> "AgentConfig":
        """Factory method for creating a variant analyzer configuration."""
        return cls(
            agent_type=AgentType.VARIANT_ANALYZER,
            feature_weights={"variants": 1.0, "conservation": 0.8},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary."""
        return {
            "agent_type": self.agent_type.name,
            "learning_rate": self.learning_rate,
            "feature_weights": self.feature_weights,
            "epsilon_start": self.epsilon_start,
            "epsilon_decay": self.epsilon_decay,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Deserialize the configuration from a dictionary."""
        return cls(
            agent_type=AgentType[config_dict.get("agent_type", "GENERALIST")],
            learning_rate=config_dict.get("learning_rate", 0.01),
            feature_weights=config_dict.get("feature_weights", {}),
            epsilon_start=config_dict.get("epsilon_start", 0.3),
            epsilon_decay=config_dict.get("epsilon_decay", 0.9),
        )


@dataclass
class AgentState:
    """Tracks agent-specific state and memory."""

    q_values: Dict[str, np.ndarray] = field(default_factory=dict)
    experience_buffer: List[Tuple[Any, ...]] = field(default_factory=list)
    confidence: float = 0.5
    history: List[Any] = field(default_factory=list)

    def update_q_values(
        self, state_hash: str, action: str, q_vector: np.ndarray
    ) -> None:
        """Update Q-values for a specific state-action pair."""
        key = f"{state_hash}_{action}"
        self.q_values[key] = q_vector

    def add_experience(self, experience: Tuple[Any, ...]) -> None:
        """Add an experience tuple to the experience buffer."""
        self.experience_buffer.append(experience)

    def update_confidence(self, new_confidence: float) -> None:
        """Update the agent's confidence with validation."""
        if not 0 <= new_confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        self.confidence = new_confidence

    def clear_experience_buffer(self) -> None:
        """Clear the experience buffer."""
        self.experience_buffer.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the agent state to a dictionary."""
        return {
            "q_values": {k: v.tolist() for k, v in self.q_values.items()},
            "experience_buffer": self.experience_buffer,
            "confidence": self.confidence,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """Deserialize the agent state from a dictionary."""
        return cls(
            q_values={
                k: np.array(v) for k, v in state_dict.get("q_values", {}).items()
            },
            experience_buffer=state_dict.get("experience_buffer", []),
            confidence=state_dict.get("confidence", 0.5),
            history=state_dict.get("history", []),
        )


class GeneticAgent:
    """HDC-powered reinforcement learning agent for genetic sequence analysis."""

    def __init__(
        self, supercomputer: "HDCVectorSpace", config: Optional[AgentConfig] = None
    ):
        """
        Initialize the genetic agent.

        Args:
            supercomputer: The HDC vector space for sequence encoding.
            config: Agent configuration (optional).
        """
        self.sc = supercomputer
        self.xp = supercomputer.xp
        self.config = config or AgentConfig.create_generalist_config()
        self.state = AgentState()
        self.action_space = ["A", "T", "G", "C"]

    def predict_base(self, state_vector: Any, epsilon: float = 0.1) -> str:
        """
        Predict the next base using vectorized Q-value computation.

        Args:
            state_vector: The current state vector.
            epsilon: Exploration rate.

        Returns:
            The predicted base.
        """
        if self.xp.random.random() < epsilon:
            return self.xp.random.choice(self.action_space)  # Exploration

        # Exploitation: Choose the action with the highest Q-value
        state_hash = hash(state_vector.tobytes())
        q_values = {
            action: self._get_q_value(state_hash, action, state_vector)
            for action in self.action_space
        }
        return max(q_values.items(), key=lambda x: x[1])[0]

    def _get_q_value(self, state_hash: int, action: str, state_vector: Any) -> float:
        """
        Get the Q-value for a state-action pair, initializing if needed.

        Args:
            state_hash: Hash of the state vector.
            action: The action to evaluate.
            state_vector: The state vector.

        Returns:
            The Q-value.
        """
        key = f"{state_hash}_{action}"
        if key not in self.state.q_values:
            self.state.q_values[key] = self.sc.base_vectors[action].copy()
        return float(self.xp.dot(state_vector, self.state.q_values[key]))

    def update_q_values(
        self, state: Any, action: str, reward: float, next_state: Any
    ) -> None:
        """
        Update Q-values using Temporal Difference (TD) learning.

        Args:
            state: The current state vector.
            action: The action taken.
            reward: The reward received.
            next_state: The next state vector.
        """
        state_hash = hash(state.tobytes())
        key = f"{state_hash}_{action}"

        # Compute max future Q-value
        next_hash = hash(next_state.tobytes())
        next_q = max(
            self._get_q_value(next_hash, a, next_state) for a in self.action_space
        )

        # TD update
        current_q = self._get_q_value(state_hash, action, state)
        td_error = reward + self.config.gamma * next_q - current_q
        delta = self.config.learning_rate * td_error

        # Update Q-value using HDC bundling
        self.state.q_values[key] = self.sc.bundle(
            self.state.q_values[key], state, alpha=1.0 - min(0.9, abs(delta))
        )

    def train_on_segment(self, context: Any, target: str) -> float:
        """
        Train the agent on a sequence segment.

        Args:
            context: The context sequence vector.
            target: The target sequence.

        Returns:
            The total reward obtained during training.
        """
        state = context.copy()
        total_reward = 0

        for target_base in target:
            # Predict the next base
            pred_base = self.predict_base(state, self.config.epsilon_start)
            reward = 1.0 if pred_base == target_base else -0.2
            total_reward += reward

            # Compute the next state
            next_vector = self.sc.encode_kmer(pred_base)
            next_state = self.sc.bundle(state, next_vector, alpha=0.7)

            # Update Q-values and store experience
            self.update_q_values(state, pred_base, reward, next_state)
            self.state.add_experience((state, pred_base, reward, next_state))

            # Trim the experience buffer if needed
            if len(self.state.experience_buffer) > self.config.buffer_size:
                self.state.experience_buffer = self.state.experience_buffer[
                    -self.config.buffer_size :
                ]

            state = next_state

        # Update agent confidence
        self.state.update_confidence(total_reward / len(target))
        return total_reward

    def train_batch(self, batch_size: Optional[int] = None) -> int:
        """
        Train the agent on a batch of experiences.

        Args:
            batch_size: The size of the batch (defaults to config.batch_size).

        Returns:
            The number of experiences processed.
        """
        batch_size = batch_size or self.config.batch_size
        if len(self.state.experience_buffer) < batch_size:
            return 0

        # Sample a batch of experiences
        indices = self.xp.random.choice(
            len(self.state.experience_buffer), batch_size, replace=False
        )
        for idx in indices:
            state, action, reward, next_state = self.state.experience_buffer[idx]
            self.update_q_values(state, action, reward, next_state)

        return len(indices)

    def specialize_in_motifs(self) -> None:
        """
        Specialize the agent in motif detection.
        """
        self.config = AgentConfig.create_motif_detector_config()
        logger.info("Agent specialized in motif detection.")

    def specialize_in_variants(self) -> None:
        """
        Specialize the agent in variant analysis.
        """
        self.config = AgentConfig.create_variant_analyzer_config()
        logger.info("Agent specialized in variant analysis.")

    def save_state(self, filepath: str) -> None:
        """
        Save the agent's state to a file.

        Args:
            filepath: The path to save the state.
        """
        state_dict = self.state.to_dict()
        with open(filepath, "w") as f:
            import json

            json.dump(state_dict, f)
        logger.info(f"Agent state saved to {filepath}.")

    def load_state(self, filepath: str) -> None:
        """
        Load the agent's state from a file.

        Args:
            filepath: The path to load the state from.
        """
        with open(filepath, "r") as f:
            import json

            state_dict = json.load(f)
        self.state = AgentState.from_dict(state_dict)
        logger.info(f"Agent state loaded from {filepath}.")


@dataclass
class SwarmConfig:
    """
    Configuration for swarm behavior, dynamics, and feature metrics.

    Combines swarm-level configurations with feature-specific metrics.
    """

    # Swarm configuration
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

    # Feature metrics
    motifs: float = 0.0
    structure: float = 0.0
    conservation: float = 0.0
    coding: float = 0.0
    boundary: float = 0.0
    gc_content: float = 0.0
    complexity: float = 0.0

    def set_swarm_size(self, size: int) -> None:
        """Set the swarm size with validation."""
        if size < self.min_agent_count:
            raise ValueError(f"Swarm size must be at least {self.min_agent_count}.")
        self.size = size

    def set_dynamics(
        self, individual_weight: float, type_weight: float, momentum: float
    ) -> None:
        """Set swarm dynamics with validation."""
        if not (
            0 <= individual_weight <= 1 and 0 <= type_weight <= 1 and 0 <= momentum <= 1
        ):
            raise ValueError("Weights and momentum must be between 0 and 1.")
        self.individual_weight = individual_weight
        self.type_weight = type_weight
        self.momentum = momentum

    def set_feature_thresholds(
        self, feature_threshold: float, interest_threshold: float
    ) -> None:
        """Set feature thresholds with validation."""
        if not (0 <= feature_threshold <= 1 and 0 <= interest_threshold <= 1):
            raise ValueError("Thresholds must be between 0 and 1.")
        self.feature_threshold = feature_threshold
        self.interest_threshold = interest_threshold

    def normalize_feature_metrics(self) -> None:
        """Normalize feature metrics to sum to 1."""
        total = sum(getattr(self, f) for f in self._get_feature_fields())
        if total > 0:
            for field in self._get_feature_fields():
                setattr(self, field, getattr(self, field) / total)

    def track_success_rates(self, success_rates: List[float]) -> None:
        """Track success rates and update swarm dynamics."""
        if len(success_rates) > self.history_size:
            success_rates = success_rates[-self.history_size :]
        avg_success_rate = np.mean(success_rates)
        self.adjust_weights_based_on_success(avg_success_rate)

    def adjust_weights_based_on_success(self, success_rate: float) -> None:
        """Adjust swarm weights based on success rate."""
        if success_rate > 0.8:
            self.individual_weight *= 1.1  # Increase individual exploration
        elif success_rate < 0.5:
            self.type_weight *= 1.1  # Increase type-based collaboration

    def _get_feature_fields(self) -> List[str]:
        """Get the list of feature metric fields."""
        return [
            "motifs",
            "structure",
            "conservation",
            "coding",
            "boundary",
            "gc_content",
            "complexity",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the swarm configuration to a dictionary."""
        return {
            "size": self.size,
            "min_agent_count": self.min_agent_count,
            "beam_width": self.beam_width,
            "window_size": self.window_size,
            "stride": self.stride,
            "individual_weight": self.individual_weight,
            "type_weight": self.type_weight,
            "momentum": self.momentum,
            "feature_threshold": self.feature_threshold,
            "interest_threshold": self.interest_threshold,
            "history_size": self.history_size,
            "cache_size": self.cache_size,
            **{field: getattr(self, field) for field in self._get_feature_fields()},
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SwarmConfig":
        """Deserialize the swarm configuration from a dictionary."""
        feature_fields = cls._get_feature_fields()
        feature_metrics = {
            field: config_dict.get(field, 0.0) for field in feature_fields
        }
        return cls(
            size=config_dict.get("size", 10),
            min_agent_count=config_dict.get("min_agent_count", 6),
            beam_width=config_dict.get("beam_width", 5),
            window_size=config_dict.get("window_size", 10),
            stride=config_dict.get("stride", 5),
            individual_weight=config_dict.get("individual_weight", 0.7),
            type_weight=config_dict.get("type_weight", 0.3),
            momentum=config_dict.get("momentum", 0.8),
            feature_threshold=config_dict.get("feature_threshold", 0.1),
            interest_threshold=config_dict.get("interest_threshold", 0.5),
            history_size=config_dict.get("history_size", 10),
            cache_size=config_dict.get("cache_size", 1000),
            **feature_metrics,
        )


class GeneticSwarm:
    """An expanded swarm intelligence system for genetic sequence analysis using HDC space navigation."""

    def __init__(
        self, supercomputer: "HDCVectorSpace", config: Optional[SwarmConfig] = None
    ):
        self.sc = supercomputer
        self.xp = supercomputer.xp
        self.config = config or SwarmConfig()

        # Core swarm components
        self.agents: List[GeneticAgent] = []
        self.positions: List[Any] = []  # Agent positions in HDC space
        self.past_updates: List[Any] = []  # Past movement vectors for sign matching
        self.local_fields: List[Any] = []  # Local sequence field each agent senses
        self.specializations: Dict[
            int, str
        ] = {}  # Track what each agent specializes in

        # Adaptive weights
        self.weights = {}  # Dynamic agent influence weights
        self.success_rates = defaultdict(list)  # Track prediction success by region

        # Global swarm state
        self.attractor = None  # Current sequence attractor state
        self.epoch = 0
        self.convergence = 0.0

        # Initialize the motif discovery system
        self.motif_discovery = HDCMotifDiscovery(
            self.sc, projection_dim=128, similarity_threshold=0.82
        )

    def initialize_swarm(self, sequences: Optional[List[str]] = None) -> None:
        """Create a diverse swarm with different specializations."""
        distribution = (
            self._analyze_distribution(sequences)
            if sequences
            else self._default_distribution()
        )

        for agent_type, count in distribution.items():
            for _ in range(count):
                # Create agent with its initial HDC space position
                agent = GeneticAgent(self.sc, agent_type=agent_type)
                initial_pos = self.sc.hdc._random_vector()

                # Initialize agent's local sensing field
                field_size = self.config.window_size
                local_field = self.xp.zeros((field_size, self.sc.hdc.dim))

                self.agents.append(agent)
                self.positions.append(initial_pos)
                self.past_updates.append(self.xp.zeros_like(initial_pos))
                self.local_fields.append(local_field)
                self.weights[len(self.agents) - 1] = 1.0 / len(self.agents)
                self.specializations[len(self.agents) - 1] = str(agent_type)

        if sequences:
            self._adapt_to_sequences(sequences[:5])  # Quick initial adaptation

    def _swarm_step(self, target_vector: Any, epsilon: float = 0.1) -> None:
        """Execute one step of swarm movement in HDC space."""
        sign_matches = []
        new_positions = []
        new_updates = []

        # Calculate repulsion between agents
        repulsions = defaultdict(list)
        for i, pos_i in enumerate(self.positions):
            for j, pos_j in enumerate(self.positions):
                if i != j:
                    dist = float(self.xp.dot(pos_i, pos_j))
                    if dist > 0.8:  # Too close
                        repulsions[i].append((pos_i - pos_j) * 0.1)

        # Update each agent's position
        for i, (pos, past_update) in enumerate(zip(self.positions, self.past_updates)):
            # Get current gradient to target
            grad = target_vector - pos

            # LION-style sign matching
            sign_match = self.xp.sign(grad) == self.xp.sign(past_update)
            sign_matches.append(float(self.xp.mean(sign_match)))

            # Compute update with sign-based confidence
            base_update = self.xp.where(
                sign_match,
                self.xp.sign(grad) * 0.1,  # Confident step
                self.xp.sign(grad) * 0.02,  # Cautious step
            )

            # Add repulsion and local field influence
            repulsion = sum(repulsions[i]) if i in repulsions else 0
            field_influence = self.xp.mean(self.local_fields[i], axis=0) * 0.05

            # Combine all influences
            update = base_update + repulsion + field_influence
            update = self.xp.clip(update, -0.15, 0.15)  # Limit step size

            # Update position and history
            new_pos = pos + update
            new_positions.append(self.sc.hdc._normalize(new_pos))
            new_updates.append(grad)

            # Update agent's local field
            self.local_fields[i] = self.xp.roll(self.local_fields[i], -1, axis=0)
            self.local_fields[i][-1] = target_vector

        # Update swarm state
        self.positions = new_positions
        self.past_updates = new_updates
        self.convergence = float(self.xp.mean(sign_matches))

        # Adapt weights based on position quality
        self._adapt_weights()

    def _adapt_weights(self) -> None:
        """Dynamically adjust agent influence based on position quality and specialization."""
        total_influence = 0

        # Calculate each agent's influence based on position and track record
        for i, pos in enumerate(self.positions):
            # How well this agent aligns with current sequence context
            alignment = float(self.xp.dot(pos, self.attractor))

            # Boost influence if agent has consistent prediction success
            success_history = (
                len(self.success_rates[i]) / 100 if self.success_rates[i] else 0
            )

            # Combine into single influence score
            self.weights[i] = alignment * (1 + success_history)
            total_influence += self.weights[i]

        # Normalize into probability distribution
        if total_influence > 0:
            self.weights = {i: w / total_influence for i, w in self.weights.items()}

    def predict_sequence(self, context: str, length: int, epsilon: float = 0.05) -> str:
        """Generate sequence using collective swarm intelligence."""
        state = self.sc.encode_sequence(context)
        self.attractor = state  # Set global attractor
        result = []

        for _ in range(length):
            # Move swarm to current state
            self._swarm_step(state, epsilon)

            # Get weighted predictions from all agents
            predictions = defaultdict(float)
            for i, agent in enumerate(self.agents):
                # Each agent predicts based on its local view
                base = agent.predict_base(self.positions[i], epsilon)

                # Weight by position quality and historic success
                weight = self.weights[i] * (1 + len(self.success_rates[i]) / 100)
                predictions[base] += weight

            # Select best base and update state
            next_base = max(predictions.items(), key=lambda x: x[1])[0]
            result.append(next_base)

            # Update swarm's global state
            kmer = "".join(result[-7:])
            next_vec = self.sc.encode_kmer(kmer)
            state = self.sc.bundle(state, next_vec, alpha=0.7)
            self.attractor = state

        return "".join(result)

    def train(self, sequences: List[str], epochs: int = 5) -> Dict:
        """Train swarm through sequence exploration and motif discovery."""
        history = []

        for epoch in range(epochs):
            epoch_reward = 0
            for seq in sequences:
                if len(seq) < 30:
                    continue

                # Initialize sequence state
                state = self.sc.encode_sequence(seq[:10])
                self.attractor = state

                # Process sequence in chunks
                for i in range(10, len(seq) - 5, 5):
                    target = seq[i : i + 5]
                    target_vec = self.sc.encode_sequence(target)

                    # Move swarm
                    self._swarm_step(target_vec)

                    # Train agents
                    chunk_reward = 0
                    for j, agent in enumerate(self.agents):
                        reward = agent.train_on_segment(self.positions[j], target)
                        chunk_reward += reward * self.weights[j]

                        # Track successful predictions
                        if reward > 0.5:
                            self.success_rates[j].append(1)
                        if len(self.success_rates[j]) > 100:
                            self.success_rates[j] = self.success_rates[j][-100:]

                    epoch_reward += chunk_reward
                    state = self.sc.encode_sequence(seq[i : i + 10])
                    self.attractor = state

            history.append(epoch_reward)
            self.epoch += 1

        return {"rewards": history, "convergence": self.convergence}


@dataclass
class HDCGenomicUIConfig:
    """Configuration for HDCGenomicUI with defaults optimized for user experience."""

    # Core HDC parameters
    dimension: int = 4096
    device: str = "auto"  # "auto", "gpu", or "cpu"
    max_kmer_size: int = 12
    bundle_decay: float = 0.85

    # UI parameters remain unchanged
    theme: str = "default"
    enable_gpu_warning: bool = True
    max_sequence_length: int = 100000
    max_upload_size_mb: int = 10

    # Visualization parameters remain unchanged
    default_plot_width: int = 10
    default_plot_height: int = 8
    color_palette: str = "viridis"
    max_display_sequences: int = 20

    # Analysis defaults remain unchanged
    default_kmer_size: int = 0  # 0 means auto-detect
    default_motif_length: int = 6
    default_similarity_threshold: float = 0.8
    default_gap_length: int = 10
    default_variant_impact_threshold: float = 0.5
    default_epigenetics_weight: float = 1.0

    # Task-Specific Parameters - RENAMED for consistency with GSCConfig
    motif_detection_enabled: bool = True
    variant_analysis_enabled: bool = True
    sequence_imputation_enabled: bool = True
    epigenetic_integration_enabled: bool = True

    motif_detection_threshold: float = 0.8
    motif_window_size: int = 10
    variant_analysis_threshold: float = 0.75
    epigenetic_integration_mode: str = "combined"

    # Advanced options remain unchanged
    cache_size: Optional[int] = None
    vector_type: str = "bipolar"
    chunk_size: Optional[int] = None

    # Expert options remain unchanged
    advanced_mode: bool = False
    log_level: str = "INFO"

    def to_gsc_config(self) -> Dict[str, Any]:
        """Convert UI config to GSCConfig parameter dict"""
        return {
            # Core parameters
            "dimension": self.dimension,
            "device": self.device,
            "max_kmer_size": self.max_kmer_size,
            "bundle_decay": self.bundle_decay,
            "vector_type": self.vector_type,
            "cache_size": self.cache_size,
            "chunk_size": self.chunk_size,
            # Task-specific parameters - ADDED all task flags
            "motif_detection_enabled": self.motif_detection_enabled,
            "variant_analysis_enabled": self.variant_analysis_enabled,
            "sequence_imputation_enabled": self.sequence_imputation_enabled,
            "epigenetic_integration_enabled": self.epigenetic_integration_enabled,
            # Task-specific thresholds
            "motif_detection_threshold": self.motif_detection_threshold,
            "motif_window_size": self.motif_window_size,
            "variant_analysis_threshold": self.variant_analysis_threshold,
            "epigenetic_integration_mode": self.epigenetic_integration_mode,
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class HDCGenomicUI:
    """
    HDC Genomic Supercomputer UI that provides an interface for analyzing genomic sequences
    using Hyperdimensional Computing. This implementation uses a feature registry to manage
    feature-specific tabs and functions.
    """

    def __init__(self, config=None):
        """
        Initialize the UI with configuration and computation engines.

        Args:
            config: Optional HDCGenomicUIConfig instance. If None, a default is created.
        """
        # Use provided config or create default one
        self.ui_config = config or HDCGenomicUIConfig()

        # Feature registry maps feature flags to tabs and functions
        self.feature_registry = {
            "motif_detection_enabled": {
                "tab_name": "Motif Discovery",
                "functions": ["discover_motifs"],
                "config_param": "motif_detection_threshold",
                "ui_display_name": "Motif Discovery",
            },
            "variant_analysis_enabled": {
                "tab_name": "Variant Analysis",
                "functions": ["analyze_variant"],
                "config_param": "variant_analysis_threshold",
                "ui_display_name": "Variant Analysis",
            },
            "sequence_imputation_enabled": {
                "tab_name": "Impute Sequence",
                "functions": ["impute_sequence", "_impute_hdc", "_impute_rl"],
                "config_param": "sequence_imputation_enabled",
                "ui_display_name": "Sequence Imputation",
            },
            "epigenetic_integration_enabled": {
                "tab_name": "Epigenetic Analysis",
                "functions": ["analyze_epigenetics"],
                "config_param": "epigenetic_integration_mode",
                "ui_display_name": "Epigenetic Integration",
            },
        }

        # Initialize GSCConfig using parameters from UI config
        self.config = GSCConfig(**self.ui_config.to_gsc_config())

        # Create the DNA supercomputer instance
        self.sc = DNASupercomputer(self.config)
        self.bio_encoder = BiologicalEncoder(self.config)

        # Keep track of the interface instance for rebuilding
        self.interface = None
        self.needs_rebuild = False
        self.logger = logging.getLogger("HDCGenomicUI")

    def _build_tab_if_enabled(self, tab_name, feature_flag, tab_builder_fn):
        """
        Build a tab if the corresponding feature is enabled.

        Args:
            tab_name: The name of the tab
            feature_flag: The feature flag to check
            tab_builder_fn: Function to build the tab content

        Returns:
            The tab content if enabled, None otherwise
        """
        if getattr(self.ui_config, feature_flag, False):
            return tab_builder_fn()
        return None

    def is_feature_enabled(self, feature_flag):
        """
        Check if a feature is enabled in the current configuration.

        Args:
            feature_flag: The feature flag to check

        Returns:
            Boolean indicating if the feature is enabled
        """
        return getattr(self.ui_config, feature_flag, False)

    def process_sequence_input(self, input_text, file_obj):
        """
        Handle input from either text area or file upload.

        Args:
            input_text: Text input from the text area
            file_obj: Uploaded file object

        Returns:
            Tuple of (sequences, sequence_ids)
        """
        if file_obj is not None:
            # Process uploaded file (FASTA/FASTQ)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                tmp.write(file_obj)
                tmp_name = tmp.name

            sequences, seq_ids = self.sc.data_loader.load_sequences(tmp_name)
            os.unlink(tmp_name)  # Clean up temp file
            return sequences, seq_ids

        elif input_text.strip():
            # Check sequence length limit
            if len(input_text) > self.ui_config.max_sequence_length:
                return [], ["Error: Sequence exceeds maximum length limit"]

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
        """
        Encode sequences and visualize the vector space.

        Args:
            input_text: Text input from the text area
            file_obj: Uploaded file object
            kmer_size: The k-mer size to use (0 for auto-detect)

        Returns:
            Tuple of (stats_text, visualization_image, info_text)
        """
        sequences, seq_ids = self.process_sequence_input(input_text, file_obj)

        if not sequences:
            return "No sequences provided", None, None

        # Check for error message in seq_ids
        if seq_ids and seq_ids[0].startswith("Error:"):
            return seq_ids[0], None, None

        # Encode sequences
        k = (
            int(kmer_size)
            if kmer_size
            else Utils.detect_optimal_kmer_size(sequences[0])
        )
        vectors = [self.sc.encode_sequence(seq, k=k) for seq in sequences]

        # Create visualization of vector space
        if len(vectors) > 1:
            # Convert to numpy arrays if on GPU
            if self.sc.hdc.device == "gpu":
                vectors_np = [v.get() for v in vectors]
            else:
                vectors_np = vectors

            # Compute pairwise similarities
            n_seqs = len(sequences)
            sim_matrix = np.zeros((n_seqs, n_seqs))
            for i in range(n_seqs):
                for j in range(i, n_seqs):
                    sim = float(self.sc.hdc.similarity(vectors_np[i], vectors_np[j]))
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            # Create heatmap
            plt.figure(
                figsize=(
                    self.ui_config.default_plot_width,
                    self.ui_config.default_plot_height,
                )
            )
            plt.imshow(sim_matrix, cmap=self.ui_config.color_palette)
            plt.colorbar(label="Similarity")
            plt.title(f"Sequence Similarity Matrix (k={k})")

            # Add sequence IDs as labels if not too many
            if n_seqs <= self.ui_config.max_display_sequences:
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
            if motif_positions and self.is_feature_enabled("motif_detection_enabled"):
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

    def requires_feature(feature_flag):
        """
        Decorator to check if a feature is enabled before running a function.

        Args:
            feature_flag: The feature flag to check

        Returns:
            Function decorator
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if not self.is_feature_enabled(feature_flag):
                    feature_name = self.feature_registry.get(feature_flag, {}).get(
                        "ui_display_name", feature_flag
                    )
                    return (
                        f"The {feature_name} feature is disabled in the current configuration.",
                        None,
                    )
                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    @requires_feature("sequence_imputation_enabled")
    def impute_sequence(self, prefix, suffix, gap_length, strategy="hdc"):
        """
        Fill in missing sequence between prefix and suffix.

        Args:
            prefix: Sequence before the gap
            suffix: Sequence after the gap
            gap_length: Length of the gap to impute
            strategy: Imputation strategy (hdc or rl)

        Returns:
            Imputation results text
        """
        if not prefix and not suffix:
            return "Need at least prefix or suffix to impute sequence"

        if strategy == "hdc":
            # Use HDC-based imputation
            return self._impute_hdc(prefix, suffix, gap_length)
        elif strategy == "rl":
            # Use reinforcement learning-based imputation
            return self._impute_rl(prefix, suffix, gap_length)
        else:
            return "Invalid imputation strategy"

    def _impute_hdc(self, prefix, suffix, gap_length):
        """
        HDC-based imputation.

        Args:
            prefix: Sequence before the gap
            suffix: Sequence after the gap
            gap_length: Length of the gap to impute

        Returns:
            Imputation results text
        """
        prefix_vec = self.sc.encode_sequence(prefix) if prefix else None
        suffix_vec = self.sc.encode_sequence(suffix) if suffix else None

        if prefix and suffix:
            gap_length = (
                int(gap_length) if gap_length else self.ui_config.default_gap_length
            )
            candidates = []
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                result = []
                state = prefix_vec
                for _ in range(gap_length):
                    scores = {}
                    for base in "ACGT":
                        base_vec = self.sc.hdc.base_vectors[base]
                        bound = self.sc.hdc.bind(state, base_vec)
                        if suffix_vec is not None:
                            scores[base] = float(
                                self.sc.hdc.similarity(bound, suffix_vec)
                            )
                        else:
                            scores[base] = float(self.sc.hdc.similarity(bound, state))
                    next_base = max(scores.items(), key=lambda x: x[1])[0]
                    result.append(next_base)
                    next_vec = self.sc.hdc.encode_kmer(next_base)
                    state = self.sc.hdc.bundle(state, next_vec, alpha=alpha)
                candidates.append(("".join(result), alpha))

            # Score candidates
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

    def _impute_rl(self, prefix, suffix, gap_length):
        """
        Reinforcement learning-based imputation.

        Args:
            prefix: Sequence before the gap
            suffix: Sequence after the gap
            gap_length: Length of the gap to impute

        Returns:
            Imputation results text
        """
        # Implementation would go here
        return "RL-based imputation not implemented in this version"

    @requires_feature("motif_detection_enabled")
    def discover_motifs(self, sequence, motif_length, similarity_threshold):
        """
        Discover motifs in a DNA sequence.

        Args:
            sequence: DNA sequence to analyze
            motif_length: Length of motifs to discover
            similarity_threshold: Similarity threshold for motif discovery

        Returns:
            Tuple of (result_text, visualization_image)
        """
        if not sequence:
            return "No sequence provided", None

        motif_discovery = HDCMotifDiscovery(self.sc.hdc)
        motifs = motif_discovery.discover_motifs(
            [sequence], k_range=(motif_length, motif_length)
        )

        # Format results
        result_text = "Discovered Motifs:\n"
        for motif, data in motifs.items():
            result_text += (
                f"- {motif}: Score={data['weight']:.3f}, Family={data['family']}\n"
            )

        # Visualize motifs
        plt.figure(figsize=(12, 3))
        for motif, data in motifs.items():
            positions = data.get("positions", {}).get(0, [])
            if positions:
                plt.scatter(positions, [data["weight"]] * len(positions), label=motif)
        plt.xlabel("Position")
        plt.ylabel("Motif Score")
        plt.title("Discovered Motifs")
        plt.legend()

        motif_path = tempfile.mktemp(suffix=".png")
        plt.tight_layout()
        plt.savefig(motif_path)
        plt.close()

        return result_text, Image.open(motif_path)

    @requires_feature("variant_analysis_enabled")
    def analyze_variant(self, sequence, variant):
        """
        Analyze the impact of a genetic variant.

        Args:
            sequence: Original DNA sequence
            variant: Variant sequence

        Returns:
            Tuple of (result_text, visualization_image)
        """
        if not sequence or not variant:
            return "No sequence or variant provided", None

        original_vec = self.sc.encode_sequence(sequence)
        variant_vec = self.sc.encode_sequence(variant)

        similarity = float(self.sc.hdc.similarity(original_vec, variant_vec))
        impact_score = 1 - similarity

        # Visualize the impact
        plt.figure(figsize=(8, 4))
        plt.bar(["Original", "Variant"], [1.0, similarity])
        plt.ylabel("Similarity")
        plt.title(f"Variant Impact: {impact_score:.3f}")

        variant_path = tempfile.mktemp(suffix=".png")
        plt.tight_layout()
        plt.savefig(variant_path)
        plt.close()

        return f"Variant Impact Score: {impact_score:.3f}", Image.open(variant_path)

    def compare_sequences(self, sequences):
        """
        Compare multiple sequences and identify conserved regions.

        Args:
            sequences: List of DNA sequences

        Returns:
            Tuple of (result_text, visualization_image)
        """
        if not sequences:
            return "No sequences provided", None

        vectors = [self.sc.encode_sequence(seq) for seq in sequences]

        n_seqs = len(sequences)
        sim_matrix = np.zeros((n_seqs, n_seqs))
        for i in range(n_seqs):
            for j in range(i, n_seqs):
                sim_matrix[i, j] = float(self.sc.hdc.similarity(vectors[i], vectors[j]))
                sim_matrix[j, i] = sim_matrix[i, j]

        # Visualize similarity matrix
        plt.figure(
            figsize=(
                self.ui_config.default_plot_width,
                self.ui_config.default_plot_height,
            )
        )
        plt.imshow(sim_matrix, cmap=self.ui_config.color_palette)
        plt.colorbar(label="Similarity")
        plt.title("Sequence Similarity Matrix")

        comparison_path = tempfile.mktemp(suffix=".png")
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()

        return "Sequence comparison completed", Image.open(comparison_path)

    @requires_feature("epigenetic_integration_enabled")
    def analyze_epigenetics(self, sequence, epigenetic_file):
        """
        Analyze epigenetic data for a sequence.

        Args:
            sequence: DNA sequence to analyze
            epigenetic_file: Epigenetic data file

        Returns:
            Tuple of (result_text, visualization_image)
        """
        if not sequence or not epigenetic_file:
            return "No sequence or epigenetic data provided", None

        epigenetic_data = self.sc.data_loader.load_epigenetic_file(epigenetic_file)

        sequence_vec = self.sc.encode_sequence(sequence)
        epigenetic_vec = self.sc.encode_sequence(epigenetic_data.get("sequence", ""))

        similarity = float(self.sc.hdc.similarity(sequence_vec, epigenetic_vec))

        # Visualize epigenetic marks
        plt.figure(figsize=(12, 3))
        plt.plot(epigenetic_data.get("scores", []))
        plt.xlabel("Position")
        plt.ylabel("Epigenetic Score")
        plt.title("Epigenetic Marks")

        epigenetic_path = tempfile.mktemp(suffix=".png")
        plt.tight_layout()
        plt.savefig(epigenetic_path)
        plt.close()

        return f"Epigenetic Similarity: {similarity:.3f}", Image.open(epigenetic_path)

    def update_configuration(self, **kwargs):
        """
        Update the UI configuration with provided parameters.

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            Status message about the configuration update
        """
        try:
            # Store original values to detect changes
            original_config = {
                k: getattr(self.ui_config, k)
                for k in kwargs.keys()
                if hasattr(self.ui_config, k)
            }

            # Update UI config values
            for key, value in kwargs.items():
                if hasattr(self.ui_config, key):
                    setattr(self.ui_config, key, value)

            # Normalize feature flag names if necessary
            # Map legacy flag names to new standardized names
            feature_flag_mapping = {
                "enable_motif_discovery": "motif_detection_enabled",
                "enable_variant_analysis": "variant_analysis_enabled",
                "enable_imputation": "sequence_imputation_enabled",
                "enable_epigenetics": "epigenetic_integration_enabled",
            }

            # Synchronize legacy flags with new flag names
            for old_flag, new_flag in feature_flag_mapping.items():
                if (
                    old_flag in kwargs
                    and hasattr(self.ui_config, old_flag)
                    and hasattr(self.ui_config, new_flag)
                ):
                    setattr(self.ui_config, new_flag, getattr(self.ui_config, old_flag))

            # Update GSCConfig using the new method
            core_params_changed = self.config.update_from_ui_config(self.ui_config)

            # Feature flags that were toggled
            toggled_features = []
            for feature in self.feature_registry.keys():
                if feature in kwargs and getattr(
                    self.ui_config, feature
                ) != original_config.get(feature):
                    toggled_features.append(
                        self.feature_registry[feature]["ui_display_name"]
                    )

                # Also check legacy flags
                legacy_flag = next(
                    (k for k, v in feature_flag_mapping.items() if v == feature), None
                )
                if (
                    legacy_flag
                    and legacy_flag in kwargs
                    and getattr(self.ui_config, feature)
                    != original_config.get(legacy_flag)
                ):
                    toggled_features.append(
                        self.feature_registry[feature]["ui_display_name"]
                    )

            # Remove duplicates
            toggled_features = list(set(toggled_features))

            if core_params_changed:
                # Reinitialize computational engines with new config
                self.sc = DNASupercomputer(self.config)
                self.bio_encoder = BiologicalEncoder(self.config)

                # Flag the interface for complete rebuild
                self.needs_rebuild = True
                return "Core configuration updated. Please refresh the page to apply changes."
            elif toggled_features:
                # Feature toggles changed but no core params
                feature_list = ", ".join(toggled_features)
                return f"Features updated: {feature_list}. Changes will apply to new operations."
            else:
                # UI-only changes
                return "Configuration updated successfully."

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return f"Error updating configuration: {str(e)}"

    def _build_encode_analyze_tab(self):
        """Build the Encode & Analyze tab."""
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
                        value=self.ui_config.default_kmer_size,
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

    def _build_imputation_tab(self):
        """Build the Impute Sequence tab."""
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
                        value=self.ui_config.default_gap_length,
                    )
                    strategy = gr.Radio(
                        label="Imputation Strategy",
                        choices=["hdc", "rl"],
                        value="hdc",
                    )
                    impute_btn = gr.Button("Impute Missing Sequence")

                with gr.Column():
                    impute_output = gr.Textbox(label="Imputation Results", lines=15)

            impute_btn.click(
                fn=self.impute_sequence,
                inputs=[prefix_input, suffix_input, gap_length, strategy],
                outputs=[impute_output],
            )

    def _build_motif_discovery_tab(self):
        """Build the Motif Discovery tab."""
        with gr.Tab("Motif Discovery"):
            with gr.Row():
                with gr.Column():
                    motif_sequence_input = gr.Textbox(
                        label="Enter DNA sequence for motif discovery",
                        placeholder="ATGCAAGTGCAATATTACGA...",
                        lines=5,
                    )
                    motif_length = gr.Slider(
                        label="Motif Length",
                        minimum=4,
                        maximum=20,
                        step=1,
                        value=self.ui_config.default_motif_length,
                    )
                    motif_threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.5,
                        maximum=1.0,
                        step=0.05,
                        value=self.ui_config.default_similarity_threshold,
                    )
                    discover_btn = gr.Button("Discover Motifs")

                with gr.Column():
                    motif_output = gr.Textbox(label="Motif Discovery Results", lines=10)
                    motif_viz = gr.Image(label="Motif Visualization")

            discover_btn.click(
                fn=self.discover_motifs,
                inputs=[motif_sequence_input, motif_length, motif_threshold],
                outputs=[motif_output, motif_viz],
            )

    def _build_variant_analysis_tab(self):
        """Build the Variant Analysis tab."""
        with gr.Tab("Variant Analysis"):
            with gr.Row():
                with gr.Column():
                    variant_sequence_input = gr.Textbox(
                        label="Enter DNA sequence",
                        placeholder="ATGCAAGTGCAATATTACGA...",
                        lines=5,
                    )
                    variant_input = gr.Textbox(
                        label="Enter variant (e.g., A10G)",
                        placeholder="A10G",
                    )
                    analyze_variant_btn = gr.Button("Analyze Variant")

                with gr.Column():
                    variant_output = gr.Textbox(
                        label="Variant Analysis Results", lines=10
                    )
                    variant_viz = gr.Image(label="Variant Impact Visualization")

            analyze_variant_btn.click(
                fn=self.analyze_variant,
                inputs=[variant_sequence_input, variant_input],
                outputs=[variant_output, variant_viz],
            )

    def _build_comparative_genomics_tab(self):
        """Build the Comparative Genomics tab."""
        with gr.Tab("Comparative Genomics"):
            with gr.Row():
                with gr.Column():
                    comparison_input = gr.Textbox(
                        label="Enter sequences (one per line)",
                        placeholder="Sequence 1\nSequence 2\n...",
                        lines=10,
                    )
                    compare_btn = gr.Button("Compare Sequences")

                with gr.Column():
                    comparison_output = gr.Textbox(label="Comparison Results", lines=10)
                    comparison_viz = gr.Image(label="Similarity Matrix")

            compare_btn.click(
                fn=self.compare_sequences,
                inputs=[comparison_input],
                outputs=[comparison_output, comparison_viz],
            )

    def _build_epigenetics_tab(self):
        """Build the Epigenetic Analysis tab."""
        with gr.Tab("Epigenetic Analysis"):
            with gr.Row():
                with gr.Column():
                    epigenetic_sequence_input = gr.Textbox(
                        label="Enter DNA sequence",
                        placeholder="ATGCAAGTGCAATATTACGA...",
                        lines=5,
                    )
                    epigenetic_file_input = gr.File(
                        label="Upload Epigenetic Data (BED/BigWig)"
                    )
                    analyze_epigenetic_btn = gr.Button("Analyze Epigenetics")

                with gr.Column():
                    epigenetic_output = gr.Textbox(
                        label="Epigenetic Analysis Results", lines=10
                    )
                    epigenetic_viz = gr.Image(label="Epigenetic Marks Visualization")

            analyze_epigenetic_btn.click(
                fn=self.analyze_epigenetics,
                inputs=[epigenetic_sequence_input, epigenetic_file_input],
                outputs=[epigenetic_output, epigenetic_viz],
            )

    def _build_about_tab(self):
        """Build the About tab."""
        with gr.Tab("About"):
            gr.Markdown(
                f"""
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
                - Variant impact analysis
                - Comparative genomics
                - Epigenetic data integration

                ## Current Configuration
                - Vector dimension: {self.config.dimension}
                - Computing device: {self.config.device}
                - Max k-mer size: {self.config.max_kmer_size}
                - Color palette: {self.ui_config.color_palette}
                {"- Running in advanced mode" if self.ui_config.advanced_mode else ""}
                """
            )

    def _build_configuration_tab(self):
        """Build the Configuration tab."""
        with gr.Tab("Configuration"):
            gr.Markdown("## Advanced Configuration")
            gr.Markdown("Changes to dimension or device require restart.")

            with gr.Row():
                with gr.Column():
                    dimension_slider = gr.Slider(
                        label="Vector Dimension",
                        minimum=1024,
                        maximum=16384,
                        step=1024,
                        value=self.ui_config.dimension,
                    )
                    device_radio = gr.Radio(
                        label="Compute Device",
                        choices=["auto", "gpu", "cpu"],
                        value=self.ui_config.device,
                    )
                    color_dropdown = gr.Dropdown(
                        label="Color Palette",
                        choices=[
                            "viridis",
                            "plasma",
                            "inferno",
                            "magma",
                            "cividis",
                        ],
                        value=self.ui_config.color_palette,
                    )

                with gr.Column():
                    # Use consistent names for feature toggles
                    enable_motif = gr.Checkbox(
                        label="Enable Motif Discovery",
                        value=self.ui_config.motif_detection_enabled,
                    )
                    enable_variant = gr.Checkbox(
                        label="Enable Variant Analysis",
                        value=self.ui_config.variant_analysis_enabled,
                    )
                    enable_imputation = gr.Checkbox(
                        label="Enable Imputation",
                        value=self.ui_config.sequence_imputation_enabled,
                    )
                    enable_epigenetics = gr.Checkbox(
                        label="Enable Epigenetics",
                        value=self.ui_config.epigenetic_integration_enabled,
                    )

            apply_btn = gr.Button("Apply Configuration")
            config_output = gr.Textbox(label="Configuration Status", value="")

            # Connect the apply button to the configuration update function
            apply_btn.click(
                fn=self.update_configuration,
                inputs={
                    dimension_slider: "dimension",
                    device_radio: "device",
                    color_dropdown: "color_palette",
                    enable_motif: "motif_detection_enabled",
                    enable_variant: "variant_analysis_enabled",
                    enable_imputation: "sequence_imputation_enabled",
                    enable_epigenetics: "epigenetic_integration_enabled",
                },
                outputs=[config_output],
            )

            # Add a refresh button for applying core changes
            refresh_btn = gr.Button("Refresh Interface")
            refresh_btn.click(fn=lambda: None, inputs=[], outputs=[]).then(
                fn=lambda: gr.Markdown("Refreshing..."),
                inputs=[],
                outputs=[config_output],
            ).then(
                fn=lambda: None,
                _js="() => { window.location.reload(); }",
                inputs=[],
                outputs=[],
            )

    def build_interface(self):
        """
        Create the Gradio interface with dynamic tabs based on enabled features.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="HDC Genomic Supercomputer") as app:
            gr.Markdown("# 🧬 HDC Genomic Supercomputer")
            gr.Markdown("Analyze DNA sequences using Hyperdimensional Computing")

            # Always include the Encode & Analyze tab
            self._build_encode_analyze_tab()

            # Build feature-specific tabs if enabled
            if self.is_feature_enabled("sequence_imputation_enabled"):
                self._build_imputation_tab()

            if self.is_feature_enabled("motif_detection_enabled"):
                self._build_motif_discovery_tab()

            if self.is_feature_enabled("variant_analysis_enabled"):
                self._build_variant_analysis_tab()

            # Always include Comparative Genomics
            self._build_comparative_genomics_tab()

            if self.is_feature_enabled("epigenetic_integration_enabled"):
                self._build_epigenetics_tab()

            # About tab
            self._build_about_tab()

            # Configuration tab if in advanced mode
            if self.ui_config.advanced_mode:
                self._build_configuration_tab()

            self.interface = app
            return app

    def save_configuration(self, filename="hdc_genomic_config.json"):
        """
        Save current configuration to a JSON file.

        Args:
            filename: Path to save the configuration

        Returns:
            Status message
        """
        import json

        config_dict = {
            k: v for k, v in self.ui_config.__dict__.items() if not k.startswith("_")
        }

        try:
            with open(filename, "w") as f:
                json.dump(config_dict, f, indent=2)
            return f"Configuration saved to {filename}"
        except Exception as e:
            return f"Error saving configuration: {str(e)}"

    def load_configuration(self, filename="hdc_genomic_config.json"):
        """
        Load configuration from a JSON file.

        Args:
            filename: Path to load the configuration from

        Returns:
            Status message
        """
        import json

        try:
            with open(filename, "r") as f:
                config_dict = json.load(f)

            # Update configuration
            for k, v in config_dict.items():
                if hasattr(self.ui_config, k):
                    setattr(self.ui_config, k, v)

            # Recreate core objects
            self.config = GSCConfig(**self.ui_config.to_gsc_config())
            self.sc = DNASupercomputer(self.config)
            self.bio_encoder = BiologicalEncoder(self.config)

            # Flag for rebuild
            self.needs_rebuild = True

            return f"Configuration loaded from {filename}"
        except Exception as e:
            return f"Error loading configuration: {str(e)}"

    def launch(self, share=False, server_name="0.0.0.0", server_port=None):
        """
        Launch the interface with the specified settings.

        Args:
            share: Whether to create a public link
            server_name: Server name to listen on
            server_port: Port to listen on
        """
        app = self.build_interface()
        app.launch(share=share, server_name=server_name, server_port=server_port)


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
    # --- Configure Logging ---
    logger = configure_logging()

    # --- Argument Parsing (Unified) ---
    parser = argparse.ArgumentParser(
        description="HDC Genomic Supercomputer - Unified CLI and UI Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # UI Mode Arguments (with defaults that match HDCGenomicUIConfig)
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio UI")
    parser.add_argument("--config", type=str, help="Path to UI configuration file")
    parser.add_argument(
        "--ui_dimension", type=int, help="UI: HDC vector dimension (overrides config)"
    )
    parser.add_argument(
        "--ui_device",
        choices=["auto", "gpu", "cpu"],
        help="UI: Compute device (overrides config)",
    )
    parser.add_argument(
        "--advanced", action="store_true", help="UI: Enable advanced mode"
    )
    parser.add_argument("--share", action="store_true", help="UI: Share the interface")

    # CLI Mode Arguments (keep your existing CLI arguments)
    parser.add_argument("--input", type=str, help="Input FASTA/FASTQ file")
    parser.add_argument("--output", type=str, help="Output file/directory")
    parser.add_argument("--dim", type=int, help="HDC vector dimension")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "gpu", "cpu"],
        help="Computation device",
    )
    parser.add_argument("--model", type=str, help="Path to saved model file")
    parser.add_argument(
        "--db_path", type=str, default="genetic_data.db", help="Path to the SQLite database file"
    )

    # Add subparsers for CLI modes
    subparsers = parser.add_subparsers(dest="mode", help="CLI operation mode")

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
    # --- Parse Arguments ---
    # Use parse_known_args to handle UI-specific args in CLI mode gracefully
    args, unknown_args = parser.parse_known_args()

    # --- Database Initialization ---
    db_schema = DatabaseSchema(db_path=args.db_path)
    db_schema.connect()  # Connect to the database
    db_schema.create_tables()  # Ensure tables exist

    try:  # Use a try...finally block to ensure database connection is closed
        # --- UI Mode ---
        if args.ui:
            if unknown_args:
                logger.warning(f"Unknown arguments for UI mode: {unknown_args}")

            # Initialize with default UI configuration
            ui_config = HDCGenomicUIConfig()

            # Override UI config with command-line arguments if provided
            if args.ui_dimension:  # Use specific flag names for UI args
                ui_config.dimension = args.ui_dimension
            if args.ui_device:
                ui_config.device = args.ui_device
            if args.advanced:
                ui_config.advanced_mode = True

            # Create UI instance
            ui = HDCGenomicUI(config=ui_config)

            # Load UI configuration file if provided
            if args.config:
                ui.load_configuration(args.config)

            # Launch the interface
            ui.launch(share=args.share)


        # --- CLI Mode ---
        else:
            if unknown_args:
                logger.warning(f"Unknown arguments for CLI mode: {unknown_args}")

            if not args.mode:
                parser.error("Please specify a mode (encode, train, predict, impute, analyze) when not using --ui")
            # Pass the database schema to the run_genetic_analysis function
            run_genetic_analysis(args, db_schema)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")  # Use logger.exception for stack trace

    finally:
        db_schema.close()  # Always close the database connection