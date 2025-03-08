# Genetic Supercomputer: HDC-Powered Genomic Analysis

This project introduces a novel approach to genomic sequence analysis using **Hyperdimensional Computing (HDC)**. It leverages HDC to efficiently encode, analyze, predict, and impute DNA/RNA sequences, providing scalable solutions for large-scale genomic datasets. The system includes key components such as HDC vector spaces, specialized DNA encoders, RL-powered agents, and swarm intelligence for advanced sequence analysis.

## Key Features

- **Hyperdimensional Computing (HDC) Core**: A flexible framework supporting both **bipolar** and **binary** vectors, optimized for **CPU and GPU** computation (via CuPy). Includes efficient vector operations like binding, bundling, permutation, and similarity calculation.
  
- **DNA/RNA Sequence Encoding**:
  - **`DNAEncoder`**: Basic **k-mer** based encoding with adaptive parameter tuning (e.g., k-mer size, stride, chunk size) and sequence statistics tracking.
  - **`BiologicalEncoder`**: Extends `DNAEncoder` to integrate **biological features** like GC content, sequence complexity, known motifs, and optional external data such as **annotations**, **conservation scores**, and **epigenetic data**.
  
- **Reinforcement Learning Agents (`GeneticAgent`)**: RL-based agents designed for **sequence imputation** and **base prediction**. Agents can specialize in tasks like **motif detection**, **structure analysis**, and **conservation tracking**.

- **Swarm Intelligence (`GeneticSwarm`)**: A collaborative system of specialized agents that dynamically adapts its agent distribution and weights based on input data characteristics, making it highly versatile for different sequence analysis tasks.

- **Unified CLI**: A single command-line interface for encoding, training, prediction, imputation, and sequence analysis.

- **Data-Driven Adaptation**: Automatically adjusts parameters (like k-mer size, chunk size, agent distribution) based on sequence characteristics and available computational resources.

- **GPU Acceleration**: Utilizes **CuPy** for **GPU-based computations** when available, with a fallback to **NumPy** for CPU-based operations.

- **Caching**: Extensive caching (e.g., for k-mers, sequences, features) to optimize performance.

- **Extensibility**: Easily extendable to support new biological feature providers, agent types, and analysis methods.

- **File Format Support**: Supports **FASTA** and **FASTQ** formats for sequence input, and various output formats including **HDF5**, **NumPy**, **JSON**, and **CSV**.

- **Meta-Learning**: Includes the `MetaHDConservation` module for learning and predicting **evolutionary conservation patterns**.

## Core Components

- **`HDCVectorSpace`**: The foundation of all HDC operations, handling vector creation, binding, bundling, permutation, similarity calculations, and batch processing.

- **`DNASupercomputer`**: Extends `HDCVectorSpace` to provide DNA-specific functionality, such as base and position vectors, k-mer encoding, and sequence encoding.

- **`BiologicalHDC`**: A further extension of `DNASupercomputer` that integrates biological features like GC content, sequence complexity, and external biological data (e.g., annotations, conservation, motifs).

- **`DNAEncoder`**: Autotuning DNA encoder for HDC-based sequence analysis.

- **`BiologicalEncoder`**: An HDC encoder that integrates biological features such as GC content, sequence complexity, motifs, and other annotations.

- **`GeneticAnalyzer`**: A high-level API that abstracts sequence analysis tasks, including analysis, training, prediction, and imputation.

- **`GeneticAgent`**: An RL agent designed for base prediction and sequence imputation using HDC vectors as state representations.

- **`GeneticSwarm`**: A system of specialized **GeneticAgent** instances that collaborate to solve sequence analysis problems.

- **`MetaHDConservation`**: A meta-learning system that learns conservation patterns across sequences.

## Getting Started

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/genetic-supercomputer.git
   cd genetic-supercomputer
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note for GPU Usage**: If you have a CUDA-enabled GPU, install **CuPy** for optimal performance:
   
   ```bash
   pip install cupy-cuda11x  # Replace 11x with your CUDA version (e.g., 10x, 12x)
   ```

   **Optional**: For BigWig file support, install additional system libraries (for Ubuntu/Debian):

   ```bash
   sudo apt-get install libcurl4-openssl-dev libssl-dev
   ```

### Usage (CLI)

The `genetic_supercomputer.py` script offers a unified CLI for all functionalities.

#### 1. **Encoding Sequences**:
Encode sequences from a FASTA file into HDC vectors and save them to an HDF5 file.

```bash
python genetic_supercomputer.py encode --input sequences.fasta --output encoded_vectors.h5 --dim 10000 --kmer 7
```

#### 2. **Training a Model**:
Train a genetic model (swarm or simple agent) on a set of sequences.

```bash
python genetic_supercomputer.py train --input training_sequences.fasta --output trained_model --epochs 10 --agent-type swarm --swarm-size 15
```

#### 3. **Predicting Sequence Continuations**:
Predict the next 20 bases from a context using a trained model.

```bash
python genetic_supercomputer.py predict --model trained_model.h5 --context "ATGCGTAGCTAGCTAG" --length 20
```

#### 4. **Imputing Missing Segments**:
Impute a gap of length 15, given a prefix and suffix, using a trained model.

```bash
python genetic_supercomputer.py impute --model trained_model.h5 --prefix "ATGCGTAGCTAG" --suffix "GCTAGCTAGCTA" --gap-length 15
```

#### 5. **Analyzing Sequences**:
Analyze sequence properties (GC content, base counts, optimal k-mer size, etc.):

```bash
python genetic_supercomputer.py analyze --input sequences.fasta --output analysis_results
```

Calculate a pairwise similarity matrix:

```bash
python genetic_supercomputer.py analyze --input sequences.fasta --output analysis_results --similarity
```

#### 6. **Help**:
To see all available options for each mode:

```bash
python genetic_supercomputer.py encode --help
python genetic_supercomputer.py train --help
python genetic_supercomputer.py predict --help
python genetic_supercomputer.py impute --help
python genetic_supercomputer.py analyze --help
```

### Example Workflow

1. **Prepare your data**: Create a FASTA file (e.g., `my_sequences.fasta`) containing DNA sequences.

2. **Train a swarm model**:

   ```bash
   python genetic_supercomputer.py train --input my_sequences.fasta --output my_trained_model --epochs 5 --agent-type swarm
   ```

3. **Analyze sequences using the trained model**:

   ```bash
   python genetic_supercomputer.py analyze --input my_sequences.fasta --output my_analysis --model my_trained_model.h5
   ```

4. **Predict a sequence continuation, given a context**:

   ```bash
   python genetic_supercomputer.py predict --model my_trained_model.h5 --context "GCTAGCTAGCTAGCTAGCTAGC" --length 25
   ```

5. **Impute a sequence**:

   ```bash
   python genetic_supercomputer.py impute --model my_trained_model.h5 --prefix "GCTAGCTAGCTAGCTAGCTAGC" --suffix "AGCTAGCTAGCTAGCTAGCTA" --gap-length 25
   ```

### API Usage (Python)

The system is organized into modular classes. Below is an example of using the `DNASupercomputer` and `GeneticSwarm` components directly:

```python
from genetic_supercomputer import DNASupercomputer, GeneticSwarm

# Initialize the supercomputer
supercomputer = DNASupercomputer(dimension=10000, device="cpu")

# Example sequences
sequences = [
    "ATGCGTAGCTAGCTAGCTAGCTAGCTA",
    "GCTAGCTAGCTAGCTAGCTAGCTAGC",
    "TTAGCTAGCTAGCTAGCTAGCTAGCT",
]

# Encode sequences
encoded_vectors = [supercomputer.encode_sequence(seq) for seq in sequences]

# Initialize the swarm
swarm = GeneticSwarm(supercomputer, swarm_size=10)

# Train the swarm
swarm.train(sequences, epochs=3)

# Analyze a sequence
analysis_results = swarm.analyze_sequence("ATGCGTAGCTAGCTAGCTAGCTAGCTA")
print(analysis_results)

# Predict a sequence
prediction = swarm.predict("ATGCGTAGCTAG", length=10)
print(f"Prediction: {prediction}")

# Impute a missing segment
imputation = swarm.impute_segment("ATGCGT", "AGCTAGCTA", gap_length=5)
print(f"Imputation: {imputation}")
```

### Project Structure

- **`genetic_supercomputer.py`**: Core classes and functions:
  - `HDCVectorSpace`: Base HDC class.
  - `DNASupercomputer`: DNA-specific HDC implementation.
  - `BiologicalHDC`: HDC with biological feature integration.
  - `DNAEncoder`: Autotuning DNA encoder.
  - `BiologicalEncoder`: HDC encoder with integrated biological features.
  - `GeneticAgent`: RL agent for prediction/imputation.
  - `GeneticSwarm`: Swarm intelligence system.
  - `MetaHDConservation`: Meta-learning conservation scorer.
  - `AgentType`: Enum for agent specializations.
  - `parse_args()`: CLI argument parsing.
  - `load_sequences()`: FASTA/FASTQ loader.
  - `save_results()`: Result saving (HDF5, NumPy, JSON).
  - `run_genetic_analysis()`: Main function for CLI.

---
