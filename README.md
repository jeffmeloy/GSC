Genetic Supercomputer: HDC-Powered Genomic Analysis

This project implements a novel approach to genomic sequence analysis using Hyperdimensional Computing (HDC). It provides a suite of tools for encoding, analyzing, predicting, and imputing DNA/RNA sequences, leveraging the power of HDC for efficient and scalable computation.  The system includes several key components: a base HDC vector space, specialized DNA encoders (with and without biological feature integration), RL-powered agents, and a swarm intelligence system for advanced sequence analysis.

## Key Features

*   **Hyperdimensional Computing (HDC) Core:** A robust and flexible HDC framework with support for bipolar and binary vectors, optimized for both CPU and GPU (via CuPy). Includes efficient binding, bundling, and permutation operations.
*   **DNA/RNA Sequence Encoding:**
    *   `DNAEncoder`:  Basic k-mer based encoding with adaptive parameter tuning (k-mer size, stride, chunk size) and sequence statistics tracking.
    *   `BiologicalEncoder`:  Extends `DNAEncoder` to integrate biological features like GC content, sequence complexity, known motifs, and optional external data (annotations, conservation scores, epigenetic data).
*   **Reinforcement Learning Agents (`GeneticAgent`):** RL agents trained to predict and impute DNA sequences, using HDC vectors as state representations.  Agents can be specialized for different tasks (motif detection, structure analysis, conservation tracking, etc.).
*   **Swarm Intelligence (`GeneticSwarm`):** A collection of specialized `GeneticAgent`s that collaborate to perform sequence analysis, prediction, and imputation.  The swarm dynamically adapts its agent distribution and weights based on input data characteristics.
*   **Unified CLI:** A single command-line interface to access all functionalities: encoding, training, prediction, imputation, and sequence analysis.
*   **Data-Driven Adaptation:**  The system automatically adjusts parameters (k-mer size, chunk size, agent distribution) based on input sequence characteristics and available computational resources.
*   **GPU Acceleration:**  Leverages CuPy for GPU acceleration when available, falling back to NumPy on CPU.
*   **Caching:** Extensive use of caching (k-mers, sequences, features) to improve performance.
*   **Extensibility:**  Designed to be easily extended with new biological feature providers, agent types, and analysis methods.
*   **File Format Support:** Handles FASTA and FASTQ files for sequence input, and supports various formats for saving results (HDF5, NumPy, JSON, CSV).
*   **Meta-Learning:** Includes a `MetaHDConservation` module for learning conservation patterns.

## Core Components

*   **`HDCVectorSpace` (Base Class):**  Foundation for all HDC operations.  Handles vector creation, binding, bundling, permutation, similarity calculation, and batch processing.
*   **`DNASupercomputer`:**  Extends `HDCVectorSpace` with DNA-specific functionality (base and position vectors, k-mer encoding, sequence encoding).
*   **`BiologicalHDC`:** Further extends `DNASupercomputer` to integrate biological knowledge (GC content, complexity, conservation, annotations, epigenetics, motifs).
*   **`DNAEncoder`:** Autotuning DNA encoder for HDC.
*   **`BiologicalEncoder`:** HDC encoder with integrated biological features.
*   **`GeneticAnalyzer`:** High-level analysis API for sequence analysis.
* **`GeneticAgent`:** Reinforcement learning based agent for base prediction and sequence imputation.
* **`GeneticSwarm`:** Swarm intelligence for advanced sequence analysis.
* **`MetaHDConservation`:** Learns conservation patterns with meta-learning.

## Getting Started

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/<your-username>/genetic-supercomputer.git
    cd genetic-supercomputer
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *   **Note for GPU Usage:** If you have a CUDA-enabled GPU, install CuPy separately for optimal performance. Follow the instructions on the [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).  For example:

        ```bash
        pip install cupy-cuda11x  # Replace 11x with your CUDA version (e.g., 10x, 12x)
        ```

    * **Note for PyBigWig (Optional)**: If using BigWig files, you will need additional system libraries. On Ubuntu/Debian: `sudo apt-get install libcurl4-openssl-dev libssl-dev`.

### Usage (CLI)

The `genetic_supercomputer.py` script provides a unified command-line interface.

**1. Encoding Sequences:**

Encode sequences from a FASTA file into HDC vectors and save them to an HDF5 file:

```bash
python genetic_supercomputer.py encode --input sequences.fasta --output encoded_vectors.h5 --dim 10000 --kmer 7
```

2. Training a Model:

Train a genetic model (swarm or simple agent) on a set of sequences:

```bash 
python genetic_supercomputer.py train --input training_sequences.fasta --output trained_model --epochs 10 --agent-type swarm --swarm-size 15
```

3. Predicting Sequence Continuations:

Predict the next 20 bases given a context from a trained model:

```bash
python genetic_supercomputer.py predict --model trained_model_model.h5 --context "ATGCGTAGCTAGCTAG" --length 20
```

You can also predict using the start of a sequence in a file:

 ```bash   
python genetic_supercomputer.py predict --model trained_model_model.h5 --input sequences.fasta --length 20 --context 20
```

4. Imputing Missing Segments:

Impute a gap of length 15, given a prefix and suffix, using a trained model:

```bash   
python genetic_supercomputer.py impute --model trained_model_model.h5 --prefix "ATGCGTAGCTAG" --suffix "GCTAGCTAGCTA" --gap-length 15
```

Impute a gap within a sequence from a file:

```bash     
python genetic_supercomputer.py impute --model trained_model_model.h5 --input sequences.fasta --gap-length 12
```

5. Analyzing Sequences:

Analyze sequence properties (GC content, base counts, optimal k-mer size, etc.):

```bash    
python genetic_supercomputer.py analyze --input sequences.fasta --output analysis_results
```

Calculate a pairwise similarity matrix:

```bash 
python genetic_supercomputer.py analyze --input sequences.fasta --output analysis_results --similarity
```
    
Analyze using a trained model (accessing advanced features like agent consensus and suggested exploration regions):

```bash 
python genetic_supercomputer.py analyze --input sequences.fasta --output analysis_results --model trained_model_model.h5 --window 20 --stride 10
```
    

6. Help:
To see all available options for each mode, use the --help flag:

```bash     
python genetic_supercomputer.py encode --help
python genetic_supercomputer.py train --help
python genetic_supercomputer.py predict --help
python genetic_supercomputer.py impute --help
python genetic_supercomputer.py analyze --help
```
 
Example Workflow

    Prepare your data: Create a FASTA file (e.g., my_sequences.fasta) containing the DNA sequences you want to analyze.

    Train a swarm model:

    ```bash 
    python genetic_supercomputer.py train --input my_sequences.fasta --output my_trained_model --epochs 5 --agent-type swarm
    ```
        
    Analyze sequences using the trained model:

    ```bash 
    python genetic_supercomputer.py analyze --input my_sequences.fasta --output my_analysis --model my_trained_model_model.h5
    ```

    Predict a sequence continuation, given a context:

    ```bash 
    python genetic_supercomputer.py predict --model my_trained_model_model.h5 --context "GCTAGCTAGCTAGCTAGCTAGC" --length 25
    ```

    Impute a sequence

    ```bash          
    python genetic_supercomputer.py impute  --model my_trained_model_model.h5 --prefix "GCTAGCTAGCTAGCTAGCTAGC" --suffix "AGCTAGCTAGCTAGCTAGCTA" --gap-length 25
    ```
    
API Usage (Python)

The code is organized into modular classes, allowing you to use the components directly in your Python scripts. Here's a basic example of using the DNASupercomputer and GeneticSwarm:
      
from genetic_supercomputer import DNASupercomputer, GeneticSwarm

Initialize the supercomputer
supercomputer = DNASupercomputer(dimension=10000, device="cpu")

Example sequences
sequences = [
    "ATGCGTAGCTAGCTAGCTAGCTAGCTA",
    "GCTAGCTAGCTAGCTAGCTAGCTAGC",
    "TTAGCTAGCTAGCTAGCTAGCTAGCT",
]

Encode sequences
encoded_vectors = [supercomputer.encode_sequence(seq) for seq in sequences]

Initialize the swarm
swarm = GeneticSwarm(supercomputer, swarm_size=10)

Train the swarm
swarm.train(sequences, epochs=3)

Analyze a sequence
analysis_results = swarm.analyze_sequence("ATGCGTAGCTAGCTAGCTAGCTAGCTA")
print(analysis_results)

Predict a sequence
prediction = swarm.predict("ATGCGTAGCTAG", length=10)
print(f"Prediction: {prediction}")

Impute a missing segment
imputation = swarm.impute_segment("ATGCGT", "AGCTAGCTA", gap_length=5)
print(f"Imputation: {imputation}")


See the class definitions in genetic_supercomputer.py for more details on available methods and parameters. The BiologicalEncoder, in particular, is designed to be easily extended to support a wide range of biological data types.
Project Structure

    genetic_supercomputer.py: Contains all the core classes and functions:

        HDCVectorSpace: Base HDC class.

        DNASupercomputer: DNA-specific HDC implementation.

        BiologicalHDC: HDC with biological feature integration.

        DNAEncoder: Autotuning DNA encoder.

        BiologicalEncoder: HDC encoder with integrated biological features.

        GeneticAgent: RL agent for prediction/imputation.

        GeneticSwarm: Swarm intelligence system.

        MetaHDConservation: Meta-learning conservation scorer.

        AgentType: Enum for agent specializations.

        parse_args(): CLI argument parsing.

        load_sequences(): FASTA/FASTQ loader.

        save_results(): Result saving (HDF5, NumPy, JSON).

        run_genetic_analysis(): Main function for CLI.
