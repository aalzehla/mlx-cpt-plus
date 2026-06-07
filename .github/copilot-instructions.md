# Chat Customization Guidelines for mlx-cpt-plus

This file provides guidance to VS Code's GitHub Copilot when working with the **mlx-cpt-plus** Python package.

## 🎯 Project Overview

**mlx-cpt-plus** is a production-grade **CPT+ (Context-Pattern Tree Plus)** implementation optimized for **Apple Silicon** using **MLX** (no external ML frameworks like PyTorch or TensorFlow required).

**Core Purpose**: Next-item prediction system that learns sequence patterns and predicts the next item in a sequence using tree-based structures and Apple Silicon hardware acceleration.

**Key Characteristics:**
- **Language**: Pure Python 3.9+
- **Hardware Acceleration**: Apple Silicon MLX kernels (NO GPU dependencies)
- **Version**: 0.1.0 (alpha stage)
- **License**: MIT
- **Target Audience**: Scientific/Engineering, ML developers working on sequence prediction

## 🏗️ Architecture Overview

The package follows a **modular architecture** with clear separation of concerns:

### **Core Infrastructure** (`mlx_cpt_plus/core/`)
- **Data structures for sequence management**: `Alphabet`, `Vocabulary`, `SequenceArena`, `SequenceStore`
- **Indexing and retrieval**: `LookupTable`, `InvertedIndex`
- **Candidate management**: `CandidateSet`, `SupportTable`

### **Tree Engine** (`mlx_cpt_plus/tree/`)
- **Tree-based pattern representation**: `Node`, `CompressedNode`, `PredictionTree`
- **Tree construction**: `TreeBuilder`, `BranchCompression`
- **Tree traversal**: `TreeIterator`

### **Prediction Engine** (`mlx_cpt_plus/prediction/`)
- **Pattern matching**: `Similarity`, `CandidateRetrieval`
- **Prediction logic**: `Voting`, `Ranking`, `Predictor`
- **Prediction tasks**: `NextItem`, `TopK`

### **MLX Acceleration** (`mlx_cpt_plus/mlx/`)
- **Kernel wrappers**: `SimilarityKernel`, `VoteKernel`, `RankingKernel`
- **Batch processing**: `BatchPredict`, `Compiled` operations
- **Tensor utilities**: Low-level MLX tensor operations

### **Advanced Features** (planned/partial)
- **Compression**: Subsequence compression and dictionary management
- **Storage**: Persistence/serialization (MessagePack, NumPy)
- **Online Learning**: Incremental updates and dynamic model updates
- **Metrics**: Evaluation metrics and model quality measures

## 🛠️ Common Development Commands

### Package Management
```bash
# Install package in editable mode
python -m pip install -e .

# Install dev dependencies
python -m pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/path/to/test_file.py

# Run with coverage
python -m pytest --cov=mlx_cpt_plus
```

### Development Pipeline Scripts
```bash
# Build/rebuild sequence indexes
python scripts/build_index.py

# Train the prediction model
python scripts/train_model.py

# Evaluate model performance
python scripts/evaluate_model.py

# Export trained model
python scripts/export_model.py
```

## 📋 Project Conventions

### Code Style & Formatting
- **Line length**: 100 characters
- **Black**: Target Python versions 3.9-3.12
- **isort**: Profile "black", line length 100
- **Ruff checks**: E (pycodestyle), F (pyflakes), W (pycodestyle warnings), I (isort), N (flake8-naming), UP (pyupgrade), B (flake8-bugbear), C4 (flake8-comprehensions)

### Type Checking
- **Mypy**: Python 3.9, warn on returning `Any`, warn on unused configs, ignore missing imports

### Testing Conventions
- **Framework**: pytest
- **Test naming**: `test_*` functions
- **Test file naming**: `test_*.py`
- **Test paths**: `tests/` directory
- **Fixtures**: Defined in `tests/conftest.py`

## 🔑 Key Design Principles

1. **Pure Python First**: The core implementation is pure Python, avoiding external ML framework dependencies
2. **Apple Silicon Optimized**: MLX kernels are specifically optimized for Apple Silicon hardware
3. **Modular Design**: Clear separation between core infrastructure, tree engine, prediction logic, and MLX acceleration
4. **Memory Efficiency**: Uses contiguous memory-based storage (`SequenceArena`) and compressed tree nodes
5. **Incremental Learning**: Supports dynamic updates through online learning mechanisms

## 📚 Helpful Documentation Links

- **README.md** — Primary repository overview and usage instructions
- **CLAUDE.md** — Architecture-oriented guidance for code assistants
- **docs/architecture.md** — Detailed architecture documentation
- **docs/algorithms.md** — Algorithm descriptions and mathematical foundations
- **docs/training.md** — Model training procedures
- **docs/prediction.md** — Prediction API and usage
- **docs/persistence.md** — Model persistence and serialization
- **docs/benchmarks.md** — Performance benchmarks and results
- **docs/api.md** — Complete API reference

## ⚠️ Common Pitfalls & Considerations

### Hardware Dependencies
- **MLX is Apple Silicon only**: Do NOT assume GPU availability or use CUDA-related code
- **MLX version**: Requires `mlx>=0.4.0`
- **Fallback behavior**: Pure Python implementations should work on non-Apple Silicon systems

### Memory Management
- `SequenceArena` uses contiguous memory blocks
- Tree compression is critical for large models
- Be aware of memory usage when processing large datasets

### Version Compatibility
- Python 3.9+ minimum
- MLX 0.4.0+ for hardware acceleration
- NumPy 1.24.0+ for tensor operations

## 🎓 Core Concepts to Understand

### CPT+ Algorithm
- **Context-Pattern Tree Plus**: An extension of CPT that uses tree structures to represent sequence patterns
- **Next-Item Prediction**: Given a sequence context, predict the next item
- **Pattern Matching**: Uses similarity metrics (Jaccard, Cosine, Overlap) to find relevant patterns

### Key Data Structures
- **Alphabet**: Character-level encoding/decoding
- **Vocabulary**: Token management with frequency tracking
- **PredictionTree**: Tree structure storing learned sequence patterns
- **InvertedIndex**: Maps items to sequences for fast retrieval

### Prediction Pipeline
1. **Input**: Sequence context (e.g., "transaction type", "amount")
2. **Retrieval**: Find similar patterns using inverted index
3. **Similarity**: Score patterns using similarity metrics
4. **Voting**: Aggregate votes from matching patterns
5. **Ranking**: Rank candidates using ranking algorithms
6. **Output**: Predicted next item(s)

## 💡 Tips for Working with the Codebase

### When Adding New Features
- Follow the modular architecture
- Prefer pure Python where possible
- Add MLX kernel only if hardware acceleration is critical
- Write tests for both pure Python and MLX paths

### When Debugging
- Start with pure Python implementations (no MLX dependency)
- Use pytest fixtures for test setup
- Check memory usage for large datasets
- Verify MLX compatibility for hardware-accelerated code

### When Writing Tests
- Test core logic independently of MLX
- Test MLX kernels separately with Apple Silicon
- Use parametrized tests for different input sizes
- Include edge cases for empty sequences and large vocabularies

## 🔄 Implementation Phases

The project is designed to be implemented in phases:

1. **Phase 1**: Core Infrastructure (alphabet, vocabulary, sequence storage, lookup)
2. **Phase 2**: Tree Engine (nodes, tree structure, builder)
3. **Phase 3**: Candidate Retrieval (inverted index, retrieval)
4. **Phase 4**: Prediction Engine (similarity, voting, ranking, predictor)
5. **Phase 5**: MLX Acceleration (kernels, batch processing)
6. **Phase 6**: CPT+ Features (compression, advanced algorithms)
7. **Phase 7**: Persistence (serialization, storage)
8. **Phase 8**: Online Learning (incremental updates)
9. **Phase 9**: Testing & Benchmarking

## 🧪 Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete pipeline (index building → training → prediction)
- **MLX Tests**: Run on Apple Silicon to verify hardware acceleration
- **Benchmark Tests**: Performance measurements for different datasets

## 📈 Performance Considerations

- **Memory**: SequenceArena uses contiguous memory for efficiency
- **Cache Locality**: Tree structures optimized for CPU cache
- **Batch Processing**: MLX kernels support batch operations
- **Compression**: Branch compression reduces memory footprint

---

*This document is living and will be updated as the project evolves.*