# Metal Backend Tutorials

This directory contains tutorials for using the Triton Metal Backend on Apple Silicon GPUs.

## List of Tutorials

1. **[Metal Backend Compatibility Tutorial](../tutorial_metal_compatibility.py)** - Check your system compatibility with the Metal backend and understand optimizations for M3 chips.

   This tutorial guides you through:
   - Checking system requirements
   - Understanding memory layout optimizations
   - Leveraging M3-specific features
   - Running a simple vector addition example

   To run: `python ../tutorial_metal_compatibility.py`

## Prerequisites

All tutorials require:
- macOS 13.5 or newer
- Apple Silicon Mac (M1, M2, or M3 series)
- MLX package installed

## Getting Started

Clone the repository and navigate to this directory:

```bash
git clone https://github.com/chenxingqiang/triton-metal.git
cd triton/third_party/metal/python/tutorials
```

Run the metal compatibility tutorial:

```bash
python ../tutorial_metal_compatibility.py
```

## Documentation

For more detailed information about the Metal backend, see the main [README.md](../README.md). 