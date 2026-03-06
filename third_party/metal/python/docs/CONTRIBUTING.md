# Contributing to Triton Metal Backend

Thank you for your interest in contributing to the Triton Metal backend! This document outlines the process for contributing to the project, including how to submit issues, create pull requests, and follow our coding standards.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Community](#community)

## Getting Started

The Triton Metal backend is designed to enable Triton kernels to run on Apple Silicon GPUs via the Metal API. Before contributing, we recommend:

1. Familiarizing yourself with the [Triton programming model](https://chenxingqiang.github.io/triton-metalmain/programming-guide/index.html)
2. Understanding the [Metal backend architecture](ARCHITECTURE.md)
3. Reviewing our [optimization guides](PERFORMANCE_OPTIMIZATION.md) and [M3-specific optimizations](M3_OPTIMIZATIONS.md)

## Development Setup

### Prerequisites

- macOS 13.5 or higher
- Apple Silicon Mac (M1/M2/M3)
- Xcode 15.0 or higher
- Python 3.8 or higher
- MLX 0.3.0 or higher

### Setting Up Your Development Environment

1. Clone the Triton repository:
   ```bash
   git clone https://github.com/chenxingqiang/triton-metal.git
   cd triton
   ```

2. Install the development dependencies:
   ```bash
   pip install -e ".[tests,metal]"
   ```

3. Build Triton with Metal support:
   ```bash
   cd python
   TRITON_BUILD_WITH_METAL=1 pip install -e .
   ```

4. Run the Metal backend tests to ensure your environment is set up correctly:
   ```bash
   cd test/unit
   python -m pytest -xvs metal/
   ```

## Contribution Workflow

### 1. Finding Issues to Work On

- Check the [Issues](https://github.com/chenxingqiang/triton-metal/issues) tab for issues labeled with `backend:metal`, `good first issue`, or `help wanted`
- Feel free to ask for clarification on any issue before starting work

### 2. Creating a New Issue

If you've found a bug or have a feature request:

1. Check if a similar issue already exists
2. If not, create a new issue using the appropriate template:
   - For bugs, include: OS version, Metal backend version, device information, and steps to reproduce
   - For features, include: clear description, use case, and potential implementation approach

### 3. Working on an Issue

1. Comment on the issue to let others know you're working on it
2. Create a new branch from the latest `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/metal-your-feature-name
   ```
3. Make your changes, following the [coding standards](#coding-standards)
4. Add tests for your changes
5. Update documentation as needed

## Pull Request Process

1. **Before Creating a PR**:
   - Ensure all tests pass: `python -m pytest -xvs test/unit/metal/`
   - Run the style checks: `pre-commit run --all-files`
   - Update any relevant documentation

2. **Creating the PR**:
   - Submit your PR against the `main` branch
   - Use the PR template to provide a clear description of your changes
   - Link the PR to any relevant issues using GitHub keywords (e.g., "Fixes #123")

3. **PR Review Process**:
   - A maintainer will review your PR
   - Address any requested changes or feedback
   - Once approved, a maintainer will merge your PR

4. **After Your PR is Merged**:
   - Delete your branch
   - Celebrate your contribution! 🎉

## Coding Standards

### Python Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use clear, descriptive variable and function names
- Document functions and classes using docstrings

### C++ Style Guidelines

- Follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- Use consistent formatting with clang-format (use `.clang-format` in the repository)
- Document public APIs with doxygen-style comments

### Metal-Specific Guidelines

- Use Metal best practices for performance
- Document any hardware-specific optimizations
- Include fallbacks for different Apple Silicon generations
- Prefer MLX abstractions over direct Metal API calls when possible

## Testing Guidelines

All contributions should include appropriate tests:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test interactions between components
- **Performance tests**: For performance-critical code, include benchmarks
- **Metal-specific tests**: Test across different Apple Silicon generations if possible

### Running Tests

```bash
# Run all Metal backend tests
python -m pytest -xvs test/unit/metal/

# Run specific test file
python -m pytest -xvs test/unit/metal/test_specific_feature.py

# Run a specific test
python -m pytest -xvs test/unit/metal/test_file.py::test_function_name
```

## Documentation Guidelines

Good documentation is crucial for the usability of the Metal backend:

- Update user-facing documentation when changing APIs
- Document M3-specific optimizations separately from general Metal backend features
- Add comments for complex algorithms or hardware-specific optimizations
- Include examples for new features

## Community

- Join the [Triton community](https://discord.gg/ZRH2Kk2ju3) on Discord
- Participate in discussions in the `#metal-backend` channel
- Help answer questions from other users
- Share your use cases and success stories

## Acknowledgments

Thank you for contributing to the Triton Metal backend! Your contributions help make GPU computing more accessible on Apple Silicon.

## License

By contributing to Triton Metal backend, you agree that your contributions will be licensed under the same license as the project (MIT License). 