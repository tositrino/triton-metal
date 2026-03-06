#!/usr/bin/env python
"""
Package preparation script for Triton Metal Backend.

This script organizes the necessary files for the PyPI package by:
1. Creating the proper directory structure
2. Copying backend files into the package directory
3. Ensuring documentation and examples are included
"""

import os
import sys
import shutil
import pathlib
from glob import glob

# Directory structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(CURRENT_DIR, "triton")
DOCS_DIR = os.path.join(PACKAGE_DIR, "docs")
TUTORIALS_DIR = os.path.join(PACKAGE_DIR, "tutorials")
TESTS_DIR = os.path.join(PACKAGE_DIR, "tests")

# Ensure directories exist
os.makedirs(PACKAGE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(TUTORIALS_DIR, exist_ok=True)
os.makedirs(TESTS_DIR, exist_ok=True)

def copy_files():
    """Copy necessary files into the package directory structure."""
    # Core implementation files
    core_files = [
        "check_system.py",
        "tutorial_metal_compatibility.py",
        "metal_hardware_optimizer.py",
        "m3_optimizations.py",
        "metal_memory_manager.py",
        "memory_layout_optimizer.py",
    ]
    
    for file in core_files:
        src = os.path.join(CURRENT_DIR, file)
        dst = os.path.join(PACKAGE_DIR, file)
        if os.path.exists(src):
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} does not exist")
    
    # Documentation files
    doc_files = [
        "README.md",
        "tutorial_README.md",
    ]
    
    for file in doc_files:
        src = os.path.join(CURRENT_DIR, file)
        dst = os.path.join(DOCS_DIR, file)
        if os.path.exists(src):
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
    
    # Tutorial files
    tutorial_src_dir = os.path.join(CURRENT_DIR, "tutorials")
    if os.path.exists(tutorial_src_dir):
        for file in os.listdir(tutorial_src_dir):
            if file.endswith(".py") or file.endswith(".md"):
                src = os.path.join(tutorial_src_dir, file)
                dst = os.path.join(TUTORIALS_DIR, file)
                print(f"Copying {src} to {dst}")
                shutil.copy2(src, dst)
    
    # Test files
    test_files = [
        "test_check_system.py",
        "test_tutorial_metal_compatibility.py",
    ]
    
    for file in test_files:
        src = os.path.join(CURRENT_DIR, file)
        dst = os.path.join(TESTS_DIR, file)
        if os.path.exists(src):
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)

def create_init_files():
    """Create necessary __init__.py files in each directory."""
    dirs = [PACKAGE_DIR, DOCS_DIR, TUTORIALS_DIR, TESTS_DIR]
    
    # Main package __init__.py is handled separately
    for dir_path in dirs[1:]:  # Skip the main package dir
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            print(f"Creating {init_file}")
            with open(init_file, "w") as f:
                f.write("# Package initialization\n")

def fix_imports():
    """Fix imports in copied files to use relative imports."""
    py_files = glob(os.path.join(PACKAGE_DIR, "**/*.py"), recursive=True)
    
    for file_path in py_files:
        if os.path.basename(file_path) == "__init__.py":
            continue
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Replace imports of local modules with relative imports
        for module in ["check_system", "tutorial_metal_compatibility", 
                     "metal_hardware_optimizer", "m3_optimizations",
                     "metal_memory_manager", "memory_layout_optimizer"]:
            # Direct imports
            content = content.replace(f"import {module}", f"from . import {module}")
            # From imports
            content = content.replace(f"from {module}", f"from .{module}")
        
        with open(file_path, "w") as f:
            f.write(content)

def main():
    """Main function to prepare the package."""
    print("Preparing Triton Metal Backend package...")
    
    # Copy files
    copy_files()
    
    # Create __init__.py files
    create_init_files()
    
    # Fix imports
    fix_imports()
    
    print("Package preparation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 