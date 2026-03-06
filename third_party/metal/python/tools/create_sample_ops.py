#!/usr/bin/env python
"""
Generate Sample Operations JSON for Testing

This script generates a sample JSON file containing various operations,
including reduction operations that would use the COALESCED memory layout.
"""

import os
import json
import random
import argparse

def generate_sample_ops(seed=None):
    """Generate sample operations for testing"""
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        
    ops = [
        # Reduction operations (will use COALESCED layout)
        {
            "type": "tt.reduce",
            "id": "sum_reduce_1d",
            "input_shapes": [[1024]],
            "args": {"axis": 0}
        },
        {
            "type": "tt.sum",
            "id": "sum_reduce_2d",
            "input_shapes": [[512, 512]],
            "args": {"axis": 1}
        },
        {
            "type": "tt.mean",
            "id": "mean_reduce_3d",
            "input_shapes": [[32, 64, 128]],
            "args": {"axis": [0, 1]}
        },
        {
            "type": "tt.argmax",
            "id": "argmax_reduce",
            "input_shapes": [[128, 256]],
            "args": {"axis": 1}
        },
        # Non-reduction operations (will NOT use COALESCED layout)
        {
            "type": "tt.matmul",
            "id": "matmul_op",
            "input_shapes": [[256, 256], [256, 256]]
        },
        {
            "type": "tt.add",
            "id": "elementwise_add",
            "input_shapes": [[128, 128], [128, 128]]
        },
        {
            "type": "tt.conv2d",
            "id": "conv2d_op",
            "input_shapes": [[1, 32, 64, 64], [64, 32, 3, 3]]
        },
        # Additional reduction operation
        {
            "type": "tt.max",
            "id": "max_reduce_3d",
            "input_shapes": [[16, 32, 64]],
            "args": {"axis": 2}
        }
    ]
    
    return ops

def generate_large_sample(seed=None):
    """Generate a larger sample with more operations"""
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        
    ops = generate_sample_ops(seed)
    
    # Add more variations of reduction operations
    additional_ops = [
        # Large 1D reduction
        {
            "type": "tt.sum",
            "id": "large_sum_1d",
            "input_shapes": [[1048576]],
            "args": {"axis": 0}
        },
        # Large 2D reduction along axis 0
        {
            "type": "tt.reduce",
            "id": "large_reduce_2d_axis0",
            "input_shapes": [[1024, 512]],
            "args": {"axis": 0}
        },
        # Min reduction
        {
            "type": "tt.min",
            "id": "min_reduce_2d",
            "input_shapes": [[256, 512]],
            "args": {"axis": 1}
        },
        # Multiple axis reduction
        {
            "type": "tt.mean",
            "id": "mean_reduce_multiple_axes",
            "input_shapes": [[32, 64, 128, 256]],
            "args": {"axis": [0, 2]}
        },
        # Any/All reductions
        {
            "type": "tt.any",
            "id": "any_reduce",
            "input_shapes": [[512, 512]],
            "args": {"axis": 1}
        },
        {
            "type": "tt.all",
            "id": "all_reduce",
            "input_shapes": [[512, 512]],
            "args": {"axis": 1}
        }
    ]
    
    ops.extend(additional_ops)
    
    # Add more non-reduction operations
    non_reduction_ops = [
        {
            "type": "tt.relu",
            "id": "relu_op",
            "input_shapes": [[512, 512]]
        },
        {
            "type": "tt.sigmoid",
            "id": "sigmoid_op",
            "input_shapes": [[256, 256]]
        },
        {
            "type": "tt.softmax",
            "id": "softmax_op",
            "input_shapes": [[128, 1024]],
            "args": {"axis": 1}
        },
        {
            "type": "tt.tanh",
            "id": "tanh_op",
            "input_shapes": [[512, 512]]
        }
    ]
    
    ops.extend(non_reduction_ops)
    return ops

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate sample operations JSON for testing the analyzer"
    )
    
    parser.add_argument("--output", type=str, default="sample_ops.json",
                       help="Output file path (default: sample_ops.json)")
    parser.add_argument("--large", action="store_true",
                       help="Generate a larger sample with more operations")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Generate operations
    if args.large:
        ops = generate_large_sample(args.seed)
        print(f"Generating large sample with {len(ops)} operations (seed: {args.seed})...")
    else:
        ops = generate_sample_ops(args.seed)
        print(f"Generating basic sample with {len(ops)} operations (seed: {args.seed})...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write to JSON file
    with open(args.output, 'w') as f:
        json.dump({"ops": ops}, f, indent=2)
    
    print(f"Sample operations written to {args.output}")
    print(f"Number of reduction operations: {sum(1 for op in ops if op['type'].startswith('tt.') and op['type'] in ['tt.reduce', 'tt.sum', 'tt.mean', 'tt.max', 'tt.min', 'tt.argmax', 'tt.argmin', 'tt.any', 'tt.all'])}")
    print(f"Number of non-reduction operations: {sum(1 for op in ops if not (op['type'].startswith('tt.') and op['type'] in ['tt.reduce', 'tt.sum', 'tt.mean', 'tt.max', 'tt.min', 'tt.argmax', 'tt.argmin', 'tt.any', 'tt.all']))}")

if __name__ == "__main__":
    main() 