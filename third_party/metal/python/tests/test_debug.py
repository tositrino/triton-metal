"""
Debug script for Metal memory layout optimization

This script provides various utilities to debug and test the memory layout
optimization in the Metal backend, with special focus on the COALESCED layout
for reduction operations.
"""


# Import memory layout optimizer
from MLX.memory_layout_optimizer import (
    MemoryLayoutOptimizer,
    LayoutOptimizationLevel,
    ReductionLayoutPattern,
    optimize_memory_layout,
    MemoryLayout as OptimizerMemoryLayout
)

# Import memory manager
from MLX.metal_memory_manager import (
    MetalMemoryManager,
    get_metal_memory_manager,
    MemoryLayout as ManagerMemoryLayout
)

def print_separator(title=""):
    """Print a separator line with optional title"""
    width = 80
    print("\n" + "=" * width)
    if title:
        print(f"{title.center(width)}")
        print("-" * width)

def check_enum_consistency():
    """Check that memory layout enums are consistent across components"""
    print_separator("CHECKING MEMORY LAYOUT ENUM CONSISTENCY")

    # Print all values from both enums
    print("Memory Layout in memory_layout_optimizer.py:")
    for layout in OptimizerMemoryLayout:
        print(f"  {layout.name} = {layout.value}")

    print("\nMemory Layout in metal_memory_manager.py:")
    for layout in ManagerMemoryLayout:
        print(f"  {layout.name} = {layout.value}")

    # Check if COALESCED is defined consistently
    optimizer_coalesced = getattr(OptimizerMemoryLayout, "COALESCED", None)
    manager_coalesced = getattr(ManagerMemoryLayout, "COALESCED", None)

    if optimizer_coalesced and manager_coalesced:
        print(f"\nCOALESCED layout defined in both modules:")
        print(f"  optimizer: {optimizer_coalesced.value}, manager: {manager_coalesced.value}")

        if optimizer_coalesced.value == manager_coalesced.value:
            print("✅ COALESCED values match!")
        else:
            print("❌ COALESCED values do not match!")
    else:
        if not optimizer_coalesced:
            print("❌ COALESCED not defined in memory_layout_optimizer.py")
        if not manager_coalesced:
            print("❌ COALESCED not defined in metal_memory_manager.py")

def test_reduction_layout_pattern():
    """Test the reduction layout pattern detection and optimization"""
    print_separator("TESTING REDUCTION LAYOUT PATTERN")

    # Create a reduction layout pattern
    pattern = ReductionLayoutPattern()

    # Define test shapes
    test_shapes = [
        [1024],
        [1024, 1024],
        [32, 64, 128],
        [1, 1024, 1024]
    ]

    # Define test operations
    test_ops = [
        {"type": "tt.reduce", "args": {"axis": 0}},
        {"type": "tt.reduce", "args": {"axis": 1}},
        {"type": "tt.sum", "args": {"axis": 2}},
        {"type": "tt.mean", "args": {"axis": [0, 1]}},
        {"type": "mlx.reduce", "args": {"axis": 0}},
        {"type": "mlx.sum", "args": {"axis": 1}},
        {"type": "other_op", "args": {}}
    ]

    # Test pattern applicability
    print("\nTesting pattern applicability:")
    for op in test_ops:
        is_applicable = pattern.is_applicable(op, None)
        print(f"  {op['type']}: {'✅ Applicable' if is_applicable else '❌ Not applicable'}")

    # Test optimal layout for different shapes
    print("\nTesting optimal layout for different shapes:")
    for shape in test_shapes:
        optimal_layout = pattern.get_optimal_layout(shape, None)
        print(f"  Shape {shape}: {optimal_layout.name}")

        # Verify that COALESCED is always returned
        if optimal_layout == OptimizerMemoryLayout.COALESCED:
            print(f"    ✅ Correctly using COALESCED layout")
        else:
            print(f"    ❌ Not using COALESCED layout!")

    # Test parameters for different shapes
    print("\nTesting parameters for different shapes:")
    for shape in test_shapes:
        params = pattern.get_parameters(shape, None)
        print(f"  Shape {shape}:")
        for key, value in params.items():
            print(f"    {key}: {value}")

def test_memory_manager_reduction_optimization():
    """Test the memory manager's reduction optimization"""
    print_separator("TESTING MEMORY MANAGER REDUCTION OPTIMIZATION")

    # Get the memory manager
    memory_manager = get_metal_memory_manager()

    # Define test reduction operations
    test_operations = [
        {
            "type": "tt.reduce",
            "input_shapes": [[1024]],
            "args": {"axis": 0}
        },
        {
            "type": "tt.reduce",
            "input_shapes": [[1024, 1024]],
            "args": {"axis": 1}
        },
        {
            "type": "tt.sum",
            "input_shapes": [[32, 64, 128]],
            "args": {"axis": 2}
        },
        {
            "type": "tt.mean",
            "input_shapes": [[1, 1024, 1024]],
            "args": {"axis": [0, 1]}
        }
    ]

    # Test each operation
    for op in test_operations:
        print(f"\nOptimizing {op['type']} with shape {op['input_shapes'][0]} along axis {op['args']['axis']}:")

        # Optimize the operation
        optimized_op = memory_manager._optimize_reduction_memory(op.copy())

        # Check if execution parameters were set
        if "execution_parameters" in optimized_op:
            print(f"  Execution parameters:")

            # Check memory layout
            layout_value = optimized_op["execution_parameters"].get("memory_layout")
            layout_match = layout_value == ManagerMemoryLayout.COALESCED.value
            print(f"    memory_layout: {layout_value} {'✅' if layout_match else '❌'}")

            # Print other parameters
            for key, value in optimized_op["execution_parameters"].items():
                if key != "memory_layout":
                    print(f"    {key}: {value}")
        else:
            print("  ❌ No execution parameters set!")

def test_optimizer_integration():
    """Test the integration of the memory layout optimizer with a full graph"""
    print_separator("TESTING OPTIMIZER INTEGRATION")

    # Create a test graph with various operations
    test_graph = {
        "ops": [
            {
                "id": "matmul1",
                "type": "tt.matmul",
                "input_shapes": [[128, 256], [256, 512]],
                "output_shape": [128, 512]
            },
            {
                "id": "reduce1",
                "type": "tt.reduce",
                "input_shapes": [[128, 512]],
                "args": {"axis": 1},
                "output_shape": [128, 1]
            },
            {
                "id": "sum1",
                "type": "tt.sum",
                "input_shapes": [[128, 1]],
                "args": {"axis": 0},
                "output_shape": [1]
            }
        ]
    }

    # Create optimizer with different optimization levels
    optimization_levels = [
        LayoutOptimizationLevel.NONE,
        LayoutOptimizationLevel.BASIC,
        LayoutOptimizationLevel.AGGRESSIVE,
        LayoutOptimizationLevel.HARDWARE_SPECIFIC
    ]

    for level in optimization_levels:
        print(f"\nOptimizing with level {level.name}:")

        # Create optimizer
        optimizer = MemoryLayoutOptimizer(optimization_level=level)

        # Optimize graph
        optimized_graph, stats = optimizer.optimize(test_graph.copy())

        # Print statistics
        print(f"  Optimization statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

        # Check reduction operations
        for op in optimized_graph["ops"]:
            if "reduce" in op["id"] or "sum" in op["id"]:
                print(f"\n  Operation {op['id']}:")

                # Check for layout hints
                if "layout_hints" in op:
                    print(f"    layout_hints:")
                    for key, value in op["layout_hints"].items():
                        print(f"      {key}: {value}")

                    # Check that layout is COALESCED
                    if op["layout_hints"].get("layout") == "COALESCED":
                        print(f"      ✅ Layout correctly set to COALESCED")
                    else:
                        print(f"      ❌ Layout not set to COALESCED!")
                else:
                    if level == LayoutOptimizationLevel.NONE:
                        print(f"    ✅ No layout hints (expected with NONE optimization level)")
                    else:
                        print(f"    ❌ No layout hints found!")

def dump_api_structure():
    """Dump the structure of key APIs for debugging"""
    print_separator("API STRUCTURE INFORMATION")

    # Get a memory manager instance
    memory_manager = get_metal_memory_manager()

    # Print the memory manager class structure
    print(f"MetalMemoryManager methods:")
    for method_name in dir(memory_manager):
        if not method_name.startswith("__") and callable(getattr(memory_manager, method_name)):
            print(f"  {method_name}")

    # Create a reduction layout pattern
    pattern = ReductionLayoutPattern()

    # Print the pattern class structure
    print(f"\nReductionLayoutPattern methods:")
    for method_name in dir(pattern):
        if not method_name.startswith("__") and callable(getattr(pattern, method_name)):
            print(f"  {method_name}")

    # Print all optimization levels
    print(f"\nLayoutOptimizationLevel values:")
    for level in LayoutOptimizationLevel:
        print(f"  {level.name} = {level.value}")

    # Print public functions from memory_layout_optimizer
    print(f"\nPublic functions in memory_layout_optimizer:")
    print(f"  optimize_memory_layout")
    print(f"  get_metal_layout_optimizer")

def run_all_tests():
    """Run all tests"""
    print_separator("RUNNING ALL METAL MEMORY LAYOUT TESTS")

    # Check enum consistency
    check_enum_consistency()

    # Test reduction layout pattern
    test_reduction_layout_pattern()

    # Test memory manager reduction optimization
    test_memory_manager_reduction_optimization()

    # Test optimizer integration
    test_optimizer_integration()

    # Dump API structure
    dump_api_structure()

    print_separator("ALL TESTS COMPLETED")

if __name__ == "__main__":
    run_all_tests()