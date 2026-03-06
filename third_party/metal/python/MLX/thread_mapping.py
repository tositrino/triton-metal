"""
Thread mapping utilities for Metal backend

This module handles the mapping of CUDA-style thread and grid configurations
to Metal's threadgroup-based execution model.
"""

from .metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
from typing import Dict, List, Tuple, Any, Optional, Union

class ThreadMapping:
    """Mapping of Triton thread hierarchy to Metal thread hierarchy"""

    def __init__(self, hardware_capabilities=None):
        """Initialize thread mapping

        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or self._get_default_hardware_capabilities()
        self.max_threads_per_threadgroup = self._get_max_threads_per_threadgroup()
        self.max_threadgroups = self._get_max_threadgroups()
        self.simd_width = self._get_simd_width()

    def _get_default_hardware_capabilities(self):
        """Get default hardware capabilities

        Returns:
            Default hardware capabilities
        """
        from .metal_hardware_optimizer import hardware_capabilities
        return hardware_capabilities

    def _get_max_threads_per_threadgroup(self) -> int:
        """Get maximum threads per threadgroup for the current hardware

        Returns:
            Maximum threads per threadgroup
        """
        if self.hardware and hasattr(self.hardware, 'max_threads_per_threadgroup'):
            return self.hardware.max_threads_per_threadgroup
        return 1024  # Default for modern Metal GPUs

    def _get_max_threadgroups(self) -> Tuple[int, int, int]:
        """Get maximum threadgroups for the current hardware

        Returns:
            Tuple of (max_x, max_y, max_z) threadgroups
        """
        if self.hardware and hasattr(self.hardware, 'max_threadgroups'):
            return self.hardware.max_threadgroups
        return (1024, 1024, 64)  # Default for modern Metal GPUs

    def _get_simd_width(self) -> int:
        """Get SIMD width for the current hardware

        Returns:
            SIMD width in threads
        """
        if self.hardware and hasattr(self.hardware, 'simd_width'):
            return self.hardware.simd_width
        return 32  # Default for most modern GPUs

    def get_optimal_block_size(self, total_threads: int) -> Tuple[int, int, int]:
        """Get optimal block size for the given thread count

        Args:
            total_threads: Total number of threads

        Returns:
            Tuple of (x, y, z) block size
        """
        # For modern Metal GPUs, block sizes that are multiples of SIMD width work best
        if total_threads % self.simd_width == 0:
            block_size = total_threads
        else:
            # Round up to the next multiple of SIMD width
            block_size = ((total_threads + self.simd_width - 1) // self.simd_width) * self.simd_width

        # Cap at the maximum threads per threadgroup
        block_size = min(block_size, self.max_threads_per_threadgroup)

        # Default to 1D blocks for simplicity
        return (block_size, 1, 1)

    def get_grid_dimensions(self, total_blocks: int) -> Tuple[int, int, int]:
        """Get grid dimensions for the given block count

        Args:
            total_blocks: Total number of blocks

        Returns:
            Tuple of (x, y, z) grid dimensions
        """
        max_x, max_y, max_z = self.max_threadgroups

        # Simple 1D grid if it fits
        if total_blocks <= max_x:
            return (total_blocks, 1, 1)

        # Try 2D grid
        if total_blocks <= max_x * max_y:
            y = (total_blocks + max_x - 1) // max_x
            return (max_x, y, 1)

        # Use 3D grid
        xy = max_x * max_y
        z = (total_blocks + xy - 1) // xy
        z = min(z, max_z)
        return (max_x, max_y, z)

    def generate_thread_id_calculation(self, dim: int = 1) -> str:
        """Generate code for thread ID calculation

        Args:
            dim: Number of dimensions (1, 2, or 3)

        Returns:
            Metal code for thread ID calculation
        """
        if dim == 1:
            return """
            // 1D thread ID calculation
            uint thread_id = threadIdx.x + blockIdx.x * blockDim.x;
            """
        elif dim == 2:
            return """
            // 2D thread ID calculation
            uint thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
            uint thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
            uint thread_id = thread_id_y * gridDim.x * blockDim.x + thread_id_x;
            """
        elif dim == 3:
            return """
            // 3D thread ID calculation
            uint thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
            uint thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
            uint thread_id_z = threadIdx.z + blockIdx.z * blockDim.z;
            uint thread_id = (thread_id_z * gridDim.y * blockDim.y + thread_id_y) * gridDim.x * blockDim.x + thread_id_x;
            """
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def generate_thread_mapping_defines(self) -> str:
        """Generate defines for thread mapping

        Returns:
            Metal code with thread mapping defines
        """
        return """
        // Thread mapping defines
        #define threadIdx metal::thread_position_in_threadgroup
        #define blockIdx metal::threadgroup_position_in_grid
        #define blockDim metal::threadgroup_size
        #define gridDim metal::grid_size

        // Thread ID calculation helpers
        #define get_thread_id() (threadIdx.x + blockIdx.x * blockDim.x)
        #define get_block_id() (blockIdx.x)
        #define get_num_threads() (blockDim.x * gridDim.x)
        #define get_num_blocks() (gridDim.x)

        // For kernel launching
        #define launch_triton_kernel(grid, block, kernel, ...) \
            kernel<<<grid, block>>>(__VA_ARGS__)
        """

    def get_kernel_attributes(self, threads_per_block: int) -> str:
        """Get kernel attributes for the given thread count

        Args:
            threads_per_block: Threads per block

        Returns:
            Metal kernel attributes
        """
        # Apple-specific attributes for optimal performance
        if self.hardware:
            # Check hardware version and enable appropriate optimizations
            if hasattr(self.hardware, 'chip_generation'):
                if self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value:
                    # M3 and newer can use additional optimizations
                    return f"[[thread_position_in_grid]] [[max_total_threads_per_threadgroup({threads_per_block})]]"
                elif self.hardware.chip_generation.value >= AppleSiliconGeneration.M2.value:
                    # M2 optimizations
                    return f"[[thread_position_in_grid]] [[max_total_threads_per_threadgroup({threads_per_block})]]"
                else:
                    # M1 and older
                    return f"[[thread_position_in_grid]]"

        # Default attributes
        return "[[thread_position_in_grid]]"

class SharedMemory:
    """Shared memory manager for Metal"""

    def __init__(self):
        """Initialize shared memory manager"""
        self.total_size = 0
        self.allocations = {}
        self.next_offset = 0

    def allocate(self, size: int, alignment: int = 16) -> int:
        """Allocate shared memory

        Args:
            size: Size in bytes
            alignment: Memory alignment

        Returns:
            Offset in shared memory
        """
        # Align the next offset
        aligned_offset = (self.next_offset + alignment - 1) & ~(alignment - 1)

        # Save the current offset
        offset = aligned_offset

        # Add the allocation
        self.allocations[offset] = size

        # Update the next offset and total size
        self.next_offset = offset + size
        self.total_size = max(self.total_size, self.next_offset)

        return offset

    def generate_declaration(self) -> str:
        """Generate shared memory declaration

        Returns:
            Metal code for shared memory declaration
        """
        if self.total_size > 0:
            return f"threadgroup char shared_memory[{self.total_size}];"
        else:
            return ""

    def generate_access_code(self, offset: int, type_name: str) -> str:
        """Generate code to access shared memory

        Args:
            offset: Offset in shared memory
            type_name: Type name for casting

        Returns:
            Metal code for accessing shared memory
        """
        return f"(({type_name}*)(&shared_memory[{offset}]))"

    def reset(self):
        """Reset the shared memory manager"""
        self.total_size = 0
        self.allocations = {}
        self.next_offset = 0

class SIMDGroupFunctions:
    """SIMD group function utilities for Metal"""

    @staticmethod
    def generate_reduce_sum(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group sum reduction

        Args:
            type_name: Data type name
            var_name: Variable name

        Returns:
            Metal code for SIMD group sum reduction
        """
        return f"simd_sum({var_name})"

    @staticmethod
    def generate_reduce_product(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group product reduction

        Args:
            type_name: Data type name
            var_name: Variable name

        Returns:
            Metal code for SIMD group product reduction
        """
        return f"simd_product({var_name})"

    @staticmethod
    def generate_reduce_min(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group min reduction

        Args:
            type_name: Data type name
            var_name: Variable name

        Returns:
            Metal code for SIMD group min reduction
        """
        return f"simd_min({var_name})"

    @staticmethod
    def generate_reduce_max(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group max reduction

        Args:
            type_name: Data type name
            var_name: Variable name

        Returns:
            Metal code for SIMD group max reduction
        """
        return f"simd_max({var_name})"

    @staticmethod
    def generate_broadcast(type_name: str, var_name: str, lane_id: int) -> str:
        """Generate code for SIMD group broadcast

        Args:
            type_name: Data type name
            var_name: Variable name
            lane_id: Source lane ID

        Returns:
            Metal code for SIMD group broadcast
        """
        return f"simd_broadcast({var_name}, {lane_id})"

    @staticmethod
    def generate_shuffle(type_name: str, var_name: str, source_lane: str) -> str:
        """Generate code for SIMD group shuffle

        Args:
            type_name: Data type name
            var_name: Variable name
            source_lane: Source lane expression

        Returns:
            Metal code for SIMD group shuffle
        """
        return f"simd_shuffle({var_name}, {source_lane})"

    @staticmethod
    def generate_shuffle_up(type_name: str, var_name: str, delta: int) -> str:
        """Generate code for SIMD group shuffle up

        Args:
            type_name: Data type name
            var_name: Variable name
            delta: Lane delta

        Returns:
            Metal code for SIMD group shuffle up
        """
        return f"simd_shuffle_up({var_name}, {delta})"

    @staticmethod
    def generate_shuffle_down(type_name: str, var_name: str, delta: int) -> str:
        """Generate code for SIMD group shuffle down

        Args:
            type_name: Data type name
            var_name: Variable name
            delta: Lane delta

        Returns:
            Metal code for SIMD group shuffle down
        """
        return f"simd_shuffle_down({var_name}, {delta})"

    @staticmethod
    def generate_shuffle_xor(type_name: str, var_name: str, mask: int) -> str:
        """Generate code for SIMD group shuffle XOR

        Args:
            type_name: Data type name
            var_name: Variable name
            mask: Lane mask

        Returns:
            Metal code for SIMD group shuffle XOR
        """
        return f"simd_shuffle_xor({var_name}, {mask})"

# Create global instances for convenience
thread_mapping = ThreadMapping()
shared_memory = SharedMemory()
simd_group_functions = SIMDGroupFunctions()

def map_kernel_launch_params(grid_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map CUDA-style kernel launch parameters to Metal parameters

    Args:
        grid_config: CUDA-style grid configuration with keys:
            'grid': Tuple of (grid_x, grid_y, grid_z)
            'block': Tuple of (block_x, block_y, block_z)

    Returns:
        Dictionary with Metal parameters:
            'grid_size': Tuple of (grid_x, grid_y, grid_z)
            'threadgroup_size': Tuple of (threads_x, threads_y, threads_z)
            'pipeline_depth': Optional pipeline depth for Metal
    """
    # Extract CUDA-style parameters
    grid = grid_config.get('grid', (1, 1, 1))
    block = grid_config.get('block', (1, 1, 1))

    # Validate dimensions
    if len(grid) != 3 or len(block) != 3:
        raise ValueError("Grid and block must be 3D tuples")

    # Map to Metal parameters
    metal_params = {
        'grid_size': grid,
        'threadgroup_size': block
    }

    # Add any Metal-specific parameters
    if 'pipeline_depth' in grid_config:
        metal_params['pipeline_depth'] = grid_config['pipeline_depth']

    # Check for shared memory
    if 'shared_mem' in grid_config:
        metal_params['shared_memory_size'] = grid_config['shared_mem']

    return metal_params

def get_max_threads_per_threadgroup(chip_generation: Optional[str] = None) -> int:
    """
    Get maximum threads per threadgroup for the given chip generation

    Args:
        chip_generation: Optional Apple Silicon chip generation identifier
            (e.g., 'M1', 'M2', 'M3')

    Returns:
        Maximum number of threads per threadgroup
    """
    # Default maximum threads per threadgroup
    default_max = 1024

    # Chip-specific maximums (may vary by generation)
    chip_maximums = {
        'M1': 1024,
        'M1_PRO': 1024,
        'M1_MAX': 1024,
        'M1_ULTRA': 1024,
        'M2': 1024,
        'M2_PRO': 1024,
        'M2_MAX': 1024,
        'M2_ULTRA': 1024,
        'M3': 1024,
        'M3_PRO': 1024,
        'M3_MAX': 1024
    }

    if chip_generation and chip_generation in chip_maximums:
        return chip_maximums[chip_generation]

    return default_max

def get_simd_width(chip_generation: Optional[str] = None) -> int:
    """
    Get SIMD width for the given chip generation

    Args:
        chip_generation: Optional Apple Silicon chip generation identifier

    Returns:
        SIMD width (typically 32 for Apple GPUs)
    """
    # Metal GPUs typically use a SIMD width of 32
    return 32

def suggest_threadgroup_size(total_threads: int) -> Tuple[int, int, int]:
    """
    Suggest an efficient threadgroup size based on total thread count

    Args:
        total_threads: Total number of threads required

    Returns:
        Tuple of (x, y, z) dimensions for threadgroup
    """
    # Maximum threads per threadgroup
    max_threads = get_max_threads_per_threadgroup()

    # Ensure we don't exceed maximum
    total_threads = min(total_threads, max_threads)

    # For small thread counts, use 1D threadgroups
    if total_threads <= 64:
        return (total_threads, 1, 1)

    # For medium thread counts, use 2D threadgroups
    if total_threads <= 256:
        # Find factors close to square
        from math import sqrt, ceil
        side = int(ceil(sqrt(total_threads)))
        return (side, total_threads // side, 1)

    # For large thread counts, use 3D threadgroups
    # Start with a square base and add layers as needed
    from math import sqrt, ceil, pow
    side = int(ceil(sqrt(min(total_threads, 1024) / 4)))
    depth = min(4, ceil(total_threads / (side * side)))
    return (side, side, depth)

def map_threadgroups_to_grid(
    thread_count: Union[int, Tuple[int, int, int]],
    threads_per_threadgroup: Optional[Union[int, Tuple[int, int, int]]] = None
) -> Dict[str, Tuple[int, int, int]]:
    """
    Map total thread count to Metal grid and threadgroup configuration

    Args:
        thread_count: Total threads or tuple of (x, y, z) dimensions
        threads_per_threadgroup: Optional threads per threadgroup or tuple

    Returns:
        Dictionary with Metal parameters:
            'grid_size': Tuple of (grid_x, grid_y, grid_z)
            'threadgroup_size': Tuple of (threads_x, threads_y, threads_z)
    """
    # Handle different input formats
    if isinstance(thread_count, int):
        thread_count = (thread_count, 1, 1)

    if isinstance(threads_per_threadgroup, int):
        threads_per_threadgroup = suggest_threadgroup_size(threads_per_threadgroup)
    elif threads_per_threadgroup is None:
        # Suggest threadgroup size based on total threads
        total = thread_count[0] * thread_count[1] * thread_count[2]
        threads_per_threadgroup = suggest_threadgroup_size(min(total, 1024))

    # Calculate grid size
    grid_size = (
        (thread_count[0] + threads_per_threadgroup[0] - 1) // threads_per_threadgroup[0],
        (thread_count[1] + threads_per_threadgroup[1] - 1) // threads_per_threadgroup[1],
        (thread_count[2] + threads_per_threadgroup[2] - 1) // threads_per_threadgroup[2]
    )

    return {
        'grid_size': grid_size,
        'threadgroup_size': threads_per_threadgroup
    }

def convert_cuda_index_to_metal(
    grid_dim: Tuple[int, int, int],
    block_dim: Tuple[int, int, int],
    block_idx: Tuple[int, int, int],
    thread_idx: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    Convert CUDA-style indexing to Metal indexing

    Args:
        grid_dim: Grid dimensions
        block_dim: Block dimensions
        block_idx: Block indices
        thread_idx: Thread indices

    Returns:
        Dictionary with Metal indices:
            'thread_position_in_threadgroup': (x, y, z)
            'threadgroup_position_in_grid': (x, y, z)
            'thread_position_in_grid': (x, y, z)
    """
    return {
        'thread_position_in_threadgroup': thread_idx,
        'threadgroup_position_in_grid': block_idx,
        'thread_position_in_grid': (
            block_idx[0] * block_dim[0] + thread_idx[0],
            block_idx[1] * block_dim[1] + thread_idx[1],
            block_idx[2] * block_dim[2] + thread_idx[2]
        )
    }