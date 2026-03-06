#!/usr/bin/env python
import os
import time
import json
import random
import math
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

# Initialize logger
logger = logging.getLogger('metal_auto_tuner')
logger.setLevel(logging.INFO)

# Try to import hardware_capabilities for hardware-aware tuning
try:
    from MLX.metal_hardware_optimizer import hardware_capabilities
    HAS_HARDWARE_OPTIMIZER = True
except ImportError:
    HAS_HARDWARE_OPTIMIZER = False
    logger.warning("metal_hardware_optimizer not found. Hardware-specific optimizations will be disabled.")

class ParamType(Enum):
    """Parameter types for auto-tuning"""
    INT = 'int'          # Integer parameter
    FLOAT = 'float'      # Floating point parameter
    CATEGORICAL = 'cat'  # Categorical parameter
    BOOL = 'bool'        # Boolean parameter
    POWER_OF_TWO = 'pow2'  # Power of two parameter

@dataclass
class TunableParam:
    """Definition of a tunable parameter"""
    name: str
    param_type: ParamType
    default_value: Any
    possible_values: Optional[List[Any]] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step: Optional[Any] = None
    log_scale: bool = False
    depends_on: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate parameter configuration"""
        if self.param_type == ParamType.CATEGORICAL and not self.possible_values:
            raise ValueError(f"Parameter '{self.name}' is categorical but no possible values provided")

        if self.param_type == ParamType.BOOL:
            self.possible_values = [True, False]

        if self.param_type == ParamType.POWER_OF_TWO:
            if not self.min_value or not self.max_value:
                raise ValueError(f"Power of two parameter '{self.name}' requires min_value and max_value")
            # Convert min and max to log2 values
            self.log2_min = int(math.log2(self.min_value))
            self.log2_max = int(math.log2(self.max_value))
            # Generate possible values as powers of 2
            self.possible_values = [2**i for i in range(self.log2_min, self.log2_max + 1)]

    def sample(self, rng=None) -> Any:
        """Sample a random value for this parameter

        Args:
            rng: Optional random number generator

        Returns:
            Sampled parameter value
        """
        if rng is None:
            rng = random.Random()

        if self.param_type == ParamType.CATEGORICAL or self.param_type == ParamType.BOOL or self.param_type == ParamType.POWER_OF_TWO:
            return rng.choice(self.possible_values)

        elif self.param_type == ParamType.INT:
            if self.log_scale:
                log_min = math.log(self.min_value)
                log_max = math.log(self.max_value)
                return int(math.exp(rng.uniform(log_min, log_max)))
            else:
                return rng.randint(self.min_value, self.max_value)

        elif self.param_type == ParamType.FLOAT:
            if self.log_scale:
                log_min = math.log(self.min_value)
                log_max = math.log(self.max_value)
                return math.exp(rng.uniform(log_min, log_max))
            else:
                return rng.uniform(self.min_value, self.max_value)

        return self.default_value

@dataclass
class ConfigurationResult:
    """Result of evaluating a parameter configuration"""
    config: Dict[str, Any]
    runtime_ms: float
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'config': self.config,
            'runtime_ms': self.runtime_ms,
            'success': self.success,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigurationResult':
        """Create from dictionary"""
        return cls(
            config=data['config'],
            runtime_ms=data['runtime_ms'],
            success=data['success'],
            metrics=data.get('metrics', {}),
            timestamp=data.get('timestamp', time.time())
        )

class SearchStrategy:
    """Base class for auto-tuning search strategies"""

    def __init__(self, tunable_params: List[TunableParam], seed: Optional[int] = None):
        """Initialize search strategy

        Args:
            tunable_params: List of tunable parameters
            seed: Optional random seed
        """
        self.tunable_params = tunable_params
        self.param_map = {param.name: param for param in tunable_params}
        self.rng = random.Random(seed)
        self.history: List[ConfigurationResult] = []

    def next_config(self) -> Dict[str, Any]:
        """Get next parameter configuration to try

        Returns:
            Dictionary with parameter values
        """
        raise NotImplementedError("Search strategies must implement next_config")

    def update(self, result: ConfigurationResult):
        """Update strategy with result of a configuration evaluation

        Args:
            result: Result of evaluating a configuration
        """
        self.history.append(result)

    def best_config(self) -> Optional[ConfigurationResult]:
        """Get best configuration found so far

        Returns:
            Best configuration result or None if no successful configurations
        """
        successful = [r for r in self.history if r.success]
        if not successful:
            return None
        return min(successful, key=lambda x: x.runtime_ms)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration using default values for all parameters

        Returns:
            Dictionary with default parameter values
        """
        return {param.name: param.default_value for param in self.tunable_params}

    def random_config(self) -> Dict[str, Any]:
        """Generate a random configuration

        Returns:
            Dictionary with random parameter values
        """
        config = {}
        for param in self.tunable_params:
            config[param.name] = param.sample(self.rng)
        return config

class GridSearchStrategy(SearchStrategy):
    """Grid search strategy that tries all combinations of parameter values"""

    def __init__(self, tunable_params: List[TunableParam], seed: Optional[int] = None):
        """Initialize grid search strategy

        Args:
            tunable_params: List of tunable parameters
            seed: Optional random seed
        """
        super().__init__(tunable_params, seed)
        self.grid = self._build_grid()
        self.remaining_configs = list(self._grid_generator())
        self.rng.shuffle(self.remaining_configs)

    def _build_grid(self) -> Dict[str, List[Any]]:
        """Build grid of parameter values

        Returns:
            Dictionary mapping parameter names to possible values
        """
        grid = {}

        for param in self.tunable_params:
            if param.param_type == ParamType.CATEGORICAL or param.param_type == ParamType.BOOL or param.param_type == ParamType.POWER_OF_TWO:
                grid[param.name] = param.possible_values

            elif param.param_type == ParamType.INT:
                if param.step is None:
                    step = 1
                else:
                    step = param.step
                grid[param.name] = list(range(param.min_value, param.max_value + 1, step))

            elif param.param_type == ParamType.FLOAT:
                if param.possible_values:
                    grid[param.name] = param.possible_values
                else:
                    if param.step is None:
                        # For floats without step, create a reasonable number of values
                        steps = 5
                        step = (param.max_value - param.min_value) / steps
                    else:
                        step = param.step

                    values = []
                    current = param.min_value
                    while current <= param.max_value:
                        values.append(current)
                        current += step
                    grid[param.name] = values

        return grid

    def _grid_generator(self):
        """Generator for all grid configurations

        Yields:
            Parameter configurations
        """
        param_names = list(self.grid.keys())

        def generate_configs(index, current_config):
            if index == len(param_names):
                yield current_config.copy()
                return

            param_name = param_names[index]
            for value in self.grid[param_name]:
                current_config[param_name] = value
                yield from generate_configs(index + 1, current_config)

        yield from generate_configs(0, {})

    def next_config(self) -> Dict[str, Any]:
        """Get next parameter configuration to try

        Returns:
            Dictionary with parameter values
        """
        if not self.remaining_configs:
            # All configurations tried, return best one or random one
            best = self.best_config()
            if best:
                return best.config
            return self.random_config()

        return self.remaining_configs.pop(0)

class RandomSearchStrategy(SearchStrategy):
    """Random search strategy that tries random parameter configurations"""

    def __init__(self, tunable_params: List[TunableParam], n_trials: int = 100, seed: Optional[int] = None):
        """Initialize random search strategy

        Args:
            tunable_params: List of tunable parameters
            n_trials: Number of trials
            seed: Optional random seed
        """
        super().__init__(tunable_params, seed)
        self.n_trials = n_trials
        self.current_trial = 0

    def next_config(self) -> Dict[str, Any]:
        """Get next parameter configuration to try

        Returns:
            Dictionary with parameter values
        """
        self.current_trial += 1

        # Include default configuration in trials
        if self.current_trial == 1:
            return self.get_default_config()

        # Generate random configurations
        return self.random_config()

try:
    import skopt
    from skopt.space import Real, Integer, Categorical
    from skopt import gp_minimize, dummy_minimize
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn("scikit-optimize not found. Bayesian optimization will not be available.")

class BayesianOptimizationStrategy(SearchStrategy):
    """Bayesian optimization search strategy using Gaussian Processes"""

    def __init__(self, tunable_params: List[TunableParam], n_initial_points: int = 10,
                 n_calls: int = 50, seed: Optional[int] = None):
        """Initialize Bayesian optimization strategy

        Args:
            tunable_params: List of tunable parameters
            n_initial_points: Number of initial random points
            n_calls: Total number of function evaluations
            seed: Optional random seed
        """
        super().__init__(tunable_params, seed)

        if not HAS_SKOPT:
            raise ImportError("scikit-optimize is required for Bayesian optimization. Install it with 'pip install scikit-optimize'")

        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.current_call = 0

        # Create search space for skopt
        self.space = []
        self.param_names = []

        for param in tunable_params:
            self.param_names.append(param.name)

            if param.param_type == ParamType.CATEGORICAL or param.param_type == ParamType.BOOL:
                self.space.append(Categorical(param.possible_values, name=param.name))

            elif param.param_type == ParamType.INT or param.param_type == ParamType.POWER_OF_TWO:
                self.space.append(Integer(
                    param.min_value if param.param_type == ParamType.INT else param.possible_values[0],
                    param.max_value if param.param_type == ParamType.INT else param.possible_values[-1],
                    name=param.name
                ))

            elif param.param_type == ParamType.FLOAT:
                self.space.append(Real(
                    param.min_value,
                    param.max_value,
                    name=param.name,
                    prior='log-uniform' if param.log_scale else 'uniform'
                ))

        # Initialize optimizer
        self.optimizer_results = None

        # List to track tried configs to avoid duplicates
        self.tried_configs = set()

    def next_config(self) -> Dict[str, Any]:
        """Get next parameter configuration to try

        Returns:
            Dictionary with parameter values
        """
        self.current_call += 1

        # First trial: default configuration
        if self.current_call == 1:
            config = self.get_default_config()
            config_hash = self._hash_config(config)
            self.tried_configs.add(config_hash)
            return config

        # Initial random points
        if self.current_call <= self.n_initial_points + 1:
            # Generate random configuration
            while True:
                config = self.random_config()
                config_hash = self._hash_config(config)
                if config_hash not in self.tried_configs:
                    self.tried_configs.add(config_hash)
                    return config

        # Use Bayesian optimization to suggest next points
        if not self.optimizer_results or len(self.history) != self.current_call - 1:
            # Not enough results to update optimizer yet
            return self.random_config()

        # Extract history in the format expected by skopt
        X = []
        y = []

        for result in self.history:
            # Skip failed evaluations
            if not result.success:
                continue

            # Extract parameters in the order of self.param_names
            x = [result.config.get(name) for name in self.param_names]
            X.append(x)
            y.append(result.runtime_ms)

        if len(X) < 2:
            # Not enough successful evaluations yet
            return self.random_config()

        # Update optimizer with observed points
        try:
            self.optimizer_results = gp_minimize(
                lambda x: 0,  # Dummy function, we're just using tell() to update
                self.space,
                n_calls=0,
                random_state=self.rng.randint(0, 2**32 - 1)
            )

            # Update optimizer with observed points
            from skopt import Optimizer
            opt = Optimizer(self.space, random_state=self.rng.randint(0, 2**32 - 1))
            opt.tell(X, y)

            # Ask for the next point
            next_x = opt.ask()

            # Convert to parameter configuration
            config = {name: value for name, value in zip(self.param_names, next_x)}

            # Convert numerical types appropriately
            for param in self.tunable_params:
                if param.name in config:
                    if param.param_type == ParamType.INT or param.param_type == ParamType.POWER_OF_TWO:
                        config[param.name] = int(config[param.name])
                    elif param.param_type == ParamType.FLOAT:
                        config[param.name] = float(config[param.name])

            # Check if this configuration has been tried
            config_hash = self._hash_config(config)
            if config_hash in self.tried_configs:
                # If duplicate, generate random config instead
                return self.random_config()

            self.tried_configs.add(config_hash)
            return config

        except Exception as e:
            # Fallback to random search if optimizer fails
            logger.warning(f"Bayesian optimization failed: {str(e)}. Falling back to random search.")
            return self.random_config()

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash for a configuration

        Args:
            config: Parameter configuration

        Returns:
            Hash string for the configuration
        """
        # Sort by key to ensure consistent order
        sorted_items = sorted(config.items())
        # Convert to string and hash
        config_str = json.dumps(sorted_items)
        return hashlib.md5(config_str.encode()).hexdigest()

class MetalAutoTuner:
    """Auto-tuner for Metal backend"""

    def __init__(self, kernel_name: str, tunable_params: List[TunableParam],
                 n_trials: int = 100, search_strategy: str = 'random',
                 cache_dir: Optional[str] = None, seed: Optional[int] = None,
                 operation_type: str = 'general', use_hardware_optimizer: bool = True):
        """Initialize Metal auto-tuner

        Args:
            kernel_name: Name of the kernel to tune
            tunable_params: List of tunable parameters
            n_trials: Number of trials to run
            search_strategy: Search strategy ('random', 'grid', or 'bayesian')
            cache_dir: Directory to cache results
            seed: Random seed
            operation_type: Type of operation ('matmul', 'conv', 'general')
            use_hardware_optimizer: Whether to use hardware-specific optimizations
        """
        self.kernel_name = kernel_name
        self.n_trials = n_trials
        self.seed = seed
        self.operation_type = operation_type

        # Apply hardware-specific optimizations to parameters if available
        if use_hardware_optimizer and HAS_HARDWARE_OPTIMIZER:
            logger.info(f"Applying hardware-specific optimizations for {hardware_capabilities.chip_generation.name}")
            self.tunable_params = hardware_capabilities.optimize_search_space(
                tunable_params, operation_type
            )
        else:
            self.tunable_params = tunable_params

        # Directory for caching results
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            cache_dir_env = os.environ.get('METAL_TUNER_CACHE_DIR')
            if cache_dir_env:
                self.cache_dir = cache_dir_env
            else:
                self.cache_dir = os.path.join(os.path.expanduser('~'), '.triton', 'metal', 'tuner_cache')

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize search strategy
        if search_strategy == 'random':
            self.strategy = RandomSearchStrategy(self.tunable_params, n_trials, seed)
        elif search_strategy == 'grid':
            self.strategy = GridSearchStrategy(self.tunable_params, seed)
        elif search_strategy == 'bayesian':
            if not HAS_SKOPT:
                logger.warning("scikit-optimize not found. Falling back to random search.")
                self.strategy = RandomSearchStrategy(self.tunable_params, n_trials, seed)
            else:
                self.strategy = BayesianOptimizationStrategy(self.tunable_params, n_calls=n_trials, seed=seed)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

        # Load cache
        self.cache_path = os.path.join(self.cache_dir, f"{self.kernel_name}.json")
        self.load_cache()

    def get_hardware_constraints(self) -> Dict[str, Any]:
        """
        Get hardware-specific constraints for the current operation

        Returns:
            Dictionary of hardware constraints or empty dict if hardware optimizer not available
        """
        if HAS_HARDWARE_OPTIMIZER:
            return hardware_capabilities.get_auto_tuner_constraints(self.operation_type)
        return {}

    def get_hardware_optimized_default_config(self) -> Dict[str, Any]:
        """
        Get a hardware-optimized default configuration

        Returns:
            Dictionary with optimized default parameter values
        """
        if not HAS_HARDWARE_OPTIMIZER:
            return self.strategy.get_default_config()

        # Get hardware constraints
        constraints = self.get_hardware_constraints()
        default_config = self.strategy.get_default_config()

        # Apply operation-specific optimizations
        if self.operation_type == 'matmul':
            # Get recommended block sizes
            if 'recommended_block_sizes' in constraints and constraints['recommended_block_sizes']:
                block_m, block_n, block_k = constraints['recommended_block_sizes'][0]
                if 'block_m' in default_config:
                    default_config['block_m'] = block_m
                if 'block_n' in default_config:
                    default_config['block_n'] = block_n
                if 'block_k' in default_config:
                    default_config['block_k'] = block_k

            # Set warps and stages
            if 'recommended_num_warps' in constraints and constraints['recommended_num_warps']:
                if 'num_warps' in default_config:
                    default_config['num_warps'] = constraints['recommended_num_warps'][0]

            if 'recommended_num_stages' in constraints and constraints['recommended_num_stages']:
                if 'num_stages' in default_config:
                    default_config['num_stages'] = constraints['recommended_num_stages'][0]

            # Set simdgroup matrix usage
            if 'uses_simdgroup_matrix' in constraints:
                if 'use_simdgroup_matrix' in default_config:
                    default_config['use_simdgroup_matrix'] = constraints['uses_simdgroup_matrix']

        elif self.operation_type == 'conv':
            # Get recommended block sizes
            if 'recommended_block_sizes' in constraints and constraints['recommended_block_sizes']:
                block_x, block_y, block_z, block_c = constraints['recommended_block_sizes'][0]
                if 'block_x' in default_config:
                    default_config['block_x'] = block_x
                if 'block_y' in default_config:
                    default_config['block_y'] = block_y
                if 'block_z' in default_config:
                    default_config['block_z'] = block_z
                if 'block_c' in default_config:
                    default_config['block_c'] = block_c

            # Set warps, stages, and filter tile size
            if 'recommended_num_warps' in constraints and constraints['recommended_num_warps']:
                if 'num_warps' in default_config:
                    default_config['num_warps'] = constraints['recommended_num_warps'][0]

            if 'recommended_num_stages' in constraints and constraints['recommended_num_stages']:
                if 'num_stages' in default_config:
                    default_config['num_stages'] = constraints['recommended_num_stages'][0]

            if 'filter_tile_sizes' in constraints and constraints['filter_tile_sizes']:
                if 'filter_tile_size' in default_config:
                    default_config['filter_tile_size'] = constraints['filter_tile_sizes'][0]

        return default_config

    def load_cache(self):
        """Load cached results"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)

                # Process cached results
                cached_results = []
                for result_data in cache_data.get('results', []):
                    cached_results.append(ConfigurationResult.from_dict(result_data))

                # Update strategy with cached results
                for result in cached_results:
                    self.strategy.update(result)

                logger.info(f"Loaded {len(cached_results)} cached results for kernel {self.kernel_name}")
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")

    def save_cache(self):
        """Save results to cache"""
        try:
            cache_data = {
                'kernel_name': self.kernel_name,
                'results': [result.to_dict() for result in self.strategy.history]
            }

            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Saved {len(self.strategy.history)} results to cache for kernel {self.kernel_name}")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")

    def tune(self, evaluation_func: Callable[[Dict[str, Any]], ConfigurationResult],
             max_trials: Optional[int] = None, timeout_seconds: Optional[float] = None,
             parallel: bool = False, num_workers: int = 4) -> Dict[str, Any]:
        """Run auto-tuning process

        Args:
            evaluation_func: Function to evaluate a configuration
            max_trials: Maximum number of trials to run (overrides n_trials)
            timeout_seconds: Maximum seconds to run tuning
            parallel: Whether to run evaluations in parallel
            num_workers: Number of worker threads when parallel is True

        Returns:
            Best parameter configuration found
        """
        n_trials = max_trials if max_trials is not None else self.n_trials
        start_time = time.time()

        # First try the hardware-optimized default configuration if available
        default_config = self.get_hardware_optimized_default_config()
        logger.info(f"Evaluating hardware-optimized default configuration: {default_config}")

        # Evaluate default configuration
        result = evaluation_func(default_config)
        self.strategy.update(result)

        # Save default result to cache
        self.save_cache()

        # Already have best config from history?
        best_from_history = self.strategy.best_config()
        if best_from_history and n_trials <= 1:
            logger.info(f"Using best configuration from history: {best_from_history.config}")
            return best_from_history.config

        # Setup mutex for parallel tuning
        if parallel:
            strategy_lock = threading.Lock()

            def worker_task(config):
                # Evaluate configuration
                result = evaluation_func(config)

                # Update strategy with mutex to prevent race conditions
                with strategy_lock:
                    self.strategy.update(result)

                return result

            # If running in parallel, generate all configs upfront
            pending_configs = []
            for i in range(n_trials - 1):  # Subtract 1 for default configuration
                # Check timeout
                if timeout_seconds and time.time() - start_time > timeout_seconds:
                    logger.info(f"Timeout reached during config generation after {i+1} trials")
                    break

                # Get next configuration to try
                config = self.strategy.next_config()
                pending_configs.append(config)
                logger.info(f"Generated config for trial {i+2}/{n_trials}: {config}")

            # Execute evaluations in parallel
            logger.info(f"Running {len(pending_configs)} configurations in parallel with {num_workers} workers")
            completed = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_config = {
                    executor.submit(worker_task, config): config
                    for config in pending_configs
                }

                for future in future_to_config:
                    try:
                        future.result()
                        completed += 1

                        # Save intermediate results every 5 completed configs
                        if completed % 5 == 0:
                            with strategy_lock:
                                self.save_cache()

                        # Check timeout
                        if timeout_seconds and time.time() - start_time > timeout_seconds:
                            logger.info(f"Timeout reached after {completed+1} completed trials")
                            executor.shutdown(wait=False)
                            break

                    except Exception as e:
                        logger.error(f"Error in worker task: {e}")

            # Final save after parallel execution
            self.save_cache()
        else:
            # Serial execution mode
            # Run trials
            for i in range(n_trials - 1):  # Subtract 1 for default configuration
                # Check timeout
                if timeout_seconds and time.time() - start_time > timeout_seconds:
                    logger.info(f"Timeout reached after {i+1} trials")
                    break

                # Get next configuration to try
                config = self.strategy.next_config()
                logger.info(f"Trial {i+2}/{n_trials}: {config}")

                # Evaluate configuration
                result = evaluation_func(config)
                self.strategy.update(result)

                # Save intermediate results
                if (i + 1) % 5 == 0:
                    self.save_cache()

            # Save final results
            self.save_cache()

        # Return best configuration
        best = self.strategy.best_config()
        if best:
            logger.info(f"Best configuration: {best.config} (runtime: {best.runtime_ms:.3f} ms)")
            return best.config
        else:
            logger.warning("No successful configurations found")
            return self.strategy.get_default_config()

# Common parameter definitions for Metal-specific tuning
def get_common_metal_params() -> List[TunableParam]:
    """Get common tunable parameters for Metal backend

    Returns:
        List of common tunable parameters
    """
    return [
        TunableParam(
            name="block_size",
            param_type=ParamType.POWER_OF_TWO,
            default_value=128,
            min_value=32,
            max_value=1024,
        ),
        TunableParam(
            name="num_warps",
            param_type=ParamType.INT,
            default_value=4,
            min_value=1,
            max_value=32,
        ),
        TunableParam(
            name="num_stages",
            param_type=ParamType.INT,
            default_value=2,
            min_value=1,
            max_value=5,
        ),
        TunableParam(
            name="use_vectorization",
            param_type=ParamType.BOOL,
            default_value=True,
        ),
        TunableParam(
            name="vectorize_load_size",
            param_type=ParamType.POWER_OF_TWO,
            default_value=4,
            min_value=1,
            max_value=16,
        ),
        TunableParam(
            name="use_prefetching",
            param_type=ParamType.BOOL,
            default_value=True,
        ),
        TunableParam(
            name="memory_layout",
            param_type=ParamType.CATEGORICAL,
            default_value="row_major",
            possible_values=["row_major", "column_major"],
        ),
        TunableParam(
            name="thread_coarsening",
            param_type=ParamType.INT,
            default_value=1,
            min_value=1,
            max_value=8,
        ),
        TunableParam(
            name="loop_unroll_factor",
            param_type=ParamType.POWER_OF_TWO,
            default_value=1,
            min_value=1,
            max_value=16,
        ),
    ]

# Matmul-specific parameter definitions for Metal-specific tuning
def get_matmul_metal_params() -> List[TunableParam]:
    """Get tunable parameters for matrix multiplication on Metal

    Returns:
        List of tunable parameters for matrix multiplication
    """
    params = [
        TunableParam(
            name="block_m",
            param_type=ParamType.POWER_OF_TWO,
            default_value=64,
            min_value=16,
            max_value=256,
        ),
        TunableParam(
            name="block_n",
            param_type=ParamType.POWER_OF_TWO,
            default_value=64,
            min_value=16,
            max_value=256,
        ),
        TunableParam(
            name="block_k",
            param_type=ParamType.POWER_OF_TWO,
            default_value=32,
            min_value=8,
            max_value=128,
        ),
        TunableParam(
            name="split_k",
            param_type=ParamType.POWER_OF_TWO,
            default_value=1,
            min_value=1,
            max_value=16,
        ),
        TunableParam(
            name="num_warps",
            param_type=ParamType.INT,
            default_value=4,
            min_value=1,
            max_value=16,
        ),
        TunableParam(
            name="num_stages",
            param_type=ParamType.INT,
            default_value=2,
            min_value=1,
            max_value=5,
        ),
        TunableParam(
            name="use_simdgroup_matrix",
            param_type=ParamType.BOOL,
            default_value=True,
        ),
    ]

    # Apply hardware-specific optimizations if available
    if HAS_HARDWARE_OPTIMIZER:
        params = hardware_capabilities.optimize_search_space(params, 'matmul')

    return params

# Convolution-specific parameter definitions for Metal-specific tuning
def get_conv_metal_params() -> List[TunableParam]:
    """Get tunable parameters for convolution operations on Metal

    Returns:
        List of tunable parameters for convolution operations
    """
    params = [
        TunableParam(
            name="block_x",
            param_type=ParamType.POWER_OF_TWO,
            default_value=16,
            min_value=8,
            max_value=64,
        ),
        TunableParam(
            name="block_y",
            param_type=ParamType.POWER_OF_TWO,
            default_value=16,
            min_value=8,
            max_value=64,
        ),
        TunableParam(
            name="block_z",
            param_type=ParamType.POWER_OF_TWO,
            default_value=4,
            min_value=1,
            max_value=16,
        ),
        TunableParam(
            name="block_c",
            param_type=ParamType.POWER_OF_TWO,
            default_value=32,
            min_value=8,
            max_value=64,
        ),
        TunableParam(
            name="num_warps",
            param_type=ParamType.INT,
            default_value=4,
            min_value=1,
            max_value=16,
        ),
        TunableParam(
            name="num_stages",
            param_type=ParamType.INT,
            default_value=2,
            min_value=1,
            max_value=4,
        ),
        TunableParam(
            name="use_shared_memory",
            param_type=ParamType.BOOL,
            default_value=True,
        ),
        TunableParam(
            name="filter_tile_size",
            param_type=ParamType.INT,
            default_value=3,
            min_value=1,
            max_value=7,
        ),
        TunableParam(
            name="use_winograd",
            param_type=ParamType.BOOL,
            default_value=False,
        ),
        TunableParam(
            name="vectorize_load_size",
            param_type=ParamType.POWER_OF_TWO,
            default_value=4,
            min_value=1,
            max_value=8,
        ),
    ]

    # Apply hardware-specific optimizations if available
    if HAS_HARDWARE_OPTIMIZER:
        params = hardware_capabilities.optimize_search_space(params, 'conv')

    return params

# Example usage
if __name__ == "__main__":
    print("Metal Auto-Tuner Sample")

    # Define tunable parameters
    params = get_common_metal_params()

    # Create auto-tuner
    tuner = MetalAutoTuner("example_kernel", params, n_trials=10, search_strategy="random")

    # Define evaluation function (dummy implementation)
    def evaluate_config(config: Dict[str, Any]) -> ConfigurationResult:
        # Simulate some computation
        time.sleep(0.1)

        # Calculate a dummy runtime based on configuration
        runtime = 10 + random.random() * 10

        # Simple heuristic: prefer larger block sizes but penalty for too many warps
        block_size = config.get("block_size", 128)
        num_warps = config.get("num_warps", 4)

        runtime -= block_size / 256  # Larger block size is better (up to a point)
        runtime += num_warps * 0.5   # More warps adds overhead

        return ConfigurationResult(
            config=config,
            runtime_ms=runtime,
            success=True,
            metrics={"throughput": 1000.0 / runtime}
        )

    # Run tuning
    best_config = tuner.tune(evaluate_config, max_trials=5)
    print(f"Best configuration: {best_config}")