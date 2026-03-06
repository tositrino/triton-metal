#!/usr/bin/env python
import os
import time
import json
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

# Initialize logger
logger = logging.getLogger('metal_instrumentation')

@dataclass
class PerformanceCounter:
    """Performance counter for a specific Metal operation"""
    name: str
    count: int = 0
    total_time_ns: int = 0
    min_time_ns: int = 0
    max_time_ns: int = 0
    avg_time_ns: int = 0

    def update(self, time_ns: int):
        """Update counter with a new measurement"""
        self.count += 1
        self.total_time_ns += time_ns
        self.min_time_ns = min(self.min_time_ns, time_ns) if self.min_time_ns > 0 else time_ns
        self.max_time_ns = max(self.max_time_ns, time_ns)
        self.avg_time_ns = self.total_time_ns // self.count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'name': self.name,
            'count': self.count,
            'total_time_ms': self.total_time_ns / 1_000_000,
            'min_time_ms': self.min_time_ns / 1_000_000,
            'max_time_ms': self.max_time_ns / 1_000_000,
            'avg_time_ms': self.avg_time_ns / 1_000_000,
        }

@dataclass
class DebugInfo:
    """Debug information for a Metal kernel"""
    kernel_name: str
    source_file: str
    line_number: int
    variable_values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'kernel_name': self.kernel_name,
            'source_file': self.source_file,
            'line_number': self.line_number,
            'variable_values': self.variable_values,
        }

class MetalInstrumentation:
    """Metal instrumentation for debugging and performance analysis"""

    def __init__(self, debug_level: int = 0):
        """Initialize Metal instrumentation

        Args:
            debug_level: Debug level (0=off, 1=errors, 2=warnings, 3=info, 4=debug)
        """
        self.debug_level = debug_level
        self.performance_counters: Dict[str, PerformanceCounter] = {}
        self.debug_info_list: List[DebugInfo] = []
        self.active_timers: Dict[str, float] = {}
        self.output_dir = os.environ.get('METAL_DEBUG_OUTPUT_DIR', '/tmp/metal_debug')

        # Initialize logging level based on debug_level
        log_levels = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
        log_level = log_levels[min(debug_level, len(log_levels) - 1)]
        logger.setLevel(log_level)

        # Create output directory if it doesn't exist
        if self.debug_level > 0 and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations

        Args:
            operation_name: Name of the operation being timed
        """
        if self.debug_level == 0:
            yield
            return

        timer_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        self.start_timer(timer_id)
        try:
            yield
        finally:
            elapsed_ns = self.stop_timer(timer_id)

            # Get or create performance counter
            if operation_name not in self.performance_counters:
                self.performance_counters[operation_name] = PerformanceCounter(operation_name)

            # Update counter
            self.performance_counters[operation_name].update(elapsed_ns)

    def start_timer(self, timer_id: str):
        """Start a timer

        Args:
            timer_id: Unique identifier for this timer
        """
        self.active_timers[timer_id] = time.time_ns()

    def stop_timer(self, timer_id: str) -> int:
        """Stop a timer and return elapsed time in nanoseconds

        Args:
            timer_id: Unique identifier for this timer

        Returns:
            Elapsed time in nanoseconds
        """
        if timer_id not in self.active_timers:
            return 0

        start_time = self.active_timers.pop(timer_id)
        return time.time_ns() - start_time

    def record_debug_info(self, kernel_name: str, source_file: str, line_number: int,
                          variable_values: Optional[Dict[str, Any]] = None):
        """Record debug information for a kernel

        Args:
            kernel_name: Name of the kernel
            source_file: Source file containing the kernel
            line_number: Line number in the source file
            variable_values: Optional dictionary of variable values to record
        """
        if self.debug_level < 3:
            return

        debug_info = DebugInfo(
            kernel_name=kernel_name,
            source_file=source_file,
            line_number=line_number,
            variable_values=variable_values or {}
        )
        self.debug_info_list.append(debug_info)

        if self.debug_level >= 4:
            logger.debug(f"Debug info recorded: {kernel_name} at {source_file}:{line_number}")

    def insert_debug_prints(self, kernel_src: str, kernel_name: str) -> str:
        """Insert debug print statements into kernel source

        Args:
            kernel_src: Kernel source code
            kernel_name: Name of the kernel

        Returns:
            Modified kernel source with debug prints
        """
        if self.debug_level < 3:
            return kernel_src

        # Simple implementation - insert debug header with timestamp and kernel name
        debug_header = f"""
        // Debug instrumentation inserted by MetalInstrumentation
        #include <metal_stdlib>
        using namespace metal;

        // Original kernel source follows
        """

        # Insert function to print variable values
        debug_print = f"""
        void _metal_debug_print(uint thread_id, int line, float value) {{
            // In a real implementation, this would use Metal's printf or logging infrastructure
            // For now, it's just a placeholder
        }}
        """

        return debug_header + debug_print + kernel_src

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report

        Returns:
            Dictionary with performance data
        """
        report = {
            'summary': {
                'total_operations': sum(counter.count for counter in self.performance_counters.values()),
                'total_time_ms': sum(counter.total_time_ns for counter in self.performance_counters.values()) / 1_000_000,
                'total_counters': len(self.performance_counters),
            },
            'counters': [counter.to_dict() for counter in self.performance_counters.values()],
        }

        # Save report to file if debug level is high enough
        if self.debug_level >= 2:
            report_path = os.path.join(self.output_dir, f"metal_perf_{int(time.time())}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {report_path}")

        return report

    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate debug report

        Returns:
            Dictionary with debug data
        """
        report = {
            'summary': {
                'total_debug_records': len(self.debug_info_list),
            },
            'debug_info': [info.to_dict() for info in self.debug_info_list],
        }

        # Save report to file if debug level is high enough
        if self.debug_level >= 3:
            report_path = os.path.join(self.output_dir, f"metal_debug_{int(time.time())}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Debug report saved to {report_path}")

        return report

    def clear(self):
        """Clear all instrumentation data"""
        self.performance_counters.clear()
        self.debug_info_list.clear()
        self.active_timers.clear()


class ErrorDiagnostics:
    """Error diagnostics system for Metal backend"""

    def __init__(self):
        """Initialize error diagnostics"""
        self.error_codes = {
            'E001': 'Metal shader compilation failed',
            'E002': 'Kernel launch failed',
            'E003': 'Unsupported data type',
            'E004': 'Memory allocation failed',
            'E005': 'Synchronization error',
            'E006': 'Resource binding error',
            'E007': 'Execution timeout',
            'E008': 'Unsupported operation',
            'E009': 'Invalid memory access',
            'E010': 'Metal API error',
        }
        self.error_log: List[Dict[str, Any]] = []

    def diagnose_error(self, error_message: str, kernel_name: Optional[str] = None,
                       source_code: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Diagnose an error and provide helpful information

        Args:
            error_message: Error message from Metal
            kernel_name: Optional kernel name
            source_code: Optional source code

        Returns:
            Tuple of (error_code, error_description, suggestions)
        """
        error_code = 'E010'  # Default to generic Metal API error
        suggestions = []

        # Diagnose common Metal errors
        if 'compilation failed' in error_message.lower():
            error_code = 'E001'
            suggestions = [
                'Check for syntax errors in Metal shader code',
                'Verify all variable types are compatible with Metal',
                'Check for unsupported Metal features or missing imports',
            ]
        elif 'out of memory' in error_message.lower():
            error_code = 'E004'
            suggestions = [
                'Reduce workgroup or grid size',
                'Split operation into multiple smaller kernels',
                'Reduce shared memory usage',
            ]
        elif 'timeout' in error_message.lower():
            error_code = 'E007'
            suggestions = [
                'Optimize kernel to reduce execution time',
                'Split long-running kernels into smaller chunks',
                'Check for infinite loops',
            ]
        elif 'unsupported' in error_message.lower():
            error_code = 'E008'
            suggestions = [
                'Check if the operation is supported on this Metal version',
                'Consider using alternative operations',
                'Check Apple Silicon generation compatibility',
            ]
        elif 'invalid memory access' in error_message.lower() or 'access violation' in error_message.lower():
            error_code = 'E009'
            suggestions = [
                'Check for out-of-bounds array accesses',
                'Verify buffer sizes are sufficient',
                'Check thread/workgroup IDs and bounds checking',
            ]

        # Log the error
        self.log_error(error_code, error_message, kernel_name, source_code)

        return error_code, self.error_codes[error_code], suggestions

    def log_error(self, error_code: str, error_message: str, kernel_name: Optional[str] = None,
                  source_code: Optional[str] = None):
        """Log an error for later analysis

        Args:
            error_code: Error code
            error_message: Detailed error message
            kernel_name: Optional kernel name
            source_code: Optional source code
        """
        error_entry = {
            'timestamp': time.time(),
            'error_code': error_code,
            'description': self.error_codes.get(error_code, 'Unknown error'),
            'message': error_message,
            'kernel_name': kernel_name,
        }

        # Store source code hash instead of full source for privacy/size
        if source_code:
            error_entry['source_hash'] = hashlib.md5(source_code.encode('utf-8')).hexdigest()

        self.error_log.append(error_entry)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of logged errors

        Returns:
            Dictionary with error summary
        """
        error_counts = {}
        for entry in self.error_log:
            code = entry['error_code']
            error_counts[code] = error_counts.get(code, 0) + 1

        return {
            'total_errors': len(self.error_log),
            'unique_error_codes': len(error_counts),
            'error_counts': error_counts,
            'most_recent': self.error_log[-1] if self.error_log else None,
        }


# Global instrumentation instance
_metal_instrumentation = None

def get_metal_instrumentation() -> MetalInstrumentation:
    """Get the global Metal instrumentation instance

    Returns:
        MetalInstrumentation instance
    """
    global _metal_instrumentation
    if _metal_instrumentation is None:
        debug_level = int(os.environ.get('METAL_DEBUG_LEVEL', '0'))
        _metal_instrumentation = MetalInstrumentation(debug_level)
    return _metal_instrumentation

# Global error diagnostics instance
_error_diagnostics = None

def get_error_diagnostics() -> ErrorDiagnostics:
    """Get the global error diagnostics instance

    Returns:
        ErrorDiagnostics instance
    """
    global _error_diagnostics
    if _error_diagnostics is None:
        _error_diagnostics = ErrorDiagnostics()
    return _error_diagnostics