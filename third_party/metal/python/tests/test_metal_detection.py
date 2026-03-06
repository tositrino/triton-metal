#!/usr/bin/env python3
"""
测试Metal后端的检测和初始化功能
"""

import os
import sys
import platform
import unittest



# 修复Metal后端路径
metal_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, metal_backend_dir)

# 模拟Triton的backends模块
class MockBaseBackend:
    pass

class MockDriverBase:
    pass

class MockGPUTarget:
    def __init__(self, backend, arch, warp_size):
        self.backend = backend
        self.arch = arch
        self.warp_size = warp_size

# 创建mock模块
import types
sys.modules['triton'] = types.ModuleType('triton')
sys.modules['triton.backends'] = types.ModuleType('triton.backends')
sys.modules['triton.backends.compiler'] = types.ModuleType('triton.backends.compiler')
sys.modules['triton.backends.driver'] = types.ModuleType('triton.backends.driver')
sys.modules['triton.backends.compiler'].BaseBackend = MockBaseBackend
sys.modules['triton.backends.compiler'].GPUTarget = MockGPUTarget
sys.modules['triton.backends.driver'].DriverBase = MockDriverBase

class TestMetalDetection(unittest.TestCase):
    """测试Metal后端的检测功能"""
    
    def test_platform_detection(self):
        """测试平台检测"""
        # 检查是否在macOS上运行
        self.assertEqual(platform.system(), "Darwin", "测试必须在macOS上运行")
        
        # 检查macOS版本
        mac_ver = platform.mac_ver()[0]
        mac_ver_parts = list(map(int, mac_ver.split('.')))
        # 修复比较逻辑：检查是否 >= 13.5
        if mac_ver_parts[0] > 13:
            # 如果主版本号大于13，则一定满足要求
            is_version_ok = True
        elif mac_ver_parts[0] == 13 and mac_ver_parts[1] >= 5:
            # 如果主版本号为13，次版本号需大于等于5
            is_version_ok = True
        else:
            is_version_ok = False
            
        self.assertTrue(is_version_ok, f"macOS版本必须≥13.5，当前为{mac_ver}")
        
        # 检查是否为Apple Silicon
        processor = platform.processor()
        self.assertEqual(processor, "arm", f"测试必须在Apple Silicon上运行，当前为{processor}")
    
    def test_mlx_availability(self):
        """测试MLX可用性"""
        try:
            import mlx.core as mx
            self.assertTrue(hasattr(mx, 'metal'), "MLX应该有metal模块")
            self.assertTrue(hasattr(mx.metal, 'is_available'), "MLX应该有metal.is_available函数")
            self.assertTrue(mx.metal.is_available(), "MLX Metal应该可用")
        except ImportError:
            self.fail("无法导入MLX，请确保已安装")
    
    def test_metal_driver(self):
        """测试Metal驱动"""
        try:
            # 直接从文件导入
            from backend.driver import MetalDriver
            
            # 创建驱动实例
            driver = MetalDriver()
            
            # 测试获取目标设备
            target = driver.get_current_target()
            self.assertEqual(target.backend, "metal", "后端应该是metal")
            self.assertEqual(target.arch, "apple-silicon", "架构应该是apple-silicon")
            
        except Exception as e:
            self.fail(f"Metal驱动测试失败: {e}")
            
if __name__ == "__main__":
    unittest.main() 