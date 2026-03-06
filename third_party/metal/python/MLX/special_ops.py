"""
特殊数学函数的MLX映射实现
包括高级数学函数、特殊函数和近似计算
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import math

# 延迟导入MLX以避免不必要的依赖
_mx = None

def _get_mlx():
    """懒加载MLX"""
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx

class SpecialMathFunctions:
    """特殊数学函数的MLX实现"""
    
    def __init__(self):
        self.mx = _get_mlx()
        
    def erf(self, x):
        """
        误差函数 (Error Function)
        erf(x) = (2/sqrt(π)) * ∫(0→x) exp(-t²) dt
        """
        # 使用MLX内置的erf（如果有）
        if hasattr(self.mx, "erf"):
            return self.mx.erf(x)
            
        # 没有内置实现时使用泰勒级数近似
        # erf(x) ≈ (2/√π) * x * (1 - x²/3 + x⁴/10 - x⁶/42 + ...)
        # 对于小x值有效，大x值使用其他方法
        
        # 分段实现
        abs_x = self.mx.abs(x)
        sign_x = self.mx.sign(x)
        
        # 小x值的泰勒级数
        def small_x_approx(x):
            mx = self.mx
            x2 = x * x
            return x * (1.0 - x2/3.0 + x2*x2/10.0 - x2*x2*x2/42.0) * (2.0 / mx.sqrt(mx.array(math.pi)))
            
        # 中等x值的近似
        def medium_x_approx(x):
            mx = self.mx
            x2 = x * x
            t = 1.0 / (1.0 + 0.5 * mx.abs(x))
            
            # 基于Abramowitz和Stegun的近似
            coeffs = mx.array([
                1.0, -0.3275911, 0.2548296, -0.2844967, 
                1.4214137, -1.4531520, 1.0614054
            ])
            
            polynomial = coeffs[0]
            for i in range(1, len(coeffs)):
                polynomial = polynomial + coeffs[i] * mx.power(t, i)
                
            return sign_x * (1.0 - polynomial * mx.exp(-x2))
            
        # 大x值的近似 (|x| >= 4)
        def large_x_approx(x):
            mx = self.mx
            return sign_x * (1.0 - mx.exp(-x * x) / (mx.abs(x) * mx.sqrt(mx.array(math.pi))))
        
        # 组合不同范围的近似
        result = self.mx.where(abs_x < 1.0, small_x_approx(x), 
                  self.mx.where(abs_x < 4.0, medium_x_approx(x), large_x_approx(x)))
                  
        return result
    
    def erfc(self, x):
        """
        互补误差函数 (Complementary Error Function)
        erfc(x) = 1 - erf(x)
        """
        # 使用MLX内置的erfc（如果有）
        if hasattr(self.mx, "erfc"):
            return self.mx.erfc(x)
            
        # 使用erf实现
        return 1.0 - self.erf(x)
    
    def digamma(self, x):
        """
        Digamma函数 (Psi函数)
        digamma(x) = d/dx[log(Gamma(x))]
        """
        # 使用MLX内置的digamma（如果有）
        if hasattr(self.mx, "digamma"):
            return self.mx.digamma(x)
            
        # 近似实现
        mx = self.mx
        
        # 使用常见特殊值的查找表
        special_values = {
            0.5: -1.96351002602,  # digamma(1/2) = -2*ln(2) - γ
            1.0: -0.57721566490,  # digamma(1) = -γ (Euler-Mascheroni constant)
            2.0: 0.42278433509,   # digamma(2) = 1 - γ
        }
        
        # 创建结果数组
        result = mx.zeros_like(x)
        
        # 对于小x值，使用反射公式
        small_x = x <= 0.5
        
        # 处理特殊输入值
        for val, psi_val in special_values.items():
            is_special_val = mx.abs(x - mx.array(val)) < 1e-10
            result = mx.where(is_special_val, psi_val, result)
            
        # 计算反射公式的部分
        reflection = mx.zeros_like(x)
        # 使用数组比较，避免isclose函数调用
        is_not_half = mx.abs(x - mx.array(0.5)) > 1e-10
        is_small_not_half = mx.logical_and(small_x, is_not_half)
        if mx.any(is_small_not_half):
            reflection = mx.where(is_small_not_half, 
                                 mx.array(math.pi) / mx.tan(mx.array(math.pi) * x),
                                 0.0)
            x_reflected = mx.where(is_small_not_half, 1.0 - x, x)
        else:
            x_reflected = x
            
        # 对于未处理的值，使用级数展开计算
        # 创建所有特殊值的掩码
        special_masks = [mx.abs(x - mx.array(val)) < 1e-10 for val in special_values.keys()]
        if len(special_masks) > 0:
            combined_mask = mx.stack(special_masks)
            unhandled = ~mx.any(combined_mask, axis=0)
        else:
            unhandled = mx.ones_like(x, dtype=bool)
        
        if mx.any(unhandled):
            # 通过递归关系将x移到大于2的范围
            n = mx.floor(x_reflected)
            x_frac = x_reflected - n
            
            # 如果分数部分接近0，将它设为1，避免数值问题
            x_frac = mx.where(mx.abs(x_frac) < 1e-10, 1.0, x_frac)
            x_large = x_frac + mx.maximum(n, 2)
            
            # 使用渐近级数 for large x (Stirling's approximation)
            large_x_approx = mx.log(x_large) - 1.0/(2.0*x_large)
            
            # 加上伯努利数的修正
            x_large_sq = x_large * x_large
            bernoulli_sum = 1.0/(12.0*x_large_sq) - 1.0/(120.0*x_large_sq*x_large_sq)
            large_x_approx = large_x_approx - bernoulli_sum
            
            # 使用递归关系回到原始的x
            steps = mx.maximum(n, 2) - n
            harmonic_sum = mx.zeros_like(x)
            
            # 这里使用向量化计算会更高效，但为清晰起见使用循环
            for i in range(1, 11):  # 假设最大步数为10，通常足够
                mask = (steps >= i)
                if mx.any(mask):
                    idx = x_large - i
                    harmonic_sum = mx.where(mask, harmonic_sum + 1.0/idx, harmonic_sum)
            
            unhandled_result = large_x_approx - harmonic_sum
            
            # 合并结果
            unhandled_not_small = mx.logical_and(unhandled, ~small_x)
            result = mx.where(unhandled_not_small, unhandled_result, result)
            
            # 应用反射公式到小x
            unhandled_small = mx.logical_and(unhandled, small_x)
            result = mx.where(unhandled_small, unhandled_result - reflection, result)
        
        # 处理小于0或接近整数的特殊情况
        is_integer = mx.abs(mx.floor(x) - x) < 1e-10
        is_negative = x <= 0
        neg_integers = mx.logical_and(is_integer, is_negative)
        if mx.any(neg_integers):
            # 对负整数，digamma是无穷大，设为一个大的负数
            result = mx.where(neg_integers, mx.array(-1e10), result)
        
        return result
    
    def lgamma(self, x):
        """
        Log-Gamma函数
        lgamma(x) = log(|Gamma(x)|)
        """
        # 使用MLX内置的lgamma（如果有）
        if hasattr(self.mx, "lgamma"):
            return self.mx.lgamma(x)
            
        # 近似实现 - 使用Lanczos近似
        mx = self.mx
        
        # 对于负值，使用反射公式
        reflection = mx.zeros_like(x)
        negative_x = x <= 0
        if mx.any(negative_x):
            # log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
            pi_x = mx.array(math.pi) * x
            log_sin_pi_x = mx.log(mx.abs(mx.sin(pi_x)))
            reflection = mx.log(mx.array(math.pi)) - log_sin_pi_x
            x = mx.where(negative_x, 1.0 - x, x)
        
        # Lanczos近似参数
        g = 7
        p = mx.array([
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        ])
        
        # 计算
        x_g = x + g - 0.5
        base_term = (x - 0.5) * mx.log(x_g) - x_g
        sum_term = p[0]
        for i in range(1, len(p)):
            sum_term = sum_term + p[i]/(x + i - 1)
            
        result = base_term + mx.log(mx.array(math.sqrt(2*math.pi)) * sum_term)
        
        # 应用反射公式
        result = mx.where(negative_x, reflection - result, result)
        
        return result
    
    def bessel_j0(self, x):
        """
        第一类贝塞尔函数 J0(x)
        J₀(x) = (1/π) * ∫₀^π cos(x*sin(θ)) dθ
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "bessel_j0"):
            return self.mx.bessel_j0(x)
        
        mx = self.mx
        abs_x = mx.abs(x)
        
        # 小x值的级数展开
        def small_x_approx(x):
            x2 = x * x
            return 1.0 - x2/4.0 + x2*x2/64.0 - x2*x2*x2/2304.0 + x2*x2*x2*x2/147456.0
        
        # 大x值的渐近展开
        def large_x_approx(x):
            abs_x = mx.abs(x)
            theta = abs_x - mx.array(math.pi/4)
            amplitude = mx.sqrt(mx.array(2.0/math.pi) / abs_x)
            return amplitude * mx.cos(theta)
        
        # 组合不同范围的近似
        result = mx.where(abs_x < 4.0, small_x_approx(abs_x), large_x_approx(x))
        
        return result
    
    def bessel_j1(self, x):
        """
        第一类贝塞尔函数 J1(x)
        J₁(x) = (1/π) * ∫₀^π cos(θ - x*sin(θ)) dθ
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "bessel_j1"):
            return self.mx.bessel_j1(x)
        
        mx = self.mx
        abs_x = mx.abs(x)
        sign_x = mx.sign(x)
        
        # 小x值的级数展开
        def small_x_approx(x):
            x2 = x * x
            return x/2.0 - x*x2/16.0 + x*x2*x2/384.0 - x*x2*x2*x2/18432.0
        
        # 大x值的渐近展开
        def large_x_approx(x, sign):
            abs_x = mx.abs(x)
            theta = abs_x - mx.array(3*math.pi/4)
            amplitude = mx.sqrt(mx.array(2.0/math.pi) / abs_x)
            return sign * amplitude * mx.cos(theta)
        
        # 组合不同范围的近似
        result = mx.where(abs_x < 4.0, 
                          small_x_approx(x), 
                          large_x_approx(x, sign_x))
        
        return result
    
    def i0e(self, x):
        """
        第一类修正贝塞尔函数 I0(x) 缩放版本
        I₀(x) * exp(-|x|)
        """
        mx = self.mx
        abs_x = mx.abs(x)
        
        # 小x值的多项式近似
        def small_x_approx(x):
            x2 = x * x
            return 1.0 + x2/4.0 + x2*x2/64.0 + x2*x2*x2/2304.0 + x2*x2*x2*x2/147456.0
        
        # 大x值的渐近展开
        def large_x_approx(x):
            abs_x = mx.abs(x)
            return mx.sqrt(mx.array(1.0/(2.0*math.pi*abs_x)))
        
        # 小x值直接计算，大x值使用渐近展开并缩放
        # 由于已经包含exp(-|x|)缩放，所以直接计算
        result = mx.where(abs_x < 4.0, 
                         small_x_approx(abs_x) * mx.exp(-abs_x), 
                         large_x_approx(abs_x))
        
        return result
    
    def i1e(self, x):
        """
        第一类修正贝塞尔函数 I1(x) 缩放版本
        I₁(x) * exp(-|x|)
        """
        mx = self.mx
        abs_x = mx.abs(x)
        sign_x = mx.sign(x)
        
        # 小x值的多项式近似
        def small_x_approx(x, sign):
            abs_x = mx.abs(x)
            x2 = abs_x * abs_x
            return sign * (abs_x/2.0 + abs_x*x2/16.0 + abs_x*x2*x2/384.0 + abs_x*x2*x2*x2/18432.0)
        
        # 大x值的渐近展开
        def large_x_approx(x, sign):
            abs_x = mx.abs(x)
            return sign * mx.sqrt(mx.array(1.0/(2.0*math.pi*abs_x)))
        
        # 小x值直接计算，大x值使用渐近展开并缩放
        result = mx.where(abs_x < 4.0, 
                         small_x_approx(x, sign_x) * mx.exp(-abs_x), 
                         large_x_approx(x, sign_x))
        
        return result
        
    def gamma(self, x):
        """
        伽马函数 Gamma(x)
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "gamma"):
            return self.mx.gamma(x)
            
        # 否则使用lgamma的指数
        return self.mx.exp(self.lgamma(x))
        
    def cosh(self, x):
        """
        双曲余弦 cosh(x) = (e^x + e^-x)/2
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "cosh"):
            return self.mx.cosh(x)
            
        mx = self.mx
        return (mx.exp(x) + mx.exp(-x)) / 2.0
        
    def sinh(self, x):
        """
        双曲正弦 sinh(x) = (e^x - e^-x)/2
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "sinh"):
            return self.mx.sinh(x)
            
        mx = self.mx
        return (mx.exp(x) - mx.exp(-x)) / 2.0
        
    def asinh(self, x):
        """
        反双曲正弦 asinh(x) = ln(x + sqrt(x^2 + 1))
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "asinh"):
            return self.mx.asinh(x)
            
        mx = self.mx
        return mx.log(x + mx.sqrt(x*x + 1))
        
    def acosh(self, x):
        """
        反双曲余弦 acosh(x) = ln(x + sqrt(x^2 - 1))
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "acosh"):
            return self.mx.acosh(x)
            
        mx = self.mx
        return mx.log(x + mx.sqrt(x*x - 1))
        
    def atanh(self, x):
        """
        反双曲正切 atanh(x) = 0.5 * ln((1+x)/(1-x))
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "atanh"):
            return self.mx.atanh(x)
            
        mx = self.mx
        return 0.5 * mx.log((1 + x) / (1 - x))
        
    def expm1(self, x):
        """
        exp(x) - 1，对于小x值提供更好的数值精度
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "expm1"):
            return self.mx.expm1(x)
            
        mx = self.mx
        # 小x值使用泰勒级数，大x值直接计算
        small_x = mx.abs(x) < 1e-5
        small_x_result = x + x*x/2.0 + x*x*x/6.0
        large_x_result = mx.exp(x) - 1.0
        
        return mx.where(small_x, small_x_result, large_x_result)
        
    def log1p(self, x):
        """
        log(1 + x)，对于小x值提供更好的数值精度
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "log1p"):
            return self.mx.log1p(x)
            
        mx = self.mx
        # 小x值使用泰勒级数，大x值直接计算
        small_x = mx.abs(x) < 1e-5
        small_x_result = x - x*x/2.0 + x*x*x/3.0
        large_x_result = mx.log(1.0 + x)
        
        return mx.where(small_x, small_x_result, large_x_result)

class NumericalFunctions:
    """数值近似和工具函数"""
    
    def __init__(self):
        self.mx = _get_mlx()
        
    def fast_exp(self, x):
        """
        指数函数的快速近似
        基于多项式近似
        """
        mx = self.mx
        # 将输入限制在合理范围内
        x = mx.clip(x, -88.0, 88.0)  # exp(88) ≈ 1.6e38，接近float32上限
        
        # 使用一系列多项式近似
        # 分段处理以提高精度
        abs_x = mx.abs(x)
        sign_x = mx.sign(x)
        
        # 小x值的泰勒级数 exp(x) ≈ 1 + x + x^2/2 + x^3/6 + ...
        def small_x_approx(x):
            return 1.0 + x * (1.0 + x * (0.5 + x * (1.0/6.0 + x * (1.0/24.0 + x * (1.0/120.0)))))
        
        # 中等x值使用更复杂的近似
        def medium_x_approx(x):
            # 使用预计算系数的多项式近似
            abs_x = mx.abs(x)
            sign_x = mx.sign(x)
            
            # 将x分解为整数和分数部分
            n = mx.floor(abs_x)
            f = abs_x - n
            
            # 计算2^n
            pow2n = mx.exp2(n)
            
            # 计算e^f的近似，其中0 <= f < 1
            ef_approx = 1.0 + f * (1.0 + f * (0.5 + f * (1.0/6.0 + f * (1.0/24.0))))
            
            # 组合结果: e^|x| = e^(n+f) = e^n * e^f
            e_approx = pow2n * ef_approx * mx.exp(mx.log(mx.array(math.e)) * 0.5)
            
            # 对于负x，取倒数
            return mx.where(x >= 0, e_approx, 1.0 / e_approx)
        
        # 默认使用MLX内置的exp，但可以切换到近似实现
        # 在此实现中，我们将直接返回MLX的exp作为参考
        return mx.exp(x)
        
        # 如果需要使用近似版本，可以取消下面的注释
        # return mx.where(abs_x < 0.5, small_x_approx(x), medium_x_approx(x))
    
    def fast_sigmoid(self, x):
        """
        Sigmoid函数的快速近似
        sigmoid(x) = 1/(1+e^(-x))
        
        使用多项式近似加速计算
        """
        mx = self.mx
        
        # 限制输入范围以避免数值问题
        x = mx.clip(x, -20.0, 20.0)
        
        # 小x值的近似
        # sigmoid(x) ≈ 0.5 + x/4 为|x| < 1
        def small_x_approx(x):
            return 0.5 + 0.25 * x - 0.0625 * x*x*x
        
        # 中等x值的近似
        # 使用有理函数近似
        def medium_x_approx(x):
            return mx.where(x >= 0, 
                          1.0 - 0.5 / (1.0 + 0.5*x),
                          0.5 / (1.0 - 0.5*x))
        
        # 大x值直接返回极限值
        def large_x_approx(x):
            return mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        
        # 组合不同范围的近似
        abs_x = mx.abs(x)
        result = mx.where(abs_x < 1.0, small_x_approx(x),
                 mx.where(abs_x < 8.0, medium_x_approx(x), large_x_approx(x)))
        
        # 如果只需要基本的fast sigmoid近似:
        # result = mx.where(x >= 0, 1.0 / (1.0 + mx.exp(-x)), mx.exp(x) / (1.0 + mx.exp(x)))
        
        return result
    
    def fast_tanh(self, x):
        """
        Tanh函数的快速近似
        tanh(x) = 2*sigmoid(2x) - 1 = (e^x - e^-x)/(e^x + e^-x)
        
        使用多项式近似加速计算
        """
        mx = self.mx
        
        # 限制输入范围以避免数值问题
        x = mx.clip(x, -10.0, 10.0)
        
        # 小x值的近似 
        # tanh(x) ≈ x - x^3/3 for small x
        def small_x_approx(x):
            x2 = x * x
            return x * (1.0 - x2 / 3.0)
        
        # 中等x值的近似
        # 使用有理函数近似
        def medium_x_approx(x):
            x2 = x * x
            return x * (27.0 + x2) / (27.0 + 9.0 * x2)
        
        # 大x值直接返回极限值
        def large_x_approx(x):
            return mx.sign(x)
        
        # 组合不同范围的近似
        abs_x = mx.abs(x)
        result = mx.where(abs_x < 0.8, small_x_approx(x),
                 mx.where(abs_x < 4.0, medium_x_approx(x), large_x_approx(x)))
        
        return result
    
    def rsqrt(self, x):
        """
        快速平方根倒数 1/sqrt(x)
        """
        # 使用MLX内置函数（如果有）
        if hasattr(self.mx, "rsqrt"):
            return self.mx.rsqrt(x)
            
        mx = self.mx
        # 使用Newton-Raphson迭代近似
        # 初始猜测（使用位操作启发式，但在浮点数下模拟）
        # y = 0.5 * (1.5 - 0.5 * x * y * y) 迭代公式
        
        # 这里我们简单地返回1.0 / mx.sqrt(x)作为参考实现
        return 1.0 / mx.sqrt(x)

# 创建全局实例
special_math = SpecialMathFunctions()
numerical = NumericalFunctions()

# 导出函数映射
def get_special_ops_map():
    """获取特殊操作的映射"""
    mx = _get_mlx()
    
    return {
        # 指数和对数
        'tt.expm1': special_math.expm1,
        'tt.log1p': special_math.log1p,
        
        # 双曲函数
        'tt.sinh': special_math.sinh,
        'tt.cosh': special_math.cosh,
        'tt.asinh': special_math.asinh,
        'tt.acosh': special_math.acosh,
        'tt.atanh': special_math.atanh,
        
        # 特殊函数
        'tt.erf': special_math.erf,
        'tt.erfc': special_math.erfc,
        'tt.gamma': special_math.gamma,
        'tt.lgamma': special_math.lgamma,
        'tt.digamma': special_math.digamma,
        'tt.bessel_j0': special_math.bessel_j0,
        'tt.bessel_j1': special_math.bessel_j1,
        'tt.i0e': special_math.i0e,
        'tt.i1e': special_math.i1e,
        
        # 快速近似
        'tt.fast_exp': numerical.fast_exp,
        'tt.fast_sigmoid': numerical.fast_sigmoid,
        'tt.fast_tanh': numerical.fast_tanh,
        'tt.rsqrt': numerical.rsqrt,
    } 