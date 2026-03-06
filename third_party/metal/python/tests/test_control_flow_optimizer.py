"""
Test module for control flow optimization in Metal backend
"""

import unittest
import MLX.metal_hardware_optimizer
import mlx.control_flow_optimizer
from typing import Dict, Any

class TestPredicateSupport(unittest.TestCase):
    """Test cases for predicate support"""

    def setUp(self):
        """Set up test fixtures"""
        self.pred_support = mlx.control_flow_optimizer.PredicateSupport()

    def test_predicate_var_generation(self):
        """Test predicate variable generation"""
        pred_var = self.pred_support.generate_predicate_var("x > 0", "is_positive")
        self.assertEqual(pred_var, "bool is_positive = x > 0;")

    def test_if_predicated_with_else(self):
        """Test predicated if statement with else"""
        if_code = self.pred_support.generate_if_predicated(
            "is_positive",
            "result = x;",
            "result = -x;"
        )
        self.assertIn("if (is_positive) {", if_code)
        self.assertIn("result = x;", if_code)
        self.assertIn("} else {", if_code)
        self.assertIn("result = -x;", if_code)

    def test_if_predicated_without_else(self):
        """Test predicated if statement without else"""
        if_code = self.pred_support.generate_if_predicated(
            "is_positive",
            "result = x;"
        )
        self.assertIn("if (is_positive) {", if_code)
        self.assertIn("result = x;", if_code)
        self.assertNotIn("else", if_code)

    def test_mask_to_predicate_conversion(self):
        """Test mask to predicate conversion"""
        int_pred = self.pred_support.convert_mask_to_predicate("mask", "int")
        self.assertEqual(int_pred, "(mask != 0)")

        float_pred = self.pred_support.convert_mask_to_predicate("mask", "float")
        self.assertEqual(float_pred, "(mask != 0.0f)")

        other_pred = self.pred_support.convert_mask_to_predicate("mask", "other")
        self.assertEqual(other_pred, "bool(mask)")

    def test_branch_divergence_optimization_ternary(self):
        """Test branch divergence optimization with ternary conversion"""
        if_stmt = {
            "condition": "x > 0",
            "then_body": "return x;",
            "else_body": "return -x;"
        }

        optimized = self.pred_support.optimize_branch_divergence(if_stmt)
        self.assertEqual(optimized, "return (x > 0) ? x : -x;")

    def test_branch_divergence_optimization_assignment(self):
        """Test branch divergence optimization with assignment"""
        if_stmt = {
            "condition": "x > 0",
            "then_body": "result = x;",
            "else_body": "result = -x;"
        }

        optimized = self.pred_support.optimize_branch_divergence(if_stmt)
        self.assertEqual(optimized, "result = (x > 0) ? x : -x;")

    def test_branch_divergence_optimization_complex(self):
        """Test branch divergence optimization with complex bodies"""
        if_stmt = {
            "condition": "x > 0",
            "then_body": "result = x;\ncount++;",
            "else_body": "result = -x;\ncount--;"
        }

        optimized = self.pred_support.optimize_branch_divergence(if_stmt)
        self.assertTrue("if (x > 0) {" in optimized)
        self.assertTrue("result = x;" in optimized)
        self.assertTrue("count++;" in optimized)
        self.assertTrue("} else {" in optimized)
        self.assertTrue("result = -x;" in optimized)
        self.assertTrue("count--;" in optimized)

class TestLoopOptimizer(unittest.TestCase):
    """Test cases for loop optimization"""

    def setUp(self):
        """Set up test fixtures"""
        self.loop_optimizer = control_flow_optimizer.LoopOptimizer()

    def test_loop_analysis_for_loop(self):
        """Test loop analysis for a for loop"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < 10",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": 10
        }

        analysis = self.loop_optimizer.analyze_loop(loop_info)
        self.assertEqual(analysis["is_for_loop"], True)
        self.assertEqual(analysis["is_while_loop"], False)
        self.assertEqual(analysis["known_trip_count"], True)
        self.assertEqual(analysis["trip_count"], 10)
        self.assertEqual(analysis["unroll_candidate"], True)

    def test_loop_analysis_while_loop(self):
        """Test loop analysis for a while loop"""
        loop_info = {
            "init": "",
            "condition": "i < n",
            "update": "i = next[i]",
            "body": "result += data[i];",
            "trip_count": None
        }

        analysis = self.loop_optimizer.analyze_loop(loop_info)
        self.assertFalse(analysis["is_for_loop"])
        self.assertTrue(analysis["is_while_loop"])
        self.assertFalse(analysis["known_trip_count"])
        self.assertFalse(analysis["unroll_candidate"])

    def test_full_loop_unrolling(self):
        """Test full loop unrolling"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < 4",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": 4
        }

        unrolled = self.loop_optimizer.unroll_loop(loop_info)
        for i in range(4):
            self.assertIn(f"// Unrolled iteration {i}", unrolled)
            self.assertIn("result += data[i];", unrolled)

        # The unrolled code should not have a loop structure
        self.assertNotIn("for (", unrolled)

    def test_partial_loop_unrolling(self):
        """Test partial loop unrolling"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < n",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": None
        }

        unrolled = self.loop_optimizer.unroll_loop(loop_info, 4)
        self.assertIn("for (;", unrolled)
        self.assertIn("i += 4", unrolled)

        # Should contain 4 copies of the body
        body_count = unrolled.count("result += data[i];")
        self.assertEqual(body_count, 4)

    def test_simdify_loop(self):
        """Test SIMD loop transformation"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < 128",
            "update": "i++",
            "body": "result[i] = a[i] + b[i];",
            "trip_count": 128
        }

        # Set up conditions for SIMD candidate
        self.loop_optimizer.simd_width = 4

        # Call simdify_loop
        simdified = self.loop_optimizer.simdify_loop(loop_info)

        # Check for key components expected in SIMD-optimized code
        self.assertIn("// SIMD-optimized loop", simdified)
        self.assertIn("simd::float4", simdified)  # Should use simd_width=4
        self.assertIn("// Cleanup loop", simdified)  # Should have cleanup loop
        self.assertIn("for (; i < 128;", simdified)  # Cleanup loop condition

        # Verify vector setup
        self.assertIn("simd_idx[s] =", simdified)

        # Verify scalar fallback and condition
        self.assertIn("if (simd_i < 128)", simdified)

        # Verify the body has i replaced with simd_i in the condition check
        # but not necessarily in the body (implementation specific)
        self.assertIn("result[", simdified)
        self.assertIn("a[", simdified)
        self.assertIn("b[", simdified)

    def test_nested_loop_optimization(self):
        """Test nested loop optimization"""
        # Define a set of nested loops (matrix multiplication example)
        outer_loop = {
            "init": "int i = 0;",
            "condition": "i < m",
            "update": "i++",
            "body": "// Inner loop body here",
            "trip_count": 10
        }

        middle_loop = {
            "init": "int j = 0;",
            "condition": "j < n",
            "update": "j++",
            "body": "// Innermost loop body here",
            "trip_count": 10
        }

        inner_loop = {
            "init": "int k = 0;",
            "condition": "k < p",
            "update": "k++",
            "body": "C[i][j] += A[i][k] * B[k][j];",
            "trip_count": 4
        }

        # Test nested loop optimization
        nested_loops = [outer_loop, middle_loop, inner_loop]
        optimized = self.loop_optimizer.optimize_nested_loops(nested_loops)

        # The loop optimization might vary in implementation
        # Just check that it contains the matrix multiplication operation
        self.assertIn("C[i][j] += A[i][k] * B[k][j];", optimized)
        self.assertIn("int i = 0", optimized)
        self.assertIn("int j = 0", optimized)
        self.assertIn("int k = 0", optimized)

    def test_empty_nested_loops(self):
        """Test nested loop optimization with empty list"""
        self.assertEqual(self.loop_optimizer.optimize_nested_loops([]), "")

    def test_single_loop_in_nested_optimization(self):
        """Test nested loop optimization with a single loop"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < 4",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": 4
        }

        optimized = self.loop_optimizer.optimize_nested_loops([loop_info])

        # Should be the same as directly optimizing the loop
        direct_optimized = self.loop_optimizer.optimize_loop(loop_info)
        self.assertEqual(optimized, direct_optimized)

class TestConditionalBranchMapper(unittest.TestCase):
    """Test cases for conditional branch mapping"""

    def setUp(self):
        """Set up test fixtures"""
        self.branch_mapper = control_flow_optimizer.ConditionalBranchMapper()

    def test_map_if_statement(self):
        """Test mapping of if statement"""
        if_stmt = {
            "condition": "x > 0",
            "then_body": "result = x;",
            "else_body": "result = -x;"
        }

        mapped = self.branch_mapper.map_if_statement(if_stmt)
        self.assertEqual(mapped, "result = (x > 0) ? x : -x;")

    def test_map_select(self):
        """Test mapping of select operation"""
        select_code = self.branch_mapper.map_select("x > 0", "x", "-x")
        self.assertEqual(select_code, "(x > 0) ? (x) : (-x)")

    def test_map_switch_small(self):
        """Test mapping of small switch statement"""
        switch_stmt = {
            "value": "flag",
            "cases": [
                (0, "result = 0;"),
                (1, "result = 1;"),
                (2, "result = 2;")
            ],
            "default": "result = -1;"
        }

        mapped = self.branch_mapper.map_switch_statement(switch_stmt)
        self.assertIn("if (flag == 0)", mapped)
        self.assertIn("else if (flag == 1)", mapped)
        self.assertIn("else if (flag == 2)", mapped)
        self.assertIn("else {", mapped)

    def test_map_switch_large(self):
        """Test mapping of large switch statement"""
        switch_stmt = {
            "value": "flag",
            "cases": [
                (0, "result = 0;"),
                (1, "result = 1;"),
                (2, "result = 2;"),
                (3, "result = 3;"),
                (4, "result = 4;")
            ],
            "default": "result = -1;"
        }

        mapped = self.branch_mapper.map_switch_statement(switch_stmt)
        self.assertIn("switch (flag) {", mapped)
        for i in range(5):
            self.assertIn(f"case {i}:", mapped)
        self.assertIn("default:", mapped)

class TestControlFlowOptimizer(unittest.TestCase):
    """Test cases for overall control flow optimization"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = control_flow_optimizer.ControlFlowOptimizer()

    def test_optimize_if_statement(self):
        """Test optimization of if statement"""
        if_stmt = {
            "condition": "x > 0",
            "then_body": "result = x;",
            "else_body": "result = -x;"
        }

        optimized = self.optimizer.optimize_if_statement(if_stmt)
        self.assertEqual(optimized, "result = (x > 0) ? x : -x;")

    def test_optimize_loop(self):
        """Test optimization of loop"""
        loop_info = {
            "init": "int i = 0;",
            "condition": "i < 4",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": 4
        }

        optimized = self.optimizer.optimize_loop(loop_info)
        self.assertIn("// Fully unrolled loop", optimized)
        self.assertIn("// Unrolled iteration", optimized)

    def test_optimize_select(self):
        """Test optimization of select operation"""
        optimized = self.optimizer.optimize_select("x > 0", "x", "-x")
        self.assertEqual(optimized, "(x > 0) ? (x) : (-x)")

    def test_optimize_switch(self):
        """Test optimization of switch statement"""
        switch_stmt = {
            "value": "flag",
            "cases": [
                (0, "result = 0;"),
                (1, "result = 1;"),
                (2, "result = 2;")
            ],
            "default": "result = -1;"
        }

        optimized = self.optimizer.optimize_switch(switch_stmt)
        self.assertIn("if (flag == 0)", optimized)
        self.assertIn("else if (flag == 1)", optimized)
        self.assertIn("else if (flag == 2)", optimized)
        self.assertIn("else {", optimized)

    def test_optimize_control_flow_if(self):
        """Test optimization of if statement control flow node"""
        ir_node = {
            "type": "if",
            "condition": "x > 0",
            "then_body": "result = x;",
            "else_body": "result = -x;"
        }

        optimized = self.optimizer.optimize_control_flow(ir_node)
        self.assertIn("metal_code", optimized)
        self.assertEqual(optimized["metal_code"], "result = (x > 0) ? x : -x;")

    def test_optimize_control_flow_for(self):
        """Test optimization of for loop control flow node"""
        ir_node = {
            "type": "for",
            "init": "int i = 0;",
            "condition": "i < 4",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": 4
        }

        optimized = self.optimizer.optimize_control_flow(ir_node)
        self.assertIn("metal_code", optimized)
        self.assertIn("// Fully unrolled loop", optimized["metal_code"])

    def test_optimize_control_flow_while(self):
        """Test optimization of while loop control flow node"""
        ir_node = {
            "type": "while",
            "init": "int i = 0;",
            "condition": "i < n",
            "update": "i++",
            "body": "result += data[i];",
            "trip_count": None
        }

        optimized = self.optimizer.optimize_control_flow(ir_node)
        self.assertIn("metal_code", optimized)
        self.assertIn("while", optimized["metal_code"])

    def test_optimize_control_flow_unknown(self):
        """Test optimization of unknown control flow node"""
        ir_node = {
            "type": "unknown_node_type"
        }

        optimized = self.optimizer.optimize_control_flow(ir_node)
        # Should return the original node with no metal_code
        self.assertEqual(optimized["type"], "unknown_node_type")
        self.assertNotIn("metal_code", optimized)

if __name__ == "__main__":
    unittest.main()