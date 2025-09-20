# Copyright (c) 2025, Wayne Chou <ck10600760@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import unittest
import numpy as np
import modmesh as mm


class GemmTestBase(mm.testing.TestBase):
    """Base class for matrix multiplication (GEMM) tests"""

    def test_square_matrix_multiplication(self):
        """Test basic square matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test matrix multiplication
        result = a.matmul(b)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_rectangular_matrix_multiplication(self):
        """Test rectangular matrix multiplication"""
        # Create 2x3 and 3x2 matrices
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=self.dtype)
        b_data = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[58, 64], [139, 154]]
        expected = np.array([[58.0, 64.0], [139.0, 154.0]], dtype=self.dtype)

        result = a.matmul(b)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_vector_multiplication(self):
        """Test matrix-vector multiplication"""
        # 3x3 matrix and 3x1 vector
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=self.dtype)
        v_data = np.array([[1.0], [2.0], [3.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        v = self.SimpleArray(array=v_data)

        # Expected result: [[14], [32], [50]]
        expected = np.array([[14.0], [32.0], [50.0]], dtype=self.dtype)

        result = a.matmul(v)

        self.assertEqual(list(result.shape), [3, 1])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_identity_matrix(self):
        """Test multiplication with identity matrix"""
        # 3x3 matrix
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=self.dtype)
        identity_data = np.ascontiguousarray(np.eye(3, dtype=self.dtype))

        a = self.SimpleArray(array=a_data)
        identity = self.SimpleArray(array=identity_data)

        result = a.matmul(identity)

        self.assertEqual(list(result.shape), [3, 3])
        np.testing.assert_array_almost_equal(result.ndarray, a_data)

    def test_zero_matrix(self):
        """Test multiplication with zero matrix"""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        zero_data = np.ascontiguousarray(np.zeros((2, 2), dtype=self.dtype))

        a = self.SimpleArray(array=a_data)
        zero = self.SimpleArray(array=zero_data)

        result = a.matmul(zero)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, zero_data)

    def test_dimension_mismatch_error(self):
        """Test error handling for incompatible dimensions"""
        a_data = np.array([[1.0, 2.0]], dtype=self.dtype)  # 1x2
        b_data = np.array([[1.0], [2.0], [3.0]], dtype=self.dtype)  # 3x1

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Should raise error: 1x2 cannot multiply with 3x1
        with self.assertRaises(RuntimeError):
            a.matmul(b)

    def test_compare_with_numpy(self):
        """Compare results with NumPy for various matrix sizes"""
        sizes = [(2, 3, 4), (5, 5, 5), (4, 6, 3)]

        for m, k, n in sizes:
            with self.subTest(m=m, k=k, n=n):
                # Generate random matrices
                np.random.seed(42)  # For reproducible results
                a_data = np.ascontiguousarray(np.random.randn(m, k).astype(self.dtype))
                b_data = np.ascontiguousarray(np.random.randn(k, n).astype(self.dtype))

                a = self.SimpleArray(array=a_data)
                b = self.SimpleArray(array=b_data)

                # Compute with our implementation
                result = a.matmul(b)

                # Compute with NumPy
                expected = np.matmul(a_data, b_data)

                self.assertEqual(list(result.shape), [m, n])
                if self.dtype == np.float32:
                    np.testing.assert_array_almost_equal(result.ndarray, expected, decimal=6)
                else:
                    np.testing.assert_array_almost_equal(result.ndarray, expected, decimal=10)


class GemmFloat32TC(GemmTestBase, unittest.TestCase):
    """Test matrix multiplication with float32"""

    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = mm.SimpleArrayFloat32


class GemmFloat64TC(GemmTestBase, unittest.TestCase):
    """Test matrix multiplication with float64"""

    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = mm.SimpleArrayFloat64

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

