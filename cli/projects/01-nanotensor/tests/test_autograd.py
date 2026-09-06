"""
Comprehensive mathematical and numerical verification of NanoTensor autograd engine.
Compares analytical backpropagation gradients directly against finite-difference numerical gradients.
"""

import math
import unittest
from nanotensor.tensor import Tensor


def grad_check(f, x: Tensor, eps: float = 1e-5, tol: float = 1e-3) -> bool:
    """
    Validates analytical gradient using two-sided finite difference numerical gradient:
    g_num = (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    # Analytical backward pass
    x.grad = None
    y = f(x)
    y.backward()
    analytical_grad = list(x.grad.contiguous()._data)

    # Numerical gradient computation
    x_orig = list(x.contiguous()._data)
    numerical_grad = []

    for i in range(len(x_orig)):
        # x + eps
        x_plus = list(x_orig)
        x_plus[i] += eps
        t_plus = Tensor(x_plus, shape=x.shape)
        y_plus = f(t_plus)
        val_plus = y_plus._data[0] if isinstance(y_plus._data, list) else y_plus._data

        # x - eps
        x_minus = list(x_orig)
        x_minus[i] -= eps
        t_minus = Tensor(x_minus, shape=x.shape)
        y_minus = f(t_minus)
        val_minus = y_minus._data[0] if isinstance(y_minus._data, list) else y_minus._data

        g_num = (val_plus - val_minus) / (2.0 * eps)
        numerical_grad.append(g_num)

    # Compare
    for i, (ga, gn) in enumerate(zip(analytical_grad, numerical_grad)):
        denom = max(abs(ga) + abs(gn), 1e-5)
        rel_err = abs(ga - gn) / denom
        if rel_err > tol:
            print(f"Grad check failed at element {i}: analytical={ga:.6f}, numerical={gn:.6f}, rel_err={rel_err:.6f}")
            return False
    return True


class TestAutograd(unittest.TestCase):

    def test_basic_arithmetic(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = (a * b + a ** 2).sum()
        c.backward()

        # dc/da = b + 2*a = [4 + 4, 5 + 6] = [8.0, 11.0]
        # dc/db = a = [2.0, 3.0]
        self.assertAlmostEqual(a.grad._data[0], 8.0, places=4)
        self.assertAlmostEqual(a.grad._data[1], 11.0, places=4)
        self.assertAlmostEqual(b.grad._data[0], 2.0, places=4)
        self.assertAlmostEqual(b.grad._data[1], 3.0, places=4)

    def test_gradcheck_arithmetic(self):
        x = Tensor([1.5, -2.0, 0.7], requires_grad=True)
        ok = grad_check(lambda t: ((t * 2.5) + (t ** 3) - 1.2).sum(), x)
        self.assertTrue(ok)

    def test_gradcheck_silu(self):
        x = Tensor([0.5, -1.2, 2.3, -0.1], requires_grad=True)
        ok = grad_check(lambda t: t.silu().sum(), x)
        self.assertTrue(ok)

    def test_gradcheck_sigmoid_relu(self):
        x = Tensor([0.8, -1.5, 2.0, 0.3], requires_grad=True)
        ok = grad_check(lambda t: (t.sigmoid() + t.relu()).sum(), x)
        self.assertTrue(ok)

    def test_broadcasting_addition(self):
        # A: (2, 3), B: (1, 3)
        A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        B = Tensor([[0.5, -0.5, 1.0]], requires_grad=True)
        loss = (A + B).sum()
        loss.backward()

        self.assertEqual(B.grad.shape, (1, 3))
        # B added to 2 rows, so grad of B should be [[2.0, 2.0, 2.0]]
        self.assertAlmostEqual(B.grad._data[0], 2.0, places=4)
        self.assertAlmostEqual(B.grad._data[1], 2.0, places=4)
        self.assertAlmostEqual(B.grad._data[2], 2.0, places=4)

    def test_matmul_simple(self):
        # A: (2, 3), B: (3, 2)
        A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        B = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
        C = A @ B
        loss = C.sum()
        loss.backward()

        # dLoss/dA = ones(2, 2) @ B^T = [[1, 1], [1, 1]] @ [[1, 0, 1], [0, 1, 1]]
        # Row 0: [1*1 + 1*0, 1*0 + 1*1, 1*1 + 1*1] = [1.0, 1.0, 2.0]
        self.assertAlmostEqual(A.grad._data[0], 1.0, places=4)
        self.assertAlmostEqual(A.grad._data[1], 1.0, places=4)
        self.assertAlmostEqual(A.grad._data[2], 2.0, places=4)

    def test_gradcheck_matmul(self):
        w = Tensor([[0.5, -0.2], [0.3, 0.8], [-0.1, 0.4]], requires_grad=True)
        x = Tensor([[1.0, 2.0, -1.0]], requires_grad=False)
        ok = grad_check(lambda W: (x @ W).silu().sum(), w)
        self.assertTrue(ok)

    def test_gradcheck_softmax(self):
        x = Tensor([[1.2, -0.5, 0.8], [0.1, 2.0, -1.0]], requires_grad=True)
        ok = grad_check(lambda t: (t.softmax(axis=-1) * 2.0).sum(), x)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
