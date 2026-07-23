import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clownpiece import Tensor, no_grad
from clownpiece.nn import Conv2D


def as_array(tensor: Tensor) -> np.ndarray:
    return np.asarray(tensor.tolist(), dtype=np.float32)


def test_unfold_fold_adjoint():
    x_array = np.arange(1, 7, dtype=np.float32).reshape(1, 1, 2, 3)
    x = Tensor(x_array.tolist(), requires_grad=False)
    columns = x.unfold(3, 5)

    column_weights_array = np.arange(
        1, columns.shape[0] * columns.shape[1] * columns.shape[2] + 1,
        dtype=np.float32,
    ).reshape(columns.shape)
    column_weights = Tensor(column_weights_array.tolist(), requires_grad=False)

    left = np.sum(as_array(columns) * column_weights_array)
    folded = column_weights.fold(list(x.shape), 3, 5)
    right = np.sum(x_array * as_array(folded))

    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)


def test_rectangular_conv2d_forward_and_backward():
    conv = Conv2D(2, 3, 3, 5, bias=True)
    with no_grad():
        conv.weight.copy_(Tensor.ones_like(conv.weight))
        conv.bias.copy_(Tensor.zeros_like(conv.bias))

    x = Tensor.ones([1, 2, 4, 6], requires_grad=True)
    output = conv(x)

    spatial_coverage = np.outer(
        np.array([2, 3, 3, 2], dtype=np.float32),
        np.array([3, 4, 5, 5, 4, 3], dtype=np.float32),
    )
    expected_output = np.broadcast_to(
        2 * spatial_coverage,
        (1, 3, 4, 6),
    )
    np.testing.assert_allclose(as_array(output), expected_output)

    output.backward(Tensor.ones_like(output))

    expected_input_grad = np.broadcast_to(
        3 * spatial_coverage,
        (1, 2, 4, 6),
    )
    np.testing.assert_allclose(as_array(x.grad), expected_input_grad)

    kernel_coverage = np.outer(
        np.array([3, 4, 3], dtype=np.float32),
        np.array([4, 5, 6, 5, 4], dtype=np.float32),
    )
    expected_weight_grad = np.broadcast_to(
        kernel_coverage,
        (3, 2, 3, 5),
    )
    np.testing.assert_allclose(as_array(conv.weight.grad), expected_weight_grad)
    np.testing.assert_allclose(as_array(conv.bias.grad), np.full(3, 24.0))


def test_conv2d_parameter_registration():
    with_bias = Conv2D(2, 3, 3, 5, bias=True)
    without_bias = Conv2D(2, 3, 3, 5, bias=False)

    assert [name for name, _ in with_bias.named_parameters()] == ["weight", "bias"]
    assert [name for name, _ in without_bias.named_parameters()] == ["weight"]
    assert without_bias.bias is None


def test_even_kernel_is_rejected():
    conv = Conv2D(1, 1, 2, 3)
    x = Tensor.ones([1, 1, 4, 4], requires_grad=False)

    try:
        conv(x)
    except ValueError as error:
        assert "odd kernel sizes" in str(error)
    else:
        raise AssertionError("Conv2D should reject even kernel sizes")


if __name__ == "__main__":
    test_unfold_fold_adjoint()
    test_rectangular_conv2d_forward_and_backward()
    test_conv2d_parameter_registration()
    test_even_kernel_is_rejected()
    print("Conv2D bonus tests passed: 4/4")
