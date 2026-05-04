import numpy as np

from fdx import utils


def test_order_of_accuracy():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    err = np.array([1.0, 0.5, 0.25, 0.125])
    order_of_accuracy = utils.compute_order_of_accuracy(h, err)
    print(order_of_accuracy)
    assert np.allclose(order_of_accuracy, np.array([0.0, 1.0, 1.0, 1.0]))
