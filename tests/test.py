import ukf
import numpy as np

def test_ukf_init():
    test_ukf = ukf.UnscentedKalmanFilter(1,1,1)
    np.testing.assert_equal(test_ukf._dim_x, 1)

