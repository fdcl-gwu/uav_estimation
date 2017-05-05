import sys
sys.path.append("ukf_uav")
import ukf_uav
import numpy as np


def test_ukf_init():
    test_ukf = ukf_uav.UnscentedKalmanFilter(1,1,1)
    np.testing.assert_equal(test_ukf._dim_x, 1)

