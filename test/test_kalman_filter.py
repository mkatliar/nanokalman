"""
This code is based on https://gitlab.syscop.de/mikhail.katliar/tmpc/-/blob/master/test/estimation/KalmanFilterTest.cpp
"""

from nanokalman import KalmanFilter
import numpy as np
import pytest
import sys


@pytest.fixture
def nx() -> int:
    return 4

@pytest.fixture
def ny() -> int:
    return 2

@pytest.fixture
def x0() -> np.ndarray:
    return np.array([0.8244,    0.9827,    0.7302,    0.3439])

@pytest.fixture
def u0() -> np.ndarray:
    return np.array([0.5841,    0.1078,    0.9063])

@pytest.fixture
def A() -> np.ndarray:
    return np.array([
        [0.1734,    0.0605,    0.6569,    0.0155],
        [0.3909,    0.3993,    0.6280,    0.9841],
        [0.8314,    0.5269,    0.2920,    0.1672],
        [0.8034,    0.4168,    0.4317,    0.1062]
    ])

@pytest.fixture
def B() -> np.ndarray:
    return np.array([
        [0.3724,    0.9516,    0.2691],
        [0.1981,    0.9203,    0.4228],
        [0.4897,    0.0527,    0.5479],
        [0.3395,    0.7379,    0.9427]
    ])

@pytest.fixture
def C() -> np.ndarray:
    return np.array([
        [0.1781,    0.9991,    0.0326,    0.8819],
        [0.1280,    0.1711,    0.5612,    0.6692]
    ])

@pytest.fixture
def Q() ->np.ndarray:
    return np.array([
        [0.4447,    0.4192,    0.5073,    0.6125],
        [0.4192,    1.1026,    1.0150,    0.8676],
        [0.5073,    1.0150,    1.0083,    0.9772],
        [0.6125,    0.8676,    0.9772,    1.4596]
    ])

@pytest.fixture
def R() -> np.ndarray:
    return np.array([
        [0.4442,    0.2368],
        [0.2368,    0.1547]
    ])

@pytest.fixture
def x_hat0() -> np.ndarray:
    return np.array([0.9436,    0.6377,    0.9577,    0.2407])

@pytest.fixture
def P0() -> np.ndarray:
    return np.array([
        [1.6604,    1.2726,    1.1100,    0.5482],
        [1.2726,    1.1217,    0.9461,    0.4314],
        [1.1100,    0.9461,    1.1303,    0.6500],
        [0.5482,    0.4314,    0.6500,    0.4815]
    ])

@pytest.fixture
def kalman(nx: int, ny: int, Q: np.ndarray, R: np.ndarray, x_hat0: np.ndarray, P0: np.ndarray) -> KalmanFilter:
    kalman = KalmanFilter(nx=nx, ny=ny)
    kalman.set_state_estimate(x_hat0)
    kalman.set_state_covariance(P0)
    return kalman


def test_get_nx(kalman: KalmanFilter, nx: int):
    assert kalman.get_nx() == nx

def test_get_ny(kalman: KalmanFilter, ny: int):
    assert kalman.get_ny() == ny

def test_get_state_covariance(kalman: KalmanFilter, P0: np.ndarray):
    np.testing.assert_array_equal(kalman.get_state_covariance(), P0)

def test_get_state_estimate(kalman: KalmanFilter, x_hat0: np.ndarray):
    np.testing.assert_array_equal(kalman.get_state_estimate(), x_hat0)

def test_update(kalman: KalmanFilter, C: np.ndarray, x0: np.ndarray, R: np.ndarray):
    y0 = np.dot(C, x0) + np.array([-0.2979, -0.1403]) - np.dot(C, kalman.get_state_estimate())
    kalman.update(y0, C, R)

    np.testing.assert_allclose(kalman.get_state_estimate(), np.array([1.0672, 0.7747, 0.7956, 0.0509]), atol=1e-4, rtol=0.)
    np.testing.assert_allclose(
        kalman.get_state_covariance(),
        np.array([
            [0.4024,    0.2068,    0.0987,    0.0031],
            [0.2068,    0.2157,    0.1147,   -0.0068],
            [0.0987,    0.1147,    0.1118,    0.0205],
            [0.0031,   -0.0068,    0.0205,    0.0672]
        ]),
        atol = 1e-4,
        rtol = 0.
    )

def test_predict(kalman: KalmanFilter, A: np.ndarray, B: np.ndarray, u0: np.ndarray, Q: np.ndarray):
    x_next = np.dot(A, kalman.get_state_estimate()) + np.dot(B, u0)
    kalman.predict(x_next, A=A, Q=Q)

    np.testing.assert_allclose(kalman.get_state_estimate(), x_next, atol=1e-4, rtol=0.)
    np.testing.assert_allclose(
        kalman.get_state_covariance(),
        np.array([
            [1.3584,    2.3924,    2.2925,    2.3721],
            [2.3924,    5.4282,    4.8832,    4.6771],
            [2.2925,    4.8832,    4.8142,    4.6958],
            [2.3721,    4.6771,    4.6958,    5.0963]
        ]),
        atol = 1e-4,
        rtol = 0.
    )


if __name__ == "__main__":
    sys.exit(pytest.main())
