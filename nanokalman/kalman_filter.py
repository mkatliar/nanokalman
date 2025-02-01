import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    @brief Kalman filter implementation.

    The code is based on https://gitlab.syscop.de/mikhail.katliar/tmpc/-/blob/master/tmpc/estimation/KalmanFilter.hpp
    """
    def __init__(self, nx: int, ny: int):
        self._nx = nx
        self._ny = ny
        self._state_estimate = np.zeros(nx)
        self._state_covariance = np.zeros((nx, nx))
        self._S = np.zeros((ny, ny))
        self._K = np.zeros((nx, ny))

    def get_state_estimate(self) -> np.ndarray:
        """
        @brief Get state estimate
        """
        return self._state_estimate

    def set_state_estimate(self, val: np.ndarray):
        """
        @brief Set state estimate
        """
        if val.shape != self._state_estimate.shape:
            raise ValueError("Invalid argument shape")

        self._state_estimate = val

    def get_state_covariance(self) -> np.ndarray:
        """
        @brief Get state covariance
        """
        return self._state_covariance

    def set_state_covariance(self, val: np.ndarray):
        """
        @brief Set state covariance
        """
        if val.shape != self._state_covariance.shape:
            raise ValueError("Invalid argument shape")

        self._state_covariance = val

    def get_process_noise_covariance(self) -> np.ndarray:
        """
        @brief Get process noise covariance
        """
        return self._process_noise_covariance

    def predict(self, x_next: np.ndarray, A: np.ndarray, Q: np.ndarray):
        """
        @brief Update state estimate based on a linear model and control input.

        @param x_next next state computed from the system dynamics
        @param A linear model matrix A = d(f)/dx, where f is the system dynamics
        @param Q process noise covariance
        """
        if A.shape != (self._nx, self._nx):
            raise ValueError("Invalid shape of A")

        self._state_estimate = x_next
        self._state_covariance = np.linalg.multi_dot((A, self._state_covariance, np.transpose(A))) + Q

    def update(self, y: np.ndarray, C: np.ndarray, R: np.ndarray):
        """
        @brief Update state estimate based on measurement residual and sensitivities.

        @param y measurement residual, the difference between measured and predicted system output.
        @param C output sensitivity matrix, C = d(y_pred)/dx, where y_pred is the predicted system output for state x.
        @param R measurement noise covariance
        """
        self._S = R + np.linalg.multi_dot((C, self._state_covariance, np.transpose(C)))
        S_chol = np.linalg.cholesky(self._S)

        # Calculate K = stateCovariance_ * trans(C) * inv(S)
        # by solving
        # S_chol * trans(S_chol) * trans(K) = C * stateCovariance_
        trans_S_chol_trans_K = scipy.linalg.solve_triangular(S_chol, np.dot(C, self._state_covariance), lower=True)
        self._K = np.transpose(scipy.linalg.solve_triangular(S_chol, trans_S_chol_trans_K, trans='T', lower=True))

        # Update state estimate and covariance
        self._state_estimate += np.dot(self._K, y)

        # Use the Cholesky decomposition of S to enforce symmetry by putting the expression in the form A*A^T.
        K_S_chol = np.dot(self._K, S_chol)
        self._state_covariance -= np.dot(K_S_chol, np.transpose(K_S_chol))

    def get_nx(self) -> int:
        """
        @brief Number of states
        """
        return self._nx

    def get_ny(self) -> int:
        """
        @brief Number of outputs
        """
        return self._ny

    def get_gain(self) -> np.ndarray:
        """
        @brief Kalman gain on the last update step
        """
        return self._K

    def get_measurement_prediction_covariance(self) -> np.ndarray:
        """
        @brief Measurement prediction covariance on the last update step
        """
        return self._S
