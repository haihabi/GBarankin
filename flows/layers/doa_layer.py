import torch
import numpy as np
import normflowpy as nfp
import pyresearchutils as pru
from torch import nn
import constants
from flows.helpers import LearnedHermitianPositiveDefiniteMatrix


class DOALayer(nfp.ConditionalBaseFlowLayer):
    def __init__(self,
                 m_sensors,
                 k_target,
                 wavelength,
                 nominal_sensors_locations,
                 signal_covariance_matrix=None,
                 noise_covariance_matrix=None,
                 is_multiple_snrs=False):
        super().__init__()
        self.k_target = k_target
        self.m_sensors = m_sensors
        self.sensor_location = nn.Parameter(nominal_sensors_locations,
                                            requires_grad=False)
        self.s = 2 * np.pi / wavelength
        self._signal_covariance_matrix = LearnedHermitianPositiveDefiniteMatrix(k_target,
                                                                                init_matrix=None if signal_covariance_matrix is None else signal_covariance_matrix)
        self._noise_covariance_matrix = LearnedHermitianPositiveDefiniteMatrix(m_sensors,
                                                                               init_matrix=None if noise_covariance_matrix is None else noise_covariance_matrix)
        self.eps = 1e-6

    def steering_matrix(self, locations):
        if self.k_target != locations.shape[1]:
            pru.logger.critical("Mismatch in number of targets")
        delay_matrix = self.s * (
                torch.sin(locations).unsqueeze(dim=-1) @ self.sensor_location[:, 0].unsqueeze(dim=0)) + (
                               torch.cos(locations).unsqueeze(dim=-1) @ self.sensor_location[:, 1].unsqueeze(dim=0))

        A = torch.exp(1j * delay_matrix)
        return torch.permute(A, [0, 2, 1])

    @property
    def signal_covariance_matrix(self):
        return self._signal_covariance_matrix()

    @property
    def noise_covariance_matrix(self):
        return self._noise_covariance_matrix()

    def compute_r_matrix(self, in_a_matrix, noise_scale):
        noise_scale = noise_scale.reshape([-1, 1, 1]) ** 2
        return ((in_a_matrix @ self.signal_covariance_matrix) @ (
            torch.permute(in_a_matrix, [0, 2, 1]).conj())) + self.noise_covariance_matrix.unsqueeze(
            dim=0) * torch.complex(noise_scale, torch.zeros_like(noise_scale))

    def forward(self, x, **kwargs):
        locations = kwargs[constants.DOAS]
        if x.shape[0] != locations.shape[0] and locations.shape[0] != 1:
            pru.logger.critical("Mismatch in number of targets")

        ns = kwargs[constants.NS]

        A = self.steering_matrix(locations)
        R = self.compute_r_matrix(A, ns)

        l_matrix = torch.linalg.cholesky(R)
        l_matrix_inv = torch.linalg.inv(l_matrix)
        return (l_matrix_inv.unsqueeze(dim=1) @ x.unsqueeze(dim=-1)).squeeze(dim=-1), x.shape[1] * torch.log(
            torch.abs(torch.linalg.det(l_matrix_inv)))

    def backward(self, z, **kwargs):
        locations = kwargs[constants.DOAS]
        if z.shape[0] != locations.shape[0] and locations.shape[0] != 1:
            pru.logger.critical("Mismatch in number of targets")
        ns = kwargs[constants.NS]
        A = self.steering_matrix(locations)
        R = self.compute_r_matrix(A, ns)

        L = torch.linalg.cholesky(R)
        return (L.unsqueeze(dim=1) @ z.unsqueeze(dim=-1)).squeeze(dim=-1), z.shape[1] * torch.log(
            torch.abs(torch.linalg.det(L)))
