import torch
import numpy as np
import normflowpy as nfp
import pyresearchutils as pru
from torch import nn
import constants


class DOALayer(nfp.ConditionalBaseFlowLayer):
    def __init__(self,
                 m_sensors,
                 k_target,
                 wavelength,
                 signal_covariance_matrix=None,
                 noise_covariance_matrix=None,
                 sensors_locations=None):
        super().__init__()
        self.k_target = k_target
        self.m_sensors = m_sensors
        init_sensors_locations = torch.randn([m_sensors, 2]) if sensors_locations is None else sensors_locations
        self.sensor_location = nn.Parameter(torch.randn([m_sensors, 2]))
        self.s = 2 * np.pi / wavelength
        init_signal_covariance_matrix = torch.diag(torch.diagonal(torch.ones(
            [k_target, k_target]))) + 0 * 1j if signal_covariance_matrix is None else signal_covariance_matrix
        init_noise_covariance_matrix = torch.diag(torch.diagonal(torch.ones(
            [m_sensors, m_sensors]))) + 0 * 1j if noise_covariance_matrix is None else noise_covariance_matrix
        self.signal_covariance_matrix = nn.Parameter(init_signal_covariance_matrix)
        self.noise_covariance_matrix = nn.Parameter(init_noise_covariance_matrix)

    def steering_matrix(self, locations):
        if self.k_target != locations.shape[1]:
            pru.logger.critical("Mismatch in number of targets")
        delay_matrix = self.s * (
                torch.sin(locations).unsqueeze(dim=-1) @ self.sensor_location[:, 0].unsqueeze(dim=0)) + (
                               torch.cos(locations).unsqueeze(dim=-1) @ self.sensor_location[:, 1].unsqueeze(dim=0))

        A = torch.exp(1j * delay_matrix)
        return torch.permute(A, [0, 2, 1])

    def compute_r_matrix(self, in_a_matrix):
        return ((in_a_matrix @ self.signal_covariance_matrix) @ (
            torch.permute(in_a_matrix, [0, 2, 1]).conj())) + self.noise_covariance_matrix.unsqueeze(dim=0)

    def forward(self, x, **kwargs):
        locations = kwargs[constants.DOAS]
        if x.shape[0] != locations.shape[0] and locations.shape[0] != 1:
            pru.logger.critical("Mismatch in number of targets")

        A = self.steering_matrix(locations)
        l_matrix = torch.linalg.cholesky(self.compute_r_matrix(A))
        l_matrix_inv = torch.linalg.inv(l_matrix)
        return (l_matrix_inv.unsqueeze(dim=1) @ x.unsqueeze(dim=-1)).squeeze(dim=-1), torch.log(
            torch.abs(torch.linalg.det(l_matrix_inv)))

    def backward(self, z, **kwargs):
        locations = kwargs[constants.DOAS]
        if z.shape[0] != locations.shape[0] and locations.shape[0] != 1:
            pru.logger.critical("Mismatch in number of targets")
        A = self.steering_matrix(locations)
        L = torch.linalg.cholesky(self.compute_r_matrix(A))
        return (L.unsqueeze(dim=1) @ z.unsqueeze(dim=-1)).squeeze(dim=-1), torch.log(torch.abs(torch.linalg.det(L)))
