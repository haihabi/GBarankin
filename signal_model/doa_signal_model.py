import numpy.random
import torch
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import numpy as np
import pyresearchutils as pru
from enum import Enum
import flows

from doatools.model.signals import SignalGenerator


class SignalType(Enum):
    ComplexGaussian = 0
    QAM4 = 1


class QAM4(SignalGenerator):

    def __init__(self, dim, C=1.0):
        self._dim = dim
        self.scale = C / np.sqrt(2)

    def _generator(self, n):
        a = 2 * np.random.randint(low=0, high=2, size=(self.dim, n)) - 1
        b = 2 * np.random.randint(low=0, high=2, size=(self.dim, n)) - 1
        return (a + 1j * b) / self.scale

    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        return self._generator(n)


class DOASignalModel:
    POWER_SOURCE = 1  # Normalized

    def __init__(self, m_sensors, n_snapshots, k_targets, in_snr, wavelength=1.0,
                 signal_type: SignalType = SignalType.ComplexGaussian):
        self.d0 = wavelength / 2
        self.wavelength = wavelength
        self.n_snapshots = n_snapshots
        self.m_sensors = m_sensors
        self.power_noise = DOASignalModel.POWER_SOURCE / (10 ** (in_snr / 10))
        self.array = model.UniformLinearArray(m_sensors, self.d0)
        self.signal_type = signal_type
        if signal_type == SignalType.ComplexGaussian:
            self.source_signal = model.ComplexStochasticSignal(k_targets, DOASignalModel.POWER_SOURCE)
        elif signal_type == SignalType.QAM4:
            self.source_signal = model.ComplexStochasticSignal(k_targets, DOASignalModel.POWER_SOURCE)
        else:
            raise Exception("")
        self.noise_signal = model.ComplexStochasticSignal(self.array.size, self.power_noise)
        self.k_targets = k_targets

    def compute_reference_bound(self, in_theta):
        sources = model.FarField1DSourcePlacement(
            [in_theta]

        )
        crb, _ = perf.crb_stouc_farfield_1d(self.array, sources, self.wavelength, DOASignalModel.POWER_SOURCE,
                                            self.power_noise, self.n_snapshots)
        bb_bound, bb_matrix, test_points = perf.barankin_stouc_farfield_1d(self.array, sources, self.wavelength,
                                                                           DOASignalModel.POWER_SOURCE,
                                                                           self.power_noise, self.n_snapshots)
        return crb, bb_bound, bb_matrix, test_points

    def get_optimal_flow_model(self):
        if self.signal_type != SignalType.ComplexGaussian:
            return None
        locations = torch.Tensor(self.array._locations)
        if locations.shape[1] == 1:
            locations = torch.cat([locations, torch.zeros_like(locations)], dim=-1)

        doa_optimal_flow = flows.DOAFlow(self.n_snapshots, self.m_sensors, self.k_targets, self.wavelength,
                                         nominal_sensors_locations=locations.to(pru.get_working_device()).float(),
                                         signal_covariance_matrix=torch.diag(
                                             torch.diag(torch.ones(self.k_targets, self.k_targets))).to(
                                             pru.get_working_device()).float() + 0 * 1j,
                                         noise_covariance_matrix=self.power_noise * torch.diag(
                                             torch.diag(torch.ones(self.m_sensors, self.m_sensors))).to(
                                             pru.get_working_device()).float() + 0 * 1j,
                                         n_flow_layer=0)
        return doa_optimal_flow.to(pru.get_working_device())

    def generate_dataset(self, number_of_samples, transform=None):
        labels_list = []
        data_list = []
        for i in range(number_of_samples):
            doas = np.pi * (1 - 0.05) * np.random.rand(self.k_targets) - np.pi * (1 - 0.05) / 2
            sources = model.FarField1DSourcePlacement(
                doas

            )
            A = self.array.steering_matrix(sources, self.wavelength)
            S = self.source_signal.emit(self.n_snapshots)
            N = self.noise_signal.emit(self.n_snapshots)
            Y = A @ S + N
            labels_list.append(doas.astype("float32"))
            data_list.append(Y.T.astype("complex64"))
        return pru.torch.NumpyDataset(data_list, labels_list, transform=transform)
