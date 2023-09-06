import numpy.random
import torch
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import numpy as np
import pyresearchutils as pru
from enum import Enum
import flows
import os
import pickle
from doatools.model.signals import SignalGenerator
from doatools.utils.math import randcn


class SignalType(Enum):
    ComplexGaussian = 0
    CorrelatedComplexGaussian = 2
    QAM4 = 1


class NoiseMatrix(Enum):
    Uncorrelated = 0
    Correlated = 1


class QAM4(SignalGenerator):

    def __init__(self, dim, C=1.0):
        self._dim = dim
        self.scale = C / np.sqrt(2)

    def _generator(self, n):
        a = 2 * np.random.randint(low=0, high=2, size=(self.dim, n)) - 1
        b = 2 * np.random.randint(low=0, high=2, size=(self.dim, n)) - 1
        return (a + 1j * b) * self.scale

    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        return self._generator(n)


class DOASignalModel:
    POWER_SOURCE = 1  # Normalized

    def __init__(self, m_sensors, n_snapshots, k_targets, in_snr, wavelength=1.0,
                 signal_type: SignalType = SignalType.ComplexGaussian,
                 noise_type: NoiseMatrix = NoiseMatrix.Uncorrelated,
                 array_perturbed_scale: float = 0.0,
                 is_multiple_snr=False,
                 snr_min=-30,
                 snr_max=10):
        self.d0 = wavelength / 2
        self.wavelength = wavelength
        self.n_snapshots = n_snapshots
        self.m_sensors = m_sensors

        self.array = model.UniformLinearArray(m_sensors, self.d0)
        self.signal_type = signal_type
        self.noise_type = noise_type
        self.signal_matrix = (np.eye(k_targets) * DOASignalModel.POWER_SOURCE).astype("complex64")
        if signal_type == SignalType.ComplexGaussian:
            self.source_signal = model.ComplexStochasticSignal(k_targets, DOASignalModel.POWER_SOURCE)
        elif signal_type == SignalType.CorrelatedComplexGaussian:
            D = np.eye(k_targets) * DOASignalModel.POWER_SOURCE
            L = np.tril(0.9 * randcn([k_targets, k_targets]), k=-1) + np.eye(k_targets)
            self.signal_matrix = (L @ D @ L.T.conj()).astype("complex64")
            self.source_signal = model.ComplexStochasticSignal(k_targets, self.signal_matrix)
        elif signal_type == SignalType.QAM4:
            self.source_signal = model.ComplexStochasticSignal(k_targets, DOASignalModel.POWER_SOURCE)
        else:
            raise Exception("")
        self.is_multiple_snr = is_multiple_snr
        if self.is_multiple_snr:
            self.power_noise = 1
        else:
            self.power_noise = DOASignalModel.POWER_SOURCE / (10 ** (in_snr / 10))
        if noise_type == NoiseMatrix.Uncorrelated:
            self.noise_matrix = (np.eye(self.array.size) * self.power_noise + 0 * 1j).astype("complex64")
            self.noise_signal = model.ComplexStochasticSignal(self.array.size, self.noise_matrix)
        elif noise_type == NoiseMatrix.Correlated:
            D = np.eye(self.array.size) * self.power_noise
            L = np.tril(0.05 * randcn([self.array.size, self.array.size]), k=-1) + np.eye(self.array.size)
            self.noise_matrix = (L @ D @ L.T.conj()).astype("complex64")
            self.noise_signal = model.ComplexStochasticSignal(self.array.size, self.noise_matrix)
        else:
            raise Exception("")

        self.array_perturbed_scale = array_perturbed_scale
        self.k_targets = k_targets

        self.snr_min = snr_min
        self.snr_max = snr_max

    def save_model(self, folder):
        with open(os.path.join(folder, "signal_model.pkl"), 'wb') as file:
            # A new file will be created
            pickle.dump({"noise_matrix": self.noise_matrix,
                         "signal_matrix": self.signal_matrix}, file)

    def load_model(self, folder):
        with open(folder, "rb") as file:
            data = pickle.load(file)
        self.noise_matrix = data["noise_matrix"]
        self.signal_matrix = data["signal_matrix"]

    @staticmethod
    def build_sources(in_theta):
        if np.isscalar(in_theta):
            in_theta = [in_theta]

        sources = model.FarField1DSourcePlacement(
            in_theta
        )
        return sources

    def mse_mle(self, in_sources, n_repeats=300):
        sources = self.build_sources(in_sources)
        if sources.size != self.k_targets:
            raise Exception("Mismatch in number of sources")
        cur_mse = 0
        estimator = estimation.RootMUSIC1D(self.wavelength)
        for r in range(n_repeats):
            if self.array_perturbed_scale > 0:
                self.array = self.array.get_perturbed_copy(
                    {'location_errors': (np.random.randn(self.array.size, 1) * self.array_perturbed_scale, True)})
            # Stochastic signal model.
            A = self.array.steering_matrix(sources, self.wavelength)
            S = self.source_signal.emit(self.n_snapshots)
            N = self.noise_signal.emit(self.n_snapshots)
            Y = A @ S + N
            # Rs = (S @ S.conj().T) / self.n_snapshots
            Ry = (Y @ Y.conj().T) / self.n_snapshots
            resolved, estimates = estimator.estimate(Ry, sources.size, self.d0)
            # In practice, you should check if `resolved` is true.
            # We skip the check here.
            cur_mse += np.mean((estimates.locations - sources.locations) ** 2)
        return cur_mse / n_repeats

    def compute_reference_bound(self, in_theta, in_snr=None):
        if self.is_multiple_snr and in_snr is None:
            raise Exception("")
        elif self.is_multiple_snr:
            noise_scale = np.asarray(DOASignalModel.POWER_SOURCE / (10 ** (in_snr / 10))).astype("float32")
        else:
            noise_scale = 1
        sources = self.build_sources(in_theta)
        crb, _ = perf.crb_stouc_farfield_1d(self.array, sources, self.wavelength, self.signal_matrix,
                                            self.noise_matrix * noise_scale, self.n_snapshots)
        bb_bound, bb_matrix, test_points = perf.barankin_stouc_farfield_1d(self.array, sources, self.wavelength,
                                                                           self.signal_matrix,
                                                                           self.noise_matrix * noise_scale,
                                                                           self.n_snapshots)
        return crb, bb_bound, bb_matrix, test_points

    def get_optimal_flow_model(self):
        locations = torch.Tensor(self.array._locations)
        if locations.shape[1] == 1:
            locations = torch.cat([locations, torch.zeros_like(locations)], dim=-1)

        doa_optimal_flow = flows.DOAFlow(self.n_snapshots, self.m_sensors, self.k_targets, self.wavelength,
                                         nominal_sensors_locations=locations.to(pru.get_working_device()).float(),
                                         signal_covariance_matrix=self.signal_matrix,
                                         noise_covariance_matrix=self.noise_matrix,
                                         n_flow_layer=0,
                                         is_multiple_snrs=self.is_multiple_snr)
        return doa_optimal_flow.to(pru.get_working_device())

    def generate_dataset(self, number_of_samples, transform=None):
        labels_list = []
        data_list = []
        noise_scale = 1
        for i in range(number_of_samples):
            if self.is_multiple_snr:
                snr = self.snr_min + (self.snr_max - self.snr_min) * np.random.rand(1)
                noise_scale = np.sqrt(DOASignalModel.POWER_SOURCE / (10 ** (snr / 10))).astype("float32")
            if self.array_perturbed_scale > 0:
                self.array = self.array.get_perturbed_copy(
                    {'location_errors': (np.random.randn(self.array.size, 1) * self.array_perturbed_scale, True)})

            doas = np.pi * (1 - 0.05) * np.random.rand(self.k_targets) - np.pi * (1 - 0.05) / 2
            sources = model.FarField1DSourcePlacement(
                doas

            )
            A = self.array.steering_matrix(sources, self.wavelength)
            S = self.source_signal.emit(self.n_snapshots)
            N = self.noise_signal.emit(self.n_snapshots)
            Y = A @ S + N * noise_scale
            labels_list.append([doas.astype("float32"), noise_scale])
            data_list.append(Y.T.astype("complex64"))
        return pru.torch.NumpyDataset(data_list, labels_list, transform=transform)
