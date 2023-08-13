import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import numpy as np
import pyresearchutils as pru


class DOASignalModel:
    POWER_SOURCE = 1  # Normalized

    def __init__(self, m_sensors, n_snapshots, k_targets, in_snr, wavelength=1.0):
        self.d0 = wavelength / 2
        self.wavelength = wavelength
        self.n_snapshots = n_snapshots
        self.power_noise = DOASignalModel.POWER_SOURCE / (10 ** (in_snr / 10))
        self.array = model.UniformLinearArray(m_sensors, self.d0)
        self.source_signal = model.ComplexStochasticSignal(k_targets, DOASignalModel.POWER_SOURCE)
        self.noise_signal = model.ComplexStochasticSignal(self.array.size, self.power_noise)
        self.k_targets = k_targets

    def get_optimal_flow_model(self):
        raise NotImplemented

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
            data_list.append(Y.astype("complex64"))
        return pru.torch.NumpyDataset(data_list, labels_list, transform=transform)
