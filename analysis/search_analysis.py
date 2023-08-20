import math

import numpy as np
import scipy

import constants
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import matplotlib.pyplot as plt
import flows
import torch
import pyresearchutils as pru
import generative_bound

wavelength = 1.0  # normalized
d0 = wavelength / 2
m_sensors = 20
k_targets = 1

# Create a 12-element ULA.
ula = model.UniformLinearArray(m_sensors, d0)
sources = model.FarField1DSourcePlacement(
    [-np.pi / 10]
)
# All sources share the same power.
power_source = 1  # Normalized
source_signal = model.ComplexStochasticSignal(sources.size, power_source)

n_snapshots = 10

snrs = constants.SNR_POINTS
s = -7
snrs = [s]
plot_snr = s
results = []
for i, snr in enumerate(snrs):
    power_noise = power_source / (10 ** (snr / 10))
    # noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)

    BB_stouc, bb_matrix, test_points, search_landscpae, test_points_search_array = perf.barankin_stouc_farfield_1d(ula,
                                                                                                                   sources,
                                                                                                                   wavelength,
                                                                                                                   power_source,
                                                                                                                   power_noise,
                                                                                                                   n_snapshots,
                                                                                                                   output_search_landscape=True)
    # print(test_points.shape)
    #################
    # Generative Bound
    #################
    # The squared errors and the deterministic CRB varies
    # for each run. We need to compute the average.
    locations = torch.Tensor(ula._locations)
    if locations.shape[1] == 1:
        locations = torch.cat([locations, torch.zeros_like(locations)], dim=-1)
    n_samples2generate = 64000
    doa_optimal_flow = flows.DOAFlow(n_snapshots, m_sensors, k_targets, wavelength,
                                     nominal_sensors_locations=locations.to(pru.get_working_device()).float(),
                                     signal_covariance_matrix=torch.diag(
                                         torch.diag(torch.ones(k_targets, k_targets))).to(
                                         pru.get_working_device()).float() + 0 * 1j,
                                     noise_covariance_matrix=power_noise * torch.diag(
                                         torch.diag(torch.ones(m_sensors, m_sensors))).to(
                                         pru.get_working_device()).float() + 0 * 1j)
    doa_optimal_flow = doa_optimal_flow.to(pru.get_working_device())
    test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
    gbarankin_with_search, _, search_landscape_gbb, test_points_search_array_gbb = generative_bound.generative_barankin_bound(
        doa_optimal_flow,
        n_samples2generate,
        parameter_name=constants.DOAS,
        doas=torch.tensor(
            sources).to(
            pru.get_working_device()).reshape(
            [1, -1]).float())
    gbarankin, _, _, _ = generative_bound.generative_barankin_bound(doa_optimal_flow,
                                                                    n_samples2generate,
                                                                    test_points=test_points,
                                                                    parameter_name=constants.DOAS,
                                                                    doas=torch.tensor(
                                                                        sources).to(
                                                                        pru.get_working_device()).reshape(
                                                                        [1, -1]).float())
    if plot_snr == snr:
        # eps = 1e-2
        # base_array = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, search_landscape_gbb.shape[0])
        peaks = scipy.signal.find_peaks(search_landscpae)[0]
        plt.plot(test_points_search_array.flatten(), 1 / search_landscpae, label="BM")
        plt.plot(test_points_search_array.flatten()[peaks], 1 / search_landscpae[peaks], "o", label="BM Peaks")
        plt.plot(test_points_search_array_gbb.cpu().numpy(), 1 / search_landscape_gbb, label="GBM")
        peaks = scipy.signal.find_peaks(search_landscape_gbb)[0]
        plt.plot(test_points_search_array_gbb.cpu().numpy()[peaks], 1 / search_landscape_gbb[peaks], "v", label="GBM Peaks")
        plt.legend()
        plt.grid()
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\mathrm{xBM}^{-1}_{i,i}$")
        plt.savefig("tp_search.svg")
        plt.show()

    # print(gbarankin, BB_stouc)
    # # print(gcrb, B_stouc)
    # gbb_np = gbb.cpu().numpy()
    #
    # print("BB RE", np.linalg.norm(gbb.cpu().numpy() - bb_matrix) / np.linalg.norm(bb_matrix))
    # print("GBB RE", np.linalg.norm(gbarankin.cpu().numpy() - BB_stouc) / np.linalg.norm(BB_stouc))
