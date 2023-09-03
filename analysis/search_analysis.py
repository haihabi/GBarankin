import math

import numpy as np
import scipy

import constants
import doatools.model as model
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
theta = -np.pi / 10
# Create a 12-element ULA.
ula = model.UniformLinearArray(m_sensors, d0)
theta = -np.pi / 3
sources = model.FarField1DSourcePlacement(
    [theta]
)
# All sources share the same power.
power_source = 1  # Normalized
source_signal = model.ComplexStochasticSignal(sources.size, power_source)

n_snapshots = 5

snrs = [6, -1, -21]

legned_type_dict = {6: "Asymptotic",
                    -1: "Threshold",
                    -21: "No Information"}
color = {6: "red",
         -1: "green",
         -21: "blue"}
plt.figure(figsize=(8, 6))
for i, snr in enumerate(snrs):
    power_noise = power_source / (10 ** (snr / 10))
    BB_stouc, bb_matrix, test_points, search_landscpae, test_points_search_array = perf.barankin_stouc_farfield_1d(ula,
                                                                                                                   sources,
                                                                                                                   wavelength,
                                                                                                                   power_source,
                                                                                                                   power_noise,
                                                                                                                   n_snapshots,
                                                                                                                   output_search_landscape=True)

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
    gbarankin_with_search, _, search_landscape_gbb, test_points_search_array_gbb, nf_bm_matrix = generative_bound.generative_barankin_bound(
        doa_optimal_flow,
        n_samples2generate,
        parameter_name=constants.DOAS,
        doas=torch.tensor(
            sources).to(
            pru.get_working_device()).reshape(
            [1, -1]).float())

    plt.plot(test_points_search_array[0] - theta, search_landscpae[0],
             label=f"SNR={snr} ({legned_type_dict[snr]})",
             color=color[snr])
    # plt.plot(test_points_search_array_gbb[0].cpu().numpy() - theta, search_landscape_gbb[0], "--",
    #          label=f"Generative SNR= SNR={snr} ({legned_type_dict[snr]})",
    #          color=color[snr])
plt.grid()
plt.legend()
plt.xlabel(r"$\psi-\theta$")
plt.ylabel(r"$\alpha$")
plt.show()
# peaks_gbb = [torch.where(test_points_search_array_gbb.flatten() == nf_bm_matrix[i, :].flatten())[0].item() for i in
#              range(nf_bm_matrix.shape[0])]
#
#
# # gbarankin, _, _, _, _ = generative_bound.generative_barankin_bound(doa_optimal_flow,
# #                                                                    n_samples2generate,
# #                                                                    test_points=test_points,
# #                                                                    parameter_name=constants.DOAS,
# #                                                                    doas=torch.tensor(
# #                                                                        sources).to(
# #                                                                        pru.get_working_device()).reshape(
# #                                                                        [1, -1]).float())
#
# def apply_transform(in_search_array, beta_array):
#     return beta_array
#
#
# plt.figure(figsize=(8, 6))
# if plot_snr == snr:
#     eps = 1e-2
#     base_array = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, nf_bm_matrix.shape[0])
#
#     delta_filter = np.diff(test_points_search_array_gbb.cpu().numpy() - theta)
#
#     loc = np.where(np.logical_not(np.isclose(np.min(delta_filter), delta_filter, atol=np.min(delta_filter))))[0]
#     loc = np.concatenate([np.zeros(1), loc + 1, np.ones(1) * len(test_points_search_array_gbb.cpu().numpy())])
#     loc = loc.astype("int")
#     peaks = scipy.signal.find_peaks(search_landscpae[0])[0]
#     plt.semilogy(test_points_search_array[0].flatten() - theta,
#                  apply_transform(test_points_search_array[0].flatten(), search_landscpae[0]),
#                  label="BB (Feasible TP)")
#     plt.semilogy(test_points_search_array[0].flatten()[peaks] - theta,
#                  apply_transform(test_points_search_array[0].flatten()[peaks], search_landscpae[0][peaks])
#                  , "o",
#                  label="BB (Selected)")
#
#     delta_b = np.diff(test_points_search_array_gbb.cpu().numpy())
#     delta_min = np.min(delta_b)
#     delta_min = delta_min * torch.ones(1, device=test_points_search_array_gbb.device)
#     for i in range(len(loc) - 1):
#         plt.semilogy(test_points_search_array_gbb.cpu().numpy()[loc[i]:loc[i + 1]] - theta,
#                      apply_transform(test_points_search_array_gbb.cpu().numpy(),
#                                      search_landscape_gbb[loc[i]:loc[i + 1]]),
#                      "red",
#                      label="GBB (Feasible TP)" if i == 0 else None)
#
#     # peaks = scipy.signal.find_peaks(search_landscape_gbb)[0]
#     # if len(peaks) > 1:
#     #     sel = test_points_search_array_gbb.reshape([1, -1])[:, peaks].reshape([-1, 1])
#     #     pos_delta = test_points_search_array_gbb.reshape([1, -1])[:, peaks + 1].reshape([-1, 1])
#     #     neg_delta = test_points_search_array_gbb.reshape([1, -1])[:, peaks - 1].reshape([-1, 1])
#     #     index_to_keep = torch.logical_and(
#     #         torch.isclose(torch.abs(pos_delta - sel), delta_min, rtol=1, atol=delta_min.item()),
#     #         torch.isclose(torch.abs(neg_delta - sel), delta_min, rtol=1,
#     #                       atol=delta_min.item())).cpu().numpy().flatten()
#     #     peaks = peaks[index_to_keep]
#     plt.semilogy(test_points_search_array_gbb.cpu().numpy()[peaks_gbb] - theta,
#                  apply_transform(test_points_search_array_gbb.cpu().numpy()[peaks_gbb],
#                                  search_landscape_gbb[peaks_gbb])
#                  , "v",
#                  label="GBB (Selected)")
#     plt.legend(fontsize=constants.FONTSIZE)
#     plt.grid()
#
#     plt.xlabel(r"$\psi-\theta$", fontsize=constants.FONTSIZE)
#     plt.ylabel(r"$\alpha$", fontsize=constants.FONTSIZE)
#     plt.tight_layout()
#     plt.savefig(f"tp_search_{snr}.svg")
#     plt.show()
