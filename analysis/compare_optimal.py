import math

import numpy as np

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
# Place 8 sources uniformly within (-pi/3, pi/4)
sources = model.FarField1DSourcePlacement(
    [-np.pi / 10]

)
# All sources share the same power.
power_source = 1  # Normalized
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
# 200 snapshots.
n_snapshots = 10
# We use root-MUSIC.
estimator = estimation.RootMUSIC1D(wavelength)

snrs = np.linspace(-30, 10, 15)
# snrs = [-12.5]
# 300 Monte Carlo runs for each SNR
n_repeats = 300

mses = np.zeros((len(snrs),))
# crbs_sto = np.zeros((len(snrs),))
barankin_stouc = np.zeros((len(snrs),))
gbarankin_stouc = np.zeros((len(snrs),))
# crbs_det = np.zeros((len(snrs),))
crbs_stouc = np.zeros((len(snrs),))
gcrbs_stouc = np.zeros((len(snrs),))

for i, snr in enumerate(snrs):
    power_noise = power_source / (10 ** (snr / 10))
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)

    B_stouc, _ = perf.crb_stouc_farfield_1d(ula, sources, wavelength, power_source,
                                            power_noise, n_snapshots)
    BB_stouc, bb_matrix, test_points = perf.barankin_stouc_farfield_1d(ula, sources, wavelength, power_source,
                                                                       power_noise, n_snapshots)
    print(test_points.shape)
    #################
    # Generative Bound
    #################
    # The squared errors and the deterministic CRB varies
    # for each run. We need to compute the average.
    locations = torch.Tensor(ula._locations)
    if locations.shape[1] == 1:
        locations = torch.cat([locations, torch.zeros_like(locations)], dim=-1)
    n_samples2generate = 8 * 64000
    doa_optimal_flow = flows.DOAFlow(n_snapshots, m_sensors, k_targets, wavelength,
                                     nominal_sensors_locations=locations.to(pru.get_working_device()).float(),
                                     signal_covariance_matrix=torch.diag(
                                         torch.diag(torch.ones(k_targets, k_targets))).to(
                                         pru.get_working_device()).float() + 0 * 1j,
                                     noise_covariance_matrix=power_noise * torch.diag(
                                         torch.diag(torch.ones(m_sensors, m_sensors))).to(
                                         pru.get_working_device()).float() + 0 * 1j)
    doa_optimal_flow = doa_optimal_flow.to(pru.get_working_device())
    gcrb = generative_bound.generative_cramer_rao_bound(doa_optimal_flow, n_samples2generate,
                                                        parameter_name=constants.DOAS,
                                                        doas=torch.tensor(sources).to(pru.get_working_device()).reshape(
                                                            [1, -1]).float())
    test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
    gbarankin, gbb = generative_bound.generative_barankin_bound(doa_optimal_flow, n_samples2generate, test_points,
                                                                parameter_name=constants.DOAS,
                                                                doas=torch.tensor(sources).to(
                                                                    pru.get_working_device()).reshape(
                                                                    [1, -1]).float())

    print(gbarankin, BB_stouc)
    # print(gcrb, B_stouc)
    gbb_np = gbb.cpu().numpy()

    one_time_one = np.ones(gbb_np.shape)
    eps = 1e-5
    cond = np.linalg.norm(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps) / np.linalg.norm(
        np.linalg.inv(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps))
    np.linalg.norm(np.linalg.inv(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps) - np.linalg.inv(
        bb_matrix - one_time_one + np.eye(gbb_np.shape[0]) * eps)) / np.linalg.norm(
        np.linalg.inv(bb_matrix - one_time_one + np.eye(gbb_np.shape[0]) * eps))

    print("BB RE", np.linalg.norm(gbb.cpu().numpy() - bb_matrix) / np.linalg.norm(bb_matrix))
    print("GBB RE", np.linalg.norm(gbarankin.cpu().numpy() - BB_stouc) / np.linalg.norm(BB_stouc))
    print("GCRB RE", np.linalg.norm(gcrb.cpu().numpy() - B_stouc) / np.linalg.norm(B_stouc))
    gcrbs_stouc[i] = np.sqrt(torch.diag(gcrb).mean().item())
    gbarankin_stouc[i] = np.sqrt(torch.diag(gbarankin).mean().item())
    cur_mse = 0.0

    for r in range(n_repeats):
        # Stochastic signal model.
        A = ula.steering_matrix(sources, wavelength)

        S = source_signal.emit(n_snapshots)
        N = noise_signal.emit(n_snapshots)
        Y = A @ S + N
        Rs = (S @ S.conj().T) / n_snapshots
        Ry = (Y @ Y.conj().T) / n_snapshots
        resolved, estimates = estimator.estimate(Ry, sources.size, d0)
        # In practice, you should check if `resolved` is true.
        # We skip the check here.
        cur_mse += np.mean((estimates.locations - sources.locations) ** 2)

    mses[i] = np.sqrt(cur_mse / n_repeats)

    crbs_stouc[i] = np.sqrt(np.mean(np.diag(B_stouc)))
    barankin_stouc[i] = np.sqrt(np.mean(np.diag(BB_stouc)))
    print('Completed SNR = {0:.2f} dB'.format(snr))
plt.figure(figsize=(8, 6))
plt.semilogy(
    snrs, mses, '-x',
    snrs, gbarankin_stouc, '--v',
    snrs, gcrbs_stouc, '--v',
    snrs, crbs_stouc, '--',
    snrs, barankin_stouc, '--'
)
plt.xlabel('SNR (dB)')
plt.ylabel(r'RMSE / $\mathrm{rad}$')
plt.grid(True)
plt.legend(['MSE',
            'GBarankin',
            'GCRB ',
            'CRB ',
            'Barankin'])
plt.title('MSE vs. CRB')
plt.margins(x=0)
plt.savefig("compare_vs_snr.svg")
plt.show()
