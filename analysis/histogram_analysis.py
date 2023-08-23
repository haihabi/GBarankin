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


def relative_error(in_est, in_ref):
    return np.linalg.norm(in_est - in_ref) / np.linalg.norm(in_ref)


def run_trails(in_flow_model, in_n_samples, in_m_trails, in_test_points, in_sources, ref_barankin, ref_bb):
    results_list = []
    for i in range(in_m_trails):
        gbarankin, gbb, _, _ = generative_bound.generative_barankin_bound(in_flow_model, in_n_samples, in_test_points,
                                                                          parameter_name=constants.DOAS,
                                                                          doas=torch.tensor(in_sources).to(
                                                                              pru.get_working_device()).reshape(
                                                                              [1, -1]).float())
        re_bb = relative_error(gbarankin.cpu().numpy(), ref_barankin)
        re_bm = relative_error(gbb.cpu().numpy(), ref_bb)
        results_list.append([re_bb, re_bm])
    return np.asarray(results_list)


###########################
# Parameters
###########################
wavelength = 1.0  # normalized
d0 = wavelength / 2
m_sensors = 20
k_targets = 1
n_samples2generate = 512
n_snapshots = 10
m_trails = 1000
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

# We use root-MUSIC.
estimator = estimation.RootMUSIC1D(wavelength)

snrs = constants.SNR_POINTS
# n_repeats = 300

# mses = np.zeros((len(snrs),))
# crbs_sto = np.zeros((len(snrs),))
# barankin_stouc = np.zeros((len(snrs),))
# gbarankin_stouc = np.zeros((len(snrs),))
# crbs_det = np.zeros((len(snrs),))
# crbs_stouc = np.zeros((len(snrs),))
# gcrbs_stouc = np.zeros((len(snrs),))
snr = -20
# for i, snr in enumerate(snrs):
power_noise = power_source / (10 ** (snr / 10))
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)

# B_stouc, _ = perf.crb_stouc_farfield_1d(ula, sources, wavelength, power_source,
#                                         power_noise, n_snapshots)
print("Compute Barankin Bound")
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

data = run_trails(doa_optimal_flow, n_samples2generate, m_trails, test_points, sources, BB_stouc, bb_matrix)
plt.subplot(1, 2, 1)
plt.hist(data[:, 0])
plt.subplot(1, 2, 2)
plt.hist(data[:, 1])
plt.show()
print("a")
