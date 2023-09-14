import numpy as np
from matplotlib import pyplot as plt

import generative_bound
import pyresearchutils as pru
import torch


def compute_noise_scale(snr, source_power):
    return np.sqrt(source_power / (10 ** (snr / 10))).astype("float32")


def align_bb_matrix(original_tp, final_tp, original_bb, final_bb):
    index_list = []
    for i in range(final_tp.shape[0]):
        j = torch.where(torch.abs(final_tp[i, :] - original_tp).sum(axis=1) == 0)[0].item()
        index_list.append((i, j))
    bb_compare = torch.zeros_like(final_bb)
    for i, j in index_list:
        bb_compare[i, i] = original_bb[j, j]
        for ii, jj in index_list:
            if i != ii:
                bb_compare[i, ii] = original_bb[j, jj]
                bb_compare[ii, i] = original_bb[jj, j]
    return bb_compare


def get_timming_function(is_apply_trimming, in_sm,snr):
    adaptive_trimming = None
    if is_apply_trimming:
        data = in_sm.generate_dataset(10000, forces_snr=snr)
        data_np = np.stack(data.data, axis=0)
        data_np = data_np.reshape([10000, -1])
        mean_vec = np.mean(data_np, axis=0)
        data_np_norm = data_np - mean_vec
        max_norm = np.linalg.norm(data_np_norm, axis=-1).max()

        adaptive_trimming = generative_bound.AdaptiveTrimming(
            generative_bound.TrimmingParameters(mean_vec, max_norm, 0),
            generative_bound.TrimmingType.MAX)
        adaptive_trimming.to(pru.get_working_device())
    return adaptive_trimming


def plot_tp_search(in_data, in_legend, in_theta):
    for i in range(len(in_data)):
        search_landscape, test_point, selet_test_points = in_data[i]
        test_point = test_point.cpu().numpy()
        name, color = in_legend[i]
        delta_filter = np.diff(test_point - in_theta)
        loc = np.where(np.logical_not(np.isclose(np.min(delta_filter), delta_filter, atol=np.min(delta_filter))))[0]
        loc = np.concatenate([np.zeros(1), loc + 1, np.ones(1) * len(test_point)])
        loc = loc.astype("int")
        ###############################
        # Apply Transfor
        ##############################
        alpha = search_landscape
        for i in range(len(loc) - 1):
            plt.semilogy(test_point[loc[i]:loc[i + 1]] - in_theta,
                         alpha[loc[i]:loc[i + 1]],
                         color,
                         label=name if i == 0 else None)

    plt.xlabel(r"$\psi-\theta$")
    plt.ylabel(r"$\alpha$")
    plt.legend()
    plt.grid()
    plt.show()


def rmse_db(x):
    return np.sqrt(x) * 180 / np.pi


def relative_error(in_est, in_ref):
    return np.linalg.norm(in_est - in_ref) / np.linalg.norm(in_ref)
