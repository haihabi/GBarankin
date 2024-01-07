import numpy as np
import generative_bound
import pyresearchutils as pru
import torch


def compute_noise_scale(snr, source_power):
    return np.sqrt(source_power / (10 ** (snr / 10))).astype("float32")


def get_timming_function(is_apply_trimming, in_sm, snr):
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


def rmse_db(x):
    return np.sqrt(x) * 180 / np.pi


def relative_error(in_est, in_ref):
    return np.linalg.norm(in_est - in_ref) / np.linalg.norm(in_ref)
