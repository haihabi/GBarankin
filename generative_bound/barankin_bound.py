import math

import torch
import copy
from torch.utils.data import DataLoader

import constants
from generative_bound import utils
import pyresearchutils as pru
from tqdm import tqdm
import scipy
import numpy as np


def barankin_matrix(in_flow_model, gamma, theta_test, parameter_name: str, search=False, **in_kwargs):
    in_param_dict = copy.copy(in_kwargs)
    nll_base = in_flow_model.nll(gamma, **in_param_dict)
    nll_test_point_list = []
    k_test_points = theta_test.shape[0]
    for i in range(k_test_points):
        in_param_dict[parameter_name] = torch.reshape(theta_test[i, :], [1, -1])
        nll_test_point = in_flow_model.nll(gamma, **in_param_dict)
        nll_test_point_list.append(nll_test_point)
    nll_test_point_matrix = torch.stack(nll_test_point_list).transpose(-1, -2).unsqueeze(-1).double()
    nll_base = nll_base.unsqueeze(-1).unsqueeze(-1).double()
    if search:
        delta_nll = 2 * (nll_base - nll_test_point_matrix)
    else:
        _nll_test_point_matrix = nll_test_point_matrix + torch.permute(nll_test_point_matrix, [0, 2, 1])
        delta_nll = 2 * nll_base - _nll_test_point_matrix
    a = torch.exp(delta_nll)

    return a


def search_test_points(in_flow_model, in_samples_data, batch_size, search_size=2048, max_test_points=30,
                       n_test_points_search=1600,
                       eps=1e-2,
                       **in_kwargs):
    maximal_tp = 10
    in_param_dict = copy.copy(in_kwargs)
    range_min = -math.pi / 2
    range_max = math.pi / 2
    base_array = torch.linspace(range_min + eps, range_max - eps, n_test_points_search).to(pru.get_working_device())
    delta = torch.diff(base_array).min()
    subset = torch.utils.data.Subset(in_samples_data, [i for i in range(search_size)])
    dl = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    # data_list = []
    bm_matrix = 0
    n = 0
    for gamma in tqdm(dl):
        gamma = gamma.to(pru.get_working_device())
        nll_base = in_flow_model.nll(gamma, **in_param_dict)
        nll_search = [in_flow_model.nll(gamma, **{"doas": tp.reshape([-1, 1]),
                                                  **{k: v for k, v in in_param_dict.items() if k != "doas"}}) for tp in
                      base_array]
        nll_search = torch.stack(nll_search, axis=0).T
        delta_nll = 2 * nll_base.unsqueeze(dim=-1) - 2 * nll_search
        bounded = torch.exp(delta_nll)
        _bm_matrix = torch.sum(bounded, dim=0)

        alpha = n / (n + gamma.shape[0])
        beta = 1 / (n + gamma.shape[0])

        bm_matrix = bm_matrix * alpha + beta * _bm_matrix
        n += gamma.shape[0]
        # data_list.append(delta_nll)

    index = bm_matrix > 1
    m = torch.sum(bm_matrix > 1)
    if m > 0:
        bm_matrix = bm_matrix[index]
        test_points_search = base_array[index]
        # delta_tp = ((test_points_search - in_kwargs['doas']).flatten())**2
        search_landscape = 1 / (bm_matrix - 1)
        search_landscape = search_landscape.cpu().numpy()

        peaks = scipy.signal.find_peaks(search_landscape)[0]
        if len(peaks) > 1:
            sel = test_points_search.reshape([1, -1])[:, peaks].reshape([-1, 1])
            pos_delta = test_points_search.reshape([1, -1])[:, peaks + 1].reshape([-1, 1])
            neg_delta = test_points_search.reshape([1, -1])[:, peaks - 1].reshape([-1, 1])
            index_to_keep = torch.logical_and(torch.isclose(torch.abs(pos_delta - sel), delta, rtol=1, atol=delta),
                                              torch.isclose(torch.abs(neg_delta - sel), delta, rtol=1,
                                                            atol=delta)).cpu().numpy().flatten()
            peaks = peaks[index_to_keep]
        peaks = peaks[np.flip(np.argsort(search_landscape[peaks]))][:maximal_tp]
        if len(peaks) > 0:
            return test_points_search.reshape([1, -1])[:, peaks].reshape([-1, 1]), search_landscape, test_points_search
        else:
            from matplotlib import pyplot as plt
            plt.plot(search_landscape)
            plt.show()
            raise NotImplemented
    else:
        raise NotImplemented


def _estimate_co_variance(in_samples_data, batch_size=512):
    train_dataloader = DataLoader(in_samples_data, batch_size=batch_size, shuffle=False)
    mu_vector = 0
    r_matrix = 0
    count = 0
    for gamma in tqdm(train_dataloader):
        gamma_v = gamma.reshape([-1, gamma.shape[-1], 1]).to(pru.get_working_device())
        _R = (gamma_v @ torch.permute(gamma_v, [0, 2, 1]).conj()).mean(dim=0)
        _mu = gamma_v.mean(dim=0)

        alpha = gamma.shape[0] / (gamma.shape[0] + count)
        beta = count / (gamma.shape[0] + count)

        mu_vector = alpha * _mu + beta * mu_vector
        r_matrix = alpha * _R + beta * r_matrix
        count += gamma.shape[0]
    R = r_matrix - mu_vector @ mu_vector.T.conj()
    return R


def compute_bp_tp(in_r, in_flow, in_n_bp_points, max_per_dim, in_eps=1e-6, **kwargs):
    doas = kwargs[constants.DOAS]
    n_sources = doas.shape[-1]
    base_array = torch.linspace(-np.pi / 2 + in_eps, np.pi / 2 - in_eps, in_n_bp_points, device=in_r.device)
    doa_layer = in_flow.find_doa_layer()
    tp_list = []
    cost_list = []
    for i in range(n_sources):
        _doas = doas.clone().detach()
        _results_list = []
        for _theta in base_array:
            _doas[i] = _theta
            _A = doa_layer.steering_matrix(_doas)
            _A = _A[0, :, :]
            _results_list.append(torch.trace(torch.real(_A.T.conj() @ in_r @ _A)).item())
        bp = np.asarray(_results_list).flatten()
        normalized_bp = bp / np.max(bp)
        peaks = scipy.signal.find_peaks(normalized_bp)[0]
        peaks = peaks[np.flip(np.argsort(normalized_bp[peaks]))[:max_per_dim]]
        for p in peaks:
            _doas = doas.clone().detach()
            _doas[i] = base_array[p]
            tp_list.append(_doas.flatten())
        cost_list.append(normalized_bp)

    return torch.stack(tp_list, dim=0), cost_list, [base_array]


def search_test_points_bp(in_flow_model, in_samples_data, max_test_points=3, n_test_points_search=3200, eps=1e-2,
                          **in_kwargs):
    r = _estimate_co_variance(in_samples_data)
    return compute_bp_tp(r, in_flow_model, n_test_points_search, max_test_points, eps, **in_kwargs)


def _generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                               temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                               **kwargs):
    search_landscape = None
    test_points_search = None
    with torch.no_grad():
        sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
        train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=False)

        if test_points is None:
            print("Start TP Search")
            test_points, search_landscape, test_points_search = search_test_points_bp(in_flow_model, sample_data,
                                                                                      **kwargs)
            print(f"End TP Search with {test_points.shape} Test Points")

        count = 0
        bb_info = torch.zeros([test_points.shape[0], test_points.shape[0]], device=test_points.device).double()
        for gamma in tqdm(train_dataloader):
            gamma = gamma.to(pru.get_working_device())
            _bb_info_i = barankin_matrix(in_flow_model, gamma, test_points, parameter_name, **kwargs)

            bb_info_i = _bb_info_i.sum(dim=0)
            future = bb_info_i / (count + _bb_info_i.shape[0])
            past = bb_info * (count / (count + _bb_info_i.shape[0]))
            bb_info = future + past  # Average
            count += _bb_info_i.shape[0]
        ##########################
        # Filter TP
        ##########################
        # print(conv)
        above_one = bb_info.diag() > 1
        NFTP = torch.sum(above_one)
        bb_info = bb_info[above_one.reshape([-1, 1]) * above_one.reshape([1, -1])].reshape(
            [NFTP, NFTP])

        bb_info_inv = torch.linalg.inv(bb_info - torch.ones_like(bb_info))
        test_points = test_points[above_one, :]
        tau_vector = (test_points - kwargs[parameter_name]).double()
    return bb_info_inv, tau_vector, bb_info, search_landscape, test_points_search, test_points


def generative_barankin_bound(in_flow_model, m, test_points=None, batch_size=512, trimming_step=None,
                              temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                              **kwargs):
    bb_info_inv, tau_vector, bb_info, search_landscape, test_points_search, test_points_selected = _generative_barankin_bound(
        in_flow_model,
        m, test_points,
        batch_size,
        trimming_step,
        temperature,
        eps,
        parameter_name,
        **kwargs)
    bound = torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
    if torch.any(bound.diagonal() < 0).item():
        print("Negative Bound!!!")
        if tau_vector.shape[0] > 1:
            status = (bb_info_inv.diag() > 0)
            if torch.sum(status) > 0:
                bb_info_filter = bb_info[status.reshape([-1, 1]) * status.reshape([1, -1])].reshape(
                    [status.sum(), status.sum()])
                tau_vector_filter = tau_vector[status, :]
            else:
                index_chosen = bb_info.diag().argmin()
                bb_info_filter = bb_info[index_chosen, index_chosen].reshape([1, 1])
                tau_vector_filter = tau_vector[index_chosen, :].reshape([1, -1])
            print(f"New Number of TP:{tau_vector_filter.shape[0]}")
            bb_info_inv_filter = torch.linalg.inv(bb_info_filter)
            bound = torch.matmul(tau_vector_filter.transpose(-1, -2),
                                 torch.matmul(bb_info_inv_filter, tau_vector_filter))

    return bound, bb_info, search_landscape, test_points_search, test_points_selected
