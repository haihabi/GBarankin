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
from generative_bound.cramer_rao_lower_bound import score_vector_function


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


def search_test_points(in_flow_model, in_samples_data, batch_size, search_size=2048, maximal_tp=30,
                       n_test_points_search=1600,
                       eps=1e-2,
                       **in_kwargs):
    # maximal_tp = 10
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
        delta_tp = ((test_points_search - in_kwargs['doas']).flatten()) ** 2
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
        # if len(peaks) > 0:
        # peaks = np.argmax(search_landscape)
        return test_points_search.reshape([1, -1])[:, peaks].reshape([-1, 1]), search_landscape, test_points_search
        # else:
        #     from matplotlib import pyplot as plt
        #     plt.plot(search_landscape)
        #     plt.show()
        #     raise NotImplemented
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


class IterativeAverage:
    def __init__(self):
        self.mean = 0
        self.power_2 = 0
        self.count = 0

    def iir(self, n, current, past):
        alpha = n / (self.count + n)
        beta = self.count / (self.count + n)
        return alpha * current + beta * past  # Average

    def update(self, in_tensor):
        bb_info_i = in_tensor.mean(dim=0)
        # alpha = in_tensor.shape[0] / (self.count + in_tensor.shape[0])
        # beta = self.count / (self.count + in_tensor.shape[0])
        self.mean = self.iir(in_tensor.shape[0], bb_info_i, self.mean)
        self.power_2 = self.iir(in_tensor.shape[0], (in_tensor ** 2).mean(dim=0), self.power_2)
        self.count += in_tensor.shape[0]

    def __call__(self):
        return self.mean, self.power_2


def _generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                               temperature: float = 1.0, eps=1e-12, parameter_name=constants.DOAS,
                               **kwargs):
    search_landscape = None
    test_points_search = None
    with torch.no_grad():
        sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
        train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=False)
        if test_points is None:
            print("Start TP Search")
            if False:
                test_points, search_landscape, test_points_search = search_test_points_bp(in_flow_model, sample_data,
                                                                                          **kwargs)
            else:
                test_points, search_landscape, test_points_search = search_test_points(in_flow_model, sample_data,
                                                                                       batch_size, search_size=512,
                                                                                       maximal_tp=1,
                                                                                       n_test_points_search=1600,
                                                                                       eps=1e-2,
                                                                                       **kwargs)
            print(f"End TP Search with {test_points.shape} Test Points")
        # count = 0
        # bb_info = torch.zeros([test_points.shape[0], test_points.shape[0]], device=test_points.device).double()

    # base_theta_tensor = kwargs[parameter_name]
    # theta_tensor = test_points * torch.ones([batch_size, test_points.shape[0]], requires_grad=True,
    #                                         device=test_points.device)
    input_dict = copy.copy(kwargs)
    # input_dict[parameter_name] = theta_tensor

    # it_fim = IterativeAverage()
    it_bm = IterativeAverage()
    it_cross = IterativeAverage()
    delta_tp = test_points - kwargs[parameter_name]

    theta_grad = kwargs[parameter_name] * torch.ones([batch_size, kwargs[parameter_name].shape[0]], requires_grad=True,
                                                     device=kwargs[parameter_name].device)

    for gamma in tqdm(train_dataloader):
        gamma = gamma.to(pru.get_working_device())
        input_dict[parameter_name] = theta_grad
        nll_tensor = in_flow_model.nll(gamma, **input_dict).reshape([-1, 1])
        s = utils.jacobian_single(nll_tensor, theta_grad)
        it_cross.update(s ** 2)
        with torch.no_grad():
            _bb_info_i = barankin_matrix(in_flow_model, gamma, test_points, parameter_name, **kwargs)
            it_bm.update(_bb_info_i)
    ##########################
    # Filter TP
    ##########################
    gfim = it_cross()[0]
    zero_delta = torch.abs(delta_tp) < torch.sqrt(1 / gfim).item()
    bb_info, _ = it_bm()
    if zero_delta:
        bb_info_inv = 1 / torch.abs((gfim * (delta_tp ** 2)).double())
    elif torch.all(bb_info - torch.ones_like(bb_info) < 1) and not zero_delta:
        bb_info_inv = torch.eye(1, device=bb_info.device).double()
    else:
        bb_info_inv = torch.linalg.inv(bb_info - torch.ones_like(bb_info))

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
    return bound, bb_info, search_landscape, test_points_search, test_points_selected
