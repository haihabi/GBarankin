import math

import torch
import copy
from torch.utils.data import DataLoader

import constants
from generative_bound import utils
import pyresearchutils as pru
from tqdm import tqdm
import scipy


def barankin_matrix(in_flow_model, gamma, theta_test, parameter_name: str, search=False, **in_kwargs):
    in_param_dict = copy.copy(in_kwargs)
    nll_base = in_flow_model.nll(gamma, **in_param_dict)
    nll_test_point_list = []
    k_test_points = theta_test.shape[0]
    for i in range(k_test_points):
        in_param_dict[parameter_name] = torch.reshape(theta_test[i, :], [-1, 1])
        nll_test_point = in_flow_model.nll(gamma, **in_param_dict)
        nll_test_point_list.append(nll_test_point)
    nll_test_point_matrix = torch.stack(nll_test_point_list).transpose(-1, -2).unsqueeze(-1)
    nll_base = nll_base.unsqueeze(-1).unsqueeze(-1).double()
    if search:
        delta_nll = 2 * (nll_base - nll_test_point_matrix)
    else:
        _nll_test_point_matrix = nll_test_point_matrix + torch.permute(nll_test_point_matrix, [0, 2, 1])
        delta_nll = 2 * nll_base - _nll_test_point_matrix
    return torch.exp(delta_nll)
    # nll_test_point_matrix = torch.stack(nll_test_point_list).transpose(-1, -2).unsqueeze(-1)
    # nll_base = nll_base.unsqueeze(-1).unsqueeze(-1).double()
    # return torch.exp(delta_nll) - 1


def search_test_points(in_flow_model, in_samples_data, batch_size, search_size=2048, max_test_points=30,
                       test_points_search=1600,
                       eps=1e-2,
                       **in_kwargs):
    print("Starting Test Point Search")
    in_param_dict = copy.copy(in_kwargs)
    range_min = -math.pi / 2
    range_max = math.pi / 2
    base_array = torch.linspace(range_min + eps, range_max - eps, test_points_search).to(pru.get_working_device())
    subset = torch.utils.data.Subset(in_samples_data, [i for i in range(search_size)])
    dl = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    data_list = []
    for gamma in tqdm(dl):
        gamma = gamma.to(pru.get_working_device())
        nll_base = in_flow_model.nll(gamma, **in_param_dict)
        nll_search = [in_flow_model.nll(gamma, **{"doas": tp.reshape([-1, 1])}) for tp in base_array]
        nll_search = torch.stack(nll_search, axis=0).T
        delta_nll = 2 * nll_base.unsqueeze(dim=-1) - 2 * nll_search
        data_list.append(delta_nll)
    # print("a")
    bm_matrix = torch.mean(torch.exp(torch.cat(data_list)), dim=0)
    search_landscape = 1 / bm_matrix
    search_landscape = search_landscape.cpu().numpy()

    peaks = scipy.signal.find_peaks(search_landscape)[0]

    # from matplotlib import pyplot as plt
    #
    # plt.plot(base_array.cpu().numpy(), search_landscape)
    # plt.plot(base_array.cpu().numpy()[peaks], search_landscape[peaks], "v")
    # plt.grid()
    # plt.show()

    return base_array.reshape([1, -1])[:, peaks].reshape([-1, 1]), search_landscape


def _generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                               temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                               **kwargs):
    search_landscape = None
    with torch.no_grad():
        sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
        train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=False)
        if test_points is None:
            test_points, search_landscape = search_test_points(in_flow_model, sample_data, batch_size, **kwargs)

        count = 0
        bb_info = torch.zeros([test_points.shape[0], test_points.shape[0]], device=test_points.device)
        for gamma in tqdm(train_dataloader):
            gamma = gamma.to(pru.get_working_device())
            _bb_info_i = barankin_matrix(in_flow_model, gamma, test_points, parameter_name, **kwargs)

            bb_info_i = _bb_info_i.sum(dim=0)
            future = bb_info_i / (count + _bb_info_i.shape[0])
            past = bb_info * (count / (count + _bb_info_i.shape[0]))
            bb_info = future + past  # Average
            count += _bb_info_i.shape[0]

        bb_info_inv = torch.linalg.inv(bb_info - torch.ones_like(bb_info))
        tau_vector = (test_points - kwargs[parameter_name]).double()
    return bb_info_inv, tau_vector, bb_info, search_landscape


def generative_barankin_bound(in_flow_model, m, test_points=None, batch_size=128, trimming_step=None,
                              temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                              **kwargs):
    bb_info_inv, tau_vector, bb_info, search_landscape = _generative_barankin_bound(in_flow_model, m, test_points,
                                                                                    batch_size,
                                                                                    trimming_step,
                                                                                    temperature, eps, parameter_name,
                                                                                    **kwargs)
    bound = torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
    if torch.any(bound.diagonal() < 0).item():
        bb_info_inv, tau_vector, bb_info, search_landscape = _generative_barankin_bound(in_flow_model, m, test_points,
                                                                                        batch_size,
                                                                                        trimming_step,
                                                                                        temperature, eps,
                                                                                        parameter_name, **kwargs)
        bound = torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
    return bound, bb_info, search_landscape
