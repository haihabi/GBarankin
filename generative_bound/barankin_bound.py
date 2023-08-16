import torch
import copy
from torch.utils.data import DataLoader

import constants
from generative_bound import utils
import pyresearchutils as pru
from tqdm import tqdm


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


def search_test_points(in_flow_model, gamma, **in_kwargs):
    in_param_dict = copy.copy(in_kwargs)
    nll_base = in_flow_model.nll(gamma, **in_param_dict)


def _generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                               temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                               **kwargs):
    with torch.no_grad():
        sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
        train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=False)

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
    return bb_info_inv, tau_vector, bb_info


def generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                              temperature: float = 1.0, eps=1e-12, parameter_name=constants.THETA,
                              **kwargs):
    bb_info_inv, tau_vector, bb_info = _generative_barankin_bound(in_flow_model, m, test_points, batch_size,
                                                                  trimming_step,
                                                                  temperature, eps, parameter_name, **kwargs)
    bound = torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
    if torch.any(bound.diagonal() < 0).item():
        bb_info_inv, tau_vector, bb_info = _generative_barankin_bound(in_flow_model, m, test_points, batch_size,
                                                                      trimming_step,
                                                                      temperature, eps, parameter_name, **kwargs)
        bound = torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
    return bound, bb_info
