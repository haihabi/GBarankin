import torch
import copy
from torch.utils.data import DataLoader

from generative_bound import utils

from torch import nn


def barankin_score_vector(in_flow_model, gamma, theta_test, parameter_name: str, **in_kwargs):
    in_param_dict = copy.copy(in_kwargs)
    nll_base = in_flow_model.nll(gamma, **in_param_dict)
    nll_test_point_list = []
    k_test_points = theta_test.shape[0]
    for i in range(k_test_points):
        in_param_dict[parameter_name] = torch.reshape(theta_test[i, :], [-1, 1])
        nll_test_point = in_flow_model.nll(gamma, **in_param_dict)
        nll_test_point_list.append(nll_test_point)
    nll_test_point_matrix = torch.stack(nll_test_point_list).transpose(-1, -2).unsqueeze(-1)
    nll_base = nll_base.unsqueeze(-1).unsqueeze(-1)
    delta_nll = nll_base - nll_test_point_matrix

    return torch.minimum(torch.exp(delta_nll),
                         torch.ones(1, device=gamma.device))
    # return torch.exp(delta_nll) - 1


def barankin_transform_function(theta_base, theta_test, bb_info_inv):
    tau_vector = theta_test - theta_base
    return torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))


def generative_barankin_bound(in_flow_model, m, test_points, batch_size=128, trimming_step=None,
                              temperature: float = 1.0, eps=1e-12,
                              **kwargs):
    sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
    train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=True)
    eta_list = []
    for gamma in train_dataloader:
        gamma = gamma.float()
        eta_i = barankin_score_vector(in_flow_model, gamma, test_points, "theta", **kwargs)
        eta_list.append(eta_i)
    eta = torch.cat(eta_list, dim=0)
    bb_info = torch.matmul(eta, eta.transpose(-1, -2)).mean(dim=0)
    bb_info_inv = torch.linalg.inv(bb_info + torch.diag(torch.ones(bb_info.shape[0]) * eps))
    tau_vector = test_points - kwargs["theta"]
    return torch.matmul(tau_vector.transpose(-1, -2), torch.matmul(bb_info_inv, tau_vector))
