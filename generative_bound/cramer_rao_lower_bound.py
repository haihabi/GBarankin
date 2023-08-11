import torch
import copy
from generative_bound import utils
import constants
from torch.utils.data import DataLoader
from tqdm import tqdm


def score_vector_function(in_model, in_gamma, in_parameter_name, **kwargs):
    theta_tensor = kwargs[in_parameter_name]
    nll_tensor = in_model.nll(in_gamma, **kwargs).reshape([-1, 1])
    return utils.jacobian_single(nll_tensor, theta_tensor)


def generative_cramer_rao_bound(in_flow_model, m, batch_size=128, trimming_step=None,
                                temperature: float = 1.0, parameter_name=constants.THETA, **kwargs):
    """

    :param in_flow_model:
    :param m:
    :param batch_size:
    :param trimming_step:
    :param temperature:
    :param parameter_name:
    :param kwargs:
    :return:
    """

    utils.input_validation(parameter_name, **kwargs)
    input_dict = copy.deepcopy(kwargs)
    input_dict_reshape = {}
    for k, v in input_dict.items():
        if v.shape[0] == 1:
            input_dict_reshape[k] = v.expand([batch_size, *[v.shape[i + 1] for i in range(len(v.shape) - 1)]])
        else:
            input_dict_reshape[k] = v
    input_dict = input_dict_reshape
    base_theta_tensor = kwargs[parameter_name]
    sample_data = utils.generate_samples(in_flow_model, m, batch_size, trimming_step, temperature, **kwargs)
    train_dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=True)
    info_matrix_collector = utils.InformationMatrixCollector(base_theta_tensor[0, :].shape[0]).to(
        base_theta_tensor.device)
    theta_tensor = base_theta_tensor * torch.ones([batch_size, base_theta_tensor.shape[0]], requires_grad=True,
                                                  device=base_theta_tensor.device)
    input_dict[parameter_name] = theta_tensor
    for gamma in tqdm(train_dataloader):
        gamma = gamma.to(base_theta_tensor.device)
        s = score_vector_function(in_flow_model, gamma, parameter_name, **input_dict)
        info_matrix_collector.append_fim(torch.matmul(s.transpose(dim0=1, dim1=2), s))
    gfim = info_matrix_collector.calculate_final_fim()
    return torch.linalg.inv(gfim)
