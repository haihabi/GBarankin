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

    input_dict = copy.copy(kwargs)

    it_fim = IterativeAverage()
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
        it_fim.update(s ** 2)
        with torch.no_grad():
            _bb_info_i = barankin_matrix(in_flow_model, gamma, test_points, parameter_name, **kwargs)
            it_bm.update(_bb_info_i.cpu().numpy())
            it_cross.update(s * torch.sqrt(_bb_info_i))
    ##########################
    # Filter TP
    ##########################
    gfim = it_fim()[0]
    zero_delta = torch.abs(delta_tp) < torch.sqrt(1 / gfim).item()
    bb_info = it_bm()[0]
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
