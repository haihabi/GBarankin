import torch
from torch.distributions import MultivariateNormal

import constants
import normflowpy as nfp
import pyresearchutils as pru

from flows.layers.doa_layer import DOALayer
from torch import nn


def add_flow_step(flows, n_flow_layers, input_vector_shape,
                  in_n_snapshots,
                  in_m_sensors,
                  n_layer_inject=2,
                  n_hidden_inject=16,
                  inject_scale=False,
                  inject_bias=False,
                  affine_coupling=False,
                  complex_affine_coupling=True,
                  affine_inject=False,
                  is_complex=True):
    for b in range(n_flow_layers):

        if is_complex:
            flows.append(
                nfp.flows.ActNorm(input_vector_shape))
            flows.append(nfp.flows.Vector2Tensor([in_n_snapshots, in_m_sensors, 2]))
            flows.append(
                nfp.flows.InvertibleFullyConnected(dim=2, random_initialization=False))
            flows.append(nfp.flows.AffineCoupling(x_shape=[in_n_snapshots, in_m_sensors, 2],
                                                  parity=0,
                                                  net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
                                                                                             non_linear_function=nn.SiLU,
                                                                                             bias=inject_bias),
                                                  nh=n_hidden_inject, scale=True))
            flows.append(nfp.flows.AffineCoupling(x_shape=[in_n_snapshots, in_m_sensors, 2],
                                                  parity=1,
                                                  net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
                                                                                             non_linear_function=nn.SiLU,
                                                                                             bias=inject_bias),
                                                  nh=n_hidden_inject, scale=True))
            flows.append(nfp.flows.Tensor2Vector([in_n_snapshots, in_m_sensors, 2]))

        else:
            raise NotImplemented
            # flows.append(
            #     nfp.flows.InvertibleFullyConnected(dim=input_vector_shape[0], random_initialization=True))
            # flows.append(nfp.flows.AffineCoupling(x_shape=input_vector_shape,
            #                                       parity=b % 2,
            #                                       net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
            #                                                                                  non_linear_function=nn.SiLU,
            #                                                                                  bias=inject_bias),
            #                                       nh=n_hidden_inject, scale=False))
            # flows.append(
            #     nfp.flows.ActNorm(input_vector_shape))
        # flows.append(
        #     nfp.flows.InvertibleFullyConnected(dim=input_vector_shape[0], random_initialization=True))
        # if affine_inject:
        #     flows.append(
        #         nfp.flows.AffineInjector(x_shape=input_vector_shape,
        #                                  cond_name_list=[constants.DOAS],
        #                                  condition_vector_size=in_d_p, n_hidden=n_hidden_inject,
        #                                  net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
        #                                                                             non_linear_function=nn.SiLU,
        #                                                                             bias=inject_bias),
        #                                  scale=inject_scale))
        # if affine_coupling:
        #     flows.append(nfp.flows.AffineCoupling(x_shape=input_vector_shape,
        #                                           parity=b % 2,
        #                                           net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_inject,
        #                                                                                      non_linear_function=nn.SiLU,
        #                                                                                      bias=inject_bias),
        #                                           nh=n_hidden_inject, scale=False))

        # if complex_affine_coupling:


class DOAFlow(nfp.NormalizingFlowModel):
    def __init__(self, n_snapshots, m_sensors, k_target, wavelength,
                 signal_covariance_matrix=None,
                 noise_covariance_matrix=None,
                 nominal_sensors_locations=None,
                 n_flow_layer=0):
        dim = 2 * n_snapshots * m_sensors
        base_distribution = MultivariateNormal(torch.zeros(dim, device=pru.get_working_device()),
                                               torch.eye(dim,
                                                         device=pru.get_working_device()))  # generate a class for base distribution

        self.flows = [nfp.flows.ToReal(),
                      nfp.flows.Tensor2Vector([n_snapshots, m_sensors, 2])]

        # add_flow_step(self.flows, n_flow_layer, [n_snapshots * m_sensors * 2], n_snapshots, m_sensors, is_complex=False)
        add_flow_step(self.flows, n_flow_layer, [n_snapshots * m_sensors * 2], n_snapshots, m_sensors, is_complex=True)
        self.flows.extend([
            nfp.flows.Vector2Tensor([n_snapshots, m_sensors, 2]),
            nfp.flows.ToComplex(),
            DOALayer(m_sensors,
                     k_target,
                     wavelength,
                     nominal_sensors_locations,
                     signal_covariance_matrix=signal_covariance_matrix,
                     noise_covariance_matrix=noise_covariance_matrix),
            nfp.flows.ToReal(),
            nfp.flows.Tensor2Vector([n_snapshots, m_sensors, 2]),

        ])
        add_flow_step(self.flows, n_flow_layer, [n_snapshots * m_sensors * 2], n_snapshots, m_sensors, is_complex=True)
        # add_flow_step(self.flows, n_flow_layer, [n_snapshots * m_sensors * 2], n_snapshots, m_sensors)

        super().__init__(base_distribution, self.flows, is_complex=True)

    def steering_matrix(self, _locations):
        return self.find_doa_layer().steering_matrix(_locations)

    def find_doa_layer(self):
        r = [flow for flow in self.flows if isinstance(flow, DOALayer)]
        if len(r) != 1:
            raise Exception("aaa")
        return r[0]


if __name__ == '__main__':
    m_targets = 4
    doa_flow = DOAFlow(30, 12, m_targets, 1).to(pru.get_working_device())
    locations = torch.rand([1, m_targets]).to(pru.get_working_device())
    x = doa_flow.sample(32, doas=locations)
    zs, prior_logprob, log_det = doa_flow(x, doas=locations)
    x_hat, _ = doa_flow.backward(zs[-1], doas=locations)
    print(torch.mean(torch.abs(x_hat[-1] - x) ** 2))
