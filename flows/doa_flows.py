import torch
from torch.distributions import MultivariateNormal
import normflowpy as nfp
import pyresearchutils as pru

from flows.layers.doa_layer import DOALayer


class DOAFlow(nfp.NormalizingFlowModel):
    def __init__(self, n_snapshots, m_sensors, k_target, wavelength,
                 signal_covariance_matrix=None,
                 noise_covariance_matrix=None,
                 nominal_sensors_locations=None):
        dim = 2 * n_snapshots * m_sensors
        base_distribution = MultivariateNormal(torch.zeros(dim, device=pru.get_working_device()),
                                               torch.eye(dim,
                                                         device=pru.get_working_device()))  # generate a class for base distribution
        self.flows = [DOALayer(m_sensors,
                               k_target,
                               wavelength,
                               nominal_sensors_locations,
                               signal_covariance_matrix=signal_covariance_matrix,
                               noise_covariance_matrix=noise_covariance_matrix),
                      nfp.flows.ToComplex(),
                      nfp.flows.Tensor2Vector([n_snapshots, m_sensors, 2]),

                      ]
        super().__init__(base_distribution, self.flows, is_complex=True)

    def steering_matrix(self, locations):
        return self.flows[0].steering_matrix(locations)


if __name__ == '__main__':
    m_targets = 4
    doa_flow = DOAFlow(30, 12, m_targets, 1).to(pru.get_working_device())
    locations = torch.rand([1, m_targets]).to(pru.get_working_device())
    x = doa_flow.sample(32, doas=locations)
    zs, prior_logprob, log_det = doa_flow(x, doas=locations)
    x_hat, _ = doa_flow.backward(zs[-1], doas=locations)
    print(torch.mean(torch.abs(x_hat[-1] - x) ** 2))
