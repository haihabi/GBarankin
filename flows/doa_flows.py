import torch
from torch.distributions import MultivariateNormal
from flows.base_iid_flow import IIDNormalizingFlowModel
import normflowpy as nfp
import pyresearchutils as pru

from flows.layers.doa_layer import DOALayer


class DOAFlow(IIDNormalizingFlowModel):
    def __init__(self, n_snapshots, m_sensors, k_target, wavelength):
        dim = 2 * n_snapshots * m_sensors
        base_distribution = MultivariateNormal(torch.zeros(dim, device=pru.get_working_device()),
                                               torch.eye(dim,
                                                         device=pru.get_working_device()))  # generate a class for base distribution
        flows = [DOALayer(m_sensors, k_target, wavelength),
                 nfp.flows.ToComplex(),
                 nfp.flows.Tensor2Vector([n_snapshots, m_sensors, 2]),

                 ]
        super().__init__(base_distribution, flows, is_complex=True)


if __name__ == '__main__':
    doa_flow = DOAFlow(30, 12, 1, 1).to(pru.get_working_device())
    locations = torch.rand([32, 1]).to(pru.get_working_device())
    x = doa_flow.sample(32, locations=locations)
    zs, prior_logprob, log_det = doa_flow(x, locations=locations)
    x_hat, _ = doa_flow.backward(zs[-1], locations=locations)
    print(torch.mean(torch.abs(x_hat[-1] - x) ** 2))
