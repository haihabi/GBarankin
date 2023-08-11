import torch
import torch.autograd as autograd


def jacobian_single(out_gen, z, create_graph=False):
    grad_list = []
    for i in range(out_gen.shape[1]):
        gradients = autograd.grad(outputs=out_gen[:, i], inputs=z,
                                  grad_outputs=torch.ones(out_gen[:, i].size(), requires_grad=True).to(
                                      z.device),
                                  create_graph=create_graph, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        if gradients is None:  # In case there is not gradients
            gradients = torch.zeros(z.shape, requires_grad=True).to(z.device)
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)
