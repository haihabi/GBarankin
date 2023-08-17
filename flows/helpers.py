import torch
from torch import nn


class LearnedHermitianPositiveDefiniteMatrix(nn.Module):
    def __init__(self, k, eps=1e-6, init_matrix=None, learning=True):
        super().__init__()
        if init_matrix is not None:
            L = torch.linalg.cholesky(torch.tensor(init_matrix))
            d_init = torch.abs(L.diag()) ** 2
            l_init = torch.tril(L / torch.sqrt(d_init.reshape([1, -1])), diagonal=-1)
        else:
            d_init = torch.complex(torch.abs(torch.randn(k)), torch.zeros(k))
            l_init = torch.randn(k, k) + 1j * torch.randn(k, k)
        self.eps = eps
        self.L = nn.Parameter(l_init, requires_grad=learning)
        self.D = nn.Parameter(d_init, requires_grad=learning)
        self.I = nn.Parameter(torch.eye(k), requires_grad=False)
        self.ind = nn.Parameter(torch.tril(torch.ones(k, k), diagonal=-1), requires_grad=False)

    def forward(self):
        L = self.L * self.ind + self.I
        D = torch.diag(torch.abs(self.D)) + self.eps * self.I + 1j * 0
        return L @ D @ L.T.conj()

#
# if __name__ == '__main__':
#     import numpy as np
#
#
#     def is_pos_def(x):
#         return np.all(np.linalg.eigvals(x) > 0)
#
#
#     k = 3
#     lhpd = LearnedHermitianPositiveDefiniteMatrix(k)
#     # diag = np.abs(np.random.randn(k))
#     # L = np.random.randn(k, k) + 1j * np.random.randn(k, k)
#     # L = L * np.tril(np.ones(L.shape), k=-1) + np.eye(L.shape[0])
#     #
#     # A = L @ (np.diag(np.abs(diag)) + 1e-6 * np.eye(L.shape[0])) @ L.T.conj()
#     A = lhpd().cpu().detach().numpy()
#     lhpd2 = LearnedHermitianPositiveDefiniteMatrix(k, init_matrix=A)
#     B = lhpd2().cpu().detach().numpy()
#     print(is_pos_def(A))
#     print(np.sum(np.isclose(A, A.T.conj())) / k ** 2)
#     print(np.sum(np.isclose(A, B)) / k ** 2)
#
#     # print("a")
#     #
#     # A = np.random.randn(k, k) + 1j * np.random.randn(k, k)
#     #
#     # B = ensure_hermitian(A)
#     # C = ensure_hermitian(B)
#     # D = ensure_psd(B)
#     # E = ensure_psd(D)
#     # print(is_pos_def(C))
#     # print(is_pos_def(D))
#     #
#     # print(np.sum(B == B.T.conj()) / k ** 2)
#     # print(np.sum(B == C) / k ** 2)
#     # print(np.sum(D == E) / k ** 2)
#     # print(np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C), np.linalg.norm(D))
