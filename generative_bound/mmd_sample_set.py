import torch


def mmd_median_sigma(x, y):
    dists = torch.pdist(x)
    sigma = dists.median() / 2
    return mmd(x, y, sigma)


def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * sigma ** 2)) * dists ** 2) + torch.eye(n + m, device=x.device) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd.item()


class FlowMMD:
    def __init__(self):
        self.sample_list_x = []
        self.sample_list_y = []

    def clear(self):
        self.sample_list_x.clear()
        self.sample_list_y.clear()

    def compute_mmd(self):
        x_array = torch.cat(self.sample_list_x, dim=0)
        y_array = torch.cat(self.sample_list_y, dim=0)
        # print(x_array.shape, y_array.shape)
        res = mmd_median_sigma(x_array, y_array)
        return res

    def add_samples(self, x, y):
        y_real = torch.real(y)
        y_img = torch.imag(y)
        x_real = torch.real(x)
        x_img = torch.imag(x)
        x_array = torch.stack([x_real, x_img], dim=-1).reshape([x.shape[0], -1])
        y_array = torch.stack([y_real, y_img], dim=-1).reshape([y.shape[0], -1])
        self.sample_list_x.append(x_array)
        self.sample_list_y.append(y_array)
