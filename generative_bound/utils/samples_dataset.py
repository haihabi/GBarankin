import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class SamplesDataset(Dataset):
    def __init__(self):
        self.data = []
        self.n = 0

    def append_items(self, data: np.ndarray):
        for i in range(data.shape[0]):
            self.data.append(data[i, :])

    def finished(self):
        self.n = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n

    def get_min_max_vector(self):
        x = np.stack(self.data)
        return np.min(x, axis=0).astype("float32"), (np.max(x, axis=0) - np.min(x, axis=0)).astype("float32")


def sample_function(in_flow_model, in_batch_size=128, trimming_step=None,
                    temperature: float = 1.0, **kwargs):
    gamma = in_flow_model.sample(in_batch_size, temperature=temperature, **kwargs)
    gamma = gamma.detach()
    if trimming_step is not None:
        trimming_status = trimming_step(gamma)
    else:
        trimming_status = torch.ones(in_batch_size).bool().to(gamma.device)
    return gamma[trimming_status, :]  # Filter data


def generate_samples(in_flow_model, m, batch_size=128, trimming_step=None,
                     temperature: float = 1.0, **kwargs):
    sample_count = 0
    sd = SamplesDataset()
    with torch.no_grad():
        while sample_count < m:
            gamma = sample_function(in_flow_model, batch_size, trimming_step, temperature, **kwargs)
            sample_count += gamma.shape[0]
            sd.append_items(gamma.cpu().detach().numpy())
    sd.finished()
    return sd
