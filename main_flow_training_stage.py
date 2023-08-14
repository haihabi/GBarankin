import torch.utils.data

import constants
import signal_model
import flows
import pyresearchutils as pru
import constants as C
import os
from tqdm import tqdm
import wandb
import numpy as np


def init_config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("base_epochs", type=int, default=360)
    _cr.add_parameter('base_dataset_size', type=int, default=200000)

    _cr.add_parameter("lr", type=float, default=1e-4)
    _cr.add_parameter("weight_decay", type=float, default=0.0)

    # _cr.add_parameter("random_padding", type=str, default="false")
    # _cr.add_parameter("padding_size", type=int, default=0)
    ###############################################
    # CNF Parameters
    ###############################################
    # _cr.add_parameter("n_blocks", type=int, default=1)
    # _cr.add_parameter("n_layer_inject", type=int, default=1)
    # _cr.add_parameter("n_hidden_inject", type=int, default=16)
    # _cr.add_parameter("inject_scale", type=str, default="false")
    # _cr.add_parameter("inject_bias", type=str, default="false")
    # _cr.add_parameter("affine_inject", type=str, default="false")
    ###############################################
    # Signal Model Parameter
    ###############################################
    # TODO:
    # 1. Add array type
    # 2.
    _cr.add_parameter("m_sensors", type=int, default=20)
    _cr.add_parameter("n_snapshots", type=int, default=5)
    _cr.add_parameter("k_targets", type=int, default=1)
    _cr.add_parameter("in_snr", type=float, default=0)
    _cr.add_parameter("wavelength", type=float, default=1)

    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)  # 200000
    _cr.add_parameter('val_dataset_size', type=int, default=20000)
    _cr.add_parameter('force_data_generation', type=str, default="false")

    return _cr


def train_model(in_run_parameters, in_run_log_folder, in_snr):
    print(f"Starting Training Stage At SNR:{in_snr}")
    sm = signal_model.DOASignalModel(in_run_parameters.m_sensors,
                                     in_run_parameters.n_snapshots,
                                     in_run_parameters.k_targets,
                                     in_snr,
                                     wavelength=in_run_parameters.wavelength)

    training_dataset = sm.generate_dataset(in_run_parameters.dataset_size)
    validation_dataset = sm.generate_dataset(in_run_parameters.val_dataset_size)

    training_data_loader = torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=in_run_parameters.batch_size,
                                                       shuffle=True)

    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                         batch_size=in_run_parameters.batch_size,
                                                         shuffle=False)
    is_sensor_location_known = True

    nominal_locations = torch.Tensor(sm.array._locations)
    if nominal_locations.shape[1] == 1:
        nominal_locations = torch.cat([nominal_locations, torch.zeros_like(nominal_locations)], dim=-1)

    flow = flows.DOAFlow(in_run_parameters.n_snapshots, in_run_parameters.m_sensors, in_run_parameters.k_targets,
                         in_run_parameters.wavelength,
                         nominal_sensors_locations=nominal_locations if is_sensor_location_known else None)
    flow.to(pru.get_working_device())

    opt = torch.optim.Adam(flow.parameters(), lr=in_run_parameters.lr, weight_decay=in_run_parameters.weight_decay)
    n_epochs = in_run_parameters.base_epochs  # TODO:Update computation
    ma = pru.MetricAveraging()
    target_signal_covariance_matrix = torch.diag(
        torch.diag(torch.ones(in_run_parameters.k_targets, in_run_parameters.k_targets))).to(
        pru.get_working_device()).float() + 0 * 1j
    target_noise_covariance_matrix = sm.power_noise * torch.diag(
        torch.diag(torch.ones(in_run_parameters.m_sensors, in_run_parameters.m_sensors))).to(
        pru.get_working_device()).float() + 0 * 1j

    for epoch in tqdm(range(n_epochs)):
        ma.clear()
        l_re = torch.linalg.norm(
            flow.flows[2].sensor_location - nominal_locations.to(pru.get_working_device())) / torch.linalg.norm(
            nominal_locations.to(pru.get_working_device()))
        scv_re = torch.linalg.norm(
            flow.flows[2].signal_covariance_matrix - target_signal_covariance_matrix) / torch.linalg.norm(
            target_signal_covariance_matrix)

        ncv_re = torch.linalg.norm(
            flow.flows[2].noise_covariance_matrix - target_noise_covariance_matrix) / torch.linalg.norm(
            target_noise_covariance_matrix)
        for x, theta in training_data_loader:
            x, theta = pru.torch.update_device(x, theta)
            opt.zero_grad()
            loss = flow.nll_mean(x, doas=theta)
            loss.backward()
            opt.step()
            ma.log(loss=loss.item())

        wandb.log({**ma.result,
                   "l_re": l_re.item(),
                   "scv_re": scv_re.item(),
                   'ncv_re': ncv_re.item(),
                   "diag_mean": np.real(flow.flows[2].noise_covariance_matrix.diag().mean().item())})

        torch.save(flow.state_dict(), os.path.join(wandb.run.dir, f"model_last_{in_snr}.pth"))
        # TODO: Add validation
        # TODO:Add save model


if __name__ == '__main__':
    cr = init_config()
    _run_parameters, _run_log_folder = pru.initialized_log(C.PROJECT, cr, enable_wandb=True)
    # for snr in constants.SNR_POINTS:
    train_model(_run_parameters, _run_log_folder, 10)
