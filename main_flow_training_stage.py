import math

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
import names


def init_config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("base_epochs", type=int, default=160)
    _cr.add_parameter('base_dataset_size', type=int, default=200000)

    _cr.add_parameter("lr", type=float, default=5e-4)
    _cr.add_parameter("weight_decay", type=float, default=0.0)
    _cr.add_parameter("group_name", type=str, default=None)

    # _cr.add_parameter("random_padding", type=str, default="false")
    # _cr.add_parameter("padding_size", type=int, default=0)
    ###############################################
    # CNF Parameters
    ###############################################
    _cr.add_parameter("n_flow_layer", type=int, default=2)
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
    _cr.add_parameter("is_sensor_location_known", type=bool, default=True)
    _cr.add_parameter("signal_type", type=str, default="QAM4", enum=signal_model.SignalType)
    _cr.add_parameter("snr", type=float, default=None)
    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)  # 200000
    _cr.add_parameter('val_dataset_size', type=int, default=20000)
    _cr.add_parameter('force_data_generation', type=str, default="false")

    return _cr


def build_flow_model(in_run_parameters, in_sm):
    nominal_locations = torch.Tensor(in_sm.array._locations)
    if nominal_locations.shape[1] == 1:
        nominal_locations = torch.cat([nominal_locations, torch.zeros_like(nominal_locations)], dim=-1)

    flow = flows.DOAFlow(in_run_parameters.n_snapshots, in_run_parameters.m_sensors, in_run_parameters.k_targets,
                         in_run_parameters.wavelength,
                         nominal_sensors_locations=nominal_locations if in_run_parameters.is_sensor_location_known else None,
                         n_flow_layer=in_run_parameters.n_flow_layer)
    flow.to(pru.get_working_device())
    return flow


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
    flow = build_flow_model(_run_parameters, sm)
    opt = torch.optim.Adam(flow.parameters(), lr=in_run_parameters.lr, weight_decay=in_run_parameters.weight_decay)
    step_in_epoch = len(training_data_loader)
    wram_up_epoch = 1
    max_lr = 5e-4
    min_lr = 1e-5

    def lr_function(in_step):
        _epoch = in_step % step_in_epoch
        if _epoch < wram_up_epoch:
            return (in_step + 1) / (step_in_epoch * wram_up_epoch)
        else:
            norm_step = in_step - step_in_epoch * wram_up_epoch
            step_left = step_in_epoch * (n_epochs - wram_up_epoch)
            return (1 - min_lr / max_lr) * math.cos(math.pi * norm_step / (2 * step_left)) + min_lr / max_lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_function)
    n_epochs = in_run_parameters.base_epochs  # TODO:Update computation
    if in_snr < -10:
        n_epochs = int(3.5 * n_epochs)
    ma = pru.MetricAveraging()
    target_signal_covariance_matrix = torch.diag(
        torch.diag(torch.ones(in_run_parameters.k_targets, in_run_parameters.k_targets))).to(
        pru.get_working_device()).float() + 0 * 1j
    target_noise_covariance_matrix = sm.power_noise * torch.diag(
        torch.diag(torch.ones(in_run_parameters.m_sensors, in_run_parameters.m_sensors))).to(
        pru.get_working_device()).float() + 0 * 1j
    # flow_ema = pru.torch.ema.ModelEma(flow)
    for epoch in tqdm(range(n_epochs)):
        ma.clear()

        scv_re = torch.linalg.norm(
            flow.find_doa_layer().signal_covariance_matrix - target_signal_covariance_matrix) / torch.linalg.norm(
            target_signal_covariance_matrix)

        ncv_re = torch.linalg.norm(
            flow.find_doa_layer()._noise_covariance_matrix - target_noise_covariance_matrix) / torch.linalg.norm(
            target_noise_covariance_matrix)
        for x, theta in training_data_loader:
            x, theta = pru.torch.update_device(x, theta)
            opt.zero_grad()
            loss = flow.nll_mean(x, doas=theta)
            loss.backward()
            opt.step()
            sch.step()
            # flow_ema.update(flow)
            ma.log(loss=loss.item())
        with torch.no_grad():
            for x, theta in validation_data_loader:
                x, theta = pru.torch.update_device(x, theta)
                val_loss = flow.nll_mean(x, doas=theta)
                ma.log(val_loss=val_loss.item())

        wandb.log({**ma.result,
                   "scv_re": scv_re.item(),
                   'ncv_re': ncv_re.item()})

        torch.save(flow.state_dict(), os.path.join(wandb.run.dir, f"model_last_{in_snr}.pth"))


if __name__ == '__main__':
    cr = init_config()
    _run_parameters, _run_log_folder = pru.initialized_log(C.PROJECT, cr, enable_wandb=False)
    group_name = names.get_full_name().lower().replace(" ",
                                                       "_") if _run_parameters.group_name is None else _run_parameters.group_name

    snr_list = constants.SNR_POINTS if _run_parameters.snr is None else [_run_parameters.snr]
    for snr in constants.SNR_POINTS:
        try:
            wandb.init(project=C.PROJECT,
                       dir=_run_parameters.base_log_folder,
                       group=group_name,
                       name=group_name + f"_{snr}")  # Set WandB Folder to log folder
            wandb.config.update(cr.get_user_arguments())  # adds all of the arguments as config variables®
            train_model(_run_parameters, _run_log_folder, snr)
            wandb.finish()
        except:
            wandb.finish()
