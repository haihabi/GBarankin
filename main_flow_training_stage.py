import math

import torch.utils.data

import constants
import generative_bound
import signal_model
import flows
import pyresearchutils as pru
import constants as C
import os
from tqdm import tqdm
import wandb
import normflowpy as nfp
import names
from CAdam import CAdam


def init_config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("base_epochs", type=int, default=250)
    _cr.add_parameter('base_dataset_size', type=int, default=200000)

    _cr.add_parameter("lr", type=float, default=1e-4)
    _cr.add_parameter("min_lr", type=float, default=5e-5)
    _cr.add_parameter("warmup_epoch", type=int, default=2)
    _cr.add_parameter("weight_decay", type=float, default=0.0)
    _cr.add_parameter("group_name", type=str, default=None)

    # _cr.add_parameter("random_padding", type=str, default="false")
    # _cr.add_parameter("padding_size", type=int, default=0)
    ###############################################
    # CNF Parameters
    ###############################################
    _cr.add_parameter("n_flow_layer", type=int, default=0)
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
    _cr.add_parameter("signal_type", type=str, default="ComplexGaussian", enum=signal_model.SignalType)
    _cr.add_parameter("noise_type", type=str, default="Correlated", enum=signal_model.NoiseMatrix)
    _cr.add_parameter("array_perturbed_scale", type=float, default=0.0)
    _cr.add_parameter("snr", type=float, default=None)
    _cr.add_parameter("snr_min", type=float, default=-30)
    _cr.add_parameter("snr_max", type=float, default=10)
    _cr.add_parameter("is_multiple_snrs", type=bool, default=True)
    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)  # 200000
    _cr.add_parameter('val_dataset_size', type=int, default=2000)
    _cr.add_parameter('force_data_generation', type=str, default="false")

    return _cr


def build_flow_model(in_run_parameters, in_sm):
    nominal_locations = torch.Tensor(in_sm.array._locations)
    if nominal_locations.shape[1] == 1:
        nominal_locations = torch.cat([nominal_locations, torch.zeros_like(nominal_locations)], dim=-1)

    flow = flows.DOAFlow(in_run_parameters.n_snapshots, in_run_parameters.m_sensors, in_run_parameters.k_targets,
                         in_run_parameters.wavelength,
                         nominal_sensors_locations=nominal_locations if in_run_parameters.is_sensor_location_known else None,
                         n_flow_layer=in_run_parameters.n_flow_layer,
                         is_multiple_snrs=in_run_parameters.is_multiple_snrs)
    flow.to(pru.get_working_device())
    flow_ema = nfp.FlowEMA(flow)
    return flow, flow_ema


def build_signal_model(in_run_parameters, in_snr):
    return signal_model.DOASignalModel(in_run_parameters.m_sensors,
                                       in_run_parameters.n_snapshots,
                                       in_run_parameters.k_targets,
                                       in_snr,
                                       wavelength=in_run_parameters.wavelength,
                                       signal_type=in_run_parameters.signal_type,
                                       noise_type=in_run_parameters.noise_type,
                                       array_perturbed_scale=in_run_parameters.array_perturbed_scale,
                                       is_multiple_snr=in_run_parameters.is_multiple_snrs,
                                       snr_min=in_run_parameters.snr_min,
                                       snr_max=in_run_parameters.snr_max,

                                       )


def train_model(in_run_parameters, in_run_log_folder, in_snr):
    if in_run_parameters.is_multiple_snrs:
        print(f"Starting Training Multiple SNRs")
    else:
        print(f"Starting Training Stage At SNR:{in_snr}")

    sm = build_signal_model(in_run_parameters, in_snr)
    training_dataset = sm.generate_dataset(in_run_parameters.dataset_size)
    validation_dataset = sm.generate_dataset(in_run_parameters.val_dataset_size)

    training_data_loader = torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=in_run_parameters.batch_size,
                                                       shuffle=True)

    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                         batch_size=in_run_parameters.batch_size,
                                                         shuffle=False)
    flow, flow_ema = build_flow_model(_run_parameters, sm)
    opt = CAdam(flow.parameters(), lr=in_run_parameters.lr, weight_decay=in_run_parameters.weight_decay)
    step_in_epoch = len(training_data_loader)

    def lr_function(in_step):
        _epoch = in_step % step_in_epoch
        if _epoch < in_run_parameters.warmup_epoch:
            return (in_step + 1) / (step_in_epoch * in_run_parameters.warmup_epoch)
        else:
            norm_step = in_step - step_in_epoch * in_run_parameters.warmup_epoch
            step_left = step_in_epoch * (n_epochs - in_run_parameters.warmup_epoch)
            return (1 - in_run_parameters.min_lr / in_run_parameters.lr) * math.cos(
                math.pi * norm_step / (2 * step_left)) + in_run_parameters.min_lr / in_run_parameters.lr

    n_epochs = in_run_parameters.base_epochs  # TODO:Update computation
    if not _run_parameters.is_multiple_snrs:
        if in_snr < 1:
            n_epochs = int(2.5 * n_epochs)
    else:
        n_epochs = int(2.5 * n_epochs)
    ma = pru.MetricAveraging()
    target_signal_covariance_matrix = torch.diag(
        torch.diag(torch.ones(in_run_parameters.k_targets, in_run_parameters.k_targets))).to(
        pru.get_working_device()).float() + 0 * 1j

    target_noise_covariance_matrix = torch.tensor(sm.noise_matrix).to(pru.get_working_device())
    mmd_metric = generative_bound.FlowMMD()
    sm.save_model(wandb.run.dir)
    for epoch in tqdm(range(n_epochs)):
        ma.clear()

        scv_re = torch.linalg.norm(
            flow.find_doa_layer().signal_covariance_matrix - target_signal_covariance_matrix) / torch.linalg.norm(
            target_signal_covariance_matrix)

        ncv_re = torch.linalg.norm(
            flow.find_doa_layer().noise_covariance_matrix - target_noise_covariance_matrix) / torch.linalg.norm(
            target_noise_covariance_matrix)
        flow.train()
        for x, (theta, ns) in training_data_loader:
            x, theta, ns = pru.torch.update_device(x, theta, ns)
            opt.zero_grad()
            loss = flow.nll_mean(x, doas=theta, noise_scale=ns)
            loss.backward()
            opt.step()

            ma.log(loss=loss.item())
        flow.eval()
        with torch.no_grad():
            for x, (theta, ns) in validation_data_loader:
                x, theta, ns = pru.torch.update_device(x, theta, ns)
                y = flow.sample(x.shape[0], doas=theta, noise_scale=ns).detach()

                mmd_metric.add_samples(x, y)
                val_loss = flow.nll_mean(x, doas=theta, noise_scale=ns)
                ma.log(val_loss=val_loss.item())
        mmd = mmd_metric.compute_mmd()
        mmd_metric.clear()
        wandb.log({**ma.result,
                   "MMD": mmd,
                   "scv_re": scv_re.item(),
                   'ncv_re': ncv_re.item()})

        torch.save(flow.state_dict(), os.path.join(wandb.run.dir, f"model_last_{in_snr}.pth"))


if __name__ == '__main__':
    random_name = names.get_full_name().lower().replace(" ", "_")
    cr = init_config()
    _run_parameters, _run_log_folder = pru.initialized_log(C.PROJECT, cr, enable_wandb=False)
    group_name = random_name if _run_parameters.group_name is None else _run_parameters.group_name
    if _run_parameters.is_multiple_snrs:
        wandb.init(project=C.PROJECT,
                   dir=_run_parameters.base_log_folder,
                   group=group_name,
                   name=group_name + f"_{_run_parameters.snr_min}_{_run_parameters.snr_max}")  # Set WandB Folder to log folder
        wandb.config.update(cr.get_user_arguments())  # adds all of the arguments as config variables®
        train_model(_run_parameters, _run_log_folder, None)
        wandb.finish()

    else:
        snr_list = constants.SNR_POINTS if _run_parameters.snr is None else [_run_parameters.snr]
        for snr in [-4, 10, 21]:
            wandb.init(project=C.PROJECT,
                       dir=_run_parameters.base_log_folder,
                       group=group_name,
                       name=group_name + f"_{snr}")  # Set WandB Folder to log folder
            wandb.config.update(cr.get_user_arguments())  # adds all of the arguments as config variables®
            train_model(_run_parameters, _run_log_folder, snr)
            wandb.finish()
