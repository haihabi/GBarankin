import torch.utils.data

import signal_model
import flows
import pyresearchutils as pru
import constants as C
import os
from tqdm import tqdm


def init_config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("base_epochs", type=int, default=360)
    _cr.add_parameter('base_dataset_size', type=int, default=200000)

    _cr.add_parameter("lr", type=float, default=2e-4)
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
    _cr.add_parameter("m_sensors", type=int, default=8)
    _cr.add_parameter("n_snapshots", type=int, default=2)
    _cr.add_parameter("k_targets", type=int, default=1)
    _cr.add_parameter("in_snr", type=float, default=0.1)
    _cr.add_parameter("wavelength", type=float, default=0.1)

    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)  # 200000
    _cr.add_parameter('val_dataset_size', type=int, default=20000)
    _cr.add_parameter('force_data_generation', type=str, default="false")

    return _cr


def main():
    cr = init_config()
    run_parameters, run_log_folder = pru.initialized_log(C.PROJECT, cr, enable_wandb=True)
    print("Starting Training Stage")
    sm = signal_model.DOASignalModel(run_parameters.m_sensors,
                                     run_parameters.n_snapshots,
                                     run_parameters.k_targets,
                                     run_parameters.in_snr, wavelength=run_parameters.wavelength)

    training_dataset = sm.generate_dataset(run_parameters.dataset_size)
    validation_dataset = sm.generate_dataset(run_parameters.val_dataset_size)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=run_parameters.batch_size,
                                                       shuffle=True)

    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=run_parameters.batch_size,
                                                         shuffle=False)
    flow = flows.DOAFlow(run_parameters.n_snapshots, run_parameters.m_sensors, run_parameters.k_targets,
                         run_parameters.wavelength)
    opt = torch.optim.Adam(flow.parameters(), lr=run_parameters.lr, weight_decay=run_parameters.weight_decay)
    n_epochs = run_parameters.base_epochs  # TODO:Update computation
    ma = pru.MetricAveraging()
    for epoch in range(n_epochs):
        for x, theta in tqdm(training_data_loader):
            opt.zero_grad()
            loss = flow.nll_mean(x, doas=theta)
            loss.baclward()
            opt.step()
            ma.log(loss=loss.item())

        # TODO: Add validation
        # TODO:Add save model


if __name__ == '__main__':
    main()
