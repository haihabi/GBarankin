import numpy as np
import constants
import matplotlib.pyplot as plt
import torch
import pyresearchutils as pru
import generative_bound
import signal_model
from main_flow_training_stage import init_config, build_signal_model
from analysis.helpers import rmse_db, plot_tp_search, compute_noise_scale, align_bb_matrix


def main():
    pru.set_seed(0)
    cr = init_config()
    run_param = cr.get_user_arguments()

    test_name = "corr_optimal_results"
    # group_name = "janice_sullivan"


    theta_value = np.asarray([np.pi / 4])
    n_samples2generate = 64000 * 8
    metric_list = pru.MetricLister()
    for snr in np.linspace(-30, 10, 41):
        sm = build_signal_model(run_param, snr)
        flow_opt = sm.get_optimal_flow_model()
        crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value, snr)
        noise_scale = compute_noise_scale(snr, sm.POWER_SOURCE)
        mle_mse = sm.mse_mle(theta_value, in_snr=snr)

        test_points = torch.tensor([[-np.pi / 2 + 1e-2]]).to(pru.get_working_device()).float().T
        print(test_points.shape)
        gbarankin, gbb, search_landscape, test_points_search, test_points_final = generative_bound.generative_barankin_bound(
            flow_opt, n_samples2generate,
            test_points=test_points,
            parameter_name=constants.DOAS,
            doas=torch.tensor([theta_value]).to(
                pru.get_working_device()).reshape(
                [1, -1]).float(),
            noise_scale=torch.tensor([noise_scale]).to(
                pru.get_working_device()).reshape(
                [1, -1]).float())
        print(gbb)

        metric_list.add_value(gbarankin=torch.trace(gbarankin).item(),
                              crb=np.trace(crb),
                              bb_bound=np.trace(bb_bound).item(),
                              mle_mse=mle_mse,
                              snr=snr)
        metric_list.print_last()
        print('Completed SNR = {0:.2f} dB'.format(snr))
    metric_list.save2disk(f"data_{test_name}_{n_samples2generate}.pkl")


if __name__ == '__main__':
    main()
