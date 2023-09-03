import numpy as np
import constants
import matplotlib.pyplot as plt
import torch
import pyresearchutils as pru
import generative_bound
import signal_model
from analysis.helpers import get_timming_function, plot_tp_search, rmse_db
from main_flow_training_stage import init_config, build_flow_model, build_signal_model
from analysis.helpers import relative_error


def main():
    pru.set_seed(0)
    cr = init_config()

    # group_name = "thomas_boyle"
    # # group_name = "janice_sullivan"
    # group_name = "brandi_mccammon"
    # group_name = "linda_lambert_-30_10"  # Pertubation
    group_name = "benjamin_ginsberg_-30_10"
    user_name = "HVH"
    apply_trimming = True
    use_ref_test_points = True
    is_multiple_snr = True
    theta_value = np.pi / 10
    n_samples2generate = 64000 * 8
    metric_list = pru.MetricLister()

    if is_multiple_snr:
        run_name = group_name
        run_config, run = pru.load_run(run_name, constants.PROJECT, user_name, cr)
        sm = build_signal_model(run_config, None)
        flow, _ = build_flow_model(run_config, sm)
        flow_opt = sm.get_optimal_flow_model()
        pru.load_model_weights(run, flow, f"model_last_{None}.pth")
        adaptive_trimming = get_timming_function(apply_trimming, sm)
        for snr in np.linspace(3, 10, 11):
            crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value, in_snr=snr)
            if use_ref_test_points:
                test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
            else:
                test_points = None
            eig_vec = np.linalg.eig(bb_matrix - np.ones(bb_matrix.shape))[0]
            cond = np.max(eig_vec) / np.min(eig_vec)

            noise_scale = np.sqrt(sm.POWER_SOURCE / (10 ** (snr / 10))).astype("float32")
            print("Learned")
            gbarankin_ntp, gbb, search_landscape_ntp, test_points_search_ntp, test_points_ntp = generative_bound.generative_barankin_bound(
                flow, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                trimming_step=adaptive_trimming)
            print("Optimal")
            gbarankin, gbb, search_landscape, test_points_search, test_points = generative_bound.generative_barankin_bound(
                flow_opt, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float())

            if False:
                plot_tp_search([(search_landscape_ntp, test_points_search_ntp, test_points_ntp),
                                (search_landscape, test_points_search, test_points)], [("Learned", "green"),
                                                                                       ("Optimal", "red")],
                               theta_value)
            re = relative_error(gbarankin.cpu().numpy(), bb_bound)
            metric_list.add_value(gbarankin=gbarankin.item(),
                                  # gbarankin_opt=gbarankin_opt.item() if flow_opt is not None else 0,
                                  re=re,
                                  cond=cond,
                                  gbarankin_ntp=gbarankin_ntp.item(),
                                  crb=crb.flatten(),
                                  bb_bound=bb_bound.flatten(),
                                  snr=snr)
            metric_list.print_last()

    else:
        for snr in constants.SNR_POINTS:
            run_name = group_name + f"_{snr}"
            run_config, run = pru.load_run(run_name, constants.PROJECT, user_name, cr)
            if run_config is None:
                print(f"Skip SNR:{snr}")
                continue
            sm = signal_model.DOASignalModel(run_config.m_sensors,
                                             run_config.n_snapshots,
                                             run_config.k_targets,
                                             snr,
                                             wavelength=run_config.wavelength,
                                             signal_type=run_config.signal_type)

            flow, _ = build_flow_model(run_config, sm)
            pru.load_model_weights(run, flow, f"model_last_{snr}.pth")
            flow_opt = sm.get_optimal_flow_model()
            adaptive_trimming = get_timming_function(apply_trimming, sm)
            crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value)
            mle_mse = sm.mse_mle(theta_value)

            if use_ref_test_points:
                test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
            else:
                test_points = None

            gbarankin_ntp, gbb, search_landscape_ntp, test_points_search_ntp, test_points_ntp = generative_bound.generative_barankin_bound(
                flow, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                trimming_step=adaptive_trimming)

            gbarankin, gbb, search_landscape, test_points_search, test_points = generative_bound.generative_barankin_bound(
                flow_opt, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float())

            metric_list.add_value(gbarankin=gbarankin.item(),
                                  # gbarankin_opt=gbarankin_opt.item() if flow_opt is not None else 0,
                                  gbarankin_ntp=gbarankin_ntp.item(),
                                  crb=crb.flatten(),
                                  bb_bound=bb_bound.flatten(),
                                  mle_mse=mle_mse,
                                  snr=snr)
            metric_list.print_last()
            if False:
                plot_tp_search([(search_landscape_ntp, test_points_search_ntp, test_points_ntp),
                                (search_landscape, test_points_search, test_points)], [("Learned", "green"),
                                                                                       ("Optimal", "red")],
                               theta_value)

            print('Completed SNR = {0:.2f} dB'.format(snr))
    metric_list.save2disk(f"data_{group_name}_{n_samples2generate}.pkl")
    plt.figure(figsize=(10, 8))
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("gbarankin_ntp")), "--v",
                 label=f"GBarankin (Learned,{run_config.signal_type.name})")
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("gbarankin")), "--o",
                 label="GBarankin (Optimal, Gaussian)")

    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("crb")), label="CRB")
    # plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("mle_mse")), "o", label="MLE")
    plt.grid()
    plt.legend(fontsize=14)
    plt.ylabel("RMSE[deg]", fontsize=14)
    plt.xlabel("SNR[dB]", fontsize=14)
    plt.savefig(f"compare_with_learned_{group_name}_{n_samples2generate}.svg")
    plt.show()


if __name__ == '__main__':
    main()
