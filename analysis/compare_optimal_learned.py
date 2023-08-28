import numpy as np
import constants
import matplotlib.pyplot as plt
import torch
import pyresearchutils as pru
import generative_bound
import signal_model
from main_flow_training_stage import init_config, build_flow_model


def rmse_db(x):
    return np.sqrt(x) * 180 / np.pi


def plot_tp_search(in_data, in_legend, in_theta):
    for i in range(len(in_data)):
        search_landscape, test_point, selet_test_points = in_data[i]
        test_point = test_point.cpu().numpy()
        name, color = in_legend[i]
        delta_filter = np.diff(test_point - in_theta)
        loc = np.where(np.logical_not(np.isclose(np.min(delta_filter), delta_filter, atol=np.min(delta_filter))))[0]
        loc = np.concatenate([np.zeros(1), loc + 1, np.ones(1) * len(test_point)])
        loc = loc.astype("int")
        ###############################
        # Apply Transfor
        ##############################
        alpha = (test_point - in_theta) ** 2 / (1 / search_landscape - 1 + 1e-6)
        for i in range(len(loc) - 1):
            plt.semilogy(test_point[loc[i]:loc[i + 1]] - in_theta,
                         alpha[loc[i]:loc[i + 1]],
                         color,
                         label=name)

    plt.xlabel(r"$\psi-\theta$")
    plt.ylabel(r"$\alpha$")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    pru.set_seed(0)
    cr = init_config()

    group_name = "thomas_boyle"
    # group_name = "janice_sullivan"
    group_name = "brandi_mccammon"
    user_name = "HVH"
    apply_trimming = False
    use_ref_test_points = False
    theta_value = np.pi / 10
    n_samples2generate = 64000
    metric_list = pru.MetricLister()
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
        adaptive_trimming = None
        if apply_trimming:
            data = sm.generate_dataset(10000)
            data_np = np.stack(data.data, axis=0)
            data_np = data_np.reshape([10000, -1])
            mean_vec = np.mean(data_np, axis=0)
            data_np_norm = data_np - mean_vec
            max_norm = np.linalg.norm(data_np_norm, axis=-1).max()

            adaptive_trimming = generative_bound.AdaptiveTrimming(
                generative_bound.TrimmingParameters(mean_vec, max_norm, 0),
                generative_bound.TrimmingType.MAX)
            adaptive_trimming.to(pru.get_working_device())
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
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("mle_mse")), "o", label="MLE")
    plt.grid()
    plt.legend(fontsize=14)
    plt.ylabel("RMSE[deg]", fontsize=14)
    plt.xlabel("SNR[dB]", fontsize=14)
    plt.savefig(f"compare_with_learned_{group_name}_{n_samples2generate}.svg")
    plt.show()


if __name__ == '__main__':
    main()
