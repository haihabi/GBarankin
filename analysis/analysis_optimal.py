import numpy as np
import constants
import matplotlib.pyplot as plt
import torch
import pyresearchutils as pru
import generative_bound
import signal_model
from main_flow_training_stage import init_config, build_signal_model


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
    run_param = cr.get_user_arguments()

    test_name = "test_noise_corr"
    # group_name = "janice_sullivan"
    # user_name = "HVH"
    apply_trimming = False
    use_ref_test_points = True
    theta_value = np.asarray([-np.pi / 3, np.pi / 3])
    n_samples2generate = 64000 * 8
    metric_list = pru.MetricLister()
    for snr in [6]:
        sm = build_signal_model(run_param, snr)
        flow_opt = sm.get_optimal_flow_model()
        crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value)
        mle_mse = sm.mse_mle(theta_value)
        if use_ref_test_points:
            test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
        else:
            test_points = None
        print(test_points.shape)
        gbarankin, gbb, search_landscape, test_points_search, test_points_final = generative_bound.generative_barankin_bound(
            flow_opt, n_samples2generate,
            test_points=test_points,
            parameter_name=constants.DOAS,
            doas=torch.tensor([theta_value]).to(
                pru.get_working_device()).reshape(
                [1, -1]).float())
        print("a")
        index_list = []
        for i in range(test_points_final.shape[0]):
            j = torch.where(torch.abs(test_points_final[i, :] - test_points).sum(axis=1) == 0)[0].item()
            index_list.append((i, j))

        bb_compare = torch.zeros_like(test_points_final)
        for i, j in index_list:
            bb_compare[i, i] = bb_matrix[j, j]
            for ii, jj in index_list:
                if i != ii:
                    bb_compare[i, ii] = bb_matrix[j, jj]
                    bb_compare[ii, i] = bb_matrix[jj, j]

        print(100 * np.linalg.norm(gbb.cpu().numpy() - bb_compare.cpu().numpy()) / np.linalg.norm(
            bb_compare.cpu().numpy()))
        metric_list.add_value(gbarankin=torch.trace(gbarankin).item(),
                              # gbarankin_opt=gbarankin_opt.item() if flow_opt is not None else 0,
                              # gbarankin_ntp=gbarankin_ntp.item(),
                              crb=np.trace(crb),
                              bb_bound=np.trace(bb_bound).item(),
                              mle_mse=mle_mse,
                              snr=snr)
        metric_list.print_last()
        if False:
            plot_tp_search([(search_landscape_ntp, test_points_search_ntp, test_points_ntp),
                            (search_landscape, test_points_search, test_points)], [("Learned", "green"),
                                                                                   ("Optimal", "red")],
                           theta_value)

        print('Completed SNR = {0:.2f} dB'.format(snr))
    metric_list.save2disk(f"data_{test_name}_{n_samples2generate}.pkl")
    plt.figure(figsize=(10, 8))
    # plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("gbarankin_ntp")), "--v",
    #              label=f"GBarankin (Learned,{run_config.signal_type.name})")
    # plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("gbarankin")), "--o",
    #              label="GBarankin (Optimal, Gaussian)")

    # plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("crb")), label="CRB")
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("mle_mse")), "o", label="MLE")
    plt.grid()
    plt.legend(fontsize=14)
    plt.ylabel("RMSE[deg]", fontsize=14)
    plt.xlabel("SNR[dB]", fontsize=14)
    plt.savefig(f"compare_with_learned_{test_name}_{n_samples2generate}.svg")
    plt.show()


if __name__ == '__main__':
    main()
