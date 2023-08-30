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

    use_ref_test_points = True
    theta_value = np.asarray([-np.pi / 3, np.pi / 3])
    n_samples2generate = 64000*8
    n_trys = 1000
    for snr in [6, -1, -21]:
        metric_list = pru.MetricLister()
        sm = build_signal_model(run_param, snr)
        flow_opt = sm.get_optimal_flow_model()
        crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value)

        if use_ref_test_points:
            test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
        else:
            test_points = None

        for j_try in range(n_trys):
            try:
                gbarankin, gbb, search_landscape, test_points_search, test_points_final = generative_bound.generative_barankin_bound(
                    flow_opt, n_samples2generate,
                    test_points=test_points,
                    parameter_name=constants.DOAS,
                    doas=torch.tensor([theta_value]).to(
                        pru.get_working_device()).reshape(
                        [1, -1]).float())

                index_list = []
                for i in range(test_points_final.shape[0]):
                    j = torch.where(torch.abs(test_points_final[i, :] - test_points).sum(axis=1) == 0)[0].item()
                    index_list.append((i, j))
                bb_compare = torch.zeros_like(gbb)
                for i, j in index_list:
                    bb_compare[i, i] = bb_matrix[j, j]
                    for ii, jj in index_list:
                        if i != ii:
                            bb_compare[i, ii] = bb_matrix[j, jj]
                            bb_compare[ii, i] = bb_matrix[jj, j]

                re_bm = 100 * np.linalg.norm(gbb.cpu().numpy() - bb_compare.cpu().numpy()) / np.linalg.norm(
                    bb_compare.cpu().numpy())
                re_bb = 100 * np.linalg.norm(gbarankin.cpu().numpy() - bb_bound) / np.linalg.norm(
                    bb_bound)
                metric_list.add_value(re_bm=re_bm,
                                      re_bb=re_bb,
                                      index=j_try)
                metric_list.print_last()
            except:
                print("Skip one")
        metric_list.save2disk(f"results_hist_{snr}.pkl")


if __name__ == '__main__':
    main()
