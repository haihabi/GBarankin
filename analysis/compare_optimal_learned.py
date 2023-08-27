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
    # pass
    # pass


def main():
    pru.set_seed(0)
    cr = init_config()
    # run_name = "clear-sun-96"
    # group_name = "thomas_boyle"
    group_name = "janice_sullivan"
    user_name = "HVH"

    theta_value = np.pi / 10
    n_samples2generate = 8*64000
    metric_list = pru.MetricLister()
    for snr in constants.SNR_POINTS:  # constants.SNR_POINTS:  # constants.SNR_POINTS
        run_name = group_name + f"_{snr}"
        run_config, run = pru.load_run(run_name, constants.PROJECT, user_name, cr)
        sm = signal_model.DOASignalModel(run_config.m_sensors,
                                         run_config.n_snapshots,
                                         run_config.k_targets,
                                         snr,
                                         wavelength=run_config.wavelength,
                                         signal_type=run_config.signal_type)

        flow, _ = build_flow_model(run_config, sm)
        pru.load_model_weights(run, flow, f"model_last_{snr}.pth")
        flow_opt = sm.get_optimal_flow_model()

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
        gbarankin_ntp, gbb, search_landscape_ntp, test_points_search_ntp, test_points_ntp = generative_bound.generative_barankin_bound(
            flow, n_samples2generate,
            parameter_name=constants.DOAS,
            doas=torch.tensor([theta_value]).to(
                pru.get_working_device()).reshape(
                [1, -1]).float(),
            trimming_step=adaptive_trimming)

        gbarankin, gbb, search_landscape, test_points_search, test_points = generative_bound.generative_barankin_bound(
            flow_opt, n_samples2generate,
            parameter_name=constants.DOAS,
            doas=torch.tensor([theta_value]).to(
                pru.get_working_device()).reshape(
                [1, -1]).float())
        # print("a")
        # plt.semilogy(test_points_search_ntp.cpu().numpy(), 1 / (1 / search_landscape_ntp - 1))
        # plt.show()
        # if flow_opt is not None:
        #     gbarankin_opt, gbb_opt, _, _ = generative_bound.generative_barankin_bound(flow_opt, n_samples2generate,
        #                                                                               test_points,
        #                                                                               parameter_name=constants.DOAS,
        #                                                                               doas=torch.tensor(
        #                                                                                   [theta_value]).to(
        #                                                                                   pru.get_working_device()).reshape(
        #                                                                                   [1, -1]).float())
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
    # if flow_opt is not None:
    #     plt.semilogy(constants.SNR_POINTS, rmse_db(metric_list.get_array("gbarankin_opt")), "--x",
    #                  label="GBarankin (Optimal)")
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("crb")), label="CRB")
    # plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("bb_bound")), label="BB")
    plt.semilogy(metric_list.get_array("snr"), rmse_db(metric_list.get_array("mle_mse")), "o", label="MLE")
    plt.grid()
    plt.legend(fontsize=14)
    plt.ylabel("RMSE[deg]", fontsize=14)
    plt.xlabel("SNR[dB]", fontsize=14)
    plt.savefig(f"compare_with_learned_{group_name}_{n_samples2generate}.svg")
    plt.show()


if __name__ == '__main__':
    main()

#
#
#
#
# wavelength = 1.0  # normalized
# d0 = wavelength / 2
# m_sensors = 20
# k_targets = 1
#
# # Create a 12-element ULA.
# ula = model.UniformLinearArray(m_sensors, d0)
# # Place 8 sources uniformly within (-pi/3, pi/4)
# sources = model.FarField1DSourcePlacement(
#     [-np.pi / 10]
#
# )
# # All sources share the same power.
# power_source = 1  # Normalized
# source_signal = model.ComplexStochasticSignal(sources.size, power_source)
# # 200 snapshots.
# n_snapshots = 10
# # We use root-MUSIC.
# estimator = estimation.RootMUSIC1D(wavelength)
#
# # snrs = np.linspace(-30, 10, 15)
# # print(np.round(snrs))
# snrs = constants.SNR_POINTS
# # print("a")
# # snrs = [-12.5]
# # 300 Monte Carlo runs for each SNR
# n_repeats = 300
#
# mses = np.zeros((len(snrs),))
# # crbs_sto = np.zeros((len(snrs),))
# barankin_stouc = np.zeros((len(snrs),))
# gbarankin_stouc = np.zeros((len(snrs),))
# # crbs_det = np.zeros((len(snrs),))
# crbs_stouc = np.zeros((len(snrs),))
# gcrbs_stouc = np.zeros((len(snrs),))
#
# for i, snr in enumerate(snrs):
#     power_noise = power_source / (10 ** (snr / 10))
#     noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
#
#     B_stouc, _ = perf.crb_stouc_farfield_1d(ula, sources, wavelength, power_source,
#                                             power_noise, n_snapshots)
#     BB_stouc, bb_matrix, test_points = perf.barankin_stouc_farfield_1d(ula, sources, wavelength, power_source,
#                                                                        power_noise, n_snapshots)
#     print(test_points.shape)
#     #################
#     # Generative Bound
#     #################
#     # The squared errors and the deterministic CRB varies
#     # for each run. We need to compute the average.
#     locations = torch.Tensor(ula._locations)
#     if locations.shape[1] == 1:
#         locations = torch.cat([locations, torch.zeros_like(locations)], dim=-1)
#     n_samples2generate = 64000
#     doa_optimal_flow = flows.DOAFlow(n_snapshots, m_sensors, k_targets, wavelength,
#                                      nominal_sensors_locations=locations.to(pru.get_working_device()).float(),
#                                      signal_covariance_matrix=torch.diag(
#                                          torch.diag(torch.ones(k_targets, k_targets))).to(
#                                          pru.get_working_device()).float() + 0 * 1j,
#                                      noise_covariance_matrix=power_noise * torch.diag(
#                                          torch.diag(torch.ones(m_sensors, m_sensors))).to(
#                                          pru.get_working_device()).float() + 0 * 1j)
#     doa_optimal_flow = doa_optimal_flow.to(pru.get_working_device())
#     gcrb = generative_bound.generative_cramer_rao_bound(doa_optimal_flow, n_samples2generate,
#                                                         parameter_name=constants.DOAS,
#                                                         doas=torch.tensor(sources).to(pru.get_working_device()).reshape(
#                                                             [1, -1]).float())
#     test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
#     gbarankin, gbb = generative_bound.generative_barankin_bound(doa_optimal_flow, n_samples2generate, test_points,
#                                                                 parameter_name=constants.DOAS,
#                                                                 doas=torch.tensor(sources).to(
#                                                                     pru.get_working_device()).reshape(
#                                                                     [1, -1]).float())
#
#     print(gbarankin, BB_stouc)
#     # print(gcrb, B_stouc)
#     gbb_np = gbb.cpu().numpy()
#
#     one_time_one = np.ones(gbb_np.shape)
#     eps = 1e-5
#     cond = np.linalg.norm(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps) / np.linalg.norm(
#         np.linalg.inv(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps))
#     np.linalg.norm(np.linalg.inv(gbb_np - one_time_one + np.eye(gbb_np.shape[0]) * eps) - np.linalg.inv(
#         bb_matrix - one_time_one + np.eye(gbb_np.shape[0]) * eps)) / np.linalg.norm(
#         np.linalg.inv(bb_matrix - one_time_one + np.eye(gbb_np.shape[0]) * eps))
#
#     print("BB RE", np.linalg.norm(gbb.cpu().numpy() - bb_matrix) / np.linalg.norm(bb_matrix))
#     print("GBB RE", np.linalg.norm(gbarankin.cpu().numpy() - BB_stouc) / np.linalg.norm(BB_stouc))
#     print("GCRB RE", np.linalg.norm(gcrb.cpu().numpy() - B_stouc) / np.linalg.norm(B_stouc))
#     gcrbs_stouc[i] = np.sqrt(torch.diag(gcrb).mean().item())
#     gbarankin_stouc[i] = np.sqrt(torch.diag(gbarankin).mean().item())
#     cur_mse = 0.0
#
#     for r in range(n_repeats):
#         # Stochastic signal model.
#         A = ula.steering_matrix(sources, wavelength)
#
#         S = source_signal.emit(n_snapshots)
#         N = noise_signal.emit(n_snapshots)
#         Y = A @ S + N
#         Rs = (S @ S.conj().T) / n_snapshots
#         Ry = (Y @ Y.conj().T) / n_snapshots
#         resolved, estimates = estimator.estimate(Ry, sources.size, d0)
#         # In practice, you should check if `resolved` is true.
#         # We skip the check here.
#         cur_mse += np.mean((estimates.locations - sources.locations) ** 2)
#
#     mses[i] = np.sqrt(cur_mse / n_repeats)
#
#     crbs_stouc[i] = np.sqrt(np.mean(np.diag(B_stouc)))
#     barankin_stouc[i] = np.sqrt(np.mean(np.diag(BB_stouc)))
#     print('Completed SNR = {0:.2f} dB'.format(snr))
# plt.figure(figsize=(8, 6))
# plt.semilogy(
#     snrs, mses, '-x',
#     snrs, gbarankin_stouc, '--v',
#     snrs, gcrbs_stouc, '--v',
#     snrs, crbs_stouc, '--',
#     snrs, barankin_stouc, '--'
# )
# plt.xlabel('SNR (dB)')
# plt.ylabel(r'RMSE / $\mathrm{rad}$')
# plt.grid(True)
# plt.legend(['MSE',
#             'GBarankin',
#             'GCRB ',
#             'CRB ',
#             'Barankin'])
# plt.title('MSE vs. CRB')
# plt.margins(x=0)
# plt.savefig("__compare_vs_snr.svg")
# plt.show()
