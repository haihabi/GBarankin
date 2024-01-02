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
import scipy


def main():
    pru.set_seed(0)
    cr = init_config()

    # group_name = "thomas_boyle"
    # # group_name = "janice_sullivan"
    # group_name = "brandi_mccammon"
    # group_name = "benjamin_ginsberg_-30_10"
    group_name = ("linda_lambert_-30_10", -8)  # Pertubation

    ref_cross = -9.0
    # group_name = ("john_zamora_-30_10", -11)  # QAM
    group_name = ("charles_mcadoo_-30_10",-9.0)  # Correlated -9.0
    user_name = "HVH"
    apply_trimming = True
    use_ref_test_points = True
    is_multiple_snr = True
    switch_threshold = False
    theta_value = np.pi / 4
    n_samples2generate = 64000 * 8
    metric_list = pru.MetricLister()
    far_point = -1.56079633
    if is_multiple_snr:
        run_name = group_name[0]
        run_config, run = pru.load_run(run_name, constants.PROJECT, user_name, cr)
        sm = build_signal_model(run_config, None)
        pru.download_file(run, "signal_model.pkl")
        sm.load_model("./signal_model.pkl")
        flow, _ = build_flow_model(run_config, sm)
        flow_opt = sm.get_optimal_flow_model()
        pru.load_model_weights(run, flow, f"model_last_{None}.pth")
        # theta_array_bp = torch.linspace(-np.pi / 2, np.pi / 2, 1000, device=pru.get_working_device())
        # bp = torch.stack([torch.abs(
        #     flow.steering_matrix(theta.reshape([1, 1])).reshape([-1, 1]).T.conj() @ flow.steering_matrix(
        #         torch.tensor([theta_value], device=pru.get_working_device()).reshape([1, 1])).reshape([-1, 1])) / 20 for
        #                   theta in theta_array_bp])
        # plt.plot(theta_array_bp.cpu().numpy().flatten(), bp.cpu().numpy().flatten())
        # index = scipy.signal.find_peaks(bp.cpu().numpy().flatten())[0]
        # plt.plot(theta_array_bp.cpu().numpy().flatten()[index], bp.cpu().numpy().flatten()[index], "o")
        # index_sort = np.flip(np.argsort(bp.cpu().numpy().flatten()[index]))
        # plt.plot(theta_array_bp.cpu().numpy().flatten()[index[index_sort[1]]], bp.cpu().numpy().flatten()[index[index_sort[1]]], "x")
        # plt.show()
        # far_point = theta_array_bp.cpu().numpy().flatten()[index[index_sort[1]]]
        for snr in np.linspace(-30, 10, 41):
            apply_trimming = False
            adaptive_trimming = get_timming_function(apply_trimming, sm, snr)
            crb, bb_bound, bb_matrix, test_points_base = sm.compute_reference_bound(theta_value, in_snr=snr)

            if use_ref_test_points:
                test_points = torch.tensor(test_points_base).to(pru.get_working_device()).float().T
            else:
                test_points = None

            far_test_points = torch.tensor([[far_point]]).to(pru.get_working_device()).float().T
            near_test_points = torch.tensor([[theta_value + 1e-5]]).to(pru.get_working_device()).float().T
            eig_vec = np.linalg.eig(bb_matrix - np.ones(bb_matrix.shape))[0]
            cond = np.max(eig_vec) / np.min(eig_vec)

            noise_scale = np.sqrt(sm.POWER_SOURCE / (10 ** (snr / 10))).astype("float32")
            mle = sm.mse_mle([theta_value], in_snr=snr)
            print("Learned")
            gbarankin_ntp, gbb_learend, search_landscape_ntp, test_points_search_ntp, test_points_ntp = generative_bound.generative_barankin_bound(
                flow, n_samples2generate,
                test_points=far_test_points if snr < group_name[1] else near_test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                trimming_step=adaptive_trimming)

            gbarankin_ntp_same, _, _, _, _ = generative_bound.generative_barankin_bound(
                flow, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float())
            print("Optimal")
            print(test_points_ntp)
            gbarankin, gbb, search_landscape, test_points_search, test_points_opt = generative_bound.generative_barankin_bound(
                flow_opt, n_samples2generate,
                test_points=far_test_points if snr < ref_cross else near_test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float())
            print(test_points)
            # if True:
            #     plot_tp_search([(search_landscape_ntp, test_points_search_ntp, test_points_ntp),
            #                     (search_landscape, test_points_search, test_points)], [("Learned", "green"),
            #                                                                            ("Optimal", "red")],
            #                    theta_value)
            re = relative_error(gbarankin.cpu().numpy(), bb_bound)
            b0 = bb_matrix.flatten() - 1
            b0_gen = (gbb.flatten() - 1).item()
            b0_lerend = (gbb_learend.flatten() - 1).item()
            # if b0_gen.shape[0] == 1:
            #     b0_gen = b0_gen.item()
            # else:
            #     b0_gen = None
            metric_list.add_value(gbarankin=gbarankin.item(),
                                  test_points=test_points_base.flatten(),
                                  mle=mle,
                                  # gbarankin_opt=gbarankin_opt.item() if flow_opt is not None else 0,
                                  b0=b0,
                                  b0_gen=b0_gen,
                                  b0_lerend=b0_lerend,
                                  re=re,
                                  cond=cond,
                                  gbarankin_ntp=gbarankin_ntp.item(),
                                  gbarankin_ntp_same=gbarankin_ntp_same.item(),
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

    metric_list.save2disk(f"data_{group_name[0]}_{n_samples2generate}_new_new.pkl")


if __name__ == '__main__':
    main()
