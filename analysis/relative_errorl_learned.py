import numpy as np
import constants
import torch
import pyresearchutils as pru
import generative_bound
from main_flow_training_stage import init_config, build_flow_model, build_signal_model
from analysis.helpers import get_timming_function
from analysis.helpers import relative_error


def main():
    pru.set_seed(0)
    cr = init_config()

    group_name = "vickie_turman_-30_10"  # "vickie_turman_-30_10"
    user_name = "HVH"
    apply_trimming = True
    use_ref_test_points = True
    is_multiple_snr = True
    # theta_value = np.pi / 10
    n_samples2generate = 64000 * 8
    metric_list = pru.MetricLister()

    run_name = group_name
    run_config, run = pru.load_run(run_name, constants.PROJECT, user_name, cr)
    sm = build_signal_model(run_config, None)

    flow, _ = build_flow_model(run_config, sm)
    pru.load_model_weights(run, flow, f"model_last_{None}.pth")
    pru.download_file(run, "signal_model.pkl")
    sm.load_model("signal_model.pkl")
    flow_opt = sm.get_optimal_flow_model()

    adaptive_trimming = get_timming_function(apply_trimming, sm)
    for snr in [6, -1, -21]:
        noise_scale = np.sqrt(sm.POWER_SOURCE / (10 ** (snr / 10))).astype("float32")
        for theta_value in np.linspace(np.pi / 3, np.pi / 4, 20):
            theta_value = [-theta_value, theta_value]
            crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value, in_snr=snr)
            if use_ref_test_points:
                test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
            else:
                test_points = None
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
            gbarankin, gbb_opt, search_landscape, test_points_search, test_points = generative_bound.generative_barankin_bound(
                flow_opt, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float())

            gcrb = generative_bound.generative_cramer_rao_bound(flow_opt, n_samples2generate,
                                                                parameter_name=constants.DOAS,
                                                                doas=torch.tensor([theta_value]).to(
                                                                    pru.get_working_device()).reshape(
                                                                    [1, -1]).float(),
                                                                noise_scale=torch.tensor([noise_scale]).to(
                                                                    pru.get_working_device()).reshape(
                                                                    [1, -1]).float()
                                                                )

            metric_list.add_value(gbarankin=torch.trace(gbarankin).item(),
                                  re_optimal=relative_error(gbarankin.cpu().numpy(), bb_bound),
                                  re=relative_error(gbarankin_ntp.cpu().numpy(), bb_bound),
                                  gbarankin_ntp=torch.trace(gbarankin_ntp).item(),
                                  crb=np.trace(crb).flatten(),
                                  bb_bound=np.trace(bb_bound).flatten(),
                                  theta=theta_value,
                                  snr=snr)
            metric_list.print_last()
    metric_list.save2disk("re_analysis.pkl")


if __name__ == '__main__':
    main()
