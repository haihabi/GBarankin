import numpy as np
import constants
import torch
import pyresearchutils as pru
import generative_bound

from main_flow_training_stage import init_config, build_signal_model
from tqdm import tqdm
from analysis.helpers import align_bb_matrix


def main():
    pru.set_seed(0)
    cr = init_config()
    run_param = cr.get_user_arguments()

    use_ref_test_points = True
    theta_value = np.asarray([-np.pi / 10])
    n_samples2generate = 64000 * 8
    n_trys = 1000
    for snr in [6, -1]:
        metric_list = pru.MetricLister()
        sm = build_signal_model(run_param, snr)
        flow_opt = sm.get_optimal_flow_model()
        crb, bb_bound, bb_matrix, test_points = sm.compute_reference_bound(theta_value, snr)

        if use_ref_test_points:
            test_points = torch.tensor(test_points).to(pru.get_working_device()).float().T
        else:
            test_points = None

        for j_try in tqdm(range(n_trys)):
            # try:
            noise_scale = np.sqrt(sm.POWER_SOURCE / (10 ** (snr / 10))).astype("float32")
            gbarankin, gbb, search_landscape, test_points_search, test_points_final = generative_bound.generative_barankin_bound(
                flow_opt, n_samples2generate,
                test_points=test_points,
                parameter_name=constants.DOAS,
                doas=torch.tensor([theta_value]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float(),
                noise_scale=torch.tensor([noise_scale]).to(
                    pru.get_working_device()).reshape(
                    [1, -1]).float()
            )
            bb_compare = align_bb_matrix(test_points, test_points_final, bb_matrix, gbb)

            eig_vec = torch.abs(torch.linalg.eig(gbb - torch.ones_like(gbb))[0])
            eig_vec_ref = torch.abs(torch.linalg.eig(torch.tensor(bb_matrix, device=gbb.device) - torch.ones_like(
                torch.tensor(bb_matrix, device=gbb.device)))[0])
            cond = torch.max(eig_vec) / torch.min(eig_vec)
            cond_ref = torch.max(eig_vec_ref) / torch.min(eig_vec_ref)
            re_bm = np.linalg.norm(gbb.cpu().numpy() - bb_compare.cpu().numpy()) / np.linalg.norm(
                bb_compare.cpu().numpy())
            re_bb = np.linalg.norm(gbarankin.cpu().numpy() - bb_bound) / np.linalg.norm(
                bb_bound)
            print(re_bm, re_bb)
            metric_list.add_value(re_bm=re_bm,
                                  re_bb=re_bb,
                                  cond=cond.item(),
                                  cond_ref=cond_ref.item(),
                                  index=j_try)
            metric_list.print_last()
        # except:
        #     print("Skip one")
        metric_list.save2disk(f"results_hist_{snr}.pkl")


if __name__ == '__main__':
    main()
