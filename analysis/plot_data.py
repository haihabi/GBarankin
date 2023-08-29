import matplotlib.pyplot as plt

import pyresearchutils as pru
import numpy as np

fontsize = 14


def project_results(in_results, in_snr, snr2remove=None):
    index = [True if snr not in snr2remove else False for snr in in_snr]

    return [in_snr[index], valie_function(180 * np.sqrt(in_results) / np.pi)[index]]


def valie_function(in_array):
    # in_array = in_array.flatten()
    #
    # results = []
    # for i in range(in_array.shape[0]):
    #     if i == 0:
    #         results.append(in_array[-1 - i])
    #     else:
    #         results.append(max(in_array[-1 - i],results[-1]))

    return np.asarray(in_array)


data_prut = r"C:\Work\repos\GBarankin\analysis\data_brandi_mccammon_512000.pkl"
ml = pru.MetricLister.load_data(data_prut)

data_qam = r"C:\Work\repos\GBarankin\analysis\data_janice_sullivan.pkl"
ml_qam = pru.MetricLister.load_data(data_qam)
print("a")
snrs = ml.get_array("snr")
# bb_bound = project_results(ml.get_array("bb_bound"))
# gbb_bound = project_results(ml.get_array("gbarankin"))

snr2remove = [-10, -6, -3, -16]
plt.figure(figsize=(12, 8))
_, y = project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove)
plt.semilogy(*project_results(ml.get_array("gbarankin_ntp"), ml.get_array("snr"), snr2remove), "blue",
             label="Location Perturbation")
# plt.semilogy(ml.get_array("snr"), project_results(ml.get_array("gbarankin")), label="1")
plt.semilogy(*project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove), "green",
             label="Reference")
plt.semilogy(*project_results(ml_qam.get_array("gbarankin_ntp"), ml_qam.get_array("snr"), snr2remove), "red",
             label="QAM4")
plt.semilogy(*project_results(ml_qam.get_array("crb"), ml_qam.get_array("snr"), snr2remove), "--v", color="black",
             label="CRB")
# plt.semilogy(*project_results(ml_qam.get_array("mle_mse"), ml.get_array("snr"), snr2remove), "o", label="MLE(Reference)")
# plt.semilogy(snrs, gbb_bound)
plt.semilogy([-7, -7], [np.min(y), np.max(y)], "r--", label="Threshold: QAM4")
plt.semilogy([-4, -4], [np.min(y), np.max(y)], "b--", label="Threshold: Location Perturbation")
plt.semilogy([4, 4], [np.min(y), np.max(y)], "g--", label="Threshold: Reference")
plt.legend(fontsize=fontsize)
plt.grid()
plt.xlabel("SNR[dB]", fontsize=fontsize)
plt.ylabel("RMSE[degree]", fontsize=fontsize)
plt.tight_layout()
plt.savefig("analysis_threshold.svg")
plt.show()
