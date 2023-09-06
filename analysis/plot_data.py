import matplotlib.pyplot as plt

import pyresearchutils as pru
import numpy as np

fontsize = 14


def project_results(in_results, in_snr, snr2remove=None):
    index = [True if snr not in snr2remove else False for snr in in_snr]

    return [in_snr[index], valie_function(180 * np.sqrt(in_results) / np.pi)[index]]


theta_value = np.pi / 4
far_point = -1.56079633
delta_far = np.power(far_point - theta_value, 2.0)
max_angle_error = 180 * np.sqrt(delta_far) / np.pi


def valie_function(in_array):
    # in_array = in_array.flatten()
    #
    # results = []
    # for i in range(in_array.shape[0]):
    #     if i == 0:
    #         results.append(in_array[-1 - i])
    #     else:
    #         results.append(max(in_array[-1 - i],results[-1]))

    return np.minimum(np.asarray(in_array), max_angle_error)


# data_prut = r"C:\Work\repos\GBarankin\analysis\data_jeffrey_seiler_-30_10_512000.pkl"
# corr = pru.MetricLister.load_data(data_prut)

print("a")
data_prut = r"C:\Work\repos\GBarankin\analysis\data_linda_lambert_-30_10_512000.pkl"

ml = pru.MetricLister.load_data(data_prut)
data_qam = r"C:\Work\repos\GBarankin\analysis\data_john_zamora_-30_10_512000.pkl"
ml_qam = pru.MetricLister.load_data(data_qam)

ax = plt.figure(figsize=(12, 8))
y = ml.get_array("b0_gen")
plt.semilogy(ml.get_array("snr"), ml.get_array("b0_gen"), label=r"$b(\Delta)-1$")
plt.semilogy([-10.5, -10.5],
             [np.min(ml.get_array("b0_gen")), np.max(ml.get_array("b0_gen"))], "--", color="black",
             label="Test Point Switch")

ax.get_axes()[0].axvspan(-30, -19, ymin=np.min(y) - 1, ymax=np.max(y), color='red', alpha=0.5,
                         label=r"No Information $\Delta^2$")
ax.get_axes()[0].axvspan(-19, -10, ymin=np.min(y) - 1, ymax=np.max(y), color='orange', alpha=0.5,
                         label=r"Threshold $\frac{\Delta^2}{\overline{b}(\Delta)-1}$")
ax.get_axes()[0].axvspan(-10, 10, ymin=0, ymax=np.max(y), color='gray', alpha=0.5,
                         label=r"Asymptotic $\mathrm{F}^{-1}_{\theta}$")
plt.grid()
plt.legend(fontsize=fontsize)
plt.xlabel(r"SNR[dB]", fontsize=fontsize)
plt.ylabel("$b(\Delta)-1$", fontsize=fontsize)
plt.tight_layout()
plt.savefig("b0_neg_one.svg")
plt.show()
print(ml.get_array("snr")[12])
# data_qam = r"C:\Work\repos\GBarankin\analysis\data_janice_sullivan.pkl"

print("a")
snrs = ml.get_array("snr")

snr2remove = []
plt.figure(figsize=(12, 8))
_, y = project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove)
plt.semilogy(*project_results(ml.get_array("gbarankin_ntp"), ml.get_array("snr"), snr2remove), "blue",
             label="Location Perturbation")
# plt.semilogy(ml.get_array("snr"), project_results(ml.get_array("gbarankin")), label="1")
plt.semilogy(*project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove), "green",
             label="Reference")
plt.semilogy(*project_results(ml_qam.get_array("bb_bound"), ml_qam.get_array("snr"), snr2remove), "yellow",
             label="A")
plt.semilogy(*project_results(ml_qam.get_array("gbarankin_ntp"), ml_qam.get_array("snr"), snr2remove), "red",
             label="QAM4")
plt.semilogy(*project_results(ml_qam.get_array("crb"), ml_qam.get_array("snr"), snr2remove), "--v", color="black",
             label="CRB")

plt.legend(fontsize=fontsize)
plt.grid()
plt.xlabel("SNR[dB]", fontsize=fontsize)
plt.ylabel("RMSE[degree]", fontsize=fontsize)
plt.tight_layout()
plt.savefig("analysis_threshold.svg")
plt.show()

ax = plt.figure(figsize=(12, 8))
_, y = project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove)
# plt.semilogy(*project_results(ml.get_array("gbarankin_ntp"), ml.get_array("snr"), snr2remove), "blue",
#              label="Location Perturbation")
# plt.semilogy(ml.get_array("snr"), project_results(ml.get_array("gbarankin")), label="1")
plt.semilogy(*project_results(ml_qam.get_array("gbarankin"), ml_qam.get_array("snr"), snr2remove), "green",
             label="Reference")
# plt.semilogy(*project_results(ml_qam.get_array("gbarankin_ntp"), ml_qam.get_array("snr"), snr2remove), "red",
#              label="QAM4")
plt.semilogy(*project_results(ml_qam.get_array("crb"), ml_qam.get_array("snr"), snr2remove), "--v", color="black",
             label="CRB")
print(np.min(y))
ax.get_axes()[0].axvspan(-30, -19, ymin=np.min(y) - 1, ymax=np.max(y), color='red', alpha=0.5,
                         label=r"No Information $\Delta^2$")
ax.get_axes()[0].axvspan(-19, -10, ymin=np.min(y) - 1, ymax=np.max(y), color='orange', alpha=0.5,
                         label=r"Threshold $\frac{\Delta^2}{\overline{b}(\Delta)-1}$")
ax.get_axes()[0].axvspan(-10, 10, ymin=0, ymax=np.max(y), color='gray', alpha=0.5,
                         label=r"Asymptotic $\mathrm{F}^{-1}_{\theta}$")

plt.legend(fontsize=fontsize)
plt.grid()
plt.xlim([-30, 10])
plt.xlabel("SNR[dB]", fontsize=fontsize)
plt.ylabel("RMSE[degree]", fontsize=fontsize)
plt.tight_layout()
plt.savefig("analysis_threshold_regiens.svg")
plt.show()
