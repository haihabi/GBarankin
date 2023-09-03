import numpy as np
from matplotlib import pyplot as plt
import pyresearchutils as pru
from constants import FONTSIZE

legned_type_dict = {6: "Asymptotic",
                    -1: "Threshold",
                    -21: "No Information"}

color = {6: "red",
         -1: "green",
         -21: "blue"}
plt.figure(figsize=(8, 6))
for snr in [-21]:
    data = pru.MetricLister.load_data(f"C:\\Work\\repos\\GBarankin\\analysis\\results_hist_{snr}.pkl")

    count, x = np.histogram(data.data["cond"], bins=20, density=True)

    plt.semilogx(x[:-1], count, "--x", color=color[snr],
                 label=f"SNR={snr}[dB] ({legned_type_dict[snr]}) Generative Model")

    count, x = np.histogram(data.data["cond_ref"], bins=20, density=True)

    plt.semilogx(x[:-1], count, color=color[snr],
                 label=f"SNR={snr}[dB] ({legned_type_dict[snr]}) Analytic")

plt.legend(fontsize=FONTSIZE)
plt.grid()
plt.xlabel("RE", fontsize=FONTSIZE)
plt.ylabel("Probability", fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig("hist_res_cond.svg")
plt.show()

plt.figure(figsize=(8, 6))
for snr in [6, -1, -21]:
    data = pru.MetricLister.load_data(f"C:\\Work\\repos\\GBarankin\\analysis\\results_hist_{snr}.pkl")

    count, x = np.histogram(data.data["re_bm"], bins=20, density=True)

    plt.semilogx(x[:-1], count, "--x", color=color[snr],
                 label=f"SNR={snr}[dB] ({legned_type_dict[snr]})")

plt.legend(fontsize=FONTSIZE)
plt.grid()
plt.xlabel("RE", fontsize=FONTSIZE)
plt.ylabel("Probability", fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig("hist_res_bm.svg")
plt.show()
plt.figure(figsize=(8, 6))
for snr in [6, -1, -21]:
    data = pru.MetricLister.load_data(f"C:\\Work\\repos\\GBarankin\\analysis\\results_hist_{snr}.pkl")

    count, x = np.histogram(data.data["re_bb"], bins=20, density=True)

    plt.semilogx(x[:-1], count, "--x", color=color[snr], label=f"SNR={snr}[dB] ({legned_type_dict[snr]})")
plt.legend(fontsize=FONTSIZE)
plt.grid()
plt.xlabel("RE", fontsize=FONTSIZE)
plt.ylabel("Probability", fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig("hist_res_bb.svg")
plt.show()
