import numpy as np
from matplotlib import pyplot as plt
import pyresearchutils as pru
from constants import FONTSIZE

legned_type_dict = {6: "Asymptotic",
                    -14: "Threshold",
                    -15: "Threshold",
                    -18: "Threshold",
                    -21: "Threshold"}

color = {6: "red",
         -14: "cyan",
         -15: "green",
         -18: "green",
         -21: "blue"}

plt.figure(figsize=(12, 8))
for snr in [6, -18]:
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
plt.figure(figsize=(12, 8))
for snr in [6, -18, ]:
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
