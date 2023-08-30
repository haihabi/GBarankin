import numpy as np
from matplotlib import pyplot as plt
import pyresearchutils as pru

legned_type_dict = {6: "Asymptotic",
                    -1: "Threshold",
                    -21: "No Information"}
for snr in [6, -1, -21]:
    data = pru.MetricLister.load_data(f"C:\\Work\\repos\\GBarankin\\analysis\\results_hist_{snr}.pkl")

    count, x = np.histogram(data.data["re_bm"], bins=20)

    plt.semilogx(x[:-1] / 100, count / np.sum(count), label=f"SNR={snr}[dB] ({legned_type_dict[snr]})")
plt.legend()
plt.grid()
plt.xlabel("RE")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()

for snr in [6, -1, -21]:
    data = pru.MetricLister.load_data(f"C:\\Work\\repos\\GBarankin\\analysis\\results_hist_{snr}.pkl")

    count, x = np.histogram(data.data["re_bb"], bins=20)

    plt.plot(x[:-1] / 100, count / np.sum(count), label=f"SNR={snr}[dB] ({legned_type_dict[snr]})")
plt.legend()
plt.grid()
plt.xlabel("RE")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()
