import pyresearchutils as pru
import numpy as np

for snr in [6, -1, -21]:
    file_name = f"C:\\Work\\repos\\GBarankin\\analysis\\re_analysis_{snr}.pkl"
    data = pru.MetricLister.load_data(file_name)
    print("a")
    print("-" * 100, snr)
    print(np.mean(data.get_array("re_optimal")), np.max(data.get_array("re_optimal")))
    print(np.mean(data.get_array("re")), np.max(data.get_array("re")))
