## conversion de PSD en ADS

import os
import numpy as np

path = "./PSD_HL_mcvt-estimate"


freq_L1, psd_L1 = np.loadtxt(
    f"{path}/L1_mcvt-estimate-psd_reference-1366934418-2678400.txt",
    unpack=True,
    skiprows=1,
)
freq_H1, psd_H1 = np.loadtxt(
    f"{path}/H1_mcvt-estimate-psd_reference-1366934418-2678400.txt",
    unpack=True,
    skiprows=1,
)


asd_L1 = np.sqrt(psd_L1)
asd_H1 = np.sqrt(psd_H1)

outdata_L1 = np.array([freq_L1, asd_L1]).T
outdata_H1 = np.array([freq_H1, asd_H1]).T

np.savetxt(X=outdata_L1, fname=os.path.join(f"{path}", "asd_L1.txt"))
np.savetxt(X=outdata_H1, fname=os.path.join(f"{path}", "asd_H1.txt"))
