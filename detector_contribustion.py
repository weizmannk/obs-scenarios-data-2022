import os
from astropy.table import Table, hstack
import matplotlib.pyplot as plt

plt.rcdefaults()
import numpy as np
import matplotlib as mpl


outdir = "Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# the runs data direction
path = "./Farah/runs/"

run_names = run_dirs = ["O4"]
pops = ["BNS"]

H1 = []
L1 = []
V1 = []
K1 = []

for run_name in run_names:
    datapath = path + run_name + "/farah/"

    coincs = Table.read(datapath + "coincs.dat", format="ascii.fast_tab")

    for i in range(len(coincs)):
        if "H1" in coincs["ifos"][i].split(","):
            H1.append(1)
        else:
            H1.append(0)

        if "L1" in coincs["ifos"][i].split(","):
            L1.append(1)
        else:
            L1.append(0)

        if "V1" in coincs["ifos"][i].split(","):
            V1.append(1)
        else:
            V1.append(0)

        if "K1" in coincs["ifos"][i].split(","):
            K1.append(1)
        else:
            K1.append(0)


# Figure Plot
plt.clf()
fig = plt.figure()
ax = fig.add_subplot()


detectors = ["H1", "L1", "V1", "K1"]
X = np.arange(len(detectors))

contributions = [H1.count(1), L1.count(1), V1.count(1), K1.count(1)]

ax.bar(X, contributions, align="center", alpha=0.5)

ax.axhline(
    len(coincs),
    color="navy",
    linestyle="--",
    alpha=0.5,
    label="total number of detections",
)

ax.axhline(H1.count(1), color="gray", linestyle="--", alpha=0.5)

ax.axhline(L1.count(1), color="gray", linestyle="--", alpha=0.5)

ax.axhline(V1.count(1), color="gray", linestyle="--", alpha=0.5)

ax.axhline(K1.count(1), color="gray", linestyle="--", alpha=0.5)


plt.yscale("log")

plt.xticks(X, detectors)


## Remobe logscale values and reput the values in bellow.
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_yticks([7063, 6457, 6143, 8258])

plt.ylabel("Number of detection")
plt.xlabel("Detectors")
plt.title(f"Detectors contribustion during the run {run_name}")

plt.legend()
plt.subplots_adjust(right=0.9, wspace=0.4, hspace=0.4)
plt.tight_layout()


plt.savefig(f"{outdir}/detectors_contribustions.png", dpi=300)
plt.close()
