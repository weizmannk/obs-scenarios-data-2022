## This script plot the detectors sensitivities curves
## Then determine and print the BNS inspiral range
## For those the Run O4 and O5 with SNR=8 and and
## 1.4 sun mass binary system.

import os
from pathlib import Path

from gwpy.frequencyseries import FrequencySeries
from gwpy.astro import inspiral_range

import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["legend.fontsize"] = 9

from matplotlib import pyplot

pyplot.rc("axes", axisbelow=True)


colors = {"L1": "#4ba6ff", "V1": "#ee0000", "H1": "#9b59b6"}


outdir = "Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# the runs data direction
path = Path("ASD")


run_names = run_dirs = ["O4"]  # , 'O5']

# Sensitivities files
# O4= ['aligo_O4high.txt', 'avirgo_O4high_NEW.txt', 'kagra_10Mpc.txt']
# O5 = ['AplusDesign.txt', 'avirgo_O5low_NEW.txt', 'kagra_128Mpc.txt']

# Get the ASD of the Hanford  Livingston Virgo and KAGRA detectors

print("======================================")
for run_name in run_names:
    if run_name == "O4":
        # freq_L1, asd_L1     = np.loadtxt(f'{path}/aligo_O4high.txt', unpack=True)
        # freq_V1, asd_V1     = np.loadtxt(f'{path}/avirgo_O4high_NEW.txt', unpack=True)
        freq_L1, asd_L1 = np.loadtxt(f"{path}/asd_O4a_messured_L1.txt", unpack=True)
        freq_V1, asd_V1 = np.loadtxt(f"{path}/asd_O4a_Virgo_78.txt", unpack=True)
        freq_H1, asd_H1 = np.loadtxt(f"{path}/asd_O4a_mesured_H1.txt", unpack=True)

        # freq_K1, asd_K1     = np.loadtxt(f'{path}/kagra_10Mpc.txt', unpack=True)
    else:
        freq_L1, asd_L1 = np.loadtxt(f"{path}/AplusDesign.txt", unpack=True)
        freq_V1, asd_V1 = np.loadtxt(f"{path}/avirgo_O5low_NEW.txt", unpack=True)
        freq_K1, asd_K1 = np.loadtxt(f"{path}/kagra_128Mpc.txt", unpack=True)

    ## conversion de PSD en ADS
    #   freq_L1, psd_L1     = np.loadtxt(f'{path}/aligo_O4high.txt', unpack=True)

    #  asd_L1 = np.sqrt(psd_L1)

    #  outdata = np.array([freq_L1, asd_L1]).T

    # np.savetxt(X = outdata, fname = os.path.join(f'{path}', 'asd_L1.txt'))

    # Plot amplitude spectral density of the detectors
    fig = pyplot.figure(figsize=(4.5, 3))
    pyplot.loglog(
        freq_L1,
        asd_L1,
        label=r"$\mathrm{LIGO}\,\mathrm{L1}$",
        color=colors["L1"],
        linewidth=1,
        alpha=0.7,
    )
    pyplot.loglog(
        freq_H1,
        asd_H1,
        label=r"$\mathrm{LIGO}\,\mathrm{H1}$",
        color=colors["H1"],
        linewidth=1,
        alpha=0.7,
    )

    pyplot.loglog(
        freq_V1,
        asd_V1,
        label=r"$\mathrm{Virgo}$",
        color=colors["V1"],
        linewidth=1,
        alpha=0.7,
    )
    # pyplot.loglog(freq_K1, asd_K1, label=r'$\mathrm{KAGRA}$', color=colors['K1'], linewidth=1, alpha=0.7)

    pyplot.legend(loc=(0.065, 0.73))
    pyplot.xlabel(r"$\mathrm{Frequency}\,\mathrm{[Hz]}$")
    pyplot.ylabel(r"$\mathrm{ASD}\,[1/\sqrt{\mathrm{Hz}}]$")
    pyplot.xlim([10, 4000])
    pyplot.ylim(1e-24, 1e-19)

    pyplot.grid()
    fig.tight_layout()
    plt.savefig(f"{outdir}/Strain_{run_name}_HLV.png", dpi=300)
    plt.close()

    # compute BNS range

    range_L1 = inspiral_range(
        FrequencySeries(asd_L1**2, f0=freq_L1[0], df=freq_L1[1] - freq_L1[0]),
        snr=8,
        fmin=10,
        mass1=1.4,
        mass2=1.4,
    ).value
    range_V1 = inspiral_range(
        FrequencySeries(asd_V1**2, f0=freq_V1[0], df=freq_V1[1] - freq_V1[0]),
        snr=8,
        fmin=10,
        mass1=1.4,
        mass2=1.4,
    ).value
    # range_K1 = inspiral_range(FrequencySeries(asd_K1**2, f0=freq_K1[0], df=freq_K1[1]-freq_K1[0]), snr =8, fmin=1,  mass1=1.4, mass2=1.4).value
    range_H1 = inspiral_range(
        FrequencySeries(asd_H1**2, f0=freq_H1[0], df=freq_L1[1] - freq_L1[0]),
        snr=8,
        fmin=10,
        mass1=1.4,
        mass2=1.4,
    ).value

    print(f"\nThe BNS range in  Run {run_name}")
    print("** mass1=1.4, mass2=1.4, SNR=8 **\n")
    print(" inspiral sensitive distance\n")
    print(f"LIGO L1 range      : {np.round(range_L1, 0)} Mpc")
    print(f"Virgo range        : {np.round(range_V1, 0)} Mpc")
    print(f"LIGO H1 range      : {np.round(range_H1, 0)} Mpc\n")
    # print(f"KAGRA range      : {np.round(range_K1, 0)} Mpc\n")
    print("======================================")

    # del  freq_L1, asd_L1, freq_V1, asd_V1, freq_K1, asd_K1
