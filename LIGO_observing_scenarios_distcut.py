import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import mplcyberpunk
from glob import glob
import pandas as pd
import sys 
import os
from astropy.table import join, Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
from astropy.coordinates import Distance
import astropy.units as u
#matplotlib.rcParams['figure.figsize'] = (18, 10.)
matplotlib.rcParams['xtick.labelsize'] = 12.0
matplotlib.rcParams['ytick.labelsize'] = 12.0
matplotlib.rcParams['axes.labelsize'] = 22.0
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 18
#matplotlib.style.use('seaborn-colorblind')

matplotlib.use('agg')
import matplotlib.gridspec as gridspec

#datapath = 'observing_scenarios_2022/O4/'

path = '/home/kiendrebeogo/weizmann-doc/OBSERVING-SCENARIOS/obs-scenarios-data-2022/Farah/runs/'

run_names = ['O4', 'O5']

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0

# read in the files
#allsky_BNS = pd.read_csv(datapath+'BNS_O4_allsky.dat',skiprows=1,delimiter=' ')
#allsky_NSBH = pd.read_csv(datapath+'NSBH_O4_allsky.dat',skiprows=1,delimiter=' ')



# How far can ZTF detect a KN assuming GW170817-like luminosity?
Mabs = -16
mlim = 22 # assuming clear weather conditions and 300s exposures
distmod = mlim - Mabs
d = Distance(distmod=distmod, unit=u.Mpc)

# generate 100,000 realizations of the NS mergers assuming the annual rate from OS 2022
# then determine the distribution of events falling within the distance cut
Number_BNS  = {'O4':36, 'O5':180}
Number_NSBH = {'O4':6,  'O5':31}


# Figure Plot 
plt.clf()
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
rows,cols=1,2
gs = gridspec.GridSpec(rows,cols)
sax = []

for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))
        

for run_name in run_names:
    
    datapath = path+run_name+'/farah/'
    
    allsky = Table.read(datapath+'allsky.dat', format='ascii.fast_tab')
    injections = Table.read(datapath+'injections.dat', format='ascii.fast_tab')

    # Split by source frame mass
    z = z_at_value(cosmo.luminosity_distance, injections['distance'] * u.Mpc).to_value(u.dimensionless_unscaled)
    zp1 = z + 1

    source_mass1 = injections['mass1'] / zp1
    source_mass2 = injections['mass2'] / zp1


    BNS =  (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    NSBH = (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    #BBH = injections[(source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)]

    allsky_BNS = allsky[BNS].to_pandas()
    allsky_NSBH = allsky[NSBH].to_pandas()

    # generate 100,000 realizations of the NS mergers assuming the annual rate from OS 2022
    # then determine the distribution of events falling within the distance cut
    N_BNS  = Number_BNS[run_name]
    N_NSBH = Number_NSBH[run_name]


    rng = np.random.default_rng(42)
    realizations_BNS = [np.sum(rng.choice(allsky_BNS['distmean'].to_numpy(),N_BNS)<d.value) for i in range(0,100000)]
    #realizations_BNS_200 = [np.sum(rng.choice(allsky_BNS['distmean'].to_numpy(),N_BNS)<200) for i in range(0,100000)]
    realizations_NSBH = [np.sum(rng.choice(allsky_NSBH['distmean'].to_numpy(),N_NSBH)<d.value) for i in range(0,100000)]

    # now we can calculate the percentiles to get errorbars on number of detected events
    Ndet_lower = np.percentile(realizations_BNS, q=5)
    Ndet_higher = np.percentile(realizations_BNS, q=95)
    Ndet_med = np.percentile(realizations_BNS, q=50)
    
    print(f"The Run {run_name}")
    print("number of detected BNS mergers: %d^{+%d}_{-%d}" % (Ndet_med, Ndet_higher-Ndet_med, Ndet_med-Ndet_lower))


    Ndet_lower = np.percentile(realizations_NSBH, q=5)
    Ndet_higher = np.percentile(realizations_NSBH, q=95)
    Ndet_med = np.percentile(realizations_NSBH, q=50)
    
    print("number of detected NSBH mergers: %d^{+%d}_{-%d}" % (Ndet_med, Ndet_higher-Ndet_med, Ndet_med-Ndet_lower))
    print(" ")

    bins = np.arange(0, 30, 1)
    
    if  run_name=='O4':
        sax[0].hist(realizations_BNS, bins=bins, label='O4 BNS in 400 Mpc', density=True, cumulative=True, histtype='step', linestyle='--', color='goldenrod', linewidth=3)
        sax[0].hist(realizations_NSBH,bins=bins, label='O4 NSBH in 400 Mpc', density=True, cumulative=True, histtype='step', linestyle=':', color='teal', linewidth=3)

        bbox = dict(facecolor = 'white', alpha = 0.7,edgecolor='teal', linestyle=':', linewidth=2)
        sax[0].text(2.8, 0.4, 'NSBH',color='k',fontsize=22, bbox = bbox)
        bbox = dict(facecolor = 'white', alpha = 0.7,edgecolor='goldenrod',linestyle='--', linewidth=2)
        sax[0].text(20, 0.4, 'BNS',color='k',fontsize=22, bbox = bbox)
        sax[0].text(2.25, 0.3, f'<N>={int(np.round(np.mean(realizations_NSBH)))}', fontsize=22)
        sax[0].text(18, 0.3,   f'<N>={int(np.round(np.mean(realizations_BNS)))}',  fontsize=22)
        sax[0].text(-2, 1.01, r'Run O4', color='blue',fontname="Times New Roman",  fontweight="bold", fontsize=22)    

    else :
        sax[1].hist(realizations_BNS, bins=bins, label='O5 BNS in 400 Mpc', density=True, cumulative=True, histtype='step', linestyle='--', color='goldenrod', linewidth=3)
        sax[1].hist(realizations_NSBH,bins=bins, label='O5 NSBH in 400 Mpc', density=True, cumulative=True, histtype='step', linestyle=':', color='teal', linewidth=3)

        bbox = dict(facecolor = 'white', alpha = 0.7,edgecolor='teal', linestyle=':', linewidth=2)
        sax[1].text(2.8, 0.4, 'NSBH',color='k',fontsize=22, bbox = bbox)
        bbox = dict(facecolor = 'white', alpha = 0.7,edgecolor='goldenrod',linestyle='--', linewidth=2)
        sax[1].text(20, 0.4, 'BNS',color='k',fontsize=22, bbox = bbox)
        sax[1].text(2.25, 0.3, f'<N>={int(np.round(np.mean(realizations_NSBH)))}', fontsize=22)
        sax[1].text(18, 0.3,   f'<N>={int(np.round(np.mean(realizations_BNS)))}',  fontsize=22)
        sax[1].text(-2, 1.01, r'Run O5', color='blue',fontname="Times New Roman", fontweight="bold", fontsize=22)    



sax[0].set_ylabel('cumulative probability density')
sax[0].set_title('Annual LVK-detected NS Mergers in O4 within 400 Mpc')
sax[0].set_xlabel('number of events')

sax[1].set_title('Annual LVK-detected NS Mergers in O5 within 400 Mpc')
sax[1].set_xlabel('number of events')
sax[0].set_xlim(-4, 25)
sax[1].set_xlim(-4, 28.9)

plt.gcf().set_size_inches(20, 10)
plt.subplots_adjust(
        right=0.9,
        wspace=0.4,
        hspace=0.4)
fig.tight_layout()



plt.savefig('ndet_22mag_allsky_NSBH_os2022_O5.pdf', dpi=500)
plt.close()

