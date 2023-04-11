import os
from tqdm.auto import tqdm
from pathlib import Path
from astropy.table import join, Table,  hstack
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
import numpy as np

import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
#plt.style.use('seaborn-v0_8')

# the runs data direction
path_dir = './Farah/runs'

run_names = run_dirs=  ['O4']
pops = ['BNS']


#bayestar_sampling = Table(resampling)

detection_table ={}

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0

for run_name, run_dir in zip(tqdm(run_names), run_dirs):
    
    print(f'the run {run_name}')
    
    detection_table[run_name] = {}
           
    path = Path(f'{path_dir}/{run_dir}/farah')
    injections = Table.read(str(path/'injections.dat'), format='ascii.fast_tab')
    coincs =     Table.read(str(path/'coincs.dat'), format='ascii.fast_tab')
     
    table = injections
    
    # Split by source frame mass
    z = z_at_value(cosmo.luminosity_distance, table['distance'] * u.Mpc).to_value(u.dimensionless_unscaled)
    zp1 = z + 1

    source_mass1 = table['mass1']/zp1
    source_mass2 = table['mass2']/zp1
    
    for pop in pops:
        print(f'population {pop}')
        SNR = []
        if pop == 'BNS': 
            data = table[(source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)]
            
        elif pop == 'NSBH':
            data = table[(source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)]
        else:
            data = table[(source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)]
            
           
        for event_id in data['simulation_id']:
            if event_id in coincs['coinc_event_id']:
                SNR.append(coincs[event_id]['snr'])
                
        snr_table = Table({'snr': SNR})
        detection_table[run_name][pop] = hstack([data, snr_table], join_type='exact')
        
        del SNR, snr_table
        
    del injections, table, source_mass1, source_mass2, z, zp1,  data
    

BNS = detection_table['O4']['BNS']

bns_color, nsbh_color, bbh_color = sns.color_palette(
    'rocket', 3)
n_bins= 150
stat = "count" 

# Figure Plot
plt.clf()
fig, axs2 = plt.subplots( tight_layout=True)

#axs1.hist(BNS['snr'], n_bins,  histtype="stepfilled", alpha=.4 , cumulative=True, color=bns_color,  label= 'BNS SNR in O4')

axs2 = sns.histplot(BNS['snr'], stat=stat, cumulative=True, bins=25, fill=False, legend=True)

axs2 = sns.ecdfplot(BNS['snr'], stat=stat)

axs2.set_ylabel('number')
axs2.set_title(' SNR BNS cummulative')

#axs1.legend(loc=2)


plt.xlabel('SNR')

plt.savefig(f'SNR_BNS_cummulative_hist.png')
plt.close()
