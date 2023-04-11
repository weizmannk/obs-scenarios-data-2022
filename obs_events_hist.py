import os
from tqdm.auto import tqdm
from pathlib import Path
from astropy.table import join, Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as  mpatches
from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

#mpl.use("agg")

fig_width_pt = 750.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.9 * fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
mpl.rcParams.update(params)


# the runs data direction
distributions = ['Farah',  'Petrov']

run_names = run_dirs=  ['O4'] #, 'O5']
pops = ['BNS', 'NSBH', 'BBH']


#### Farah sample
farah_initial = {'BNS': 892762, 'NSBH': 35962, 'BBH': 71276}

initial = { 'O3':{'BNS': 892762, 'NSBH': 35962, 'BBH': 71276}, 'O4':{'BNS': 892762, 'NSBH': 35962, 'BBH': 71276}, 'O5': {'BNS': 892762, 'NSBH': 35962, 'BBH': 71276}}

detection = {'O3':{'BNS': 460, 'NSBH': 79, 'BBH': 4891}, 'O4':{'BNS': 1004, 'NSBH': 184, 'BBH': 7070}, 'O5':{'BNS': 2003, 'NSBH': 356, 'BBH': 9809}}

resampling = {'O3':{'BNS': 390079, 'NSBH': 49177, 'BBH': 560744} , 'O4':{'BNS': 587016, 'NSBH': 60357, 'BBH': 352627} , 'O5':{'BNS': 768739, 'NSBH': 54642, 'BBH': 176619}
}

###Petrov
bns_astro = []
nsbh_astro = []
bbh_astro =[]

### spliting Farah data 
detection_table ={}

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0

for dist in distributions:
    if dist == 'Farah':
        for run_name, run_dir in zip(tqdm(run_names), run_dirs):
            
            print(f'the Farah  run {run_name}')
            
            detection_table[run_name] = {}
                
            path = Path(f'./Farah/runs/{run_dir}/farah')
            injections = Table.read(str(path/'injections.dat'), format='ascii.fast_tab')
            
            table = injections
            
            # Split by source frame mass
            z = z_at_value(cosmo.luminosity_distance, table['distance'] * u.Mpc).to_value(u.dimensionless_unscaled)
            zp1 = z + 1

            source_mass1 = table['mass1']/zp1
            source_mass2 = table['mass2']/zp1
            
            for pop in pops:
                print(f'population {pop}')
                
                if pop == 'BNS': 
                    data = table[(source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)]
                    
                elif pop == 'NSBH':
                    data = table[(source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)]
                else:
                    data = table[(source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)]
                
                print(len(data))
                
                detection_table[run_name][pop] = len(data)
                
                
            del injections, table, source_mass1, source_mass2, z, zp1,  data
    
    else:
        for run_name, run_dir in zip(tqdm(run_names), run_dirs):
            for pop in pops:
                      
                print(f'the  Petrov {pop} in the  run {run_name}')
                    
                path = Path(f'./Petrov/runs/{run_dir}/{pop.lower()}_astro')
                injections = Table.read(str(path/'injections.dat'), format='ascii.fast_tab')
                
                table = injections
                
                if pop == 'BNS':
                    bns_astro.append(len(table))
                elif pop == 'NSBH':
                    nsbh_astro.append(len(table))
                else:
                    bbh_astro.append(len(table))
                
                del table, injections



#### Farah data             
BNS_detect =[]
NSBH_detect = []
BBH_detect = []

BNS_resamp =[]
NSBH_resamp = []
BBH_resamp = []

for run_name in run_names:
    BNS_detect.append(detection_table[run_name]['BNS'])
    NSBH_detect.append(detection_table[run_name]['NSBH'])
    BBH_detect.append(detection_table[run_name]['BBH'])
    
    BNS_resamp.append(resampling[run_name]['BNS'])
    NSBH_resamp.append(resampling[run_name]['NSBH'])
    BBH_resamp.append(resampling[run_name]['BBH'])
    
    
BNS_detect  = np.array(BNS_detect)
NSBH_detect = np.array(NSBH_detect)
BBH_detect  = np.array(BBH_detect)

BNS_resamp  = np.array(BNS_resamp)
NSBH_resamp = np.array(NSBH_resamp)
BBH_resamp  = np.array(BBH_resamp)


#### Petrov data
bns_astro  = np.array(bns_astro)
nsbh_astro = np.array(nsbh_astro)
bbh_astro  =  np.array(bbh_astro)


## Plot 

bns_color, nsbh_color, bbh_color = sns.color_palette(
    'rocket', 3)

largeur_barre = 0.3

x1 = range(len(distributions))
x2 = [i + largeur_barre for i in x1[0]]
x3 = [i + 2*largeur_barre for i in x1[0]]

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

fig = plt.figure()
ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=3, colspan=4)
ax2 = plt.subplot2grid((4, 5), (3, 0), colspan=4, sharex=ax1)
ax1.set_yscale('log')

#ax1.set_xlabel('Observation Runs')

#Plot Farah Detection Number in each Run
ax1.bar(x1, NSBH_detect, width = largeur_barre, color = nsbh_color, edgecolor = ['black' for i in NSBH_detect], linewidth = 1)
ax1.bar(x2, BNS_detect, width = largeur_barre,  color = bns_color,  edgecolor = ['black' for i in BNS_detect], linewidth = 1)
ax1.bar(x3, BBH_detect, width = largeur_barre,  color = bbh_color,  edgecolor = ['black' for i in BBH_detect], linewidth = 1)

# Add annotation to bars
for i in range(len(x1[0])):
    ax1.text(i,      NSBH_detect[i]+20,  NSBH_detect[i], ha = 'center', color='navy')
    ax1.text(x2[i],  BNS_detect[i]+110,   BNS_detect[i],  ha = 'center', color='navy')
    ax1.text(x3[i],  BBH_detect[i]+300,   BBH_detect[i],  ha = 'center', color='navy')

label_position = []
for r in range(len(distributions)):
    label_position += [x1[r], x2[r], x3[r]]
    #run_pop += pops + ['\n\n'+run_names[r]]
    
    

#plt.setp(ax1.get_xticklabels() ha="right", rotation_mode="anchor")
run_pop = []
ax1.set_xticks(label_position, len(distributions)*['NSBH', 'BNS', 'BBH'],  ha='center', rotation_mode='anchor')

# rotate labels with A
#for label in ax1.get_xmajorticklabels():
 #   if  label.get_text() in pops:
  #      label.set_rotation(0)
    


#ax1.legend(pops,loc=2)
    
#Plot injection Number in each Run

ax1.scatter(x1[0], NSBH_resamp,color = nsbh_color)
ax1.scatter(x2[0], BNS_resamp, color = bns_color)
ax1.scatter(x3[0], BBH_resamp, color = bbh_color)

#ax1.axhline(0, color='grey', linewidth=0.8)

#ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

#ax2.set_ylabel(r'Detection Number')
#ax2.invert_yaxis()

#Plot Petrov Detection Number in each Run
ax1.bar(x1[1], nsbh_astro, width = largeur_barre, color = nsbh_color, edgecolor = ['black' for i in NSBH_detect], linewidth = 1)
ax1.bar(x2[1], bns_astro, width = largeur_barre,  color = bns_color,  edgecolor = ['black' for i in BNS_detect], linewidth = 1)
ax1.bar(x3[1], bbh_astro, width = largeur_barre,  color = bbh_color,  edgecolor = ['black' for i in BBH_detect], linewidth = 1)

# Add Petrov annotation to bars
for i in range(len(x1[1])):
    ax1.text(i,     nsbh_astro[i]+20,    nsbh_astro[i], ha = 'center', color='navy')
    ax1.text(x2[i],  bns_astro[i]+110,   bns_astro[i],  ha = 'center', color='navy')
    ax1.text(x3[i],  bbh_astro[i]+300,   bbh_astro[i],  ha = 'center', color='navy')


ax1.set_xticks([r + largeur_barre  for r in range(len(run_names))], run_names, ha='center', rotation_mode='anchor')
    
#Plot Petrov injection Number in each Run
ax1.scatter(x1[1], [1e6], color = nsbh_color)
ax1.scatter(x2[1], [1e6], color = bns_color)
ax1.scatter(x3[1],    [1e6], color = bbh_color)

#ax1.legend(pops,loc=2)
ax1.set_yscale('log')

#ax1.set_yticks(list(NSBH_resamp)+list(BNS_resamp)+list(BBH_resamp))

#plt.gcf().set_size_inches(8, 12)
plt.subplots_adjust(left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4)
fig.tight_layout()

#plt.savefig(f'obs_events_hist_plot.pdf')
plt.savefig(f'obs_events_hist_plot.png')
plt.close()
