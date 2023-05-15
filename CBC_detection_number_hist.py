## if you get , TypeError: set_ticks() got an unexpected keyword argument 'ha'
## Please use maplotlib >= 3.7.1

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
#plt.style.use('seaborn-v0_8-darkgrid')

sns.set_style("whitegrid")
#mpl.use("agg")

fig_width_pt = 700.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 2*0.9 * fig_width * golden_mean  # height in inches
fig_size = [11.5, 10.5] #[fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
mpl.rcParams.update(params)


outdir = 'Plots'
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# the runs data direction
distributions = ['Farah',  'Petrov']

run_names = run_dirs=  ['O5'] # 'O4']
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


 #########
#percentage 
petrov_event_nsbh = np.array([nsbh_astro])/1e6*100
petrov_event_bns = np.array([bns_astro])/1e6*100
petrov_event_bbh = np.array([bbh_astro])/1e6*100

petrov_tot = (bns_astro+nsbh_astro+bbh_astro)/(3*1e6)*100

#petrov_event = np.array([petrov_event_nsbh, petrov_event_bns, petrov_event_bbh])

gwtc_3_event_nsbh = np.array([NSBH_detect/NSBH_resamp])*100
gwtc_3_event_bns = np.array([BNS_detect/BNS_resamp])*100
gwtc_3_event_bbh = np.array([BBH_detect/BBH_resamp])*100

gwtc_3_tot= (BNS_detect + NSBH_detect +BBH_detect )/1e6*100

#gwtc_3_event = np.array([gwtc_3_event_nsbh, gwtc_3_event_bns, gwtc_3_event_bbh])
    

## Plot 

bns_color, nsbh_color, bbh_color = sns.color_palette(
    'rocket', 3)

largeur_barre = 0.3

x1 = range(len(distributions))
x2 = [i + largeur_barre for i in x1]
x3 = [i + 2*largeur_barre for i in x1]

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

fig = plt.figure()
ax1 = plt.subplot2grid((5, 8), (0, 0), rowspan=4, colspan=7)
ax2 = plt.subplot2grid((5, 8), (4, 0), rowspan=1, colspan=7, sharex=ax1)
ax1.set_yscale('log')

#ax1.set_xlabel('Observation Runs')

i=0
#Plot Farah Detection Number in each Run
ax1.bar(x1[i], NSBH_detect, width = largeur_barre, color = nsbh_color, edgecolor = ['black' for i in NSBH_detect], linewidth = 1)
ax1.bar(x2[i], BNS_detect, width = largeur_barre,  color = bns_color,  edgecolor = ['black' for i in BNS_detect], linewidth = 1)
ax1.bar(x3[i], BBH_detect, width = largeur_barre,  color = bbh_color,  edgecolor = ['black' for i in BBH_detect], linewidth = 1)

# Add annotation to bars
#for i in range(len([x1[0]])):
ax1.text(x1[i],      NSBH_detect[0]+20,  NSBH_detect[0], ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x2[i],      BNS_detect[0]+110,   BNS_detect[0],  ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x3[i],      BBH_detect[0]+320,   BBH_detect[0],  ha = 'center', color='navy', fontsize = 18, fontweight ='bold')


#run_pop  = []
label_position = []
for r in range(len(distributions)):
    label_position += [x1[r], x2[r], x3[r]]
    

#plt.setp(ax1.get_xticklabels() ha="right", rotation_mode="anchor")
run_pop = []
#ax2.set_xticks(label_position, len(distributions)*['NSBH', 'BNS', 'BBH'],  ha='center', rotation_mode='anchor')

# rotate labels with A
#for label in ax1.get_xmajorticklabels():
 #   if  label.get_text() in pops:
  #      label.set_rotation(0)
    


handles = [mpatches.Patch(facecolor=nsbh_color, label='NSBH'),
           mpatches.Patch(facecolor=bns_color, label='BNS'),
           mpatches.Patch(facecolor=bbh_color, label='BBH')]

ax1.set_ylabel(r'Detection Number')
ax1.legend(handles=handles, shadow=True, loc='upper left')

ax2.set_ylabel(r'$\%$ of detection')
    
#Plot injection Number in each Run

ax1.scatter(x1[i], NSBH_resamp[0],color = nsbh_color)
ax1.scatter(x2[i], BNS_resamp[0], color = bns_color)
ax1.scatter(x3[i], BBH_resamp[0], color = bbh_color)

ax1.text(x1[i]+0.03,    NSBH_resamp[0]-6000, NSBH_resamp[0], ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x2[i]+0.03,    BNS_resamp[0]-55000,  BNS_resamp[0],  ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x3[i]+0.03,    BBH_resamp[0]-35000,  BBH_resamp[0],  ha = 'left', color='navy', fontsize = 18, fontweight ='bold')




i = 1

#Plot Petrov Detection Number in each Run
ax1.bar(x1[i], nsbh_astro, width = largeur_barre, color = nsbh_color, edgecolor = ['black' for i in NSBH_detect], linewidth = 1)
ax1.bar(x2[i], bns_astro, width = largeur_barre,  color = bns_color,  edgecolor = ['black' for i in BNS_detect], linewidth = 1)
ax1.bar(x3[i], bbh_astro, width = largeur_barre,  color = bbh_color,  edgecolor = ['black' for i in BBH_detect], linewidth = 1)

# Add Petrov annotation to bars
#for i in range(len([x1[1]])):

ax1.text(i,     nsbh_astro[0]+180,    nsbh_astro[0], ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x2[i],  bns_astro[0]+110,   bns_astro[0],  ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax1.text(x3[i],  bbh_astro[0]+300,   bbh_astro[0],  ha = 'center', color='navy', fontsize = 18, fontweight ='bold')


ax2.set_xticks([r + largeur_barre  for r in range(len(distributions))], ['GWTC-3', 'LRR'], ha='center', rotation_mode='anchor')
plt.setp(ax1.get_xticklabels(), visible=False)
     
#Plot Petrov injection Number in each Run
ax1.scatter(x1[i], 1e6, color = nsbh_color)
ax1.scatter(x2[i], 1e6, color = bns_color)
ax1.scatter(x3[i], 1e6, color = bbh_color)

#fig.subplots_adjust(hspace=0.4)
ax_gwtc=np.array([x1[0], x2[0], x3[0]])
ax_petrov  = np.array([x1[1], x2[1], x3[1]])



ax2.scatter(x1[0], gwtc_3_event_nsbh, color = nsbh_color,  marker="o")
ax2.scatter(x2[0], gwtc_3_event_bns, color = bns_color,  marker="o")
ax2.scatter(x3[0], gwtc_3_event_bbh, color = bbh_color,  marker="o")

ax2.scatter(x1[1], petrov_event_nsbh, color = nsbh_color,  marker="o")
ax2.scatter(x2[1],  petrov_event_bns, color = bns_color,  marker="o")
ax2.scatter(x3[1],  petrov_event_bbh, color = bbh_color,  marker="o")


############
#Text
##################
ax2.text(x1[0]+0.03, gwtc_3_event_nsbh-0.09, f'{np.round(gwtc_3_event_nsbh[0][0], 2)}%',   ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax2.text(x2[0]+0.03, gwtc_3_event_bns-0.025,  f'{np.round(gwtc_3_event_bns[0][0], 2)}% ',  ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax2.text(x3[0]+0.03, gwtc_3_event_bbh-1.2,  f'{np.round(gwtc_3_event_bbh[0][0], 1)}% ',  ha = 'left', color='navy', fontsize = 18, fontweight ='bold')

ax2.text(x1[1]+0.03, petrov_event_nsbh-0.09, f'{np.round(petrov_event_nsbh[0][0], 2)}%',   ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax2.text(x2[1]+0.03,  petrov_event_bns-0.025, f'{np.round(petrov_event_bns[0][0], 2)}% ',   ha = 'left', color='navy', fontsize = 18, fontweight ='bold')
ax2.text(x3[1]+0.03,  petrov_event_bbh-0.14, f'{np.round(petrov_event_bbh[0][0], 2)}%',   ha = 'left', color='navy', fontsize = 18, fontweight ='bold')



ax2.set_yscale('log')


ax1.axvline(0.8, color="grey", linestyle="--", alpha=0.5)
ax2.axvline(0.8, color="grey", linestyle="--", alpha=0.5)


ax2.axhline(petrov_tot,  color="green", linestyle="--", alpha=0.5)
ax2.axhline(gwtc_3_tot,   color="blue", linestyle="--", alpha=0.5)

ax2.text(x2[1],  petrov_tot+0.015,  'LRR',   ha = 'left', fontsize = 18,fontweight ='bold')
ax2.text(x2[0],  gwtc_3_tot +0.03, 'GWTC-3',   ha = 'left',fontsize = 18, fontweight ='bold')


## Remobe logscale values and reput the values in bellow.
ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#ax2.set_yticks([0.1, np.round(petrov_tot[0], 2),np.round(gwtc_3_tot[0], 2),   2])
if run_name=='O4':
    ax2.set_yticks([0.1, np.round(petrov_tot[0], 2),np.round(gwtc_3_tot[0], 2),   2])
else:
    ax2.set_yticks([0.1, np.round(petrov_tot[0], 2),np.round(gwtc_3_tot[0], 2),   5.6])
    
plt.subplots_adjust(hspace=0.1)

#fig.tight_layout()

#plt.savefig(f'obs_events_hist_plot.pdf')
plt.savefig(f'{outdir}/CBC_detection_number_{run_name}_white_grid.png')
plt.close()
