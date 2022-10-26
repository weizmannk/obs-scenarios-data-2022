import os
from tqdm.auto import tqdm
from pathlib import Path
import shutil
from astropy.table import join, Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
import numpy as np


# the distribution flders
distribution = ['Farah', 'Petrov']

pops = ['BNS', 'NSBH', 'BBH']
run_names = run_dirs=  ['O3', 'O4', 'O5']


for dist in distribution:
    outdir = f'GW_EM_joint/{dist}_subpops/runs'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if dist == 'Farah':

        # For splitting into BNS, NSBH, and BBH populations
        ns_max_mass = 3

        for run_name, run_dir in zip(tqdm(run_names), run_dirs):

            path = Path(f'{dist}/runs/{run_dir}/farah')
            injections = Table.read(str(path/'injections.dat'), format='ascii.fast_tab')

            table = injections

            # Split by source frame mass
            z = z_at_value(cosmo.luminosity_distance, table['distance'] * u.Mpc).to_value(u.dimensionless_unscaled)
            zp1 = z + 1


            source_mass1 = table['mass1']/zp1
            source_mass2 = table['mass2']/zp1

            print("===============================================================")
            print(f'The number of subpopulation in {run_name} is : ')

            for pop in pops:
                pop_dir = Path(f"{outdir}/{run_dir}/{pop.lower()}_farah")
                if not os.path.isdir(pop_dir):
                    os.makedirs(pop_dir)

                if pop == 'BNS':
                    data = table[(source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)]
                elif pop == 'NSBH':
                    data= table[(source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)]
                else:
                    data = table[(source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)]

                data.write(Path(f"{pop_dir}/injections.dat"), format='ascii.tab', overwrite=True)

                print(f'{pop} {len(data)} ; ')


            del injections, table, source_mass1, source_mass2, z, zp1,  data
            print("***************************************************************")

    ### Petrov distribution just copy files
    else:

        for run_name, run_dir in zip(tqdm(run_names), run_dirs):

            print("===============================================================")
            print(f'In the {run_name} :')

            for pop in pops:
                path = Path(f'{dist}/runs/{run_dir}/{pop.lower()}_astro')

                pop_outdir = Path(f"{outdir}/{run_dir}/{pop.lower()}_astro")

                if not os.path.isdir(pop_outdir):
                    os.makedirs(pop_outdir)


                shutil.copy(f'{path}/injections.dat', f'{pop_outdir}')

                print(f'{path}/injections.dat is copy in {pop_outdir}')
            print("***************************************************************")
