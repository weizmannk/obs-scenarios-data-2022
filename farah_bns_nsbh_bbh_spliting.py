### This code use to split Farah or GWTC-3 CBCS distribustion
## which passed the SNR, record in injections.dat file.
## In this file the masses of CBCs are detector masses
## So we have to divide those masses by 1+z,
## where z is the readshift, before split,
## CBC in BNS , NSBH and BBH.

## For Petrov (LRR) distribustion,
## the code can just copy , injections.dat in another folder ,
## If you want to use them just add the name of the in ''distribution'' list
## Then uncomment the part concern this , in the code

import os
from tqdm.auto import tqdm
from pathlib import Path
import shutil
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value


# the distribution folders
distribution = ["Farah"]  # , 'Petrov']

pops = ["BNS", "NSBH", "BBH"]
run_names = run_dirs = ["O3", "O4", "O5"]

## creat outdir to save

for dist in distribution:
    outdir = f"nmma_GWdata/{dist}_data/runs"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if dist == "Farah":
        print("\nFarah or GWTC-3 distribustion\n")

        # For splitting into BNS, NSBH, and BBH populations
        ns_max_mass = 3.0

        for run_name, run_dir in zip(tqdm(run_names), run_dirs):

            path = Path(f"{dist}/runs/{run_dir}/farah")
            injections = Table.read(
                str(path / "injections.dat"), format="ascii.fast_tab"
            )

            table = injections

            # Split by source frame mass
            z = z_at_value(
                cosmo.luminosity_distance, table["distance"] * u.Mpc
            ).to_value(u.dimensionless_unscaled)
            zp1 = z + 1

            source_mass1 = table["mass1"] / zp1
            source_mass2 = table["mass2"] / zp1

            print("=============================================================== \n")
            print(
                f"The number of subpopulation in {run_name} in within the radius of 15740 Mpc : "
            )

            for pop in pops:
                pop_dir = Path(f"{outdir}/{run_dir}/{pop.lower()}_farah")
                if not os.path.isdir(pop_dir):
                    os.makedirs(pop_dir)

                if pop == "BNS":
                    data = table[
                        (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
                    ]
                elif pop == "NSBH":
                    data = table[
                        (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
                    ]
                else:
                    data = table[
                        (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)
                    ]

                data.write(
                    Path(f"{pop_dir}/injections.dat"),
                    format="ascii.tab",
                    overwrite=True,
                )

                print(f"{pop} {len(data)} ; ")

            del injections, table, source_mass1, source_mass2, z, zp1, data
        print("***************************************************************\n")


"""
    ### Petrov distribution just copy files
    else:
        print("\nPetrov or LRR distribustion\n")

        for run_name, run_dir in zip(tqdm(run_names), run_dirs):

            print("===============================================================\n")
            print(f'In the {run_name} :')

            for pop in pops:
                path = Path(f'{dist}/runs/{run_dir}/{pop.lower()}_astro')

                injections = Table.read(str(path/'injections.dat'), format='ascii.fast_tab')
                data = injections


                pop_outdir = Path(f"{outdir}/{run_dir}/{pop.lower()}_astro")
                if not os.path.isdir(pop_outdir):
                    os.makedirs(pop_outdir)

                data.write(Path(f"{pop_outdir}/injections.dat"), format='ascii.tab', overwrite=True)

                print(f'{pop} {len(data)} ; ')

                #shutil.copy(f'{path}/injections.dat', f'{pop_outdir}')
                #print(f'{path}/injections.dat is copy in {pop_outdir}')

                del injections, table, data
        print("***************************************************************")

"""
