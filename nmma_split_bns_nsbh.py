## we select data with redshift  z < 1.98 that means d < 15740 Mpc
## because nmma could not simulate multimessenger with z close to 2.
import os
from tqdm.auto import tqdm
from pathlib import Path
import shutil
from astropy.table import Table, join
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value

from ligo.skymap.io import read_sky_map
from gwpy.table import Table as gwpy_Table
import numpy as np


# Set 'nmma' to False if these parameters are not used for inferences with 'nmma'.
# For 'nmma' inference, the readshift should be less than z<2, hence we constrain
# the CBCs within a radius of 15740 Mpc.
nmma = True

# If you require adding GPS time to your parameter, set 'GPS_TIMES' (gps_time) to True.
# However, note that GPS times are already included in the '.fits' skymap events,
# resulting in extended reading times. Therefore, it may not be practical to include them
# since gps_time ≈ geocent_end_time + (geocent_end_time_ns)^(10^(-9)),
# where geocent_end_time is in seconds (s) and geocent_end_time_ns is in nanoseconds.
# then easy to get them in the xml files.
GPS_TIME = True


# data dir to the runs
datapath = "/home/kiendrebeogo/weizmann-doc/GitHub/runs_HL_O4a_SNR_8_PSD_Ideal/runs/"

## creat outdir to save
outdir = f"nmma_GWdata"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# the distribution folders
distribution = ["Farah"]

pops = ["BNS", "NSBH", "BBH"]
run_names = run_dirs = ["O4a", "O4", "O5"]


with tqdm(total=len(run_names) * len(pops) * len(distribution)) as progress:
    for dist in distribution:
        output = f"{outdir}/{dist}/runs"
        if not os.path.isdir(output):
            os.makedirs(output)

        if dist == "Farah":
            print("\nFarah or GWTC-3 distribustion\n")

            # For splitting into BNS, NSBH, and BBH populations
            ns_max_mass = 3.0

            for run_name, run_dir in zip(run_names, run_dirs):
                path = Path(f"{datapath}/{run_dir}/farah")
                print(path)

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

                print(
                    "=============================================================== \n"
                )

                for pop in pops:
                    pop_dir = Path(f"{output}/{run_dir}/{pop.lower()}_farah")
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
                            (source_mass1 >= ns_max_mass)
                            & (source_mass2 >= ns_max_mass)
                        ]

                    # select data with z <= 1.98
                    if pop == "BNS" or "NSBH":
                        if nmma:
                            print(
                                f"The number of subpopulation in {run_name} in within the radius of 15740 Mpc : "
                            )
                            data = data[data["distance"] <= 15740]

                    ### Add gps_time and geocent_time
                    gps_time = []
                    geocent_end_time = []
                    geocent_end_time_ns = []

                    ## Extensions parameters for bilby process
                    polarization = []
                    coa_phase = []

                    # Read XML files to extract the geocent_time
                    xml_data = gwpy_Table.read(
                        str(path / "events.xml.gz"),
                        format="ligolw",
                        tablename="sim_inspiral",
                    )

                    simulation_ID = data["simulation_id"]
                    for ID in simulation_ID:
                        geocent_end_time.append(xml_data[ID]["geocent_end_time"])
                        geocent_end_time_ns.append(xml_data[ID]["geocent_end_time_ns"])
                        polarization.append(xml_data[ID]["polarization"])
                        coa_phase.append(xml_data[ID]["coa_phase"])

                        if GPS_TIME:
                            # read skymap events to get gps_time
                            gps_time.append(
                                read_sky_map(f"{path}/allsky/{ID}.fits")[1]["gps_time"]
                            )

                    if GPS_TIME:
                        time_dict = {
                            "simulation_id": simulation_ID,
                            "polarization": polarization,
                            "coa_phase": coa_phase,
                            "geocent_end_time": geocent_end_time,
                            "geocent_end_time_ns": geocent_end_time_ns,
                            "gps_time": gps_time,
                        }

                    else:
                        time_dict = {
                            "simulation_id": simulation_ID,
                            "polarization": polarization,
                            "coa_phase": coa_phase,
                            "geocent_end_time": geocent_end_time,
                            "geocent_end_time_ns": geocent_end_time_ns,
                        }

                    time_table = Table(time_dict)
                    tables = join(data, time_table)

                    tables.write(
                        Path(f"{pop_dir}/{pop.lower()}_{run_name}_injections.dat"),
                        format="ascii.tab",
                        overwrite=True,
                    )

                    print(f"{pop} {len(tables)} ; ")
                    print(
                        "\n===============================================================\n"
                    )

                    progress.update()

                del (
                    injections,
                    table,
                    source_mass1,
                    source_mass2,
                    z,
                    zp1,
                    data,
                    time_dict,
                    time_table,
                    tables,
                    path,
                )
