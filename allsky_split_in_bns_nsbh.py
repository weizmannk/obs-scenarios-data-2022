import os
from astropy.table import  Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value


path = 'Farah/runs/'


outdir = 'allsky_output'
if not os.path.isdir(outdir):
    os.makedirs(outdir)


run_names = ['O4', 'O5']

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0


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


    # If you want this convert astropy table  in pandas DataFrame format.
    # Please add to_pandas()
    
    allsky_BNS  = allsky[BNS]  # .to_pandas()
    allsky_NSBH = allsky[NSBH] #   .to_pandas()
    
    allsky_BNS.write(f"{outdir}/allsky_bns.dat", format='ascii.tab', overwrite=True)
    
    allsky_NSBH.write(f"{outdir}/allsky_nsbh.dat", format='ascii.tab', overwrite=True)
    
    del  allsky, injections, datapath, z, zp1, source_mass1, source_mass2, allsky_BNS, allsky_NSBH