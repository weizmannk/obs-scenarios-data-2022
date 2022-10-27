# Perform `combined analyses`

NMMA is capable of performing combined analyses to constrain the neutron star equation of state (EOS) and Hubble Constant. In the following, we will take as an example the EOS analysis.

## **`Create an oudir to put all data we will generate`**

First of all, you need to create an output directory, this output will host all the data that will be used to constrain the EOS.

	mkdir -p ./GW_EM_joint



## Copy and split files in `GW_EM_joint` directory

For  **[Farah-distribution]**: In each Run (O3, O4 and O5) we have all populations(BNS, NSBH and BBH) together in the `injections.dat` file. So we need to split them in BNS, NSBH and BBH. Run the following command.
**[Petrov-distribution]** are already in BNS, NSBH and BBH. So the script will just copy `injections.dat` files
and create a new directory containg what we and with the same name.

	python split_farah_populations.py

 IN `GW_EM_joint/EMdata` you normally get and new folders names `Petrov_subpops` and `Farah_subpops` inside which the `runs` folder contains the subpopulations `farah_bns`, `farah_nsbh` and `farah_bbh` of each , Run `O3`, `O4` and `O5`.



## **Generate a `simulation set`**

Here we will want to only works with `BNS` in `Run O4` and **[Farah-distribution]**.
You could Replace `O4` by `O3` or `O5` according to your wishes.

`Note` that for **[Petrov-distribution]** replace `Farah_subpops/O4/bns_farah` by `Petrov_subpops/O4/bns_astro`.


1. ###  **Create injection file**

	Please create an `output` name in `./GW_EM_joint/EMdata` relative to the distribution you want to use. Here we used **[Farah-distribution]** so we will create this direction.

		mkdir -p ./GW_EM_joint/EMdata/farah_lc

	So for whom want to used  **[Petrov-distribution]** please create this directory

		mkdir -p ./GW_EM_joint/EMdata/petrov_lc

	Running the following command  (`BNS` or `NSBH`) line will generate a json file (injection.json)  with the BILBY processing of compact binary merging events. We take here binaries of type BNS, NSBH is also an option. This injection contents a simulation set of parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, KNtimeshift, geocent_time for the Bu2019lm model. This creates an `Bu2019lm_injection.json` for **BNS**  or `Bu2019nsbh_injection.jon` for **NSBH** type in the `./GW_EM_joint/EMdata/farah_lc/outdir_BNS` or `/GW_EM_joint/EMdata/farah_lc/outdir_NSBH` (for **NSBH** type ) directory.
extension json. In all the following, you have to replace `farah_lc` by `petrov_lc` when you work with **[Petrov-distribution]**.

	Don't forget to complete your **[nmma]** directory in where need it, for example by `~/..../nma/...` or `/home/..../nmma/...`

	* **`BNS` type**

			nmma_create_injection --prior-file ./nmma/priors/Bu2019lm.prior --injection-file ./GW_EM_joint/Farah_subpops/runs/O4/bns_farah/injections.dat --eos-file  ./nmma/example_files/eos/ALF2.dat --binary-type BNS --n-injection 2500 --original-parameters --extension json --aligned-spin --binary-type BNS -f ./GW_EM_joint/EMdata/farah_lc/outdir_BNS/Bu2019lm_injection

	* **`NSBH` type**

			nmma_create_injection --prior-file ./nmma/priors/Bu2019nsbh.prior --injection-file ./GW_EM_join/Farah_subpops/runs/O4/nsbh_farah/injections.dat --eos-file  ./nmma/example_files/eos/ALF2.dat --binary-type NSBH --n-injection 2500 --original-parameters --extension json --aligned-spin -f ./GW_EM_joint/EMdata/farah_lc/outdir_NSBH/Bu2019nsbh_injection


2. ### **Generate `lightcurve posteriors` using condor process to submit jobs**


	Here, to get `lightcurve posteriors`  you  just run the followinga command lines. './GW_EM_joint/EMdata/farah_lc/outdir_BNS/BNS' or './GW_EM_joint/EMdata/farah_lc/utdir_NSBH/NSBH' according to the type of `binary` population,  will house the posteriors of the electromagnetic data you will produce: in particular the, where with injection about `25,000` injections for `ZTF` ligthcurve and '12,500' for 'Rubin' lightcurve. We now compute posteriors using **[nmma]** on this simulated set of  events, of which we assume a fraction is detectable by `ZTF` or `Rubin`. The result can be find at  `/GW_EM_jointEMdata/farah_lc/outdir_BNS/BNS` or `./GW_EM_joint/EMdata/farah_lc/outdir_NSBH` when you use 'NSBH'.


	## **For `ZTF` telescope**

	Create just a folder to run the condor process `light_curve_analysis_condor.....` , just because condor files create alot of files,

	Feel free to create another folder if useful but make sure  to check out the right direcories of your data.

	* **`BNS` type**

			light_curve_analysis_condor --model Bu2019lm --prior  ./nmma/priors/Bu2019lm.prior --svd-path   ./nmma/svdmodels --outdir ./GW_EM_joint/EMdata/farah_lc/outdir_BNS --label injection_Bu2019lm --injection ./GW_EM_joint/EMdata/farah_lc/outdir_BNS/Bu2019lm_injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh

	then submit the job

		condor_submit_dag -f condor.dag

	* **`NSBH` type**

			light_curve_analysis_condor --model Bu2019nsbh --prior ./nmma/priors/Bu2019nsbh.prior --svd-path  ./nmma/svdmodels --outdir ./GW_EM_joint/EMdata/farah_lc/outdir_NSBH --label injection_Bu2019nsbh --injection ./GW_EM_joint/EMdata/farah_lc/outdir_BNS/Bu2019lnsbh_injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh

	then submit the job

		condor_submit_dag -f condor.dag


	## **For `Rubin` telescope**

	Use the same command as `ZTF` case, but remove `--ztf-ToO 180 300`


	* **`BNS` type**

			light_curve_analysis_condor --model Bu2019lm --prior  ./nmma/priors/Bu2019lm.prior --svd-path   ./nmma/svdmodels --outdir ./GW_EM_joint/farah_lc/outdir_BNS --label injection_Bu2019lm --injection ./GW_EM_joint/EMdata/farah_lc/outdir_BNS/Bu2019lm_injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh

	then submit the job

		condor_submit_dag -f condor.dag

	to see the evolution of condor process :

		condor_q

	to observe the process in realtime

		tail -f condor.dag.dagman.out


	* **`NSBH` type**

			light_curve_analysis_condor --model Bu2019nsbh --prior ./nmma/priors/Bu2019nsbh.prior --svd-path  ./nmma/svdmodels --outdir outdir_NSBH --label injection --injection ../GW_EM_join/EMdata/farah_lc/outdir_BNS/Bu2019lnsbh_injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh

	then submit the job

		condor_submit_dag -f condor.dag


# GW posteriors
















[nmma]: https://github.com/nuclear-multimessenger-astronomy/nmma
[Farah-distribution]: https://doi.org/10.3847/1538-4357/ac5f03
[Petrov-distribution]: http://dx.doi.org/10.3847/1538-4357/ac366d
[Farah data]: https://zenodo.org/record/7026209
