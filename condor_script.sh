# Here I use 2500 injections and 5 seeds so that means
# for ZTF : 2500*len(EXposure)*len(seed_list) = 25000 jobs
# for Rubin :2500*len(seed_list) = 12500 jobs
# Exposure is ZTF_ToO = [180, 300]

# People could increase the number of injections by replace '--n-injection 2500'
# in nmma_create_injection command line arguments by the number that they want.
# then do the same at the --injection-num in  'light_curve_analysis_condor'
# command line arguments.

#============================================
#BNS light_curve_analysis
#============================================

# create injection file with bilby

nmma_create_injection --prior-file ./nmma/priors/Bu2019lm.prior --injection-file ./Farah_data/runs/O4/bns_farah/injections.dat --eos-file  ./nmma/example_files/eos/ALF2.dat --binary-type BNS --n-injection 2500 --original-parameters --extension json --aligned-spin -f ./outdir_BNS/injection_Bu2019lm_O4

#============================================
# ZTF light_curve_analysis
#============================================

## Run ZTF sample , if --ztf-ToO , that use automatically all ZTF samples ,
# like --remove-nondetections --ztf-sampling --ztf-uncertainties --ztf-ToO 300

light_curve_analysis_condor --model Bu2019lm --prior ./nmma/priors/Bu2019lm.prior --svd-path  ./nmma/svdmodels --outdir outdir_BNS --label injection_Bu2019lm_O4 --injection ./outdir_BNS/injection_Bu2019lm_O4.json --injection-num 2500 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh


########## Then Run condor process
condor_submit_dag -f condor.dag


#============================================
# Rubin light_curve_analysis
#============================================

# Here without  --ztf-ToO argument , that means use Rubin sampling

light_curve_analysis_condor --model Bu2019lm --prior  ./nmma/priors/Bu2019lm.prior --svd-path   ./nmma/svdmodels --outdir outdir_BNS --label injection_Bu2019lm_O4 --injection ./outdir_BNS/injection_Bu2019lm_O4. --injection-num 2500 --generation-seed 816 323 364 564 851 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh


#============================================
# NSBHlight_curve_analysis
#============================================

# For NSBH jsut replace by the right arguments



nmma_create_injection --prior-file ./nmma/priors/Bu2019nsbh.prior --injection-file ./Farah_data/runs/O4/nsbh_farah/injections.dat --eos-file  ./nmma/example_files/eos/ALF2.dat --binary-type NSBH --n-injection 2500 --original-parameters --extension json --aligned-spin -f ./outdir_NSBH/injection_Bu2019nsbh_O4

# ZTF

light_curve_analysis_condor --model Bu2019nsbh --prior  ./nmma/priors/Bu2019nsbh.prior --svd-path ./nmma/svdmodels --outdir outdir_NSBH --label injection_Bu2019nsbh_O4 --injection ./outdir_NSBH/injection_Bu2019nsbh_O4.json --injection-num 5000 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh


## Then the same for Rubin .............................

# Note that this condor process is only available for ZTF and Rubin files ,
# So For non-ZTF or non-Rubin files we need to remove this arguments directly on
# https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/analysis_condor.py
