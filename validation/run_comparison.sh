# This script comapares
# (1) ESA Cloud_cci L3C variables to CMSAF CLAAS3
# (2) ESA Cloud_cci L3C SEVIRI to SLSTR
#
# Daniel Philipp (DWD), 2021
#

YEAR=2019
PROC=1

CCI_BASE=/cmsaf/cloud_cci/esa_cloud_cci/data/cloud_cci+phase1/demo_v1
OUTPUT_PATH_CCI_CLAAS=/cmsaf/cmsaf-cld1/dphilipp/compare_cci_claas3/proc${PROC}_SEVIRI_CLAAS
OUTPUT_PATH_SEVIRI_SLSTR=/cmsaf/cmsaf-cld1/dphilipp/compare_cci_claas3/proc${PROC}_SEVIRI_SLSTR

mkdir -p $OUTPUT_PATH_CCI_CLAAS
mkdir -p $OUTPUT_PATH_SEVIRI_SLSTR

for MONTH in 02 07; do

    CCI_SEVIRI_L3C=${CCI_BASE}/SEVIRI/${YEAR}${MONTH}/averages/${YEAR}${MONTH}-ESACCI-L3C_CLOUD-CLD_PRODUCTS-SEVIRI_MSG-fv1.0.nc
    CCI_S3A_L3C=${CCI_BASE}/SLSTR/${YEAR}${MONTH}/${YEAR}${MONTH}-ESACCI-L3C_CLOUD-CLD_PRODUCTS-SLSTR_Sentinel-3a-fv3.1.nc
    CCI_S3B_L3C=${CCI_BASE}/SLSTR/${YEAR}${MONTH}/${YEAR}${MONTH}-ESACCI-L3C_CLOUD-CLD_PRODUCTS-SLSTR_Sentinel-3b-fv3.1.nc

    CLAAS3_L3=/cmsaf/cmsaf-cld10/SEVIRI/TCDR_edition3/level3

    ####################### COMPARE AGAINST CLAAS3 #######################
    python3 make_comparison_cci_claas.py --seviri_l3c ${CCI_SEVIRI_L3C} \
                                         --claas3_l3 ${CLAAS3_L3} \
                                         --opath ${OUTPUT_PATH_CCI_CLAAS} \
                                         --year ${YEAR} \
                                         --month ${MONTH}

    ####################### COMPARE SEVIRI AGAINST SLSTR #######################
    python3 make_comparison_seviri_slstr.py --seviri_l3c ${CCI_SEVIRI_L3C} \
                                            --s3a_l3c ${CCI_S3A_L3C} \
                                            --s3b_l3c ${CCI_S3B_L3C} \
                                            --opath ${OUTPUT_PATH_SEVIRI_SLSTR} \
                                            --year ${YEAR} \
                                            --month ${MONTH}
done
