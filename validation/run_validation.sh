##############################
LWP_SOURCE=nsidc
YEAR=2019
MONTH=07
PROC=proc1
############################

# PATH TO CALIPSO MATCHUP FILE
CALIPSO_PATH=/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files_merged_calipso/${PROC}/${YEAR}/${MONTH}/
CALIPSO_FILE=5km_msg_${YEAR}${MONTH}_0000_99999_calipso_avhrr_match_merged_final.h5

# PATH TO AMSR2 MATCHUP FILE
AMSR2_PATH=/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files_merged_amsr2_${LWP_SOURCE}/${PROC}/${YEAR}/${MONTH}/
AMSR2_FILE=5km_msg_${YEAR}${MONTH}_0000_99999_amsr_avhrr_match_merged_final.h5

# PATH TO OUPUT FILE FOR STATISTICS
OUTPUT_PATH=/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/VALIDATION_RESULTS/${PROC}/${MONTH}${YEAR}/
OUTPUT_FILE=${PROC}_CFC_CPH_CTX_LWP-${LWP_SOURCE}_validation.txt

if [[ ! -d $OUTPUT_PATH ]]; then
    mkdir -p $OUTPUT_PATH
fi

# VALIDATE UNCERTAINTIES
#ipython3 -W ignore validate_uncertainties.py \
#         --matchup_file_calipso ${CALIPSO_PATH}${CALIPSO_FILE} \
#         --output_dir ${OUTPUT_PATH}/uncertainty

# VALIDATE VARIABLES
python3 -W ignore validate_all.py \
	--matchup_file_calipso ${CALIPSO_PATH}${CALIPSO_FILE} \
	--matchup_file_amsr ${AMSR2_PATH}${AMSR2_FILE} \
	--cot_limits 0.0 0.15 1.0 \
	--satz_limit 70 \
	--output_txt_file ${OUTPUT_PATH}${OUTPUT_FILE}

#python3 -W ignore validate_all.py \
#        --matchup_file_calipso ${CALIPSO_PATH}${CALIPSO_FILE} \
#        --cot_limits 0.0 0.15 1.0 \
#        --satz_limit 70 \
#        --output_txt_file ${OUTPUT_PATH}${OUTPUT_FILE}
##################################################################################
##################################################################################

echo " "
echo " "
echo " "

YEAR=2019
MONTH=02
PROC=proc1

# PATH TO DARDAR MATCHUP FILE
DARDAR_PATH=/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files_merged_dardar/${PROC}/${YEAR}/${MONTH}/
DARDAR_FILE=5km_msg_${YEAR}${MONTH}_0000_99999_dardar_avhrr_match_merged_final.h5

# PATH TO OUPUT FILE FOR STATISTICS
OUTPUT_PATH=/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/VALIDATION_RESULTS/${PROC}/${MONTH}${YEAR}/
OUTPUT_FILE=${PROC}_CER-DARDAR_IWP-DARDAR_validation.txt

# VALIDATE VARIABLES
python3 -W ignore validate_all.py \
        --matchup_file_dardar ${DARDAR_PATH}${DARDAR_FILE} \
        --cot_limits 0.0 0.15 1.0 \
        --satz_limit 70 \
        --output_txt_file ${OUTPUT_PATH}${OUTPUT_FILE}
