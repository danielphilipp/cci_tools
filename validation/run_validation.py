import argparse
import os
import validate_cfc as cfc_val
import validate_cth as cth_val
import validate_cph as cph_val
import validate_lwp as lwp_val
from datetime import datetime as time
from atrain_plot.atrain_plot import run as aprun

def show_and_save_results(cfc_results, cph_results, cth_results, lwp_results, args):
    s = '\n#############################################\n' + \
        '############### CFC VALIDATION ##############\n' + \
        '#############################################\n' + \
        cfc_results + \
        '\n#############################################\n' + \
        '############### CPH VALIDATION ##############\n' + \
        '#############################################\n' + \
        cph_results + \
        '\n#############################################\n' + \
        '############### CTH VALIDATION ##############\n' + \
        '#############################################\n' + \
        cth_results + \
        '\n#############################################\n' + \
        '############### LWP VALIDATION ##############\n' + \
        '#############################################\n' + \
        lwp_results + \
        '\n#############################################\n'

    print(s)

    if os.path.isfile(args.output_txt_file):
        os.remove(args.output_txt_file)

    with open(args.output_txt_file, 'w') as fh:
        fh.write(s)

if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mc', '--matchup_file_calipso', type=str, required=True)
    parser.add_argument('-mf', '--matchup_file_amsr2', type=str, required=True)
    parser.add_argument('-cl', '--cot_limits', required=True, nargs='+')
    parser.add_argument('-sl', '--satz_limit', required=True,  type=float)
    parser.add_argument('-tf', '--output_txt_file', required=True, type=str)
    args = parser.parse_args()

    cot_limits = [float(j) for j in args.cot_limits]

    # print('\nVALIDATING CFC')
    # start = time.now()
    # cfc_results = cfc_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    # print('Runtime was {}\n'.format(time.now()-start))
    #
    # print('\nVALIDATING CPH')
    # start = time.now()
    # cph_results = cph_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    # print('Runtime was {}\n'.format(time.now()-start))
    #
    # print('\nVALIDATING CTH')
    # start = time.now()
    # cth_results = cth_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    # print('Runtime was {}\n'.format(time.now()-start))
    #
    # print('\nVALIDATING LWP')
    # start = time.now()
    # lwp_results = lwp_val.run_validation(args.matchup_file_amsr2, args.output_txt_file)
    # print('Runtime was {}\n'.format(time.now()-start))
    #
    # show_and_save_results(cfc_results, cph_results, cth_results, lwp_results, args)

    if '201907' in args.matchup_file_calipso:
        year = '2019'
        month = '07'
    elif '201902' in args.matchup_file_calipso:
        year = '2019'
        month = '02'

    print('\nRUNNING ATRAIN_PLOT')
    start = time.now()
    aprun(
        ifilepath_calipso=args.matchup_file_calipso,
        opath=os.path.dirname(args.output_txt_file),
        dnts=['ALL'],#, 'DAY', 'NIGHT', 'TWILIGHT'],
        satzs=[None],#, 70],
        year=year,
        month=month,
        dataset='CCI',
        ifilepath_amsr=args.matchup_file_amsr2,
        chunksize=100000,
        plot_area='pc_world',
        FILTER_STRATOSPHERIC_CLOUDS=False
    )
    print('Runtime was {}\n'.format(time.now() - start))

    print('\n\n*************** VALIDATION SUCCESSFUL ***************\n\n')
