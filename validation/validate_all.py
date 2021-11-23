import argparse
import os
import validate_cfc as cfc_val
import validate_cth as cth_val
import validate_cph as cph_val
import validate_lwp as lwp_val
import validate_iwp as iwp_val
import validate_cer as cer_val
from datetime import datetime as time
from atrain_plot.atrain_plot import run as aprun


class Results:
    def __init__(self):
        """ Initiate results class. """
        self.cfc_results = 'Not validated!'
        self.cph_results = 'Not validated!'
        self.cth_results = 'Not validated!'
        self.lwp_results = 'Not validated!'
        self.iwp_results = 'Not validated!'
        self.cer_results = 'Not validated!'


def _show_and_save_results(results, args):
    """ Print scores and write them to file. """

    s = '\n#############################################\n' + \
        '############### CFC VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.cfc_results + \
        '\n#############################################\n' + \
        '############### CPH VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.cph_results + \
        '\n#############################################\n' + \
        '############### CTH VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.cth_results + \
        '\n#############################################\n' + \
        '############### LWP VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.lwp_results + \
        '\n#############################################\n' + \
        '############### IWP VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.iwp_results + \
        '\n#############################################\n' + \
        '############### CER VALIDATION ##############\n' + \
        '#############################################\n' + \
        results.cer_results + \
        '\n#############################################\n'

    print(s)

    if os.path.isfile(args.output_txt_file):
        os.remove(args.output_txt_file)

    with open(args.output_txt_file, 'w') as fh:
        fh.write(s)

if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mc', '--matchup_file_calipso', type=str, default=None)
    parser.add_argument('-mf', '--matchup_file_amsr2', type=str, default=None)
    parser.add_argument('-md', '--matchup_file_dardar', type=str, default=None)
    parser.add_argument('-cl', '--cot_limits', required=True, nargs='+')
    parser.add_argument('-sl', '--satz_limit', required=True,  type=float)
    parser.add_argument('-tf', '--output_txt_file', required=True, type=str)
    args = parser.parse_args()

    # convert from str to float
    cot_limits = [float(j) for j in args.cot_limits]
    # initiate results  class
    results = Results()

    # ----------------------------------- VALIDATE CFC --------------------------------------
    print('\nVALIDATING CFC')
    start = time.now()
    cfc_results = cfc_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    if cfc_results is not None:
        results.cfc_results = cfc_results
        print('Runtime was {}\n'.format(time.now() - start))
    else:
        print('No CALIOP matchup file. Skipping.')

    # ----------------------------------- VALIDATE CPH --------------------------------------
    print('\nVALIDATING CPH')
    start = time.now()
    cph_results = cph_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    if cph_results is not None:
        results.cph_results = cph_results
        print('Runtime was {}\n'.format(time.now()-start))
    else:
        print('No CALIOP matchup file. Skipping.')

    # ----------------------------------- VALIDATE CTH --------------------------------------
    print('\nVALIDATING CTH')
    start = time.now()
    cth_results = cth_val.run_validation(args.matchup_file_calipso, cot_limits, float(args.satz_limit))
    if cth_results is not None:
        results.cth_results = cth_results
        print('Runtime was {}\n'.format(time.now()-start))
    else:
        print('No CALIOP matchup file. Skipping.')

    # ----------------------------------- VALIDATE LWP --------------------------------------
    print('\nVALIDATING LWP')
    start = time.now()
    lwp_results = lwp_val.run_validation(args.matchup_file_amsr2, args.output_txt_file)
    if lwp_results is not None:
        results.lwp_results = lwp_results
        print('Runtime was {}\n'.format(time.now()-start))
    else:
        print('No AMSR matchup file. Skipping.')

    # ----------------------------------- VALIDATE IWP --------------------------------------
    print('\nVALIDATING IWP')
    start = time.now()
    iwp_results = iwp_val.run_validation(args.matchup_file_dardar, args.output_txt_file)
    if iwp_results is not None:
        results.iwp_results = iwp_results
        print('Runtime was {}\n'.format(time.now()-start))
    else:
        print('No DARDAR matchup file. Skipping.')

    # ----------------------------------- VALIDATE CER --------------------------------------
    print('\nVALIDATING CER')
    start = time.now()
    cer_results = cer_val.run_validation(args.matchup_file_dardar, args.output_txt_file)
    if cer_results is not None:
        results.cer_results = cer_results
        print('Runtime was {}\n'.format(time.now()-start))
    else:
        print('No DARDAR matchup file. Skipping.')

    _show_and_save_results(results, args)

    # select non-None matchup file for mm and yyyy string determination
    if args.matchup_file_calipso is not None:
        pat_search = args.matchup_file_calipso
    else:
        if args.matchup_file_amsr2 is not None:
            pat_search = args.matchup_file_amsr2
        else:
            if args.matchup_file_dardar is not None:
                pat_search = args.matchup_file_dardar
            else:
                raise Exception('All matchup files are None. Aborting!')

    # determine yyyy and mm string from matchup file
    if '201907' in pat_search:
        year = '2019'
        month = '07'
    elif '201902' in pat_search:
        year = '2019'
        month = '02'

    print('\nRUNNING ATRAIN_PLOT')
    start = time.now()
    aprun(
        opath=os.path.dirname(args.output_txt_file),
        year=year,
        month=month,
        dataset='CCI',
        dnts=[],#['ALL', 'DAY', 'NIGHT', 'TWILIGHT'],
        satzs=[],#[None],
        ifilepath_calipso=None,#args.matchup_file_calipso,
        ifilepath_amsr=args.matchup_file_amsr2,
        ifilepath_dardar=None,#args.matchup_file_dardar,
        chunksize=100000,
        plot_area='pc_world',
        FILTER_STRATOSPHERIC_CLOUDS=False
    )
    print('Runtime was {}\n'.format(time.now() - start))
    print('\n\n*************** VALIDATION SUCCESSFUL ***************\n\n')
