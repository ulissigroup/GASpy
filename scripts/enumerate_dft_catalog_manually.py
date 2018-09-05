'''
This script bypasses Luigi to enumerate our catalog of sites. We bypass
Luigi's management because we know we're going to have millions of sites
and therefore multiples of millions of dependencies, and when this happens,
Luigi spends more time handling overhead than it does actually running things.

So we use this script to enumerate faster at the risk of doing redundant jobs.
We're ok with the risk because we don't plan to enumerate sites often.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__email__ = 'ktran@andrew.cmu.edu'

import sys
import warnings
import multiprocess as mp
from gaspy.tasks import EnumerateAlloys
from gaspy.utils import evaluate_luigi_task
import tqdm


whitelist = ['Pd', 'Cu', 'Au', 'Ag', 'Pt', 'Rh', 'Re', 'Ni', 'Co', 'Ir',
             'W', 'Al', 'Ga', 'In', 'H', 'N', 'Os', 'Fe', 'V', 'Si', 'Sn',
             'Sb', 'Mo', 'Mn', 'Cr', 'Ti', 'Zn', 'Ge', 'As', 'Ru', 'Pb',
             'Nb', 'Ca', 'Na', 'S', 'C', 'Cd', 'K']
alloy_enumerator = EnumerateAlloys(whitelist=whitelist, max_to_submit=100000, max_index=2, dft=True)
tasks = list(alloy_enumerator.requires())


def evaluate_luigi_task_and_bypass_errors(task):
    '''
    Many of the surface enumerations will fail for various reasons.
    We don't want to stop when that happens.
    This function will catch these errors and move on.

    Arg:
        task    Instance of a Luigi task
    '''
    try:
        # Ignore some ASE/pymatgen warnings that'll spam our output
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            evaluate_luigi_task(task, force=False)

    # Bypass the errors, but still report them
    except:     # noqa: E722
        print('Problem with task:\n    %s' % str(task))
        print('This is the error:')
        error_type, error, traceback = sys.exc_info()
        sys.excepthook(error_type, error, traceback)


with mp.Pool(32) as pool:
    multithreaded_iterator = pool.imap(evaluate_luigi_task_and_bypass_errors, tasks, chunksize=10)
    _ = list(tqdm.tqdm(multithreaded_iterator, total=len(tasks)))
