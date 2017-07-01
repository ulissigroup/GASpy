import copy
from collections import OrderedDict
from calc_settings import calc_settings


def parameter_bulk(mpid, settings='beef-vdw', encutBulk=500.):
    ''' Generate some default parameters for a bulk and expected relaxation settings '''
    if isinstance(settings, str):
        settings = calc_settings(settings)
    # We're getting a handle to a dictionary, so need to copy before modifying
    settings = copy.deepcopy(settings)
    settings['encut'] = encutBulk
    return OrderedDict(mpid=mpid,
                       relaxed=True,
                       max_atoms=50,
                       vasp_settings=OrderedDict(ibrion=1,
                                                 nsw=100,
                                                 isif=7,
                                                 isym=0,
                                                 ediff=1e-8,
                                                 kpts=[10, 10, 10],
                                                 prec='Accurate',
                                                 **settings))
